"""
Same pipeline as build_vi_dataset.py but targets OpenRouter instead of NIM
(OpenRouter has far more generous rate limits on a funded key, and cost for
meta-llama/llama-3.1-8b-instruct is negligible: ~$0.0164 per 1000 questions).

Requires: OPENROUTER_API_KEY environment variable.
"""
import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# Chat-completions endpoint. Overridable so the same caller can talk to a
# self-hosted vLLM server instead of OpenRouter: vLLM exposes the identical
# OpenAI-compatible shape, so nothing below changes. Self-hosting matters
# because Qwen3-8B is open-weights -- renting it per token is ~68% of our bill
# for a model a single GPU serves for the price of the GPU.
#   export TECQA_LLM_BASE_URL=https://<pod>-8000.proxy.runpod.net/v1/chat/completions
#   export TECQA_LLM_MODEL=qwen/qwen3-8b     # ONLY this model goes to the pod
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SELF_HOSTED_URL = os.environ.get("TECQA_LLM_BASE_URL", "")
# A vLLM server serves exactly one model. TECQA's pipeline calls two -- Stage 1
# parses with a hosted model, Stage 3 reasons with the open-weights one -- so
# naming which model is self-hosted is what keeps Stage 1 on OpenRouter. Leave
# it unset and every call goes to the pod, which is the old all-or-nothing
# behaviour and 404s Stage 1.
SELF_HOSTED_MODEL = os.environ.get("TECQA_LLM_MODEL", "")
# Anything sent to the pod must NOT carry the OpenRouter key: the pod is a
# rented box we do not control, and vLLM ignores the header anyway.
SELF_HOSTED_KEY = "self-hosted"
OR_BASE_URL = SELF_HOSTED_URL or OPENROUTER_URL


def endpoint_for(model: str) -> tuple:
    """(url, api_key_or_None) for one model id. None means "use the caller's
    OpenRouter key". Routing by model, not by a global switch, is what lets one
    run split its stages across two providers."""
    if SELF_HOSTED_URL and (not SELF_HOSTED_MODEL or model == SELF_HOSTED_MODEL):
        return SELF_HOSTED_URL, SELF_HOSTED_KEY
    return OPENROUTER_URL, None
RAW_DIR = Path(__file__).parent / "raw" / "extracted" / "MultiTQ"
OUT_DIR = Path(__file__).parent / "vi"

# USD per token, from OpenRouter's public /api/v1/models pricing (checked live,
# not hardcoded blindly) — used to compute REAL spend from each response's
# actual token usage, not estimates.
PRICING_USD_PER_TOKEN = {
    "meta-llama/llama-3.1-8b-instruct": {"prompt": 0.00000005, "completion": 0.00000008},
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.0000001, "completion": 0.00000032},
    "qwen/qwen-2.5-72b-instruct": {"prompt": 0.00000036, "completion": 0.0000004},
    "deepseek/deepseek-v3.2": {"prompt": 0.000000269, "completion": 0.0000004},
    "anthropic/claude-haiku-4.5": {"prompt": 0.000001, "completion": 0.000005},
    "deepseek/deepseek-v4-flash-0731": {"prompt": 0.00000009, "completion": 0.00000018},
}
# safety fallback for any model not in the table above: assume an expensive
# per-token rate so the hard budget stop still kicks in reliably.
FALLBACK_PRICING = {"prompt": 0.000003, "completion": 0.00001}


# Fraction of hard_cap_usd at which an ntfy.sh push fires (once per threshold
# per process) so a long-running VPS job doesn't silently hit the hard stop
# with nobody watching. NTFY_TOPIC env var must be set; if unset, alerts are
# skipped (no hard dependency on the alert channel to run the job).
BUDGET_ALERT_THRESHOLDS = (0.8, 0.95)


class BudgetTracker:
    """Thread-safe running cost tracker with a hard stop + near-cap alerts."""

    def __init__(self, hard_cap_usd: float):
        self.hard_cap_usd = hard_cap_usd
        self._lock = threading.Lock()
        self.total_usd = 0.0
        self.stopped = False
        self._alerted = set()

    def _send_alert(self, frac: float):
        topic = os.environ.get("NTFY_TOPIC")
        if not topic:
            return
        try:
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=f"Cron-VI translate job: đã dùng {frac*100:.0f}% ngân sách "
                     f"(${self.total_usd:.2f}/${self.hard_cap_usd:.2f}).".encode("utf-8"),
                headers={"Title": "Budget alert", "Priority": "high"},
                timeout=5,
            )
        except Exception:
            pass  # alert delivery is best-effort, never blocks translation

    def add(self, model: str, prompt_tokens: int, completion_tokens: int, real_cost: float = None) -> float:
        # Prefer OpenRouter's own authoritative usage.cost from the response --
        # our static PRICING_USD_PER_TOKEN table was measured to undercount
        # real cost by ~35% for deepseek-v4-flash-0731 (reasoning tokens billed
        # into completion_tokens at a rate our table didn't capture), which let
        # the hard cap overshoot by $0.258 on a $2.0 budget before it tripped.
        if real_cost is not None:
            cost = real_cost
        else:
            pricing = PRICING_USD_PER_TOKEN.get(model, FALLBACK_PRICING)
            cost = prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]
        with self._lock:
            self.total_usd += cost
            if self.total_usd >= self.hard_cap_usd:
                self.stopped = True
            frac_now = self.total_usd / self.hard_cap_usd
            to_fire = [t for t in BUDGET_ALERT_THRESHOLDS if frac_now >= t and t not in self._alerted]
            for t in to_fire:
                self._alerted.add(t)
            total = self.total_usd
        for t in to_fire:
            self._send_alert(t)
        return total

    def over_budget(self) -> bool:
        with self._lock:
            return self.stopped

VN_CHARS = re.compile(r"[àáâãèéêìíòóôõùúăđĩũơưẠ-ỹ]", re.IGNORECASE)
LEAK_MARKERS = ("dịch sang", "chú thích", "tôi không thể", "i cannot", "as an ai", "i'm sorry")
# common English function words that should never survive translation (word-boundary match).
# Proper nouns (people/orgs/places) are fine to keep in English — this only flags
# grammatical leftovers like "the", "who", "leader", "which", "of", "with" appearing
# mid-sentence in what's supposed to be a Vietnamese sentence.
ENGLISH_LEFTOVER = re.compile(
    r"\b(the|who|whom|when|which|leader|minister|president|before|after|first|last|"
    r"with|from|of|and|country|countries|government)\b",
    re.IGNORECASE,
)


def has_question_mark(en_question: str, vi_question: str) -> bool:
    # If the English source is interrogative, the Vietnamese translation must be too.
    if en_question.strip().endswith("?") and not vi_question.strip().endswith("?"):
        return False
    return True

SYSTEM_TRANSLATE = (
    "Ban la bien dich vien Anh-Viet chuyen nghiep cho bo du lieu hoi-dap tren do thi tri "
    "thuc thoi gian (Temporal KGQA). Dich CAU HOI sang tieng Viet co dau day du, tu nhien, "
    "chinh xac ve mat logic thoi gian (before/after/first/last...). "
    "QUAN TRONG: GIU NGUYEN moi ten rieng (nguoi, to chuc, dia danh) y het chu Anh goc, "
    "khong dich hay phien am ten rieng. Giu nguyen dinh dang ngay thang. "
    "TUYET DOI KHONG duoc them loi dan, chu thich, hay nhan xet nao ve viec dich; "
    "khong duoc tu choi du cau hoi nghe la hay nhay cam. "
    "CHI xuat ra DUNG MOT cau tieng Viet da dich, khong co gi khac."
)


# A reasoning model emits its scratchpad before its answer. OpenRouter returns
# that scratchpad in a separate `reasoning` field, so `content` is already just
# the answer; a plain vLLM server inlines it as <think>...</think> at the front
# of `content` instead. Callers parse `content`, so without this the two servers
# disagree about what a "response" even is -- measured on 129 pod answers, 107
# had a perfectly good answer sitting after </think> that the Stage-3 parser
# never saw. Text with no closing tag ran out of tokens mid-thought and holds no
# answer at all; returning "" lets llm_cache's is_valid retry it instead of
# caching a truncated deliberation forever.
THINK_CLOSE = "</think>"


def strip_reasoning(content: str) -> str:
    """The model's answer, with any inlined <think> block removed."""
    if THINK_CLOSE not in content:
        return "" if content.lstrip().startswith("<think>") else content
    return content.split(THINK_CLOSE)[-1].strip()


def call_or(api_key, model, prompt, system="", temperature=0.0, retries=3, timeout=30, budget=None,
            max_tokens=1024, extra_params=None):
    url, override_key = endpoint_for(model)
    headers = {"Authorization": f"Bearer {override_key or api_key}",
               "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if extra_params:
        payload.update(extra_params)
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            if budget is not None:
                budget.add(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                           real_cost=usage.get("cost"))
            return strip_reasoning(data["choices"][0]["message"]["content"].strip())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)
    raise RuntimeError("unreachable")


def clean_translation(text: str) -> str:
    text = text.strip()
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        text = parts[-1]
    elif "\n" in text:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        text = parts[-1]
    text = re.sub(r"^(câu hỏi\s*(đã\s*)?dịch\s*sang\s*tiếng\s*việt\s*(là)?\s*:?\s*)", "", text, flags=re.IGNORECASE)
    return text.strip()


def is_good_translation(text: str, en_question: str = "") -> bool:
    if not text:
        return False
    low = text.lower()
    if any(m in low for m in LEAK_MARKERS):
        return False
    if not VN_CHARS.search(text):
        return False
    if ENGLISH_LEFTOVER.search(text):
        return False
    if en_question and not has_question_mark(en_question, text):
        return False
    return True


def translate_relations(api_key, model, budget):
    cache_path = OUT_DIR / "relations_vi.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    rel_ids = json.loads((RAW_DIR / "kg" / "relation2id.json").read_text(encoding="utf-8"))
    relations = list(rel_ids.keys())
    print(f"Translating {len(relations)} unique relations...")
    system = (
        "Ban la chuyen gia dich thuat Anh-Viet cho co so tri thuc su kien chinh tri/xa hoi "
        "(theo phong cach CAMEO/ICEWS). Dich cum quan he sang tieng Viet ngan gon, tu nhien, "
        "giu dung sac thai hanh dong. CHI tra ve ban dich, khong giai thich them."
    )
    rel_vi = {}
    batch_size = 20
    for i in range(0, len(relations), batch_size):
        if budget.over_budget():
            print(f"  !! BUDGET CAP HIT during relation translation (${budget.total_usd:.4f}) — stopping early.")
            break
        batch = relations[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {r.replace('_', ' ')}" for j, r in enumerate(batch))
        prompt = "Dich cac cum tu quan he sau sang tieng Viet, giu nguyen thu tu so, moi dong mot ban dich:\n\n" + numbered
        out = call_or(api_key, model, prompt, system, budget=budget)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        for j, r in enumerate(batch):
            vi = lines[j] if j < len(lines) else r.replace("_", " ")
            vi = vi.split(".", 1)[-1].strip() if vi[:1].isdigit() else vi
            rel_vi[r] = vi
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rel_vi, ensure_ascii=False, indent=2), encoding="utf-8")
    return rel_vi


def translate_one(api_key, model, q, budget):
    if budget.over_budget():
        return None
    for attempt in range(3):
        if budget.over_budget():
            return None
        temperature = 0.0 if attempt == 0 else 0.3 + attempt * 0.15
        try:
            raw = call_or(api_key, model, f"Dich cau hoi sau sang tieng Viet:\n\n{q['question']}",
                          SYSTEM_TRANSLATE, temperature=temperature, retries=2, timeout=30, budget=budget)
        except Exception:
            time.sleep(1)
            continue
        vi = clean_translation(raw)
        if is_good_translation(vi, q["question"]):
            item = dict(q)
            item["question_en"] = q["question"]
            item["question"] = vi
            return item
    return None


def translate_questions_parallel(api_key, model, targets, workers, save_every, questions_path, existing_map, budget):
    done_count = 0
    dropped = 0
    budget_stopped_at = None
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(translate_one, api_key, model, q, budget): q["quid"] for q in targets}
        for fut in as_completed(futures):
            quid = futures[fut]
            try:
                item = fut.result()
            except Exception:
                item = None
            if budget.over_budget() and budget_stopped_at is None:
                budget_stopped_at = done_count + dropped
                print(f"  !! HARD BUDGET CAP (${budget.hard_cap_usd:.2f}) REACHED at ${budget.total_usd:.4f} "
                      f"after {budget_stopped_at} questions. Cancelling remaining work and saving now.")
                for f2 in futures:
                    f2.cancel()
            if item:
                existing_map[quid] = item
                done_count += 1
            else:
                dropped += 1
            total_done = done_count + dropped
            if total_done % save_every == 0 or budget_stopped_at is not None:
                elapsed = time.time() - t0
                rate = elapsed / total_done
                remaining = (len(targets) - total_done) * rate
                print(f"  {total_done}/{len(targets)} processed (ok={done_count}, dropped={dropped}, "
                      f"cost=${budget.total_usd:.4f}, {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
                questions_path.write_text(
                    json.dumps(list(existing_map.values()), ensure_ascii=False, indent=2), encoding="utf-8"
                )
            if budget_stopped_at is not None:
                break
    print(f"Done: {done_count} good, {dropped} dropped out of {len(targets)}. Total spend: ${budget.total_usd:.4f}")
    return existing_map


def build_fact_subgraph(questions_vi, rel_vi):
    wanted_entities = set()
    for q in questions_vi:
        for a in q.get("answers", []):
            wanted_entities.add(a)
    facts_vi = []
    with open(RAW_DIR / "kg" / "full.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            s, r, o, t = parts
            if s in wanted_entities or o in wanted_entities:
                facts_vi.append({
                    "subject": s, "relation_en": r,
                    "relation_vi": rel_vi.get(r, r.replace("_", " ")),
                    "object": o, "timestamp": t,
                })
    return facts_vi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--n-samples", type=int, default=54300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--budget-usd", type=float, default=5.0, help="hard spend cap in USD; run stops as soon as it's hit")
    args = ap.parse_args()

    key_path = Path(__file__).parent / ".or_key"
    api_key = os.environ.get("OPENROUTER_API_KEY") or key_path.read_text().strip()

    budget = BudgetTracker(args.budget_usd)
    print(f"Hard budget cap: ${args.budget_usd:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rel_vi = translate_relations(api_key, args.model, budget)
    print(f"Relations translated: {len(rel_vi)} | spend so far: ${budget.total_usd:.4f}")

    if budget.over_budget():
        print("Budget already exhausted after relation translation — stopping before questions.")
        return

    questions_path = OUT_DIR / f"questions_{args.split}_vi.json"
    existing = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.exists() else []
    existing_map = {q["quid"]: q for q in existing}
    exclude = set(existing_map.keys())

    all_questions = json.loads((RAW_DIR / "questions" / f"{args.split}.json").read_text(encoding="utf-8"))
    pool = [q for q in all_questions if q["quid"] not in exclude]
    random.seed(args.seed)
    targets = random.sample(pool, min(args.n_samples, len(pool)))
    print(f"Target: {len(targets)} new questions (already have {len(existing_map)})")

    merged_map = translate_questions_parallel(api_key, args.model, targets, args.workers,
                                               args.save_every, questions_path, existing_map, budget)
    merged = list(merged_map.values())
    questions_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(merged)} total questions -> {questions_path.name} | FINAL SPEND: ${budget.total_usd:.4f}")

    facts_vi = build_fact_subgraph(merged, rel_vi)
    (OUT_DIR / f"facts_{args.split}_vi.json").write_text(
        json.dumps(facts_vi, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(facts_vi)} facts -> facts_{args.split}_vi.json")


if __name__ == "__main__":
    main()
