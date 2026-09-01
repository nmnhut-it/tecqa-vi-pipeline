"""
Parallel evaluation runner for TECQA on MultiTQ test set (English & Vietnamese).

Supports:
  --language en (default: original paper benchmark on English test.json)
  --language vi (Vietnamese benchmark on questions_test_vi.json with Option B cross-lingual grounding)

Usage:
  python3 run_eval_parallel.py --language vi --n 100 --workers 16 --seed 42
  python3 run_eval_parallel.py --language en --n 100 --workers 16 --seed 42
"""
import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
from pathlib import Path

# Unbuffered stdout for live progress streaming
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Prevent tokenizers deadlock
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TECQA_ROOT = Path(__file__).parent
if str(TECQA_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(TECQA_ROOT.parent))

from tecqa import config
from tecqa.data.kg_multitq import MultiTQGraph
from tecqa.pipeline import TECQA




def normalize(s: str) -> str:
    return str(s).strip().lower().replace("_", " ")


def hits_at_1(pred: list, gold: list) -> bool:
    if not pred:
        return False
    gold_norm = {normalize(g) for g in gold}
    return normalize(pred[0]) in gold_norm


def eval_one(item: dict, tecqa: TECQA, language: str = "en") -> dict:
    q = item["question"]
    qtype = item.get("question_type", item.get("qtype", "simple"))
    qlabel = item.get("qlabel", "Single")
    atype = item.get("answer_type", "entity")
    gold = item["answers"]

    t0 = time.time()
    try:
        pred, meta = tecqa.answer(q, answer_type=atype, qlabel=qlabel, language=language)
        dur = time.time() - t0
        hit = hits_at_1(pred, gold)
        return {
            "quid": item.get("quid", ""),
            "question": q,
            "qtype": qtype,
            "qlabel": qlabel,
            "answer_type": atype,
            "gold": gold,
            "pred": pred,
            "hit@1": hit,
            "meta": meta,
            "latency_s": dur,
            "status": "ok",
        }
    except Exception as e:
        dur = time.time() - t0
        return {
            "quid": item.get("quid", ""),
            "question": q,
            "qtype": qtype,
            "qlabel": qlabel,
            "answer_type": atype,
            "gold": gold,
            "pred": [],
            "hit@1": False,
            "error": str(e),
            "latency_s": dur,
            "status": "error",
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TECQA in parallel (English & Vietnamese).")
    parser.add_argument("--language", choices=["en", "vi"], default="en", help="Language mode (en or vi).")
    parser.add_argument("--n", type=int, default=100, help="Number of questions to evaluate.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker threads.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument("--out", type=str, default="", help="Output JSON path.")
    parser.add_argument("--kg-path", type=str, default="data/kg/full.txt", help="Path to KG full.txt")
    parser.add_argument("--data-path", type=str, default="", help="Path to questions JSON")
    args = parser.parse_args()

    if not args.data_path:
        args.data_path = "data/questions/test_vi.json" if args.language == "vi" else "data/questions/test_en.json"
        
    data_path = Path(args.data_path)
    kg_path = Path(args.kg_path)
    out_file = args.out or str(TECQA_ROOT / f"eval_results_{args.n}_{args.language}.json")

    print(f"=================================================================")
    print(f"TECQA BENCHMARK RUNNER: Language={args.language.upper()} (n={args.n}, workers={args.workers}, seed={args.seed})")
    print(f"Dataset : {data_path}")
    print(f"Models  : Extract={config.MODEL_EXTRACT} | Reason={config.MODEL_REASON}")
    print(f"=================================================================\n")

    print("Loading full MultiTQ graph (461K facts)...")
    t0 = time.time()
    graph = MultiTQGraph(kg_path=kg_path).load()
    print(f"  {len(graph.facts)} facts, {len(graph.entities)} entities, {len(graph.relations)} relations loaded in {time.time()-t0:.2f}s\n")

    tecqa = TECQA(graph)

    raw_data = json.loads(data_path.read_text(encoding="utf-8"))
    random.seed(args.seed)
    samples = random.sample(raw_data, min(args.n, len(raw_data)))

    print(f"Evaluating {len(samples)} questions in parallel (workers={args.workers}, seed={args.seed})...\n")

    results = []
    t_start = time.time()
    hits_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(eval_one, sample, tecqa, args.language): idx
            for idx, sample in enumerate(samples)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            res = future.result()
            results.append(res)
            if res["hit@1"]:
                hits_count += 1
            status_mark = "HIT ✓" if res["hit@1"] else "MISS ✗"
            cur_acc = (hits_count / len(results)) * 100
            print(f"  [{len(results):>3}/{len(samples)}] [{status_mark}] in {res['latency_s']:.1f}s (running Hits@1: {cur_acc:.1f}%) | "
                  f"Q: {res['question'][:45]}... | pred={res['pred'][:2]} gold={res['gold'][:2]}")

    elapsed = time.time() - t_start
    hits = sum(1 for r in results if r["hit@1"])
    accuracy = (hits / len(results)) * 100 if results else 0.0

    print("\n" + "=" * 65)
    print(f"SUMMARY ({args.language.upper()}): {hits}/{len(results)} correct (Hits@1 = {accuracy:.1f}%) in {elapsed:.1f}s ({elapsed/len(results):.1f}s/query)")
    print("=" * 65)

    out_path = Path(out_file)
    out_path.write_text(json.dumps({
        "summary": {
            "language": args.language,
            "total": len(results),
            "hits": hits,
            "hits@1": accuracy,
            "elapsed_s": elapsed,
            "avg_latency_s": elapsed / len(results) if results else 0,
            "seed": args.seed,
            "workers": args.workers,
            "model_extract": config.MODEL_EXTRACT,
            "model_reason": config.MODEL_REASON,
        },
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
