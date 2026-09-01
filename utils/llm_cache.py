"""
Disk-cached OpenRouter LLM caller.

Keyed by SHA-256 of (model, temperature, max_tokens, prompt, extra_params).
Prevents paying twice or waiting twice for identical prompts across runs.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add dataset_vi so we can reuse build_vi_dataset_or.call_or
DATASET_VI_DIR = Path(__file__).parent.parent.parent / "dataset_vi"
if str(DATASET_VI_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_VI_DIR))

try:
    from build_vi_dataset_or import call_or
except ImportError:
    # Fallback to local import if called from other root
    from dataset_vi.build_vi_dataset_or import call_or

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DISABLED_VALUES = {"0", "false", "no"}

# One process runs one evaluation, so the spending cap is process-wide state
# rather than an argument threaded through all three stages. Cache hits never
# reach call_or, so they correctly cost nothing.
_BUDGET = None


def set_budget(tracker) -> None:
    """Install the run's BudgetTracker. Every uncached call then reports its
    real OpenRouter cost into it (see tecqa/eval/budget.py)."""
    global _BUDGET
    _BUDGET = tracker


def over_budget() -> bool:
    return _BUDGET is not None and _BUDGET.over_budget()


def spent_usd() -> float:
    return _BUDGET.total_usd if _BUDGET is not None else 0.0


def is_enabled() -> bool:
    """TECQA_CACHE=0 forces every call back onto the network. Needed when a run
    must measure a model's real latency, or after a prompt template changed but
    the key did not (`run_eval.py --no-cache`)."""
    return os.environ.get("TECQA_CACHE", "1").lower() not in CACHE_DISABLED_VALUES


def _cache_key(model: str, prompt: str, temperature: float,
               max_tokens: int, extra_params: dict | None) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_params": extra_params or {},
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cached_call(api_key: str, model: str, prompt: str, *,
                temperature: float = 0.0,
                max_tokens: int = 1024,
                timeout: int = 60,
                extra_params: dict | None = None,
                is_valid=None,
                attempts: int = 1) -> str:
    """Check the disk cache before calling OpenRouter.

    `is_valid` decides whether a response is worth keeping. Reasoning backbones
    occasionally spend the whole token budget deliberating and get cut off
    before emitting an answer; that is transient, but caching it would make the
    failure permanent — every later run would replay the truncated text instead
    of retrying. So an invalid response is retried up to `attempts` times and is
    never written to the cache. The last one is still returned, so the caller
    can record the failure rather than having it silently swallowed.
    """
    key = _cache_key(model, prompt, temperature, max_tokens, extra_params)
    cache_file = CACHE_DIR / f"{key}.json"
    enabled = is_enabled()

    if enabled and cache_file.exists():
        try:
            entry = json.loads(cache_file.read_text(encoding="utf-8"))
            return entry.get("response", "")
        except Exception:
            pass

    response = ""
    for attempt in range(max(1, attempts)):
        response = call_or(
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_params=extra_params,
            budget=_BUDGET,
        )
        if is_valid is None or is_valid(response):
            _store(cache_file, model, prompt, response)
            return response
        if over_budget():
            break  # a retry we cannot pay for
    return response


def _store(cache_file, model: str, prompt: str, response: str) -> None:
    """Best-effort write; a failed cache write must never fail the run."""
    try:
        cache_file.write_text(json.dumps({
            "model": model, "prompt": prompt,
            "response": response, "timestamp": time.time(),
        }, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
