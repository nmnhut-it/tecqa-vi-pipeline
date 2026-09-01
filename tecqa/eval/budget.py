"""
Hard spending cap for an evaluation run.

OWNER: EVAL (docs/TEAM_PLAN.md H5).

Two ceilings, whichever binds first:

  1. what the operator asked for (`--max-usd`);
  2. what the OpenRouter account can actually still spend, minus a margin.

The second matters because a run that hits the account limit does not stop
politely — every remaining request fails, and a half-finished results file full
of API errors looks exactly like a pipeline that is broken. Reading the real
headroom up front turns that into a clean, resumable stop.

Cost accounting reuses dataset_vi.build_vi_dataset_or.BudgetTracker, which
already prefers OpenRouter's authoritative `usage.cost` over a local price
table -- that table was measured undercounting reasoning-heavy models by ~35%,
which is exactly the direction that busts a cap.

Input:  a dollar ceiling and an API key.
Output: an object that accumulates real per-call cost and answers .over_budget().

Related: tecqa/utils/llm_cache.py (threads it into call_or), run_eval.py (checks
it between questions).
"""
import json
import sys
import urllib.request
from pathlib import Path

DATASET_VI_DIR = Path(__file__).parent.parent.parent / "dataset_vi"
if str(DATASET_VI_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_VI_DIR))

from build_vi_dataset_or import BudgetTracker  # noqa: E402

KEY_INFO_URL = "https://openrouter.ai/api/v1/key"
# Leave this much of the account's remaining credit untouched, so the last
# in-flight requests cannot tip the account over its limit mid-write.
HEADROOM_MARGIN_USD = 0.10
REQUEST_TIMEOUT = 15


def account_headroom(api_key: str) -> float:
    """Credit the account can still spend, or infinity when it is uncapped.
    Never raises: a monitoring endpoint being down must not block a run."""
    request = urllib.request.Request(KEY_INFO_URL,
                                     headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            info = json.loads(response.read().decode("utf-8")).get("data", {})
    except Exception:
        return float("inf")
    remaining = info.get("limit_remaining")
    if remaining is None:
        limit, usage = info.get("limit"), info.get("usage") or 0.0
        remaining = float("inf") if limit is None else float(limit) - float(usage)
    return float(remaining)


def effective_cap(requested_usd: float, api_key: str, already_spent: float = 0.0) -> tuple:
    """(cap, headroom). The cap is the TOTAL the run may spend, including what
    earlier sessions of the same run already spent.

    `already_spent` is what makes resuming work. The account's headroom shrinks
    as the run spends, so recomputing the cap from headroom alone would return a
    smaller number every time — eventually smaller than the run's own cumulative
    spend, at which point a resumed run would declare itself over budget before
    answering a single question and could never finish. Anchoring the ceiling at
    `already_spent + what is still spendable` keeps it monotonic in the right
    direction while still honouring the requested total.
    """
    headroom = account_headroom(api_key)
    if headroom == float("inf"):
        return requested_usd, headroom
    still_spendable = max(0.0, headroom - HEADROOM_MARGIN_USD)
    return max(0.0, min(requested_usd, already_spent + still_spendable)), headroom


def make_tracker(requested_usd: float, api_key: str) -> tuple:
    """Build the tracker the run will be held to, plus a line explaining which
    ceiling won — the operator needs to know whether the run stopped because
    they said so or because the account ran dry."""
    cap, headroom = effective_cap(requested_usd, api_key)
    if headroom == float("inf"):
        note = f"cap ${cap:.2f} (requested; account has no credit limit set)"
    elif cap < requested_usd:
        note = (f"cap ${cap:.2f} — LIMITED BY ACCOUNT: ${headroom:.2f} credit remains, "
                f"you asked for ${requested_usd:.2f}")
    else:
        note = f"cap ${cap:.2f} (requested; ${headroom:.2f} credit available)"
    return BudgetTracker(cap), note
