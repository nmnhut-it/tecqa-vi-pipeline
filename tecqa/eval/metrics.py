"""
Scoring for every TECQA-VI evaluation run (paper Sec 5.1 + Appendix D).

OWNER: EVAL (docs/TEAM_PLAN.md H5). Full rationale: docs/EVAL_DESIGN.md Sec 3.

Pure functions only — no network, no API key, no numpy. The notebook's replay
mode imports this on a clean Colab to re-score committed results/*.jsonl, which
is also the check that keeps the paper's numbers and the notebook's numbers from
drifting apart.

Metrics:
    hits_at_1        the paper's primary metric (Table 1). Strict string match.
    hit_granular     diagnostic only: same, after trimming a predicted time to
                     the granularity of the gold answer.
    answer_recall    Appendix D: gold answer present in a fact set.
    anchor_recall    Appendix D: gold temporal anchor present in a fact set.
    chain_recall     Appendix D: both present at once.
    discordant_pairs how two runs over the same questions agree and disagree.

Input:  fact tuples (s, p, o, t) and result rows (results/README.md schema).
Output: bools, and (hits, total) pairs for breakdowns.
"""
from collections import defaultdict

from .. import config
from ..utils.timeutil import (GRANULARITY_DAY, GRANULARITY_MONTH, GRANULARITY_YEAR,
                              asked_granularity, granularity_of, parse_ts,
                              trim_to_granularity)

ANSWER_TYPE_TIME = config.ANSWER_TYPE_TIME
UNKNOWN_BUCKET = "unknown"

# Ordering of the three units, so "coarser than" is a comparison and not a
# chain of if-statements.
_PARTS = {GRANULARITY_YEAR: 1, GRANULARITY_MONTH: 2, GRANULARITY_DAY: 3}


def normalize(value) -> str:
    """Canonical form for answer comparison: MultiTQ entity ids are
    underscore-joined, model output is not."""
    return str(value).strip().lower().replace("_", " ")


def hits_at_1(pred, gold) -> bool:
    """The paper's Hits@1: the FIRST prediction must be in the gold set."""
    if not pred:
        return False
    return normalize(pred[0]) in {normalize(g) for g in gold}


def _time_matches(predicted: str, gold: str) -> bool:
    """"2013-05-08" counts against gold "2013" only at the gold's granularity."""
    return trim_to_granularity(predicted, granularity_of(gold)) == str(gold)


def hit_granular(pred, gold, answer_type: str) -> bool:
    """Diagnostic sibling of hits_at_1 that forgives a time answer given at a
    finer granularity than the gold. Never report this as Hits@1 — it exists to
    size the "right answer, wrong format" error class in Sec 5.7."""
    if not pred:
        return False
    if answer_type != ANSWER_TYPE_TIME:
        return hits_at_1(pred, gold)
    return any(_time_matches(str(pred[0]), str(g)) for g in gold)


def _fact_entities(fact) -> tuple:
    return normalize(fact[config.FACT_SUBJECT]), normalize(fact[config.FACT_OBJECT])


def answer_recall(facts, gold, answer_type: str) -> bool:
    """Appendix D: does this fact set contain the ground-truth answer at all?
    Measured after Stage 1 and again after Stage 2 — the drop between them is
    the price of pruning (paper reports 99.33% -> 94.12%)."""
    if not facts or not gold:
        return False
    if answer_type == ANSWER_TYPE_TIME:
        return any(_time_matches(str(f[config.FACT_TIME]), str(g)) for f in facts for g in gold)
    gold_norm = {normalize(g) for g in gold}
    return any(gold_norm.intersection(_fact_entities(f)) for f in facts)


def anchor_recall(facts, anchors) -> bool:
    """Appendix D: does the context still contain the temporal reference point
    the question hangs off? `anchors` is a list of date objects; an empty list
    means the anchor was not derivable and the question must be excluded from
    the denominator (docs/EVAL_DESIGN.md Sec 3.3)."""
    if not facts or not anchors:
        return False
    anchor_set = set(anchors)
    return any(parse_ts(f[config.FACT_TIME]) in anchor_set for f in facts)


def chain_recall(facts, gold, answer_type: str, anchors) -> bool:
    """Appendix D: anchor AND answer present together — the pair that makes a
    2-step temporal comparison (after_first, before_last) resolvable."""
    return anchor_recall(facts, anchors) and answer_recall(facts, gold, answer_type)


def breakdown(rows, key: str, metric: str = "hit") -> dict:
    """{bucket: (hits, total)} over one row field — qtype, qlabel, answer_type
    or time_level. Rows missing the field land in a single "unknown" bucket
    rather than being dropped silently."""
    buckets = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = buckets[row.get(key) or UNKNOWN_BUCKET]
        bucket[0] += int(bool(row.get(metric)))
        bucket[1] += 1
    return {name: (hits, total) for name, (hits, total) in buckets.items()}


def rate(rows, predicate) -> float:
    """Share of rows satisfying `predicate`. 0.0 on an empty list, so callers
    never have to guard before formatting."""
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def meta_rate(rows, meta_key: str) -> float:
    """Share of rows whose meta flag is truthy — the diagnostics of Sec 3.4."""
    return rate(rows, lambda row: bool(row.get("meta", {}).get(meta_key)))


def meta_rate_where(rows, meta_key: str, condition_key: str) -> tuple:
    """(share, denominator) restricted to rows where `condition_key` holds.
    Anchor Recall needs this: it is only defined where an anchor was derivable,
    and reporting it over the full sample would understate it."""
    eligible = [row for row in rows if row.get("meta", {}).get(condition_key)]
    return meta_rate(eligible, meta_key), len(eligible)


def discordant_pairs(rows_a, rows_b, metric: str = "hit") -> dict:
    """How two runs over the SAME questions agree and disagree, question by
    question.

    Returns {n_paired, only_a, only_b, agree}. This is the raw count the error
    analysis argues over: two conditions can post near-identical aggregates
    while disagreeing on a large share of individual questions, and only the
    join shows it.

    No significance test is computed here on purpose. An earlier version ran
    McNemar's exact test; the paper now reports the counts instead, so the
    machinery for a p-value nobody prints was removed rather than left to rot.
    """
    by_qid_b = {row["qid"]: row for row in rows_b}
    paired = [(a, by_qid_b[a["qid"]]) for a in rows_a if a["qid"] in by_qid_b]
    only_a = sum(1 for a, b in paired if a.get(metric) and not b.get(metric))
    only_b = sum(1 for a, b in paired if b.get(metric) and not a.get(metric))
    return {"n_paired": len(paired), "only_a": only_a, "only_b": only_b,
            "agree": len(paired) - only_a - only_b}


def abstained(row) -> bool:
    """The model read a non-empty evidence chain, produced a well-formed answer
    list, and that list was empty.

    Distinct from a parse failure (output unreadable) and from an empty subgraph
    (nothing to read). Under Hits@1 an abstention is scored wrong with
    certainty, so it is strictly worse than a guess — which is why the `strict`
    condition retries it (docs/EVAL_DESIGN.md Sec 9.3). Under `strict` this rate
    therefore counts only the questions still empty AFTER that retry."""
    meta = row.get("meta", {})
    return bool(meta.get("chain_size")) and meta.get("parse_ok") is True and not row.get("pred")


def granularity_mismatch(row) -> bool:
    """The question asks for a coarser unit than its own gold answer carries.

    Only translation can produce this: MultiTQ's English "When did X visit Y?"
    is a day-level question, and a Vietnamese rendering as "vào năm nào?" (in
    what year?) makes the pipeline answer the year — correctly, for the question
    it was actually given — against a gold answer that is a full date. Every
    such question is unwinnable no matter how good the model is, so it belongs
    in the dataset-quality column of the error analysis and not the reasoning
    one (docs/EVAL_DESIGN.md Sec 9.5)."""
    if row.get("answer_type") != ANSWER_TYPE_TIME or not row.get("gold"):
        return False
    asked = asked_granularity(str(row.get("question", "")))
    if asked is None:
        return False
    finest = min(_PARTS[granularity_of(str(g))] for g in row["gold"])
    return finest > _PARTS[asked]


def reasoning_loss(row) -> bool:
    """A miss whose answer was sitting in the evidence chain. Retrieval did its
    job and Stage 3 did not, so this is the headroom a better reasoner or prompt
    can still recover — the denominator the Sec 5.7 error analysis argues over."""
    return not row.get("hit") and bool(row.get("meta", {}).get("answer_recall_chain"))


def summarize(rows) -> dict:
    """Aggregate every headline and diagnostic number for one run's rows.
    Used by run_eval.py for its console summary and by the notebook to re-derive
    a committed run's score from the raw rows."""
    total = len(rows)
    hits = sum(1 for row in rows if row.get("hit"))
    anchor, anchor_n = meta_rate_where(rows, "anchor_recall", "anchor_derivable")
    chain, _ = meta_rate_where(rows, "chain_recall", "anchor_derivable")
    return {
        "n_scored": total,
        "hits_at_1": hits / total if total else 0.0,
        "hits_at_1_granular": rate(rows, lambda row: row.get("hit_gran")),
        "answer_recall_sg": meta_rate(rows, "answer_recall_sg"),
        "answer_recall_chain": meta_rate(rows, "answer_recall_chain"),
        "anchor_recall": anchor,
        "chain_recall": chain,
        "anchor_derivable_rate": anchor_n / total if total else 0.0,
        "empty_subgraph_rate": rate(rows, lambda r: not r.get("meta", {}).get("subgraph_size")),
        "parse_fail_rate": rate(rows, lambda r: r.get("meta", {}).get("parse_ok") is False),
        "abstain_rate": rate(rows, abstained),
        "reasoning_loss_rate": rate(rows, reasoning_loss),
        "relation_expanded_rate": rate(rows, relation_expanded),
        "granularity_mismatch_rate": rate(rows, granularity_mismatch),
    }


def relation_expanded(row) -> bool:
    """Stage 1 grounded the question to more than one relation.

    Only the Vietnamese branch does this, so the rate is also the size of an
    asymmetry between the two language arms: English always retrieves from one
    relation. Reporting it keeps a Vietnamese-English gap from being read as a
    pure language effect (docs/EVAL_DESIGN.md Sec 9.4)."""
    return len(row.get("meta", {}).get("relation_ids") or []) > 1
