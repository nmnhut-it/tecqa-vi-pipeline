"""
Stage 2 — Temporal Evidence Chain Construction (paper Sec 4.3).

Three steps:
  1. Temporal anchor extraction (Sec 4.3.1):
       T_exp: explicit dates via tecqa.utils.timeutil (English & Vietnamese),
              one anchor per date mention.
       T_imp: timestamps of top-N semantically similar facts (embedding-based, Eq. 6).
       T_anchor = T_exp ∪ T_imp  (paper Sec 4.3.1 — BOTH always computed and unioned).
  2. K-nearest temporal neighbour pruning (Eq. 7-8):
       For each anchor, keep the K facts closest in absolute time distance.
  3. Proximity-based serialization (Eq. 9):
       Sort retained facts ascending by minimum distance to any anchor.

For Vietnamese (language == "vi"):
  - Extreme-Time Anchor Injection (min/max date anchors for 'đầu tiên' / 'cuối cùng')
  - Entity-Pair Intersection Prioritization (guarantees direct pair interaction facts are preserved)
"""
from datetime import date

from .. import config
from ..utils.grounding import embed_batch, cos_sim_matrix
# parse_ts and extract_explicit_anchors are re-exported: pipeline.py and the
# debug scripts import them from this module, and the scorer must use the very
# same implementations (see tecqa/utils/timeutil.py).
from ..utils.timeutil import (  # noqa: F401
    DEFAULT_DAY, DEFAULT_MONTH, extract_explicit_anchors, parse_ts,
)


# Vietnamese subgraphs above this size get a larger pruning budget (see
# effective_k_for): VI relation expansion can retrieve two relations, not one.
LARGE_SUBGRAPH = 1000
K_LARGE_SUBGRAPH = 80

# Superlative cues that pin an anchor to the extreme of the subgraph's timeline.
FIRST_WORDS = ("đầu tiên", "lần đầu tiên", "sớm nhất", "first", "earliest")
LAST_WORDS = ("cuối cùng", "lần cuối cùng", "gần đây nhất", "muộn nhất", "last", "latest")


def midpoint_ts(t_start: str, t_end: str) -> date:
    """
    Paper Sec 5.1 — CronQuestions interval midpoint formula: t_m = (t_s + t_e) / 2.
    """
    d_start = parse_ts(t_start)
    d_end = parse_ts(t_end)
    mid_ordinal = (d_start.toordinal() + d_end.toordinal()) // 2
    return date.fromordinal(mid_ordinal)


def fact_ts(f: tuple, *, is_cron: bool = False) -> date:
    """
    Return the effective date of a fact for temporal distance computation.
    """
    if is_cron and len(f) >= 5:
        return midpoint_ts(f[config.FACT_TIME], f[4])
    return parse_ts(f[config.FACT_TIME])


def extract_implicit_anchors(graph, question: str, f_q: list, n_implicit: int,
                              *, is_cron: bool = False) -> list:
    """
    T_imp (Sec 4.3.1, Eq. 6): Timestamps of top-N semantically similar facts in F_q.
    """
    if not f_q or n_implicit <= 0:
        return []

    # 1. Linearize facts to natural-language descriptions
    descriptions = [
        f"{f[config.FACT_SUBJECT]} {f[config.FACT_PREDICATE]} {f[config.FACT_OBJECT]}"
        for f in f_q
    ]

    # 2. Dense cosine similarity with question
    q_emb = embed_batch([question])
    f_emb = embed_batch(descriptions)
    sims = cos_sim_matrix(q_emb, f_emb)[0]

    # 3. Top-N facts -> collect their timestamps
    top_indices = sorted(range(len(f_q)), key=lambda i: sims[i], reverse=True)[:n_implicit]
    anchors = []
    seen = set()
    for idx in top_indices:
        d = fact_ts(f_q[idx], is_cron=is_cron)
        if d not in seen:
            seen.add(d)
            anchors.append(d)
    return anchors


def prune_knn(f_q: list, anchors: list, k: int, *, is_cron: bool = False) -> list:
    """
    K-nearest temporal neighbour pruning (Sec 4.3.2, Eq. 7-8).
    """
    if not anchors or not f_q:
        return list(f_q)

    retained = set()
    for anchor in anchors:
        sorted_by_dist = sorted(
            f_q,
            key=lambda f: abs((fact_ts(f, is_cron=is_cron) - anchor).days)
        )
        retained.update(sorted_by_dist[:k])

    # sorted for determinism: serialize_by_proximity's sort is stable, so ties
    # would otherwise inherit this set's per-process iteration order.
    return sorted(retained)


def serialize_by_proximity(f_retained: list, anchors: list,
                            *, is_cron: bool = False) -> list:
    """
    Proximity-based serialization (Sec 4.3.3, Eq. 9).
    """
    if not anchors:
        return list(f_retained)

    def min_distance_to_anchors(f: tuple) -> int:
        d_fact = fact_ts(f, is_cron=is_cron)
        return min(abs((d_fact - a).days) for a in anchors)

    return sorted(f_retained, key=min_distance_to_anchors)


def effective_k_for(f_q: list, k: int, language: str) -> int:
    """Dynamic K for large Vietnamese subgraphs: cross-lingual relation
    expansion can double |F_q|, so the pruning budget grows with it."""
    if language == "vi" and len(f_q) > LARGE_SUBGRAPH:
        return max(k, K_LARGE_SUBGRAPH)
    return k


def anchors_for(graph, question: str, f_q: list, n_implicit: int, *,
                is_cron: bool = False, language: str = "en", trace=None) -> list:
    """T_anchor = T_exp ∪ T_imp (Sec 4.3.1), plus the Vietnamese extreme-time
    injection for superlative questions.

    Split out of temporal_evidence_chain so the w/o-KNTN ablation
    (docs/EVAL_DESIGN.md Sec 2.2) can reuse the identical anchor set and differ
    only in HOW facts are selected against it.
    """
    record = trace if trace is not None else {}
    t_exp = extract_explicit_anchors(question)
    t_imp = extract_implicit_anchors(graph, question, f_q, n_implicit, is_cron=is_cron)
    anchors = list({*t_exp, *t_imp})
    record.update(n_explicit_anchors=len(t_exp), n_implicit_anchors=len(t_imp))

    # Proposal 1: Extreme-Time Anchor Injection (for Vietnamese)
    if language == "vi" and f_q:
        q_low = question.lower()
        if any(w in q_low for w in FIRST_WORDS):
            anchors.append(min(fact_ts(f, is_cron=is_cron) for f in f_q))
        if any(w in q_low for w in LAST_WORDS):
            anchors.append(max(fact_ts(f, is_cron=is_cron) for f in f_q))
        anchors = list(set(anchors))
    record.update(n_anchors=len(anchors))
    return anchors


def temporal_evidence_chain(graph, question: str, f_q: list, *,
                            k: int = config.K_NEIGHBORS,
                            n_implicit: int = config.N_IMPLICIT_MULTITQ,
                            is_cron: bool = False,
                            grounded_entities: set = None,
                            language: str = "en", trace=None) -> list:
    """
    Full Stage 2. Returns facts sorted ascending by temporal distance to nearest anchor.

    `trace` is an optional dict for the evaluator's diagnostics (anchor counts);
    passing None leaves behaviour unchanged.
    """
    if not f_q:
        return []

    effective_k = effective_k_for(f_q, k, language)
    anchors = anchors_for(graph, question, f_q, n_implicit,
                          is_cron=is_cron, language=language, trace=trace)

    # Step 2: K-NN pruning (Eq. 7-8)
    f_pruned = prune_knn(f_q, anchors, effective_k, is_cron=is_cron)

    # Proposal 2: Entity-Pair Intersection Prioritization (for Vietnamese)
    if language == "vi" and grounded_entities and len(grounded_entities) >= 2:
        pair_facts = [
            f for f in f_q
            if f[config.FACT_SUBJECT] in grounded_entities and f[config.FACT_OBJECT] in grounded_entities
        ]
        if pair_facts:
            f_pruned = sorted(set(f_pruned).union(set(pair_facts)))

    # Step 3: Proximity-based serialization (Eq. 9)
    return serialize_by_proximity(f_pruned, anchors, is_cron=is_cron)
