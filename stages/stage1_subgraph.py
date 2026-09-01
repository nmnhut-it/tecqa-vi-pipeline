"""
Stage 1 — Structure-Guided Subgraph Construction (paper Sec 4.2).

Algorithm:
  1. LLM extracts entity mentions E_q and relation phrase R_q from the question (Eq. 1).
  2. Each mention is grounded to a canonical KG entity/relation via cosine similarity (Eq. 2).
  3. F_main = facts touching the main entity with relation r_q (Eq. 3-4).
  4. F_context = facts connecting pairs of auxiliary entities with relation r_q (Eq. 5).
  5. Return F_q = F_main ∪ F_context.

Input:  Question (en or vi), a MultiTQGraph, EntityGrounder, RelationGrounder.
Output: (main_entity, f_q: list[fact], grounded_entities: set).
"""
import ast
import re

from .. import config
from ..prompts.stage1 import (
    ENTITY_EXTRACTION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    MAIN_ENTITY_PROMPT,
    ENTITY_EXTRACTION_VI_PROMPT,
    RELATION_EXTRACTION_VI_PROMPT,
    MAIN_ENTITY_VI_PROMPT,
)
from ..utils.llm_cache import cached_call

MIN_ENTITIES_FOR_CONTEXT = 2  # Eq. 5 requires at least a pair of auxiliary entities


def _parse_list(raw: str) -> list:
    cleaned = raw.strip()
    # Strip markdown code fences if present
    if "```" in cleaned:
        m = re.search(r"```(?:json|python)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
        if m:
            cleaned = m.group(1)
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback regex search for bracketed list
    m = re.search(r"\[([^\[\]]*)\]", cleaned)
    if m:
        try:
            parsed = ast.literal_eval("[" + m.group(1) + "]")
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def extract_entities(api_key, model, question, language="en") -> list:
    """Eq. 1 (entity half): LLM extracts topic entity mentions."""
    tmpl = ENTITY_EXTRACTION_VI_PROMPT if language == "vi" else ENTITY_EXTRACTION_PROMPT
    raw = cached_call(api_key, model, tmpl.format(question=question),
                      temperature=0.0, timeout=config.TIMEOUT_EXTRACT,
                      max_tokens=512, extra_params=None)
    return _parse_list(raw)


def extract_relation(api_key, model, question, entities, relation_set, language="en") -> str:
    """Eq. 1 (relation half): LLM picks one relation phrase from the full vocabulary."""
    tmpl = RELATION_EXTRACTION_VI_PROMPT if language == "vi" else RELATION_EXTRACTION_PROMPT
    return cached_call(api_key, model, tmpl.format(
        question=question, entities_list=entities, relation_set=relation_set,
    ), temperature=0.0, timeout=config.TIMEOUT_RELATION,
       max_tokens=512, extra_params=None).strip()


def extract_main_entity(api_key, model, question, entities, language="en") -> str:
    """Eq. 1 (main-entity): LLM identifies the pivot node connecting to the answer."""
    tmpl = MAIN_ENTITY_VI_PROMPT if language == "vi" else MAIN_ENTITY_PROMPT
    return cached_call(api_key, model, tmpl.format(
        question=question, entities_list=entities,
    ), temperature=0.0, timeout=config.TIMEOUT_EXTRACT,
       max_tokens=512, extra_params=None).strip()


def retrieve_facts(graph, main_entity, others, rel_id) -> list:
    """
    F_q = F_main ∪ F_context (Eq. 3-5).

    F_main (Eq. 3): facts touching main_entity with the grounded relation rel_id.
    F_context (Eq. 5): facts connecting auxiliary entities with relation rel_id to capture temporal anchors.
    """
    f_main = [f for f in graph.facts_touching(main_entity)
              if f[config.FACT_PREDICATE] == rel_id]
    f_context = []
    seen_facts = set(f_main)
    for ent in others:
        for f in graph.facts_touching(ent):
            if f[config.FACT_PREDICATE] == rel_id and f not in seen_facts:
                seen_facts.add(f)
                f_context.append(f)
    # sorted, not list(set(...)): str hashing is salted per process, so an
    # unsorted set would hand Stage 3 a differently ordered prompt on every
    # run — defeating the LLM cache and making runs unreproducible.
    return sorted({*f_main, *f_context})


def structure_guided_subgraph(graph, entity_grounder, relation_grounder, question,
                              *, api_key, model, language="en", trace=None):
    """
    Full Stage 1. Returns (main_entity, f_q, grounded_entity_set).
    Returns (None, [], set()) if no entities could be grounded.

    `trace` is an optional dict the evaluator passes in to collect the error
    -analysis fields of docs/EVAL_DESIGN.md Sec 3.4 (which mention was extracted,
    which relation it grounded to and how confidently). Purely observational:
    leaving it None changes nothing about what this function computes.
    """
    record = trace if trace is not None else {}

    # Step 1: LLM extraction (Eq. 1) with Cross-lingual translation for Vietnamese
    mentions = extract_entities(api_key, model, question, language=language)
    grounded = [g for g in (entity_grounder.ground(e) for e in mentions) if g]
    record.update(mentions=list(mentions), grounded=list(grounded))
    if not grounded:
        return None, [], set()

    # Step 2: Relation grounding (Eq. 2)
    # sorted, not list(): graph.relations is a set, and str hashing is salted
    # per process, so list() ordered the 251 relations differently in every
    # process. The relation prompt is the most expensive Stage-1 call, and an
    # order that changes per process gives it a fresh cache key every run --
    # measured as ~$0.002 per question re-paid on a fully cached replay.
    rel_phrase = extract_relation(api_key, model, question, grounded,
                                  sorted(graph.relations), language=language)

    if language == "vi":
        # Multi-relation candidate expansion for Vietnamese cross-lingual adaptation
        candidates = relation_grounder.ground_topk(rel_phrase, k=2)
        if len(candidates) >= 2 and (candidates[0][1] - candidates[1][1] < 0.15 or candidates[1][1] >= 0.75):
            rel_ids = [candidates[0][0], candidates[1][0]]
        else:
            rel_ids = [candidates[0][0]]
        rel_sim = candidates[0][1]
    else:
        # Paper Appendix F.3 verbatim: single relation grounding
        rel_id, rel_sim = relation_grounder.ground(rel_phrase)
        rel_ids = [rel_id]
    record.update(relation_phrase=rel_phrase, relation_id=rel_ids[0],
                  relation_ids=list(rel_ids), relation_sim=float(rel_sim))

    # Step 3: Main entity identification (Eq. 1, third call)
    main_raw = extract_main_entity(api_key, model, question, grounded, language=language)
    main_entity = entity_grounder.ground(main_raw) or grounded[0]
    others = [e for e in grounded if e != main_entity]
    record.update(main_entity_raw=main_raw)

    # Steps 4-5: Retrieve F_q = F_main ∪ F_context (Eq. 3-5)
    f_q_all = []
    for r_id in rel_ids:
        f_q_all.extend(retrieve_facts(graph, main_entity, others, r_id))
    return main_entity, sorted(set(f_q_all)), set(grounded)
