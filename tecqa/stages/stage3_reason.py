"""
Stage 3 — LLM Reasoning over the temporal evidence chain (paper Sec 4.4).

Feeds the proximity-ordered evidence chain from Stage 2 to the reasoning LLM (Eq. 10).
Includes Grounded KG Entities context for cross-lingual resolving and time granularity normalization.

Input:  Question, Stage-2 chain, MultiTQGraph for text rendering.
Output: predicted answer list (empty list on unparseable output).
"""
import ast
import re

from .. import config
from ..prompts.stage3 import (DEFAULT_STYLE, PROMPT_STRICT, fallback_suffix,
                              template_for)
from ..utils.llm_cache import cached_call
from ..utils.timeutil import (GRANULARITY_MONTH, GRANULARITY_YEAR,
                              asked_granularity, trim_to_granularity)

# Paper Table 7: Single = one answer expected {equal, before_after, first_last};
# Multiple = several valid answers {equal_multi, after_first, before_last}.
QLABEL_MULTIPLE = "Multiple"
TOPK_SINGLE = 1
TOPK_MULTIPLE = 5

# One retry when the backbone returns no answer list at all. Reasoning models
# occasionally exhaust max_tokens mid-deliberation; that is transient, and a
# retry is far cheaper than losing the question. Failures are never cached.
REASON_ATTEMPTS = 2


def topk_for(qlabel: str) -> int:
    return TOPK_MULTIPLE if qlabel == QLABEL_MULTIPLE else TOPK_SINGLE


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json|python)?\s*(\[.*?\])\s*```", re.DOTALL)
LIST_RE = re.compile(r"\[[^\[\]]*\]")


def _as_str_list(text: str):
    """literal_eval `text` if it is a Python list of scalars, else None."""
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    return [str(x) for x in parsed] if isinstance(parsed, list) else None


def extract_answer_list(raw: str):
    """
    Pull the answer list out of the model's output, or None if there is none.

    Reasoning backbones (Qwen3-8B in thinking mode is the paper's choice) often
    return the whole deliberation before the answer, and that deliberation
    quotes evidence facts in the same [a, b, c, d] shape as the answer. So the
    answer is taken from the END, not the beginning: the last bracketed span
    that actually evaluates to a Python list wins. Taking the first one let the
    model's own scratch work outrank its conclusion.

    None (no list at all) is kept distinct from [] (the model concluded that no
    fact answers the question). Collapsing them would book every honest "not in
    the evidence" as a parser failure and inflate the Sec 5.7 error class.
    """
    cleaned = THINK_RE.sub("", raw).strip()

    fenced = FENCE_RE.findall(cleaned)
    if fenced:
        answers = _as_str_list(fenced[-1])
        if answers is not None:
            return answers

    answers = _as_str_list(cleaned)
    if answers is not None:
        return answers

    # Fact renderings like [China, Make_a_visit, Iraq, 2013-05-08] are not valid
    # literals, so they fail _as_str_list and get skipped automatically.
    for candidate in reversed(LIST_RE.findall(cleaned)):
        answers = _as_str_list(candidate)
        if answers is not None:
            return answers
    return None


def parse_answer_list(raw: str) -> list:
    """extract_answer_list, with a failure flattened to an empty prediction."""
    answers = extract_answer_list(raw)
    return [] if answers is None else answers


def normalize_time_granularity(answers: list, question: str, answer_type: str) -> list:
    """Trim a time answer to the unit the question actually asked for.

    'When did X visit Y?' keeps YYYY-MM-DD; 'In what year...' becomes YYYY. The
    question wording is the only signal available at inference time -- so when a
    translation has quietly changed the unit the question asks for, this
    faithfully answers the translated question and the gold answer no longer
    matches. metrics.granularity_mismatch() counts exactly those cases rather
    than letting them look like reasoning errors.
    """
    if answer_type != config.ANSWER_TYPE_TIME or not answers:
        return answers
    wanted = asked_granularity(question)
    if wanted is None:
        return [str(a).strip() for a in answers]
    return [trim_to_granularity(str(a).strip(), wanted) for a in answers]


def reason(graph, question: str, chain: list, *, api_key, model,
           answer_type="entity", qlabel="Single", topk=None,
           grounded_entities=None, language="en", params=None,
           prompt_style=DEFAULT_STYLE, trace=None) -> list:
    """
    Eq. 10 — answer the question from the proximity-ordered evidence chain.
    Returns a list of predicted answers (strings).

    `params` overrides config.REASON_PARAMS so the backbone sweep can turn
    thinking mode off (paper Table 4). `prompt_style` selects the template:
    'paper' is the published prompt, 'strict' is ours (tecqa/prompts/stage3.py).
    It is part of the cache key, so switching styles cannot read back an answer
    produced by the other one. `trace` is an optional dict collecting
    whether the model's output actually parsed — that is the numerator of the
    parse-failure rate in docs/EVAL_DESIGN.md Sec 3.4, and an empty prediction
    caused by a malformed list is a very different error from a wrong answer.
    """
    record = trace if trace is not None else {}
    if topk is None:
        topk = topk_for(qlabel)
    facts_text = "\n".join(graph.fact_tuple_str(f) for f in chain)

    grounded_str = ", ".join(sorted(list(grounded_entities))) if grounded_entities else "None"

    prompt = template_for(prompt_style, language).format(
        question=question, answer_type=answer_type,
        topk=topk, grounded_entities=grounded_str,
        facts=facts_text,
    )
    raw = _call(api_key, model, prompt, params)
    answers = extract_answer_list(raw)
    record.update(parse_ok=answers is not None, raw_len=len(raw),
                  prompt_style=prompt_style, abstain_retry=False)
    if prompt_style == PROMPT_STRICT and chain and not answers:
        answers = _retry_after_abstention(api_key, model, prompt, language,
                                          params, record)
    return normalize_time_granularity(answers or [], question, answer_type)


def _call(api_key: str, model: str, prompt: str, params) -> str:
    """One Stage-3 request to the reasoning backbone.

    `params` overrides config.REASON_PARAMS (the backbone sweep turns thinking
    mode off through it). A response with no parseable answer list is retried
    up to REASON_ATTEMPTS times and is never written to the cache — see
    utils/llm_cache.cached_call. Returns the raw text, empty string included.
    """
    return cached_call(api_key, model, prompt, temperature=0.0,
                       timeout=config.TIMEOUT_REASON,
                       max_tokens=config.REASON_MAX_TOKENS,
                       extra_params=config.REASON_PARAMS if params is None else params,
                       is_valid=lambda text: extract_answer_list(text) is not None,
                       attempts=REASON_ATTEMPTS)


def _retry_after_abstention(api_key: str, model: str, prompt: str, language: str,
                            params, record: dict):
    """Second and final Stage-3 attempt, for the 'strict' style only.

    Sent only when the first pass produced no answer over a NON-EMPTY evidence
    chain. It re-sends the same prompt with fallback_suffix() appended, so the
    model sees identical facts and is granted permission to guess only here —
    putting that permission in the main prompt made the model drop the relation
    and time constraints (docs/EVAL_DESIGN.md Sec 9.3). Hits@1 already scored
    the empty answer wrong, so this pass can gain a point and cannot lose one.

    Records `abstain_retry` / `abstain_recovered` in `record`. Returns the
    parsed answer list, or None if the retry did not parse either.
    """
    raw = _call(api_key, model, prompt + fallback_suffix(language), params)
    answers = extract_answer_list(raw)
    record.update(abstain_retry=True, abstain_recovered=bool(answers))
    return answers
