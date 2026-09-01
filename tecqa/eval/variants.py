"""
Everything that makes one evaluation run differ from another (docs/EVAL_DESIGN.md
Sec 2): language, ablation, hyperparameters, backbone.

OWNER: EVAL (docs/TEAM_PLAN.md H5).

`tecqa.pipeline.TECQA` stays the only implementation of Algorithm 1. Every
variant is a SUBCLASS that overrides the three stage methods `answer()`
dispatches through, so the English condition and the three ablations can never
become private forks that drift when a stage is fixed.

Language is NOT handled here: the pipeline already threads `language=` through
every stage and both prompt sets live in tecqa/prompts/. This runner only pins
that choice for the whole run so a caller cannot accidentally mix conditions.

Input:  a loaded MultiTQGraph plus a variant spec.
Output: a runner whose .answer() obeys the (pred, meta) contract, with extra
        diagnostic keys added to meta and none of the originals removed.
"""
import copy
import hashlib
import random

from .. import config
from .. import pipeline as pipeline_mod
from ..pipeline import TECQA
from ..stages import stage1_subgraph as stage1
from ..stages import stage2_chain as stage2
from ..prompts.stage3 import DEFAULT_STYLE, PROMPT_STYLES  # noqa: F401
from ..stages import stage3_reason as stage3
from ..utils.timeutil import LANG_EN, LANG_VI, extract_explicit_anchors, parse_ts
from .global_retrieval import GlobalSemanticRetriever

ABLATION_NONE = ""
ABLATION_NO_SG = "no_sg"
ABLATION_NO_KNTN = "no_kntn"
ABLATION_NO_PS = "no_ps"
ABLATIONS = (ABLATION_NONE, ABLATION_NO_SG, ABLATION_NO_KNTN, ABLATION_NO_PS)

_HASH_BITS = 16  # hex chars of the digest used to seed the no_ps shuffle


def _seeded_random(seed: int, question: str) -> random.Random:
    """Python's hash() is salted per process, so derive the shuffle seed from a
    stable digest instead — otherwise `no_ps` is unreproducible across runs."""
    digest = hashlib.sha256(f"{seed}:{question}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:_HASH_BITS], 16))


class TECQARunner(TECQA):
    """One configured variant of TECQA.

    Not safe to share across threads: per-question state (the trace, the last
    subgraph and chain) lives on the instance. Use .clone() per worker; the
    graph, grounders and retriever are shared, only the state is fresh.
    """

    def __init__(self, graph, *, lang=LANG_VI, ablation=ABLATION_NONE,
                 k=config.K_NEIGHBORS, n_implicit=config.N_IMPLICIT_MULTITQ,
                 model=None, model_extract=None, seed=42, reason_params=None,
                 prompt_style=DEFAULT_STYLE, is_cron=False):
        super().__init__(graph, model_extract=model_extract, model_reason=model,
                         k_neighbors=k, n_implicit=n_implicit, is_cron=is_cron)
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {ABLATIONS}")
        self.lang = lang
        self.ablation = ablation
        self.seed = seed
        # None keeps config.REASON_PARAMS (thinking mode);
        # config.REASON_PARAMS_OFF is the paper's "no thinking mode" row of
        # Table 4. Do not pass {} for that -- an empty dict sends no reasoning
        # key and lets the provider decide, which for some backbones means
        # thinking stays on. See the note on REASON_PARAMS_OFF in config.py.
        self.reason_params = reason_params
        # Which Stage-3 template to use. Lives here rather than on TECQA because
        # pipeline.py is frozen (docs/TEAM_PLAN.md H2) and every condition of the
        # paper is a subclass, not a second copy of the pipeline.
        self.prompt_style = prompt_style
        self._retriever = GlobalSemanticRetriever(graph) if ablation == ABLATION_NO_SG else None
        self._reset()

    def _reset(self) -> None:
        self.trace = {}
        self.last_subgraph = []
        self.last_chain = []

    def clone(self) -> "TECQARunner":
        """Shallow copy with fresh per-question state. The graph, grounders and
        BM25 index are large and read-only, so they stay shared."""
        twin = copy.copy(self)
        twin._reset()
        return twin

    # -- Stage 1 ------------------------------------------------------------
    def structure_guided_subgraph(self, question: str, language: str = None):
        if self.ablation == ABLATION_NO_SG:
            facts = self._retriever.retrieve(question)
            self.trace.update(mentions=[], grounded=[], retrieval="global_semantic")
            self.last_subgraph = facts
            return None, facts, set()
        main_entity, f_q, grounded = stage1.structure_guided_subgraph(
            self.graph, self.entity_grounder, self.relation_grounder, question,
            api_key=pipeline_mod.api_key(), model=self.model_extract,
            language=language or self.lang, trace=self.trace)
        self.last_subgraph = f_q
        return main_entity, f_q, grounded

    # -- Stage 2 ------------------------------------------------------------
    def temporal_evidence_chain(self, question: str, f_q: list, *, k=None,
                                n_implicit=None, grounded_entities=None,
                                language: str = None):
        lang = language or self.lang
        neighbours = k or self.k_neighbors
        anchors_wanted = n_implicit or self.n_implicit
        if self.ablation == ABLATION_NO_KNTN:
            chain = self._semantic_chain(question, f_q, neighbours, anchors_wanted, lang)
        else:
            chain = stage2.temporal_evidence_chain(
                self.graph, question, f_q, k=neighbours, n_implicit=anchors_wanted,
                is_cron=self.is_cron, grounded_entities=grounded_entities,
                language=lang, trace=self.trace)
        if self.ablation == ABLATION_NO_PS:
            chain = list(chain)
            _seeded_random(self.seed, question).shuffle(chain)
        self.last_chain = chain
        return chain

    def _semantic_chain(self, question: str, f_q: list, k: int,
                        n_implicit: int, lang: str) -> list:
        """w/o KNTN (Appendix C): keep proximity serialization, but choose the
        facts by semantic similarity instead of temporal distance. The budget is
        k per anchor, matching what temporal pruning would have returned, so the
        ablation isolates WHICH facts are kept rather than how many."""
        if not f_q:
            return []
        from ..utils.grounding import cos_sim_matrix, embed_batch
        anchors = stage2.anchors_for(self.graph, question, f_q, n_implicit,
                                     is_cron=self.is_cron, language=lang,
                                     trace=self.trace)
        texts = [self.graph.fact_text(fact) for fact in f_q]
        sims = cos_sim_matrix(embed_batch([question], input_type="query"),
                              embed_batch(texts, input_type="passage"))[0]
        budget = k * max(1, len(anchors))
        selected = [f_q[i] for i in sorted(range(len(f_q)), key=lambda i: -sims[i])[:budget]]
        if not anchors:
            return selected
        return sorted(selected, key=lambda fact: min(
            abs((stage2.fact_ts(fact, is_cron=self.is_cron) - anchor).days)
            for anchor in anchors))

    # -- Stage 3 ------------------------------------------------------------
    def reason(self, question: str, chain: list, answer_type="entity",
               qlabel="Single", topk=None, grounded_entities=None,
               language: str = None):
        return stage3.reason(self.graph, question, chain,
                             api_key=pipeline_mod.api_key(), model=self.model_reason,
                             answer_type=answer_type, qlabel=qlabel, topk=topk,
                             grounded_entities=grounded_entities,
                             language=language or self.lang,
                             params=self.reason_params,
                             prompt_style=self.prompt_style, trace=self.trace)

    # -- contract -----------------------------------------------------------
    def answer(self, question: str, answer_type="entity", qlabel="Single",
               language: str = None):
        """(pred, meta) contract. meta gains the diagnostic keys of
        docs/EVAL_DESIGN.md Sec 3.4; none of the base keys are removed."""
        self._reset()
        pred, meta = super().answer(question, answer_type=answer_type, qlabel=qlabel,
                                    language=language or self.lang)
        meta.update(self.trace)
        meta.update(lang=self.lang, ablation=self.ablation, k=self.k_neighbors,
                    n_implicit=self.n_implicit, model=self.model_reason,
                    prompt_style=self.prompt_style)
        return pred, meta

    def gold_anchors(self, question: str, main_entity) -> list:
        """The anchor a correct answer has to be measured against
        (docs/EVAL_DESIGN.md Sec 3.3). Explicit dates when the question names
        one; otherwise the timestamps of facts linking the pivot to an auxiliary
        entity. Empty list means "not derivable" — the caller must then drop the
        question from the Anchor Recall denominator rather than scoring it 0."""
        explicit = extract_explicit_anchors(question, self.lang)
        if explicit:
            return explicit
        others = {e for e in self.trace.get("grounded", []) if e != main_entity}
        if not main_entity or not others:
            return []
        return [parse_ts(fact[config.FACT_TIME])
                for fact in self.graph.facts_touching(main_entity)
                if others.intersection((fact[config.FACT_SUBJECT],
                                        fact[config.FACT_OBJECT]))]
