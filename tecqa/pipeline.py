"""
TECQA Pipeline with Multi-language support (English and Vietnamese).

Orchestrates all three stages:
  1. Structure-Guided Subgraph Construction (paper Sec 4.2)
  2. Temporal Evidence Chain Construction (paper Sec 4.3)
  3. LLM Reasoning (paper Sec 4.4)

Supports:
  - English: language="en" (default, Paper Appendix F.3 verbatim)
  - Vietnamese: language="vi" (Option B cross-lingual grounding + Vietnamese temporal engine)

This is the ONLY implementation of Algorithm 1. Evaluation variants (ablations,
K/N sweeps, backbone swaps) subclass it in tecqa/eval/variants.py rather than
copying it; see docs/EVAL_DESIGN.md Sec 2.

Usage:
  from tecqa.data import MultiTQGraph
  from tecqa.pipeline import TECQA

  graph = MultiTQGraph().load()
  tecqa = TECQA(graph)
  answers, meta = tecqa.answer("Lúc nào Mallam Isa Yuguda đi thăm Ethiopia?",
                               answer_type="time", qlabel="Single", language="vi")
"""
from . import config
from .data.kg_multitq import MultiTQGraph
from .stages.stage1_subgraph import structure_guided_subgraph
from .stages.stage2_chain import temporal_evidence_chain
from .stages.stage3_reason import reason
from .utils.grounding import EntityGrounder, RelationGrounder

# Module-level so a caller can monkeypatch the backbone for one run
# (scripts/experiment_matrix.py, the Table-4 sweep). Every read below goes
# through the module global at CALL time — binding these as __init__ defaults
# would freeze them at import and make patching silently do nothing.
MODEL_EXTRACT = config.MODEL_EXTRACT
MODEL_REASON = config.MODEL_REASON

EMPTY_META_TEMPLATE = {
    "subgraph_size": 0,
    "chain_size": 0,
}


def api_key() -> str:
    """Resolved per call, not at import: --dry-run, the offline scorer and the
    notebook's replay mode all import this module on machines with no key."""
    return config.load_api_key()


class TECQA:
    """End-to-end TECQA QA system over a temporal knowledge graph."""

    def __init__(self, graph: MultiTQGraph, *,
                 model_extract: str = None,
                 model_reason: str = None,
                 k_neighbors: int = config.K_NEIGHBORS,
                 n_implicit: int = config.N_IMPLICIT_MULTITQ,
                 is_cron: bool = False):
        self.graph = graph
        self._model_extract = model_extract
        self._model_reason = model_reason
        self.k_neighbors = k_neighbors
        self.n_implicit = n_implicit
        self.is_cron = is_cron  # True for CronQuestions: enables midpoint normalization

        self.entity_grounder = EntityGrounder(graph.entities)
        self.relation_grounder = RelationGrounder(graph.relations)

    @property
    def model_extract(self) -> str:
        return self._model_extract or MODEL_EXTRACT

    @property
    def model_reason(self) -> str:
        return self._model_reason or MODEL_REASON

    def structure_guided_subgraph(self, question: str, language: str = "en"):
        """Stage 1: returns (main_entity, f_q, grounded_entities)."""
        return structure_guided_subgraph(
            self.graph, self.entity_grounder, self.relation_grounder,
            question, api_key=api_key(), model=self.model_extract,
            language=language,
        )

    def temporal_evidence_chain(self, question: str, f_q: list, *,
                                k=None, n_implicit=None,
                                grounded_entities=None,
                                language: str = "en"):
        """Stage 2: returns proximity-ordered evidence chain."""
        return temporal_evidence_chain(
            self.graph, question, f_q,
            k=k or self.k_neighbors,
            n_implicit=n_implicit or self.n_implicit,
            is_cron=self.is_cron,
            grounded_entities=grounded_entities,
            language=language,
        )

    def reason(self, question: str, chain: list, answer_type="entity",
               qlabel="Single", topk=None, grounded_entities=None, language: str = "en"):
        """Stage 3: returns predicted answer list."""
        return reason(
            self.graph, question, chain, api_key=api_key(), model=self.model_reason,
            answer_type=answer_type, qlabel=qlabel, topk=topk,
            grounded_entities=grounded_entities, language=language,
        )

    def answer(self, question: str, answer_type="entity", qlabel="Single",
               language: str = "en"):
        """
        Full pipeline (Algorithm 1). Returns (predictions, meta).
        meta keys: main_entity, subgraph_size, chain_size.

        Subclasses may ADD meta keys; nothing here may be removed or renamed,
        because results/ files and scripts/make_tables.py read them by name.
        """
        main_entity, f_q, grounded = self.structure_guided_subgraph(question, language=language)
        if not f_q:
            return [], {"main_entity": main_entity, **EMPTY_META_TEMPLATE}
        chain = self.temporal_evidence_chain(question, f_q, grounded_entities=grounded, language=language)
        pred = self.reason(question, chain, answer_type=answer_type, qlabel=qlabel,
                           grounded_entities=grounded, language=language)
        return pred, {
            "main_entity": main_entity,
            "subgraph_size": len(f_q),
            "chain_size": len(chain),
        }
