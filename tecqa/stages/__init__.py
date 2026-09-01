"""
The 3 algorithmic stages of TECQA.
"""
from .stage1_subgraph import structure_guided_subgraph
from .stage2_chain import temporal_evidence_chain
from .stage3_reason import reason

__all__ = [
    "structure_guided_subgraph",
    "temporal_evidence_chain",
    "reason",
]
