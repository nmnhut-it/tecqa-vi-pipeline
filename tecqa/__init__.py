"""
TECQA — Temporal Evidence Chain-based Question Answering over Knowledge Graphs.

Layout:
    config.py    models, hyperparameters, API key (paper Sec 5.1)
    pipeline.py  Algorithm 1 orchestrator — the ONLY implementation
    data/        knowledge-graph loaders
    prompts/     Stage-1 and Stage-3 templates, English and Vietnamese
    stages/      the three algorithmic stages
    utils/       grounding, caches, timestamp parsing
    eval/        the evaluation harness (docs/EVAL_DESIGN.md)

Submodules are imported explicitly rather than re-exported here: the scorer and
the notebook's replay mode import tecqa.eval.metrics on machines with no API key
and no torch, and an eager import chain would drag both in.
"""
