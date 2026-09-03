"""
The TECQA evaluation harness (docs/EVAL_DESIGN.md).

    data.py     dataset loading + stratified nested sampling
    metrics.py  Hits@1, the Appendix D recalls, paired agreement counts
    record.py   the results/ writer every run shares (TEAM_PLAN H3)
    variants.py language / ablation / hyperparameter / backbone variants
    run_eval.py the single CLI entry point
    global_retrieval.py  BM25 + dense rerank, for the w/o-SG ablation

`data`, `metrics` and `record` are pure and import without an API key or torch.
`variants` and `run_eval` need the full pipeline, so they are NOT imported here.
"""
