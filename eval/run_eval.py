"""
The single evaluation entry point for TECQA (docs/EVAL_DESIGN.md).

OWNER: EVAL (docs/TEAM_PLAN.md H5).

Every experiment in the paper's Section 5 is this one script with different
flags — language, ablation, backbone, K/N, sample — so there is exactly one
implementation of Algorithm 1 being measured. Variants live in
tecqa/eval/variants.py, which subclasses tecqa.pipeline.TECQA rather than
copying it.

    python -m tecqa.eval.run_eval --dry-run --n 200          # plan + cost, no API
    python -m tecqa.eval.run_eval --lang vi --sample multitq_n200_seed42
    python -m tecqa.eval.run_eval --lang en --sample multitq_n200_seed42
    python -m tecqa.eval.run_eval --ablation no_ps --sample multitq_n200_seed42
    python -m tecqa.eval.run_eval --model deepseek/deepseek-chat --k 10

Work is queued in Redis, so the command is safe to interrupt and safe to repeat:
re-running resumes, and a question already answered is never paid for twice.
Run the same command in a second shell to add a worker to a run in progress.

Output: results/<run_id>.jsonl + results/<run_id>.meta.json (schema:
results/README.md). Scoring lives in metrics.py, sampling in data.py.
"""
import argparse
import json
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .. import config
from ..prompts import stage3 as prompts_stage3
from . import data as eval_data
from . import metrics
from .record import QuestionResult, RunConfig, RunRecorder, make_run_id

# Worst-case LLM calls per question, ignoring the cache. Stage 1 spends three
# (entity, relation, main entity) and Stage 3 one; w/o SG has no Stage-1 LLM.
CALLS_STAGE1 = 3
CALLS_STAGE3 = 1
ABLATION_NO_SG = "no_sg"
DEFAULT_SEED = 42
DEFAULT_N = 200
DEFAULT_MAX_USD = 5.0
TOPK_SINGLE = 1
TOPK_MULTIPLE = 5

# Paper Sec 5.4: point-in-time MultiTQ facts need fewer implicit anchors than
# CronQuestions intervals.
N_IMPLICIT_BY_DATASET = {eval_data.DATASET_MULTITQ: config.N_IMPLICIT_MULTITQ,
                         eval_data.DATASET_CRON: config.N_IMPLICIT_CRON}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset", default=eval_data.DATASET_MULTITQ,
                        choices=eval_data.DATASETS)
    parser.add_argument("--lang", default=eval_data.LANG_VI,
                        choices=[eval_data.LANG_VI, eval_data.LANG_EN])
    parser.add_argument("--ablation", default="",
                        choices=["", "no_sg", "no_kntn", "no_ps"])
    parser.add_argument("--model", default=None, help="Stage-3 reasoning model")
    parser.add_argument("--model-extract", default=None, help="Stage-1 parsing model")
    parser.add_argument("--stage3-prompt", default=prompts_stage3.DEFAULT_STYLE,
                        choices=list(prompts_stage3.PROMPT_STYLES),
                        help="Stage-3 template: 'paper' reproduces the published "
                             "prompt, 'strict' is ours (matched across languages)")
    parser.add_argument("--embed-backend", default=config.EMBED_BACKEND,
                        choices=config.EMBED_BACKENDS,
                        help="grounding encoder: 'local' is the paper's "
                             "bge-base-en-v1.5, 'openrouter' needs no torch")
    parser.add_argument("--embed-model", default=None,
                        help="embedding model id for --embed-backend openrouter")
    parser.add_argument("--no-thinking", action="store_true",
                        help="disable reasoning traces (paper Table 4 comparison)")
    parser.add_argument("--k", type=int, default=config.K_NEIGHBORS)
    parser.add_argument("--n-implicit", type=int, default=None,
                        help="defaults to 2 for MultiTQ, 5 for CronQuestions (paper 5.4)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="sample size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample", default=None, help="manifest name or path")
    parser.add_argument("--quids", nargs="*", default=None, help="score these ids only")
    parser.add_argument("--limit", type=int, default=None, help="truncate after N questions")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--label", default="", help="LaTeX-safe name for the paper table")
    parser.add_argument("--data-base", default=None, help="public HTTP root (Colab, H1)")
    parser.add_argument("--kg-path", default=None, help="override the MultiTQ full.txt path")
    parser.add_argument("--freeze-sample", action="store_true",
                        help="write the sample manifest to results/samples/ and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the call estimate; touch no API")
    parser.add_argument("--no-cache", action="store_true", help="bypass the disk caches")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD,
                        help="hard spending cap; the run also stops before the "
                             "OpenRouter account's own remaining credit")
    parser.add_argument("--fresh", action="store_true",
                        help="discard this run_id's stored answers and start over")
    parser.add_argument("--dump-only", action="store_true",
                        help="write results/ from what is already stored, run nothing")
    parser.add_argument("--no-redis", action="store_true",
                        help="skip the shared queue; disk journal only, single process")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the legacy eval_results.json shape")
    return parser


def resolve_sample(args) -> tuple:
    """(questions, sample_id). --quids beats --sample beats --n/--seed."""
    questions = eval_data.load_questions(args.dataset, args.data_base)
    if args.quids:
        return eval_data.select(questions, args.quids), f"adhoc_{len(args.quids)}"
    if args.sample:
        manifest = eval_data.read_manifest(args.sample)
        return eval_data.select(questions, manifest["quids"]), manifest["sample_id"]
    sample = eval_data.build_sample(questions, args.n, args.seed)
    return sample, eval_data.make_sample_id(args.dataset, len(sample), args.seed)


def estimate_calls(n: int, ablation: str) -> dict:
    stage1 = 0 if ablation == ABLATION_NO_SG else CALLS_STAGE1
    return {"llm_calls": n * (stage1 + CALLS_STAGE3),
            "stage1_calls": n * stage1, "stage3_calls": n * CALLS_STAGE3}


def print_plan(args, sample, sample_id: str) -> None:
    estimate = estimate_calls(len(sample), args.ablation)
    print(f"dataset={args.dataset} lang={args.lang} ablation={args.ablation or 'none'} "
          f"K={args.k} N={args.n_implicit} model={args.model or 'config default'}")
    print(f"sample={sample_id} n={len(sample)}")
    print(f"strata: {json.dumps(eval_data.strata_counts(sample), ensure_ascii=False)}")
    print(f"worst-case LLM calls (cache cold): {estimate['llm_calls']} "
          f"= {estimate['stage1_calls']} Stage-1 + {estimate['stage3_calls']} Stage-3")
    print("Stage-1 calls are shared with any other run over the same sample, "
          "language and parsing model — they hit the cache and cost nothing.")


def load_graph(dataset: str, kg_path: str = None):
    if dataset == eval_data.DATASET_CRON:
        raise SystemExit("CronQuestions needs a tecqa/data/kg_cron.py, which does not "
                         "exist yet (interval midpoints, N=5 — docs/EVAL_DESIGN.md).")
    from ..data.kg_multitq import KG_PATH, MultiTQGraph
    return MultiTQGraph(Path(kg_path) if kg_path else KG_PATH).load()


def make_runner(graph, args):
    from .variants import TECQARunner
    return TECQARunner(graph, lang=args.lang, ablation=args.ablation, k=args.k,
                       n_implicit=args.n_implicit, model=args.model,
                       model_extract=args.model_extract, seed=args.seed,
                       reason_params=config.REASON_PARAMS_OFF if args.no_thinking else None,
                       prompt_style=args.stage3_prompt,
                       is_cron=args.dataset == eval_data.DATASET_CRON)


def apply_embed_settings(args) -> None:
    """Point config at the requested encoder before any grounder is built.

    Module-level assignment rather than a constructor argument, because
    embedder.get_embedder() reads config at call time (CLAUDE.md: the same rule
    that keeps MODEL_EXTRACT/MODEL_REASON monkeypatchable for a sweep).
    """
    config.EMBED_BACKEND = args.embed_backend
    if args.embed_model:
        config.EMBED_MODEL_OPENROUTER = args.embed_model


def embed_tag(args) -> str:
    """Run-id fragment naming the encoder, empty for the paper's default.

    Results from two encoders must never share a journal: resume would treat
    the other model's answers as already paid for, and the meta.json would
    average two incomparable retrieval systems into one score.
    """
    if args.embed_backend == config.EMBED_BACKEND_LOCAL:
        return ""
    slug = (args.embed_model or config.EMBED_MODEL_OPENROUTER).split("/")[-1]
    return "_emb-" + slug


def run_id_for(args, sample, model: str) -> str:
    """Only non-default hyperparameters enter the id, so the K/N sweep does not
    overwrite the main run and the main run keeps its short name."""
    swept_k = args.k if args.k != config.K_NEIGHBORS else None
    default_n = N_IMPLICIT_BY_DATASET[args.dataset]
    swept_n = args.n_implicit if args.n_implicit != default_n else None
    run_id = make_run_id(args.dataset, args.lang, model, len(sample), args.ablation,
                         swept_k, swept_n)
    if args.stage3_prompt != prompts_stage3.DEFAULT_STYLE:
        run_id = f"{run_id}_{args.stage3_prompt}"
    run_id += embed_tag(args)
    if config.SELF_HOSTED_LLM:
        run_id += "_" + config.ENDPOINT_SELF_HOSTED
    return f"{run_id}_nothink" if args.no_thinking else run_id


def diagnostics(runner, question, meta: dict) -> dict:
    """The retrieval-side metrics of Appendix D, computed from fact sets the run
    already produced — no extra API call (docs/EVAL_DESIGN.md Sec 3.2)."""
    gold = list(question.answers)
    anchors = runner.gold_anchors(question.text(runner.lang), meta.get("main_entity"))
    return {
        "answer_recall_sg": metrics.answer_recall(runner.last_subgraph, gold,
                                                  question.answer_type),
        "answer_recall_chain": metrics.answer_recall(runner.last_chain, gold,
                                                     question.answer_type),
        "anchor_derivable": bool(anchors),
        "anchor_recall": metrics.anchor_recall(runner.last_chain, anchors),
        "chain_recall": metrics.chain_recall(runner.last_chain, gold,
                                             question.answer_type, anchors),
    }


def score_question(runner, question) -> QuestionResult:
    text = question.text(runner.lang)
    try:
        pred, meta = runner.answer(text, answer_type=question.answer_type,
                                   qlabel=question.qlabel)
    except Exception as exc:  # one bad question must not abandon the run
        pred, meta = [], {"error": str(exc)}
    meta.update(diagnostics(runner, question, meta))
    gold = list(question.answers)
    return QuestionResult(qid=question.qid, question=text, gold=gold, pred=pred,
                          hit=metrics.hits_at_1(pred, gold),
                          hit_gran=metrics.hit_granular(pred, gold, question.answer_type),
                          qtype=question.qtype, qlabel=question.qlabel,
                          qgroup=question.qgroup, answer_type=question.answer_type,
                          time_level=question.time_level, meta=meta)


_local = threading.local()


def _worker_runner(base):
    """One runner per thread: the runner keeps per-question state, so sharing a
    single instance across threads would interleave traces."""
    if not hasattr(_local, "runner"):
        _local.runner = base.clone()
    return _local.runner


def evaluate(runner, store, by_qid: dict, budget, workers: int) -> int:
    """Pull questions off the Redis queue until it drains or the budget is gone.

    Every answer is written to Redis the moment it is produced, so killing this
    process at any point loses at most the questions currently in flight — and
    those go back on the queue for the next run. Threads rather than processes:
    Stage 3 spends its time blocked on a slow reasoning model, and a second
    interpreter would need another copy of the 461k-fact graph and the encoder.
    """
    started = time.time()
    lock = threading.Lock()
    stopped = threading.Event()
    scored = {"n": 0, "hits": 0}
    total = sum(store.counts()[k] for k in ("queued", "inflight"))

    def announce_stop() -> None:
        if not stopped.is_set():
            stopped.set()
            print(f"\n!! BUDGET REACHED at ${budget.total_usd:.3f} of "
                  f"${budget.hard_cap_usd:.2f} — stopping.\n"
                  f"   Everything answered is safe in Redis. Re-run the same "
                  f"command to continue; nothing is ever re-sent.", flush=True)

    def worker() -> None:
        while not stopped.is_set():
            if budget.over_budget():
                return announce_stop()
            qid = store.claim()
            if qid is None:
                return  # queue drained
            question = by_qid.get(qid)
            if question is None:  # sample changed under us; drop it, don't spend
                store.complete(qid, {"qid": qid, "skipped": "not in current sample"})
                continue
            if budget.over_budget():
                store.release(qid)
                return announce_stop()
            result = score_question(_worker_runner(runner), question)
            store.complete(qid, asdict(result))
            with lock:
                scored["n"] += 1
                scored["hits"] += int(result.hit)
                print(f"  [{scored['n']}/{total}] hit={result.hit} "
                      f"acc={scored['hits'] / scored['n']:.3f} "
                      f"spent=${budget.total_usd:.3f} "
                      f"({time.time() - started:.0f}s)", flush=True)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return scored["n"]


def make_config(args, sample, sample_id: str, model: str) -> RunConfig:
    from ..utils.grounding import active_model_id
    return RunConfig(dataset=args.dataset, lang=args.lang, model=model, n=len(sample),
                     k=args.k, n_implicit=args.n_implicit, ablation=args.ablation,
                     label=args.label, seed=args.seed, sample_id=sample_id,
                     model_extract=args.model_extract or config.MODEL_EXTRACT,
                     embed_model=active_model_id(),
                     topk_single=TOPK_SINGLE, topk_multiple=TOPK_MULTIPLE,
                     prompt_style=args.stage3_prompt,
                     no_thinking=args.no_thinking,
                     llm_endpoint=config.llm_endpoint())


def write_outputs(args, store, sample, sample_id: str, model: str, budget) -> None:
    """Snapshot Redis into the files everything downstream reads (contract H3).

    Written even when the run stopped early, because a partial run is still a
    real measurement of the questions it did answer — and scripts/make_tables.py
    reports n alongside every number.
    """
    # Read the answers BEFORE opening the recorder. RunRecorder(resume=False)
    # truncates results/<run_id>.jsonl, and for the no-Redis store that file IS
    # the journal it answers from -- so constructing the recorder first deleted
    # every answer the run had just paid for, and the run reported n=0 with no
    # error. The Redis store survived it only because rows() merges Redis in.
    rows = store.rows()
    recorder = RunRecorder(run_id_for(args, sample, model),
                           make_config(args, sample, sample_id, model), resume=False)
    scored = [row for row in rows if "skipped" not in row]
    for row in scored:
        recorder.add(QuestionResult(**row))
    summary = metrics.summarize(scored)
    meta_path = recorder.close(summary)
    print(f"\n=== Hits@1: {recorder.hits}/{recorder.total} = {recorder.hits_at_1:.3f} ===")
    print(f"=== spent on this run: ${budget.total_usd:.4f} ===")
    print(json.dumps(summary, indent=2))
    if args.out:
        write_legacy(args.out, [QuestionResult(**row) for row in scored], recorder.hits_at_1)
    print(f"Saved -> {recorder.path.name} and {meta_path.name}")


def write_legacy(path: Path, rows, hits_at_1: float) -> None:
    """The old eval_results.json shape, kept so the rerun_*.py helpers still
    load. Only written when --out is given."""
    payload = [{"quid": row.qid, "question": row.question, "qtype": row.qtype,
                "gold": row.gold, "pred": row.pred, "hit@1": row.hit, "meta": row.meta}
               for row in rows]
    path.write_text(json.dumps({"hits_at_1": hits_at_1, "n": len(rows), "results": payload},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.no_cache:
        os.environ["TECQA_CACHE"] = "0"
    args.n_implicit = args.n_implicit or N_IMPLICIT_BY_DATASET[args.dataset]
    apply_embed_settings(args)
    sample, sample_id = resolve_sample(args)
    if args.limit:
        sample = sample[:args.limit]
    print_plan(args, sample, sample_id)
    if args.freeze_sample:
        path = eval_data.write_manifest(sample, args.dataset, len(sample), args.seed, sample_id)
        return print(f"frozen -> {path}")
    if args.dry_run:
        return print("dry run: no API call made, no file written.")

    from .. import pipeline
    from ..utils import embedder, llm_cache
    from .budget import effective_cap
    from .store import ReadOnlyBudget, open_store, spent_so_far

    model = args.model or config.MODEL_REASON
    run_id = run_id_for(args, sample, model)
    store, budget_class, queue_note = open_store(run_id, prefer_redis=not args.no_redis)
    print(queue_note)
    if args.fresh:
        store.clear()
        print(f"cleared previous state for {run_id}")

    # --dump-only is read-only: it must not touch the cap of a run that another
    # process is executing right now.
    if args.dump_only:
        budget = ReadOnlyBudget(store)
    else:
        spent = spent_so_far(store)
        cap, headroom = effective_cap(args.max_usd, pipeline.api_key(), spent)
        if cap < args.max_usd:
            print(f"budget: cap ${cap:.2f} — LIMITED BY ACCOUNT (${headroom:.2f} credit left, "
                  f"you asked for ${args.max_usd:.2f})")
        else:
            print(f"budget: cap ${cap:.2f} (${headroom:.2f} credit available)")
        if cap <= spent:
            raise SystemExit(f"Already spent ${spent:.2f} on this run and the account has "
                             f"${headroom:.2f} left; nothing further to run.")
        budget = budget_class(store, cap)
        llm_cache.set_budget(budget)
        embedder.set_budget(budget)  # hosted encoders bill; keep --max-usd honest

    if args.dump_only:
        # Strictly read-only: enqueue() rewrites the queue, and doing that under
        # a running worker is what duplicated work the first time.
        pending = 0
    else:
        reclaimed = store.requeue_stale()
        if reclaimed:
            print(f"requeued {reclaimed} question(s) left in flight by an earlier worker")
        pending = store.enqueue(sample)
    already = store.counts()["done"]
    print(f"run {run_id}: {already} already scored, {pending} pending, "
          f"${budget.total_usd:.3f} spent so far")

    if pending and not args.dump_only:
        runner = make_runner(load_graph(args.dataset, args.kg_path), args)
        evaluate(runner, store, {q.qid: q for q in sample}, budget, args.parallel)
    elif not pending:
        print("nothing pending — every question in this sample is already scored.")

    write_outputs(args, store, sample, sample_id, model, budget)


if __name__ == "__main__":
    main()
