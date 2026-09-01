"""
Shared writer for evaluation results (docs/TEAM_PLAN.md H3).

Every eval — whoever runs it, whatever variant — writes the same two files, so
scripts/make_tables.py can turn any of them into paper numbers without anyone
coordinating. Drop a file into results/ and you are done; no merge, no message.

Usage:
    from tecqa.eval.record import RunRecorder, RunConfig, make_run_id
    rec = RunRecorder(make_run_id("multitq", "vi", model, n), RunConfig(...))
    rec.add(QuestionResult(qid=..., question=..., gold=[...], pred=[...], ...))
    rec.close()   # -> results/<run_id>.jsonl + results/<run_id>.meta.json

Related: results/README.md (schema), scripts/make_tables.py (consumer).
"""
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
UNKNOWN_COMMIT = "unknown"


@dataclass
class QuestionResult:
    qid: str
    question: str
    gold: list
    pred: list
    hit: bool
    qtype: str = ""
    qlabel: str = ""
    answer_type: str = ""
    meta: dict = field(default_factory=dict)
    # Added for the paper's breakdowns and the Sec 3.4 diagnostics; older files
    # simply lack them and every consumer reads with .get().
    hit_gran: bool = False
    time_level: str = ""
    qgroup: str = ""


@dataclass
class RunConfig:
    """Everything needed to reproduce the run. `ablation` is "" for full TECQA,
    otherwise one of: no_sg, no_kntn, no_ps (paper Table 2)."""
    dataset: str
    lang: str
    model: str
    n: int
    k: int
    n_implicit: int
    ablation: str = ""
    label: str = ""  # short LaTeX-safe name for the paper table; defaults to run_id
    # Reproducibility fields (docs/EVAL_DESIGN.md Sec 6). Additive: the two
    # pilot runs in results/ predate them.
    seed: int = 0
    sample_id: str = ""
    model_extract: str = ""
    embed_model: str = ""
    topk_single: int = 0
    topk_multiple: int = 0
    # 'paper' or 'strict' (tecqa/prompts/stage3.py). Empty on the runs that
    # predate the strict prompt; those were all 'paper'.
    prompt_style: str = ""
    # Paper Table 4's thinking-mode contrast. Recorded because a condition that
    # only shows up in the run id cannot be paired or grouped downstream.
    no_thinking: bool = False
    # Which server produced the Stage-3 answers: "openrouter" or "vllm"
    # (tecqa/config.llm_endpoint). Same model id, different server, is not the
    # same condition -- precision, sampler defaults and how thinking mode is
    # requested all differ -- so a run that omits this cannot be paired with
    # one that has it. Empty on the runs that predate the self-hosted path;
    # all of those were openrouter.
    llm_endpoint: str = ""


def make_run_id(dataset: str, lang: str, model: str, n: int,
                ablation: str = "", k: int = None, n_implicit: int = None) -> str:
    """e.g. multitq_vi_qwen3-8b_200 — model vendor prefix dropped for brevity.

    Ablation and non-default hyperparameters are appended so that variants of
    the same sample never overwrite each other's files."""
    parts = [dataset, lang, model.split("/")[-1], str(n)]
    if ablation:
        parts.append(ablation)
    if k is not None:
        parts.append(f"k{k}")
    if n_implicit is not None:
        parts.append(f"n{n_implicit}")
    return "_".join(parts)


def _git_commit() -> str:
    """The commit that produced this run, for provenance.

    TECQA_GIT_COMMIT wins when set: the deployed copy on the VPS is a file
    tree, not a clone, so `git rev-parse` there fails and every result it
    produced would be stamped "unknown" -- unattributable to any code version.
    The deploy script writes the commit it shipped and the runner exports it.
    """
    stamped = os.environ.get("TECQA_GIT_COMMIT", "").strip()
    if stamped:
        return stamped
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=RESULTS_DIR.parent, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return UNKNOWN_COMMIT


class RunRecorder:
    """Streams one JSON object per question, then writes the config sidecar.

    Resumable by default. A run that stops early — budget exhausted, network
    died, laptop closed — still leaves a complete .jsonl of everything scored so
    far, and re-invoking the same run_id appends only the questions that are
    missing. That is what makes growing the dataset cheap: questions already
    answered are never re-sent, whatever state the LLM cache is in.
    """

    def __init__(self, run_id: str, run_config: RunConfig, out_dir: Path = RESULTS_DIR,
                 resume: bool = True):
        self.run_id = run_id
        self.run_config = run_config
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{run_id}.jsonl"
        previous = self._read_existing() if resume else []
        self.scored_qids = {row["qid"] for row in previous}
        self.hits = sum(1 for row in previous if row.get("hit"))
        self.total = len(previous)
        self._handle = self.path.open("a" if resume else "w", encoding="utf-8")

    def _read_existing(self) -> list:
        """Rows already on disk for this run_id. A truncated final line (killed
        mid-write) is dropped rather than aborting the whole resume."""
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def pending(self, questions) -> list:
        """The subset of `questions` this run has not scored yet."""
        return [q for q in questions if q.qid not in self.scored_qids]

    def all_rows(self) -> list:
        """Every row for this run_id, resumed and new. The summary has to cover
        both, not just what this process happened to add."""
        self._handle.flush()
        return self._read_existing()

    def add(self, result: QuestionResult) -> None:
        if result.qid in self.scored_qids:
            return  # never double-count a resumed question
        self._handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        self._handle.flush()  # partial runs stay usable if the API budget runs out
        self.scored_qids.add(result.qid)
        self.total += 1
        self.hits += int(result.hit)

    @property
    def hits_at_1(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def close(self, summary: dict = None) -> Path:
        """`summary` carries the aggregate diagnostics from eval_metrics
        (recalls, empty-subgraph and parse-failure rates) so the table generator
        does not have to recompute them from the rows."""
        self._handle.close()
        meta = {**asdict(self.run_config),
                "label": self.run_config.label or self.run_id,
                "hits_at_1": self.hits_at_1, "n_scored": self.total,
                **(summary or {}),
                "git_commit": _git_commit(),
                "created_at": datetime.now(timezone.utc).isoformat()}
        path = self.out_dir / f"{self.run_id}.meta.json"
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
