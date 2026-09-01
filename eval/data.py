"""
Question loading and sampling for evaluation (docs/EVAL_DESIGN.md Sec 4).

OWNER: EVAL (docs/TEAM_PLAN.md H5).

Two jobs:

1. Normalize MultiTQ and CronQuestions records into one `Question` type, so the
   runner, the scorer and the table generator never branch on dataset again.
   The two corpora disagree on field names (`quid` vs `uniq_id`, `qtype` vs
   `type`) and on category vocabulary (Single/Multiple vs Simple/Complex).
2. Draw a STRATIFIED, NESTED sample with a committed manifest. Nested matters
   for money: the 200-question pilot is a subset of the 600-question run, so
   growing the sample only pays for the new questions -- the rest are already in
   the LLM cache.

No API key and no third-party imports: the notebook loads this in replay mode.

Input:  dataset name (+ optional HuggingFace base for Colab/Kaggle), size, seed.
Output: list[Question], and a manifest under results/samples/.
"""
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).parent.parent.parent
SAMPLES_DIR = REPO_ROOT / "results" / "samples"

DATASET_MULTITQ = "multitq"
DATASET_CRON = "cron"
DATASETS = (DATASET_MULTITQ, DATASET_CRON)

LANG_VI = "vi"
LANG_EN = "en"

QLABEL_SINGLE = "Single"
QLABEL_MULTIPLE = "Multiple"
QGROUP_SIMPLE = "Simple"
QGROUP_COMPLEX = "Complex"

# Paper Table 8 splits CronQuestions into Simple (one hop) and Complex
# (before/after, first/last, time join). Only the Complex ones expect several
# answers, which is what drives Stage-3 top-k.
_CRON_SIMPLE_TYPES = frozenset({"simple_entity", "simple_time"})

_RELATIVE_PATHS = {
    DATASET_MULTITQ: "dataset_vi/vi/questions_test_vi.json",
    DATASET_CRON: "dataset_vi_cron/vi/questions_test_vi.json",
}
# Layout published on HuggingFace (docs/TEAM_PLAN.md H1, scripts/publish_hf.py):
# ONE dataset repo per corpus, so the questions file sits at the repo root and a
# base is already corpus-specific. That costs nothing here -- run_eval evaluates
# one --dataset per invocation and carries one --data-base with it -- and it
# keeps the two upstream licences in separate repos.
_REMOTE_PATHS = {
    DATASET_MULTITQ: "questions_test_vi.json",
    DATASET_CRON: "questions_test_vi.json",
}
# Set once in a notebook session and every entry point finds the repo, the same
# way TECQA_EMBED_CACHE_DIR works for the Eq. 2 caches. An explicit `base`
# argument still wins.
_ENV_DATA_BASE = {
    DATASET_MULTITQ: "TECQA_DATA_BASE_MULTITQ",
    DATASET_CRON: "TECQA_DATA_BASE_CRON",
}

# Fields the sample is stratified on. time_level is empty for CronQuestions,
# which collapses that dimension for it rather than needing a separate rule.
STRATUM_FIELDS = ("qtype", "answer_type", "time_level")


@dataclass(frozen=True)
class Question:
    """One evaluation item, identical in shape across both corpora.

    qlabel  drives Stage-3 top-k (Single -> 1, Multiple -> 5).
    qgroup  is the reporting label of the source paper: Single/Multiple for
            MultiTQ, Simple/Complex for CronQuestions (Table 1).
    """
    qid: str
    question_vi: str
    question_en: str
    answers: tuple
    answer_type: str
    qtype: str
    qlabel: str
    qgroup: str
    time_level: str = ""

    def text(self, lang: str) -> str:
        return self.question_en if lang == LANG_EN else self.question_vi


def _from_multitq(record: dict) -> Question:
    return Question(qid=str(record["quid"]), question_vi=record["question"],
                    question_en=record.get("question_en", ""),
                    answers=tuple(record["answers"]), answer_type=record["answer_type"],
                    qtype=record["qtype"], qlabel=record["qlabel"], qgroup=record["qlabel"],
                    time_level=record.get("time_level", ""))


def _from_cron(record: dict) -> Question:
    qtype = record["type"]
    simple = qtype in _CRON_SIMPLE_TYPES
    return Question(qid=str(record["uniq_id"]), question_vi=record["question"],
                    question_en=record.get("question_en", ""),
                    answers=tuple(record["answers"]), answer_type=record["answer_type"],
                    qtype=qtype,
                    qlabel=QLABEL_SINGLE if simple else QLABEL_MULTIPLE,
                    qgroup=QGROUP_SIMPLE if simple else QGROUP_COMPLEX)


_ADAPTERS = {DATASET_MULTITQ: _from_multitq, DATASET_CRON: _from_cron}


def remote_base(dataset: str, base: str = None) -> str:
    """HTTP root holding `dataset`: the explicit argument, else the environment.
    Empty means read the repo checkout instead."""
    if dataset not in _ENV_DATA_BASE:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    return (base or os.environ.get(_ENV_DATA_BASE[dataset], "")).strip()


def _read_json(dataset: str, base: str = None):
    root = remote_base(dataset, base)
    if root:
        with urlopen(f"{root.rstrip('/')}/{_REMOTE_PATHS[dataset]}") as response:
            return json.loads(response.read().decode("utf-8"))
    return json.loads((REPO_ROOT / _RELATIVE_PATHS[dataset]).read_text(encoding="utf-8"))


def from_records(dataset: str, records: list) -> list:
    """Normalize records that are already in memory. The notebook pulls the
    questions file through huggingface_hub, which caches it; routing those
    records through here instead of load_questions saves a second 22 MB download
    of the same file over plain HTTP."""
    if dataset not in _ADAPTERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    adapt = _ADAPTERS[dataset]
    return [adapt(record) for record in records]


def load_questions(dataset: str, base: str = None) -> list:
    """Every test question of `dataset`, normalized. `base` is the public root of
    that corpus's HuggingFace repo (H1); omit it to use TECQA_DATA_BASE_* or,
    failing that, the repo checkout."""
    return from_records(dataset, _read_json(dataset, base))


def stratum_of(question: Question) -> tuple:
    return tuple(getattr(question, field) for field in STRATUM_FIELDS)


def _rank(seed: int, qid: str) -> str:
    """Stable per-question order. A hash rather than random.shuffle so the
    sample never depends on RNG state or on the order of the source file."""
    return hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).hexdigest()


def _stratified_order(questions: list, seed: int) -> list:
    """Interleave strata by relative position so that ANY prefix of the result
    is a proportional stratified sample. That is what makes samples nested:
    prefix(200) is a subset of prefix(600) by construction."""
    by_stratum = {}
    for question in questions:
        by_stratum.setdefault(stratum_of(question), []).append(question)
    keyed = []
    for stratum, items in sorted(by_stratum.items()):
        items.sort(key=lambda q: _rank(seed, q.qid))
        for position, question in enumerate(items):
            keyed.append(((position + 0.5) / len(items), stratum, question))
    keyed.sort(key=lambda entry: (entry[0], entry[1]))
    return [question for _, _, question in keyed]


def build_sample(questions: list, n: int, seed: int) -> list:
    """The first `n` questions of the stratified order (docs/EVAL_DESIGN.md 4)."""
    return _stratified_order(questions, seed)[:max(0, min(n, len(questions)))]


def make_sample_id(dataset: str, n: int, seed: int) -> str:
    return f"{dataset}_n{n}_seed{seed}"


def strata_counts(sample: list) -> dict:
    counts = {}
    for question in sample:
        key = "|".join(stratum_of(question))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_manifest(sample: list, dataset: str, n: int, seed: int, name: str = "") -> Path:
    """Freeze a sample so every run and every re-run scores the same questions.
    Committed to the repo; `.meta.json` of each run points back at it."""
    sample_id = name or make_sample_id(dataset, n, seed)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"{sample_id}.json"
    path.write_text(json.dumps({
        "sample_id": sample_id, "dataset": dataset, "split": "test",
        "n": len(sample), "seed": seed, "strata": strata_counts(sample),
        "quids": [question.qid for question in sample],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_manifest(name_or_path) -> dict:
    path = Path(name_or_path)
    if not path.exists():
        path = SAMPLES_DIR / f"{name_or_path}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select(questions: list, qids) -> list:
    """Pick questions by id, preserving the order given in `qids` so a manifest
    replays identically."""
    wanted = {str(qid) for qid in qids}
    by_id = {question.qid: question for question in questions if question.qid in wanted}
    return [by_id[str(qid)] for qid in qids if str(qid) in by_id]


def as_row(question: Question) -> dict:
    """Plain dict form, for writing a sample into a log or a notebook table."""
    return asdict(question)
