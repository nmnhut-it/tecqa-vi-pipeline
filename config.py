"""
Shared TECQA configuration — original English implementation following the paper.

Models, hyperparameters, and API key loading are all kept here as the single
source of truth. Pipeline and stage modules read config at call time.

Paper: "TECQA: Temporal Evidence Chain-based Question Answering over Knowledge Graphs"
Sec 5.1 — Implementation Details.
"""
import os
from pathlib import Path

_DOTENV_PATH = Path(__file__).parent / ".env"

try:
    from dotenv import load_dotenv
except ImportError:  # scorer / notebook replay: no key needed, no dep needed
    load_dotenv = None
else:
    # Load tecqa/.env so OPENROUTER_API_KEY is available via os.environ.
    # override=False means a value already set in the shell takes precedence.
    load_dotenv(_DOTENV_PATH, override=False)

# Backbones, taken verbatim from paper Sec 5.1 so our numbers are comparable to
# its reported 0.811 on MultiTQ. Grounding uses bge-base-en-v1.5 (also the
# paper's choice); it works for Vietnamese only because Stage 1 already
# translates mentions into the English ICEWS schema before grounding runs.
MODEL_EXTRACT = "google/gemini-2.5-flash"
MODEL_REASON = "qwen/qwen3-8b"

# Where the chat-completions calls go. Set TECQA_LLM_BASE_URL to a vLLM server
# to self-host the reasoner instead of renting it per token, and TECQA_LLM_MODEL
# to the one model that server holds -- Stage 1 still needs OpenRouter, so a
# base URL alone would point its parsing calls at a server that has never heard
# of that model (see dataset_vi/build_vi_dataset_or.endpoint_for). Read once:
# the endpoint does not move mid-process the way MODEL_REASON does during a
# backbone sweep.
SELF_HOSTED_URL = os.environ.get("TECQA_LLM_BASE_URL", "")
SELF_HOSTED_MODEL = os.environ.get("TECQA_LLM_MODEL", "")

# True when Stage 3's default backbone is the one the pod serves. Drives the
# thinking-mode flag below and the run's endpoint tag, so a self-hosted result
# is never filed next to a rented one under the same name.
SELF_HOSTED_LLM = bool(SELF_HOSTED_URL) and SELF_HOSTED_MODEL in ("", MODEL_REASON)

# Free-text tag for meta.json and the run id: which server produced the
# Stage-3 answers. Two runs of the same model on different servers are not
# interchangeable -- weights load at a different precision, the sampler
# defaults differ, and thinking mode is requested through a different field.
ENDPOINT_OPENROUTER = "openrouter"
ENDPOINT_SELF_HOSTED = "vllm"


def llm_endpoint() -> str:
    return ENDPOINT_SELF_HOSTED if SELF_HOSTED_LLM else ENDPOINT_OPENROUTER

# Thinking-mode flag: OpenRouter's `reasoning` field vs vLLM's chat-template
# kwarg. A vLLM server silently ignores the wrong one instead of erroring, so
# sending the right one per endpoint matters -- that mismatch once made our
# "no thinking" condition secretly measure the thinking condition.
REASON_PARAMS = ({"chat_template_kwargs": {"enable_thinking": True}}
                 if SELF_HOSTED_LLM else {"reasoning": {"enabled": True}})

# Must be explicit: omitting `reasoning` leaves the choice to the provider,
# and the default isn't always off (DeepSeek-V4-Flash: 117s/3,429 tokens
# when omitted vs 5.5s/5 tokens when disabled by name).
REASON_PARAMS_OFF = ({"chat_template_kwargs": {"enable_thinking": False}}
                     if SELF_HOSTED_LLM
                     else {"reasoning": {"enabled": False, "exclude": True}})

REASON_MAX_TOKENS = 8192

# Embedding backend for Eq. 2 grounding (utils/embedder.py). 'local' is the
# paper's bge-base-en-v1.5, in-process. 'openrouter' sends the same texts to a
# hosted model instead -- no torch, so the pipeline fits a small VPS. The two
# are NOT interchangeable mid-experiment: different vector spaces mean a score
# difference could be the encoder, not the thing under test. Corpus caches are
# keyed by model id, and the active id is written into every run's meta.json.
EMBED_BACKEND_LOCAL = "local"
EMBED_BACKEND_OPENROUTER = "openrouter"
EMBED_BACKENDS = (EMBED_BACKEND_LOCAL, EMBED_BACKEND_OPENROUTER)

EMBED_BACKEND = os.environ.get("TECQA_EMBED_BACKEND", EMBED_BACKEND_LOCAL)

EMBED_MODEL_LOCAL = "BAAI/bge-base-en-v1.5"      # paper Sec 5.1, verbatim
EMBED_MODEL_OPENROUTER = os.environ.get("TECQA_EMBED_MODEL", "baai/bge-m3")

# MPS hangs on M1 inside sentence-transformers encode(), so CPU is used.
EMBED_DEVICE = "cpu"
EMBED_BATCH_LOCAL = 128
EMBED_BATCH_REMOTE = 256    # texts per HTTP request when building the corpus
EMBED_RETRIES = 4
EMBED_BACKOFF_BASE = 1.5
TIMEOUT_EMBED = 120


def embed_model_id() -> str:
    """The embedding model for the active backend. Read at call time so a sweep
    can switch backends by assigning to EMBED_BACKEND."""
    if EMBED_BACKEND == EMBED_BACKEND_OPENROUTER:
        return EMBED_MODEL_OPENROUTER
    return EMBED_MODEL_LOCAL


# Hyperparameters (paper Sec 5.4). K: top-K temporal neighbours per anchor
# (Eq. 7-8). N: implicit anchor facts (Eq. 6) -- CronQuestions uses 5 instead
# of MultiTQ's 2 since interval timestamps need broader anchor coverage.
K_NEIGHBORS = 40
N_IMPLICIT_MULTITQ = 2
N_IMPLICIT_CRON = 5

# Positional layout of a KG fact tuple (subject, predicate, object, timestamp).
FACT_SUBJECT, FACT_PREDICATE, FACT_OBJECT, FACT_TIME = 0, 1, 2, 3

# MultiTQ/CronQuestions answer_type values. Stage 3 formats a time answer to the
# unit the question asks for and the scorer compares at the gold's granularity,
# so both need the same literal.
ANSWER_TYPE_TIME = "time"
ANSWER_TYPE_ENTITY = "entity"

# Stage-1/Stage-3 request timeouts (seconds).
TIMEOUT_EXTRACT = 30
TIMEOUT_RELATION = 45
# Stage 3 waits longer on our own pod than on OpenRouter. One rented GPU answers
# every request in turn, so N questions in flight queue behind each other and
# per-request latency grows with concurrency even though total throughput does
# not. OpenRouter fans the same load across many machines, where 120s is ample.
# Measured before this split: 51 of 151 questions died on a read timeout -- a
# third of the GPU time paid for and then discarded, since a timed-out request
# still ran to completion on the pod.
TIMEOUT_REASON = 600 if SELF_HOSTED_LLM else 120


# The repo already keeps the OpenRouter key here (gitignored); reuse it rather
# than making everyone maintain a second copy in tecqa/.env.
_SHARED_KEY_PATH = Path(__file__).parent.parent / "dataset_vi" / ".or_key"


def load_api_key() -> str:
    """OpenRouter key, in order: the environment, tecqa/.env (loaded above by
    dotenv), then dataset_vi/.or_key. Never log or print the return value."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and _SHARED_KEY_PATH.exists():
        key = _SHARED_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Export it, add it to tecqa/.env, "
            f"or place it in {_SHARED_KEY_PATH.name}."
        )
    return key
