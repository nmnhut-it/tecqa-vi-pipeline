"""
Embedding backends for Eq. 2 grounding.

Two backends behind one interface:

  local       bge-base-en-v1.5 through sentence-transformers, run in-process on
              CPU. The paper's choice (Sec 5.1) and the DEFAULT, so a run with
              no extra flags still reproduces the paper.
  openrouter  Any model on OpenRouter's /embeddings endpoint. Needs no torch,
              so the pipeline fits on a 1GB VPS or a bare notebook runtime
              instead of pulling ~2GB of wheels.

Two different embedding models share no coordinate system, so a cosine score
computed between them is meaningless -- it would silently corrupt grounding
rather than fail. Every cached corpus matrix is therefore filed under the model
that produced it (see cache_slug), and switching backends builds a fresh cache
instead of reading the other model's vectors.

Input:  list[str] + whether they are queries or corpus passages.
Output: np.ndarray (n, d), float32, L2-normalized so cosine == dot product.

Related: utils/grounding.py (only caller), config.py (backend selection),
eval/run_eval.py (records the active model id into the run's meta.json).
"""
import os
import threading
import time

# Must be set BEFORE sentence_transformers imports tokenizers -- otherwise
# encode() can deadlock and hang the run indefinitely.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import requests

from .. import config

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

INPUT_QUERY = "query"
INPUT_PASSAGE = "passage"

# bge-*-en-v1.5 is asymmetric: the query side is trained with this instruction
# prefix and the corpus side without it. sentence-transformers applies it via
# prompt_name="query"; over HTTP we have to prepend it ourselves or the two
# sides land in slightly different regions of the space. bge-m3 and the OpenAI
# models are symmetric and must NOT get a prefix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_EN_MARKER = "bge-base-en"


def _query_prefix(model_id: str) -> str:
    return BGE_QUERY_PREFIX if _BGE_EN_MARKER in model_id.lower() else ""


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cos_sim_matrix can stay a plain dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class LocalEmbedder:
    """bge-base-en-v1.5 in-process (paper Sec 5.1). Needs torch."""

    def __init__(self, model_id: str, device: str = None):
        self.model_id = model_id
        # MPS hangs on M1 inside sentence-transformers encode(), so CPU it is.
        self.device = device or config.EMBED_DEVICE
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Imported here, not at module scope: a box running the
                    # openrouter backend has no torch installed at all, and
                    # importing this module must still work there.
                    from sentence_transformers import SentenceTransformer
                    print(f"Loading embedding model {self.model_id} on {self.device}...")
                    self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def encode(self, texts: list, input_type: str = INPUT_QUERY) -> np.ndarray:
        model = self._load()
        kwargs = {"batch_size": config.EMBED_BATCH_LOCAL, "normalize_embeddings": True}
        if input_type == INPUT_QUERY:
            kwargs["prompt_name"] = INPUT_QUERY
        return np.asarray(model.encode(list(texts), **kwargs), dtype=np.float32)


class OpenRouterEmbedder:
    """Any OpenRouter embedding model, over HTTP. No torch, no model download.

    Costs are read from the response's own usage.cost and handed to the run
    budget when one is set, so embedding spend counts against --max-usd exactly
    like Stage 1 and Stage 3 spend does.
    """

    def __init__(self, model_id: str, api_key: str = None):
        self.model_id = model_id
        self._api_key = api_key
        self._prefix = _query_prefix(model_id)
        self._budget = None
        self._lock = threading.Lock()

    def set_budget(self, budget) -> None:
        self._budget = budget

    @property
    def api_key(self) -> str:
        if not self._api_key:
            self._api_key = config.load_api_key()
        return self._api_key

    def _post(self, batch: list) -> list:
        """One request, retried on transient failure. Returns vectors in the
        order they were sent -- the API may not, so index is honoured."""
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model_id, "input": batch,
                   "encoding_format": "float"}
        last = None
        for attempt in range(config.EMBED_RETRIES):
            try:
                resp = requests.post(EMBED_URL, headers=headers, json=payload,
                                     timeout=config.TIMEOUT_EMBED)
                resp.raise_for_status()
                body = resp.json()
                self._charge(body.get("usage") or {})
                rows = sorted(body["data"], key=lambda item: item.get("index", 0))
                return [row["embedding"] for row in rows]
            except Exception as exc:  # transient: rate limit, 5xx, dropped socket
                last = exc
                if attempt == config.EMBED_RETRIES - 1:
                    break
                time.sleep(config.EMBED_BACKOFF_BASE ** attempt)
        raise RuntimeError(f"embedding request failed for {self.model_id}: {last}")

    def _charge(self, usage: dict) -> None:
        if self._budget is None:
            return
        with self._lock:
            self._budget.add(self.model_id, usage.get("prompt_tokens", 0), 0,
                             real_cost=usage.get("cost"))

    def encode(self, texts: list, input_type: str = INPUT_QUERY) -> np.ndarray:
        prefix = self._prefix if input_type == INPUT_QUERY else ""
        prepared = [prefix + str(text) for text in texts]
        vectors = []
        for start in range(0, len(prepared), config.EMBED_BATCH_REMOTE):
            vectors.extend(self._post(prepared[start:start + config.EMBED_BATCH_REMOTE]))
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


_ACTIVE = None
_ACTIVE_KEY = None
_BUILD_LOCK = threading.Lock()


def get_embedder():
    """The embedder for the CURRENT config, rebuilt if the config changed.

    config.EMBED_BACKEND is read here rather than bound at import time, for the
    same reason MODEL_EXTRACT/MODEL_REASON are (see CLAUDE.md): a sweep that
    monkeypatches the module-level setting must actually take effect.
    """
    global _ACTIVE, _ACTIVE_KEY
    key = (config.EMBED_BACKEND, config.embed_model_id())
    if _ACTIVE_KEY == key:
        return _ACTIVE
    with _BUILD_LOCK:
        if _ACTIVE_KEY != key:
            backend, model_id = key
            if backend == config.EMBED_BACKEND_OPENROUTER:
                _ACTIVE = OpenRouterEmbedder(model_id)
            elif backend == config.EMBED_BACKEND_LOCAL:
                _ACTIVE = LocalEmbedder(model_id)
            else:
                raise ValueError(
                    f"unknown embedding backend {backend!r}; "
                    f"expected one of {config.EMBED_BACKENDS}")
            _ACTIVE_KEY = key
    return _ACTIVE


def set_budget(budget) -> None:
    """Charge embedding spend to a run's budget, when the backend bills."""
    embedder = get_embedder()
    if hasattr(embedder, "set_budget"):
        embedder.set_budget(budget)


def active_model_id() -> str:
    """The model actually producing vectors right now -- this is what goes into
    the run's meta.json, so a result can never be attributed to the wrong
    encoder."""
    return get_embedder().model_id


def cache_slug() -> str:
    """Filesystem-safe tag for the active model, used in every corpus cache
    filename. This is the guard that stops one model's cached matrix from being
    read back as another's."""
    tail = active_model_id().split("/")[-1]
    return "".join(char if char.isalnum() or char in "-._" else "-" for char in tail)
