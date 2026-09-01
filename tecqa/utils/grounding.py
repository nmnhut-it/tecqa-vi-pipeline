"""
Entity and Relation grounding for the original English TECQA implementation.

Paper Sec 5.1 (Implementation Details):
  "The embedding model utilizes bge-base-en-v1.5."
  https://huggingface.co/BAAI/bge-base-en-v1.5

Entity grounding (paper Eq. 2, entity half):
  - Exact string match on underscore/space normalized form
  - Dense cosine similarity top-1 nearest neighbor over 10,488 entities

Relation grounding (paper Eq. 2, relation half):
  - Embed all canonical relation strings ONCE via bge-base-en-v1.5 (cached to disk)
  - Embed the LLM-extracted relation phrase
  - Top-1 cosine similarity nearest neighbor

Which encoder actually runs is chosen by config.EMBED_BACKEND: the paper's
bge-base-en-v1.5 in-process (default), or a hosted model over OpenRouter for
machines that cannot carry torch. See utils/embedder.py.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

from . import embedder
from .embedder import INPUT_PASSAGE, INPUT_QUERY

# Overridable so the test suite can point at a scratch directory. It must:
# building a grounder writes a corpus cache, and the tests build theirs from a
# six-fact toy graph with a stubbed encoder. Pointed at the real directory,
# that overwrites a genuine cache with two fake vectors -- which is exactly how
# tecqa/data/relation_embeddings_en.json came to hold 2 relations instead of 251.
DATA_DIR = Path(os.environ.get("TECQA_EMBED_CACHE_DIR",
                               Path(__file__).parent.parent / "data"))

# Cache files carry the model that produced them, because vectors from two
# encoders are not comparable and a silent mix would corrupt grounding without
# raising. The two files below predate that rule; both were produced by the
# paper's bge-base-en-v1.5, so they are adopted under its slug on first use
# rather than paying to re-embed 10,488 entities.
LEGACY_JSON = {"entity": DATA_DIR / "entity_embeddings_en.json",
               "relation": DATA_DIR / "relation_embeddings_en.json"}
LEGACY_MODEL_SLUG = "bge-base-en-v1.5"

KEYS_ENTITY = "entities"
KEYS_RELATION = "rel_ids"

# Evaluation threads all reach grounding at the same instant on the first
# question. Without this, each one sees an unpopulated cache, rebuilds the
# embeddings, and rewrites the file while its siblings are mid-read.
_CACHE_LOCK = threading.Lock()


def active_model_id() -> str:
    """The encoder currently producing vectors, for the run's meta.json."""
    return embedder.active_model_id()


def embed_batch(texts: list, input_type: str = INPUT_QUERY) -> np.ndarray:
    """Embed texts with the configured backend. Rows come back L2-normalized,
    so cos_sim_matrix stays a dot product. `input_type` distinguishes the query
    side from the corpus side, which asymmetric encoders like bge-*-en-v1.5
    treat differently."""
    return embedder.get_embedder().encode(list(texts), input_type)


def _cache_path(kind: str) -> Path:
    return DATA_DIR / f"{kind}_embeddings_{embedder.cache_slug()}.npz"


def _read_cache(path: Path, keys_field: str, keys: list):
    """The cached matrix, or None if the cache is absent or was built for a
    different key list (the KG changed under us)."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as blob:
            if list(blob[keys_field]) != list(keys):
                return None
            return blob["embeddings"]
    except Exception:
        return None  # truncated or unreadable: rebuild rather than crash


def _adopt_legacy(kind: str, keys_field: str, keys: list):
    """Migrate the pre-versioning JSON cache, but only when the active model is
    the one that wrote it. Under any other encoder it must be ignored."""
    path = LEGACY_JSON.get(kind)
    if embedder.cache_slug() != LEGACY_MODEL_SLUG or not path or not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if cached.get(keys_field) != list(keys):
        return None
    return np.asarray(cached["embeddings"], dtype=np.float32)


def load_or_build(kind: str, keys_field: str, keys: list) -> np.ndarray:
    """Corpus embeddings for `keys`, from cache when possible.

    The one place either grounder gets its matrix, so the model-keyed filename,
    the legacy migration and the rebuild cost live in a single spot. Texts are
    the keys with underscores turned back into spaces, matching how the KG
    writes entity and relation ids.
    """
    path = _cache_path(kind)
    emb = _read_cache(path, keys_field, keys)
    if emb is not None:
        return emb
    emb = _adopt_legacy(kind, keys_field, keys)
    if emb is None:
        print(f"Building {kind} embedding cache for {len(keys)} items "
              f"(one-time, {active_model_id()})...")
        emb = embed_batch([key.replace("_", " ") for key in keys], INPUT_PASSAGE)
    emb = np.asarray(emb, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{keys_field: np.array([str(k) for k in keys]),
                      "embeddings": emb})
    return emb


def cos_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity: (n, d) x (m, d) -> (n, m).
    Since bge embeddings are already L2-normalized, this is a dot product."""
    return a @ b.T


class EntityGrounder:
    """
    Paper Eq. 2 (entity half): map LLM-extracted mention -> canonical KG entity ID
    via dense cosine similarity over bge-base-en-v1.5 embeddings.

    Caches all entity embeddings to data/entity_embeddings_en.json for instant reuse.
    """

    def __init__(self, entities: set):
        self.entities = sorted(entities)
        self.norm_to_id = {e.replace("_", " ").lower(): e for e in self.entities}
        self._emb: np.ndarray | None = None

    def _load_or_build_cache(self):
        with _CACHE_LOCK:
            if self._emb is None:  # another thread may have won the race
                self._build_entity_cache()

    def _build_entity_cache(self):
        self._emb = load_or_build("entity", KEYS_ENTITY, self.entities)

    def ground(self, mention: str) -> str | None:
        if not mention:
            return None
        mention = mention.strip()
        key = mention.replace("_", " ").lower()

        # Fast exact string match shortcut
        if key in self.norm_to_id:
            return self.norm_to_id[key]
        if mention in self.norm_to_id.values():
            return mention

        # Dense embedding cosine similarity (Paper Eq. 2)
        if self._emb is None:
            self._load_or_build_cache()

        q_emb = embed_batch([mention], INPUT_QUERY)
        sims = cos_sim_matrix(q_emb, self._emb)[0]
        best_idx = int(np.argmax(sims))
        return self.entities[best_idx]


class RelationGrounder:
    """
    Paper Eq. 2 (relation half): map free-text relation phrase -> canonical KG relation.

    Embeds all relation strings once with bge-base-en-v1.5 (cached to
    data/relation_embeddings_en.json), then returns the top-1 cosine neighbor.
    """

    def __init__(self, relations: set):
        self.rel_ids = sorted(relations)
        self.norm_to_id = {r.replace("_", " ").lower(): r for r in self.rel_ids}
        self._emb: np.ndarray | None = None

    def _load_or_build_cache(self):
        with _CACHE_LOCK:
            if self._emb is None:  # another thread may have won the race
                self._build_relation_cache()

    def _build_relation_cache(self):
        self._emb = load_or_build("relation", KEYS_RELATION, self.rel_ids)

    def ground(self, relation_phrase: str) -> tuple[str, float]:
        """
        Paper Eq. 2: single top-1 nearest-neighbour cosine match.
        Returns (relation_id, similarity_score).
        """
        phrase = relation_phrase.strip().lower()
        if phrase in self.norm_to_id:
            return self.norm_to_id[phrase], 1.0

        if self._emb is None:
            self._load_or_build_cache()
        q_emb = embed_batch([relation_phrase], INPUT_QUERY)
        sims = cos_sim_matrix(q_emb, self._emb)[0]
        best_idx = int(np.argmax(sims))
        return self.rel_ids[best_idx], float(sims[best_idx])

    def ground_topk(self, relation_phrase: str, k: int = 2) -> list[tuple[str, float]]:
        """
        Multi-candidate relation grounding for Vietnamese cross-lingual adaptation.
        Returns list of (relation_id, similarity_score).
        """
        phrase = relation_phrase.strip().lower()
        if phrase in self.norm_to_id:
            top1 = (self.norm_to_id[phrase], 1.0)
            if k == 1:
                return [top1]

        if self._emb is None:
            self._load_or_build_cache()
        q_emb = embed_batch([relation_phrase], INPUT_QUERY)
        sims = cos_sim_matrix(q_emb, self._emb)[0]
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self.rel_ids[i], float(sims[i])) for i in top_indices]
