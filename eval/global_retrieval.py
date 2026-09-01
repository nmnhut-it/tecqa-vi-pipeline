"""
Global retrieval over the whole TKG — the `w/o SG` ablation (paper Appendix C).

OWNER: CODE (docs/TEAM_PLAN.md H5).

The paper replaces Stage 1 with "a standard dense retrieval method based on
semantic similarity" over the entire graph. Embedding all 461,329 MultiTQ facts
through a hosted encoder costs more than the whole rest of the experiment suite,
so this is a two-stage approximation: a BM25 lexical shortlist, then a dense
rerank of that shortlist with the same encoder the real pipeline grounds with.
Declared as deviation 3 in docs/EVAL_DESIGN.md Sec 9.

What matters for the ablation is preserved: no entity grounding, no relation
filter, no main-entity pivot — retrieval sees only the question string.

Input:  a graph exposing .facts and .fact_text().
Output: a list of facts, ranked, to hand to Stage 2 unchanged.
"""
import math
import re
from collections import Counter, defaultdict

from ..utils.grounding import cos_sim_matrix, embed_batch

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Standard Okapi BM25 constants; not tuned, and not worth tuning for a baseline
# whose whole job is to be the unstructured comparison point.
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_SHORTLIST = 200  # lexical candidates per question
DEFAULT_POOL = 100       # facts handed to Stage 2 after the dense rerank


def tokenize(text: str) -> list:
    return TOKEN_RE.findall(text.lower())


class GlobalSemanticRetriever:
    """BM25 shortlist + dense rerank over every fact in the graph.

    The index is built once per process and reused across questions. The
    indexed text is the English rendering of each fact, which is also what the
    Vietnamese condition matches against: Stage 1 translates mentions to the
    ICEWS English schema before grounding, so both conditions share one index.
    """

    def __init__(self, graph, shortlist=DEFAULT_SHORTLIST, pool=DEFAULT_POOL):
        self.graph = graph
        self.shortlist = shortlist
        self.pool = pool
        self._postings = None
        self._doc_len = []
        self._avg_len = 0.0
        self._texts = []

    def _build_index(self) -> None:
        postings = defaultdict(list)
        for doc_id, fact in enumerate(self.graph.facts):
            text = self.graph.fact_text(fact)
            tokens = tokenize(text)
            self._texts.append(text)
            self._doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                postings[term].append((doc_id, freq))
        self._postings = postings
        self._avg_len = sum(self._doc_len) / max(1, len(self._doc_len))

    def _idf(self, term: str) -> float:
        n_docs = len(self._doc_len)
        n_term = len(self._postings.get(term, ()))
        return math.log(1 + (n_docs - n_term + 0.5) / (n_term + 0.5))

    def _bm25_scores(self, query_tokens) -> dict:
        scores = defaultdict(float)
        for term in set(query_tokens):
            idf = self._idf(term)
            for doc_id, freq in self._postings.get(term, ()):
                norm = 1 - BM25_B + BM25_B * self._doc_len[doc_id] / self._avg_len
                scores[doc_id] += idf * freq * (BM25_K1 + 1) / (freq + BM25_K1 * norm)
        return scores

    def _lexical_shortlist(self, question: str) -> list:
        if self._postings is None:
            self._build_index()
        scores = self._bm25_scores(tokenize(question))
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [doc_id for doc_id, _ in ranked[:self.shortlist]]

    def _dense_rerank(self, question: str, doc_ids: list) -> list:
        """Second stage: cosine against the same encoder Eq. 2 grounds with."""
        sims = cos_sim_matrix(
            embed_batch([question], input_type="query"),
            embed_batch([self._texts[i] for i in doc_ids], input_type="passage"))[0]
        order = sorted(range(len(doc_ids)), key=lambda i: -sims[i])[:self.pool]
        return [doc_ids[i] for i in order]

    def retrieve(self, question: str) -> list:
        """Ranked facts for `question`, with no structural constraint at all."""
        doc_ids = self._lexical_shortlist(question)
        if not doc_ids:
            return []
        return [self.graph.facts[i] for i in self._dense_rerank(question, doc_ids)]
