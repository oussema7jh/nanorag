"""Retrievers: BM25 (lexical), vector (dense) and hybrid (RRF fusion).

Why hybrid? Pure lexical retrieval misses paraphrases; pure dense retrieval
misses exact identifiers (product codes, acronyms). Production RAG systems
combine both and fuse the ranked lists. The standard fusion method is
Reciprocal Rank Fusion (Cormack et al., 2009):

    score(d) = sum over retrievers  w_r / (k + rank_r(d))

which rewards documents that appear near the top of *multiple* retrievers
without requiring the underlying scores to be comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .chunking import Chunk
from .embeddings import Embedder, tokenize
from .store import VectorStore


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int


class BM25:
    """Okapi BM25 over a fixed chunk collection."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: List[Chunk] = []
        self._doc_len: List[int] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._docs)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """(Re)build the index from a chunk list."""
        self._docs = list(chunks)
        self._doc_len = []
        self._tf = []
        self._df = {}
        for chunk in self._docs:
            counts: Dict[str, int] = {}
            for tok in tokenize(chunk.text):
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            self._doc_len.append(sum(counts.values()))
            for tok in counts:
                self._df[tok] = self._df.get(tok, 0) + 1
        total = sum(self._doc_len)
        self._avgdl = (total / len(self._docs)) if self._docs else 0.0

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self._docs:
            return []
        scores: List[float] = [0.0] * len(self._docs)
        n_docs = len(self._docs)
        for term in set(tokenize(query)):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for i in range(n_docs):
                tf = self._tf[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_len[i] if self._doc_len[i] else 1
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        ranked = sorted(zip(self._docs, scores), key=lambda p: p[1], reverse=True)
        return [(chunk, score) for chunk, score in ranked[:k] if score > 0.0]


class VectorRetriever:
    """Dense retrieval via the vector store."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if len(self.store) == 0:
            return []
        return self.store.search(self.embedder.embed(query), k)


class HybridRetriever:
    """Fuses several retrievers with Reciprocal Rank Fusion."""

    def __init__(self, retrievers: Sequence, rrf_k: int = 60, weights: Optional[Sequence[float]] = None) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        if weights is None:
            weights = [1.0] * len(self.retrievers)
        if len(weights) != len(self.retrievers):
            raise ValueError("weights must match the number of retrievers")
        self.weights = list(weights)

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        pool: Dict[str, Tuple[Chunk, float]] = {}
        fused: Dict[str, float] = {}
        depth = max(k * 4, 20)
        for retriever, weight in zip(self.retrievers, self.weights):
            for rank, (chunk, _score) in enumerate(retriever.search(query, depth), start=1):
                pool[chunk.id] = (chunk, _score)
                fused[chunk.id] = fused.get(chunk.id, 0.0) + weight / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(pool[cid][0], score) for cid, score in ranked]
