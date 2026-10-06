"""A tiny persistent vector store.

Chunks and their embedding vectors are serialised to a single JSON index.
This keeps the project dependency-free while preserving the exact interface
a production store would expose (add / save / load / cosine search), so
swapping in SQLite, FAISS or a hosted vector DB later is a drop-in change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .chunking import Chunk
from .embeddings import Embedder, cosine

INDEX_FILE = "index.json"


class VectorStore:
    """In-memory vector index with JSON persistence."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []
        self._by_id = {c.id: i for i, c in enumerate(self.chunks)}

    # ------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: Sequence[Chunk], vectors: Optional[Sequence[List[float]]] = None) -> None:
        """Add chunks (embedding them with the store's embedder if needed)."""
        for i, chunk in enumerate(chunks):
            if chunk.id in self._by_id:
                continue  # idempotent re-ingest
            vec = vectors[i] if vectors is not None else self.embedder.embed(chunk.text)
            self._by_id[chunk.id] = len(self.chunks)
            self.chunks.append(chunk)
            self._vectors.append(vec)

    def replace_source(self, source: str, chunks: Sequence[Chunk]) -> dict:
        """Replace one local file's passages; embed before mutating the index."""
        canonical = str(Path(source).resolve())
        matches = [i for i, c in enumerate(self.chunks)
                   if c.source and str(Path(c.source).resolve()) == canonical]
        old = [self.chunks[i] for i in matches]
        if old == list(chunks):
            return {"added": 0, "removed": 0}
        if any(str(Path(c.source).resolve()) != canonical for c in chunks):
            raise ValueError("replacement chunks must belong to the source")
        vectors = [self.embedder.embed(c.text) for c in chunks]
        removed = set(matches)
        kept = [(c, v) for i, (c, v) in enumerate(zip(self.chunks, self._vectors))
                if i not in removed]
        ids = [c.id for c, _ in kept] + [c.id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("replacement would create duplicate chunk IDs")
        self.chunks = [c for c, _ in kept] + list(chunks)
        self._vectors = [v for _, v in kept] + vectors
        self._by_id = {c.id: i for i, c in enumerate(self.chunks)}
        return {"added": len(chunks), "removed": len(matches)}

    def search(self, query_vector: Sequence[float], k: int = 5) -> List[Tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity (brute force — fine to ~100k)."""
        scored: List[Tuple[Chunk, float]] = []
        for chunk, vec in zip(self.chunks, self._vectors):
            scored.append((chunk, cosine(query_vector, vec)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------ #

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "format": "nanorag-index/1",
            "dim": self.embedder.dim,
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": [[round(v, 6) for v in vec] for vec in self._vectors],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, embedder: Embedder) -> "VectorStore":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != "nanorag-index/1":
            raise ValueError(f"unrecognised index format in {path}")
        store = cls(embedder)
        chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        vectors = payload["vectors"]
        if len(chunks) != len(vectors):
            raise ValueError("corrupt index: chunk/vector mismatch")
        store.chunks = chunks
        store._vectors = vectors
        store._by_id = {c.id: i for i, c in enumerate(chunks)}
        return store

    def stats(self) -> dict:
        sources = {c.source for c in self.chunks}
        return {
            "chunks": len(self.chunks),
            "documents": len(sources),
            "dim": self.embedder.dim,
        }
