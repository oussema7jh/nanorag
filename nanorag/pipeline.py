"""The RAG pipeline: ingest -> retrieve -> generate.

Wires the subsystems together:

    documents --(chunker)--> chunks --(embedder)--> store (persisted)
    question --(hybrid retriever)--> passages --(provider)--> cited answer
"""

from __future__ import annotations

import os
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .chunking import Chunk, RecursiveChunker
from .embeddings import Embedder, HashedNGramEmbedder
from .providers import AnswerProvider, resolve_provider
from .retrieval import BM25, HybridRetriever, VectorRetriever
from .store import INDEX_FILE, VectorStore

DEFAULT_DATA_DIR = ".nanorag"
DEFAULT_PATTERNS = ("*.md", "*.txt")


@dataclass
class Passage:
    """A retrieved chunk with its fused score and final rank."""

    chunk: Chunk
    score: float
    rank: int

    @property
    def source(self) -> str:
        return self.chunk.source


@dataclass
class Answer:
    """The full result of one question."""

    question: str
    text: str
    passages: List[Passage] = field(default_factory=list)
    provider: str = ""
    latency_ms: float = 0.0

    def sources(self) -> List[str]:
        seen, out = set(), []
        for p in self.passages:
            label = p.source or p.chunk.doc_id
            if label not in seen:
                seen.add(label)
                out.append(label)
        return out


class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        chunker: RecursiveChunker,
        embedder: Embedder,
        provider: AnswerProvider,
        index_path: Optional[str] = None,
    ) -> None:
        self.store = store
        self.chunker = chunker
        self.embedder = embedder
        self.provider = provider
        self.index_path = index_path
        self._bm25 = BM25()
        self._sync_index()

    # ------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------ #

    @classmethod
    def open(
        cls,
        data_dir: str = DEFAULT_DATA_DIR,
        provider: Optional[AnswerProvider] = None,
        chunk_size: int = 480,
        overlap: int = 64,
        dim: int = 512,
    ) -> "RAGPipeline":
        """Load (or create) a pipeline rooted at ``data_dir``."""
        os.makedirs(data_dir, exist_ok=True)
        index_path = os.path.join(data_dir, INDEX_FILE)
        embedder = HashedNGramEmbedder(dim=dim)
        if os.path.exists(index_path):
            store = VectorStore.load(index_path, embedder)
        else:
            store = VectorStore(embedder)
        return cls(
            store=store,
            chunker=RecursiveChunker(chunk_size=chunk_size, overlap=overlap),
            embedder=embedder,
            provider=provider if provider is not None else resolve_provider(),
            index_path=index_path,
        )

    def _sync_index(self) -> None:
        self._bm25.index(self.store.chunks)

    # ------------------------------------------------------------ #
    # ingest
    # ------------------------------------------------------------ #

    def ingest_path(
        self,
        path: str,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
    ) -> dict:
        """Chunk, embed and index every matching file under ``path``."""
        root = Path(path).resolve()
        if not root.exists():
            raise FileNotFoundError(path)
        if root.is_file():
            files = [root]
        else:
            files = sorted(
                {p.resolve() for pattern in patterns for p in root.rglob(pattern) if p.is_file()}
            )
        docs = 0
        new_chunks = 0
        removed_chunks = 0
        for file in files:
            text = file.read_text(encoding="utf-8", errors="replace")
            chunks = self.chunker.chunk(
                text,
                doc_id=file.stem,
                metadata={"source": str(file)},
            )
            # Keep human-readable doc_id for evaluation; namespace storage IDs.
            source_id = hashlib.sha256(str(file).encode("utf-8")).hexdigest()
            for chunk in chunks:
                chunk.id = source_id + ":" + chunk.id
            changes = self.store.replace_source(str(file), chunks)
            new_chunks += changes["added"]
            removed_chunks += changes["removed"]
            docs += 1
        if new_chunks or removed_chunks:
            self._sync_index()
            self.persist()
        return {"documents": docs, "chunks": len(self.store),
                "new_chunks": new_chunks, "removed_chunks": removed_chunks}

    def persist(self) -> None:
        if self.index_path:
            self.store.save(self.index_path)

    # ------------------------------------------------------------ #
    # retrieve & answer
    # ------------------------------------------------------------ #

    def retrieve(self, question: str, k: int = 5) -> List[Passage]:
        """Hybrid retrieval: BM25 + vector fused with RRF."""
        if len(self.store) == 0:
            return []
        hybrid = HybridRetriever(
            [
                self._bm25,
                VectorRetriever(self.store, self.embedder),
            ]
        )
        hits = hybrid.search(question, k)
        return [Passage(chunk=c, score=s, rank=r) for r, (c, s) in enumerate(hits, start=1)]

    def ask(self, question: str, k: int = 5) -> Answer:
        """End-to-end: retrieve, then generate a cited answer."""
        if len(self.store) == 0:
            return Answer(
                question=question,
                text="The index is empty — run `nanorag ingest <path>` first.",
                provider=self.provider.name,
            )
        started = time.perf_counter()
        passages = self.retrieve(question, k)
        tuples = [(p.chunk, p.score) for p in passages]
        text = self.provider.answer(question, tuples)
        return Answer(
            question=question,
            text=text,
            passages=passages,
            provider=self.provider.name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
