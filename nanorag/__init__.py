"""nanorag — a from-scratch Retrieval-Augmented Generation engine.

No frameworks. No API keys required. Pure Python (>= 3.9), zero runtime
dependencies.

Subsystems:

* :mod:`nanorag.chunking`     — recursive text splitting with overlap
* :mod:`nanorag.embeddings`   — feature-hashing n-gram embedder (+ OpenAI adapter)
* :mod:`nanorag.store`        — persistent vector store with cosine search
* :mod:`nanorag.retrieval`    — BM25, vector and hybrid (RRF) retrieval
* :mod:`nanorag.providers`    — answer providers: offline extractive,
  OpenAI, Anthropic, Ollama
* :mod:`nanorag.pipeline`     — ingest -> retrieve -> generate with citations
* :mod:`nanorag.evals`        — golden-set evaluation (recall@k, MRR)
"""

from .chunking import Chunk, RecursiveChunker
from .embeddings import HashedNGramEmbedder, cosine
from .store import VectorStore
from .retrieval import BM25, HybridRetriever, VectorRetriever
from .providers import resolve_provider, ExtractiveProvider
from .pipeline import Answer, Passage, RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "BM25",
    "Chunk",
    "ExtractiveProvider",
    "HashedNGramEmbedder",
    "HybridRetriever",
    "Passage",
    "RAGPipeline",
    "RecursiveChunker",
    "VectorRetriever",
    "VectorStore",
    "cosine",
    "resolve_provider",
    "__version__",
]
