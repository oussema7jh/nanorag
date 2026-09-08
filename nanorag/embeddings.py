"""Text embeddings with zero dependencies.

The default :class:`HashedNGramEmbedder` implements the *feature-hashing
trick* (Weinberger et al., 2009) over word and character n-grams: every
feature is hashed into a fixed number of buckets and accumulated with a
sign derived from the hash, then the vector is L2-normalised. It is fast,
deterministic, needs no model download, and gives a solid lexical-similarity
signal — which is exactly what hybrid retrieval wants from its dense leg.

The :class:`Embedder` protocol lets you swap in a real neural embedder
(e.g. :class:`OpenAIEmbedder` below) without touching the rest of the stack.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from typing import List, Protocol, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9']+")

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while until because as of in on at to
    for with from by into over under about after before between during above
    below up down out off again further once here there all any both each few
    more most other some such only own same so than too very can could should
    would will shall may might must do does did done have has had having not no
    nor isn aren wasn weren be been being am is are was were i you he she it we
    they them their his her its our your my me him us what which who whom whose
    why how s t don don't ll re ve
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens (digits kept, punctuation dropped)."""
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Anything that can turn a string into a fixed-size float vector."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> List[float]: ...


class HashedNGramEmbedder:
    """Feature-hashing embedder over word + character n-grams."""

    def __init__(
        self,
        dim: int = 512,
        char_ngrams: Sequence[int] = (2, 3),
        word_ngrams: Sequence[int] = (1, 2),
    ) -> None:
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim
        self.char_ngrams = tuple(char_ngrams)
        self.word_ngrams = tuple(word_ngrams)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._content_tokens(text)
        for feature in self._features(tokens):
            h = int.from_bytes(
                hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(),
                "big",
            )
            vec[h % self.dim] += -1.0 if (h >> 63) & 1 else 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            inv = 1.0 / norm
            vec = [v * inv for v in vec]
        return vec

    @staticmethod
    def _content_tokens(text: str) -> List[str]:
        """Tokens with English function words removed (kept if all would drop)."""
        raw = tokenize(text)
        kept = [t for t in raw if t not in STOPWORDS]
        return kept if kept else raw

    def _features(self, tokens: Sequence[str]):
        """Yield hashed feature strings for a token list."""
        for tok in tokens:
            padded = f"<{tok}>"
            for n in self.char_ngrams:
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        yield "c" + padded[i : i + n]
        for n in self.word_ngrams:
            if n == 1:
                for tok in tokens:
                    yield "w" + tok
            else:
                for i in range(len(tokens) - n + 1):
                    yield "w" + " ".join(tokens[i : i + n])


class OpenAIEmbedder:
    """Adapter for the OpenAI embeddings API (used only when a key is set).

    Implemented with :mod:`urllib` so the package keeps zero dependencies.
    """

    def __init__(self, model: str = "text-embedding-3-small", timeout: int = 30) -> None:
        self.model = model
        self.timeout = timeout
        self._key = os.environ.get("OPENAI_API_KEY", "")
        if not self._key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    @property
    def dim(self) -> int:
        return 1536

    def embed(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model, "input": text}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["embedding"]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))
