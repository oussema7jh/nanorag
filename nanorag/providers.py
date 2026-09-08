"""Answer providers: the "G" in RAG.

The pipeline is model-agnostic. Every provider implements one method:

    answer(question, passages) -> str

* :class:`ExtractiveProvider` composes an answer directly from the retrieved
  passages — fully offline, deterministic, zero cost. It makes the whole
  stack demonstrable without any API key.
* :class:`OpenAIProvider`, :class:`AnthropicProvider`, :class:`OllamaProvider`
  call a real chat model, instructing it to ground its answer in the numbered
  context and cite passages like ``[2]``.

All HTTP calls use :mod:`urllib` from the standard library, so the package
ships with zero dependencies. ``resolve_provider()`` picks the best available
provider from explicit choice or environment.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from typing import Optional, Protocol, Sequence, Tuple

from .chunking import Chunk
from .embeddings import STOPWORDS, tokenize

TIMEOUT = 60

SYSTEM_PROMPT = (
    "You are a precise retrieval assistant. Answer the user's question using "
    "ONLY the numbered context passages provided. Cite every claim with the "
    "passage number in square brackets, like [1] or [2]. If the context does "
    "not contain the answer, say so plainly."
)


class AnswerProvider(Protocol):
    """Anything that can answer a question from retrieved passages."""

    name: str

    def answer(self, question: str, passages: Sequence[Tuple[Chunk, float]]) -> str: ...


# ---------------------------------------------------------------------- #
# Offline extractive provider
# ---------------------------------------------------------------------- #

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+\S")


class ExtractiveProvider:
    """Grounded extractive answering — no model, no network, no cost.

    Each sentence of the retrieved passages is scored by the weighted
    overlap of its terms with the question (rare question terms count more,
    using an IDF estimate computed over the retrieved set). The best
    sentences are stitched together in reading order with citations.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, passages: Sequence[Tuple[Chunk, float]]) -> str:
        if not passages:
            return "I found no relevant passages in the index."

        raw_terms = set(tokenize(question))
        q_terms = (raw_terms - STOPWORDS) or raw_terms
        if not q_terms:
            return _clean(passages[0][0].text.split(". ")[0]) + " [1]"

        # sentences: (passage_rank, sentence_index, text)
        sentences = []
        for rank, (chunk, _score) in enumerate(passages, start=1):
            body = _body_text(chunk.text)
            s_idx = 0
            for para in body.split("\n\n"):
                for sent in _SENT_RE.split(para):
                    if not _is_answer_sentence(sent):
                        continue
                    sent = _clean(sent)
                    if 20 <= len(sent) <= 400:
                        sentences.append((rank, s_idx, sent))
                    s_idx += 1

        if not sentences:
            return _clean(passages[0][0].text)[:300] + " [1]"

        # IDF over the retrieved sentences (rare question terms matter more)
        df = {}
        for _rank, _s_idx, sent in sentences:
            for term in set(tokenize(sent)):
                df[term] = df.get(term, 0) + 1
        n_sent = len(sentences)

        def weight(term: str) -> float:
            d = df.get(term, 0)
            return 0.0 if d == 0 else math.log(1.0 + n_sent / d)

        def sentence_score(sent: str) -> float:
            terms = set(tokenize(sent))
            overlap = sum(weight(t) for t in q_terms & terms)
            norm = 1.0 + 0.02 * len(terms)  # gentle length penalty
            return overlap / norm

        scored = sorted(
            ((sentence_score(s), rank, s_idx, s) for rank, s_idx, s in sentences),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        # keep only sentences that share informative terms with the question
        scored = [t for t in scored if t[0] > 0.0]
        if not scored:
            # no term overlap at all — fall back to the top passage opener
            first = _SENT_RE.split(passages[0][0].text)[0]
            return f"{_clean(first)} [1]"
        top_score = scored[0][0]
        strong = [t for t in scored[: self.max_sentences] if t[0] >= 0.35 * top_score]
        picked = sorted(strong, key=lambda t: (t[1], t[2]))

        parts = []
        for _score, rank, _s_idx, sent in picked:
            if sent not in parts:
                parts.append(f"{sent} [{rank}]")
        return " ".join(parts)


def _body_text(text: str) -> str:
    """Drop a leading markdown heading from a chunk."""
    lines = text.splitlines()
    if lines and _HEADING_LINE_RE.match(lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines)


def _is_answer_sentence(sent: str) -> bool:
    """True if the string looks like prose, not a heading or label line."""
    s = sent.strip()
    if not s:
        return False
    if _HEADING_LINE_RE.match(s):
        return False
    if s.startswith("**") and s.endswith("**"):
        return False
    return True


def _clean(text: str) -> str:
    """Collapse whitespace and strip stray markdown heading markers."""
    text = " ".join(text.split())
    while text.startswith("#"):
        text = text.lstrip("#").strip()
    return text


# ---------------------------------------------------------------------- #
# HTTP helpers
# ---------------------------------------------------------------------- #


def _post_json(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:  # pragma: no cover - network guard
        body = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"provider HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network guard
        raise RuntimeError(f"provider unreachable: {exc.reason}") from exc


def build_context(passages: Sequence[Tuple[Chunk, float]]) -> str:
    """Render retrieved passages as a numbered, cited context block."""
    blocks = []
    for rank, (chunk, _score) in enumerate(passages, start=1):
        header = f"[{rank}] source: {chunk.source or chunk.doc_id}"
        if chunk.section:
            header += f" — {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------- #
# Chat-model providers
# ---------------------------------------------------------------------- #


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._key = os.environ.get("OPENAI_API_KEY", "")

    def answer(self, question: str, passages: Sequence[Tuple[Chunk, float]]) -> str:
        if not self._key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self._key}"},
            {
                "model": self.model,
                "temperature": 0.2,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}",
                    },
                ],
            },
        )
        return data["choices"][0]["message"]["content"].strip()


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        self.model = model
        self._key = os.environ.get("ANTHROPIC_API_KEY", "")

    def answer(self, question: str, passages: Sequence[Tuple[Chunk, float]]) -> str:
        if not self._key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01",
            },
            {
                "model": self.model,
                "max_tokens": 500,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}",
                    }
                ],
            },
        )
        return "".join(block.get("text", "") for block in data.get("content", [])).strip()


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str = "llama3.2", url: Optional[str] = None) -> None:
        self.model = model
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")

    def answer(self, question: str, passages: Sequence[Tuple[Chunk, float]]) -> str:
        data = _post_json(
            f"{self.url}/api/chat",
            {},
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0.2},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}",
                    },
                ],
            },
        )
        return data["message"]["content"].strip()


PROVIDERS = {
    "extractive": ExtractiveProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def resolve_provider(preferred: Optional[str] = None):
    """Pick a provider: explicit name, else first configured by environment."""
    if preferred:
        if preferred not in PROVIDERS:
            raise ValueError(f"unknown provider {preferred!r}; options: {sorted(PROVIDERS)}")
        return PROVIDERS[preferred]()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    if os.environ.get("OLLAMA_URL"):
        return OllamaProvider()
    return ExtractiveProvider()
