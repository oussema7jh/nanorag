"""Recursive text chunking with character overlap.

The chunker mirrors the strategy used by production RAG stacks: split
hierarchically on natural boundaries (paragraphs, lines, sentences, words),
then greedily pack the resulting units into chunks of at most ``chunk_size``
characters, carrying a small tail of each chunk into the next one as
overlap so that sentences cut near a boundary stay retrievable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n\n", "\n\n", "\n", ". ", " ")
_HEADING_RE = re.compile(r"^#{1,6}\s+.+")


def _join(parts: List[str]) -> str:
    """Concatenate units, inserting a newline at bare boundaries.

    Without this, two units packed next to each other can glue together
    ("...behind it.## Daemons") because the separator they were split on
    was consumed during splitting.
    """
    out: List[str] = []
    for part in parts:
        if out and not out[-1][-1:].isspace() and not part[:1].isspace():
            out.append("\n")
        out.append(part)
    return "".join(out)


def stable_hash(text: str) -> str:
    """Short deterministic digest used for chunk ids."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=5).hexdigest()


@dataclass
class Chunk:
    """A retrievable unit of text."""

    id: str
    doc_id: str
    seq: int
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "seq": self.seq,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            doc_id=d["doc_id"],
            seq=d["seq"],
            text=d["text"],
            metadata=d.get("metadata", {}),
        )


class RecursiveChunker:
    """Splits text into overlapping chunks on natural boundaries."""

    def __init__(
        self,
        chunk_size: int = 480,
        overlap: int = 64,
        separators: Optional[Sequence[str]] = None,
    ) -> None:
        if chunk_size < 40:
            raise ValueError("chunk_size must be >= 40")
        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size // 4)
        self.separators = tuple(separators) if separators else DEFAULT_SEPARATORS

    # ------------------------------------------------------------------ #

    def split_text(self, text: str) -> List[str]:
        """Split ``text`` into packed, overlapping chunk strings."""
        units = self._split(text, self.separators)
        units = [u for u in units if u.strip()]
        packed: List[str] = []
        current: List[str] = []

        for unit in units:
            if current and len(_join(current)) + len(unit) > self.chunk_size:
                packed.append(_join(current).strip())
                # carry a tail of the emitted chunk into the next one
                tail: List[str] = []
                tail_len = 0
                for prev in reversed(current):
                    if tail_len + len(prev) > self.overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev)
                current = tail
            current.append(unit)

        if current:
            tail_text = _join(current).strip()
            if tail_text:
                packed.append(tail_text)
        return [p for p in packed if p]

    def chunk(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """Split ``text`` and wrap each piece in a :class:`Chunk`."""
        metadata = dict(metadata or {})
        pieces = self.split_text(text)
        chunks: List[Chunk] = []
        cursor = 0
        for seq, piece in enumerate(pieces):
            # locate the piece in the original text to recover section context
            at = text.find(piece[:64], cursor)
            if at < 0:
                at = cursor
            section = self._heading_before(text, at)
            cursor = at + max(1, len(piece) // 2)
            chunk_meta = dict(metadata)
            if section:
                chunk_meta["section"] = section
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{seq}:{stable_hash(piece)}",
                    doc_id=doc_id,
                    seq=seq,
                    text=piece,
                    metadata=chunk_meta,
                )
            )
        return chunks

    # ------------------------------------------------------------------ #

    def _split(self, text: str, separators: Sequence[str]) -> List[str]:
        """Recursively split ``text`` into units no larger than chunk_size."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sep = self._pick_separator(text, separators)
        if sep is None:  # no separator present — hard split
            step = self.chunk_size
            return [text[i : i + step] for i in range(0, len(text), step)]

        parts = self._split_keep(text, sep)
        rest = tuple(s for s in separators if s != sep)
        units: List[str] = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    units.append(part)
            else:
                units.extend(self._split(part, rest or (sep,)))
        return units

    @staticmethod
    def _pick_separator(text: str, separators: Sequence[str]) -> Optional[str]:
        for sep in separators:
            if sep in text:
                return sep
        return None

    @staticmethod
    def _split_keep(text: str, sep: str) -> List[str]:
        """Split on ``sep`` but keep it attached to the preceding piece."""
        parts = text.split(sep)
        kept = [p + sep for p in parts[:-1]]
        if parts[-1]:
            kept.append(parts[-1])
        return kept

    @staticmethod
    def _heading_before(text: str, pos: int) -> str:
        """Return the closest markdown heading above ``pos``."""
        best = ""
        for line in text[:pos].splitlines():
            if _HEADING_RE.match(line):
                best = line.lstrip("#").strip()
        return best
