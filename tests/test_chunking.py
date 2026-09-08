"""Tests for the recursive chunker."""

from nanorag.chunking import RecursiveChunker


LONG_TEXT = "\n\n".join(
    f"Section {i} heading.\n" + ("Sentence about topic %d. " % i) * 30 for i in range(6)
)


def test_chunks_respect_max_size():
    chunker = RecursiveChunker(chunk_size=400, overlap=50)
    for piece in chunker.split_text(LONG_TEXT):
        assert len(piece) <= 400


def test_all_content_is_preserved():
    chunker = RecursiveChunker(chunk_size=300, overlap=40)
    pieces = chunker.split_text(LONG_TEXT)
    # every original sentence must appear in some chunk
    for i in range(6):
        assert f"topic {i}" in " ".join(pieces)


def test_overlap_carries_context():
    chunker = RecursiveChunker(chunk_size=300, overlap=60)
    pieces = chunker.split_text(LONG_TEXT)
    assert len(pieces) > 1
    # consecutive chunks should share some trailing/leading content
    shared = 0
    for a, b in zip(pieces, pieces[1:]):
        tail = a[-60:]
        if tail and tail.split() and any(w in b for w in tail.split() if len(w) > 4):
            shared += 1
    assert shared >= len(pieces) - 2  # allow minor boundary slack


def test_short_text_is_single_chunk():
    chunker = RecursiveChunker(chunk_size=480, overlap=64)
    assert chunker.split_text("Hello world, this is short.") == ["Hello world, this is short."]


def test_chunk_metadata_has_ids_and_sections():
    chunker = RecursiveChunker(chunk_size=300, overlap=40)
    doc = "# Overview\n\nAll about widgets.\n\n" + LONG_TEXT
    chunks = chunker.chunk(doc, doc_id="doc", metadata={"source": "doc.md"})
    assert all(c.doc_id == "doc" for c in chunks)
    assert [c.seq for c in chunks] == list(range(len(chunks)))
    assert all(len(c.id) > 5 for c in chunks)
    assert any(c.section == "Overview" for c in chunks)
    assert chunks[0].source == "doc.md"


def test_reingest_is_idempotent_when_ids_stable():
    chunker = RecursiveChunker(chunk_size=300, overlap=40)
    chunks_a = chunker.chunk("Some stable text. " * 40, doc_id="d")
    chunks_b = chunker.chunk("Some stable text. " * 40, doc_id="d")
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
