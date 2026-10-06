"""Regression coverage for changing local knowledge bases."""
from pathlib import Path

import pytest

from nanorag.pipeline import RAGPipeline
from nanorag.providers import ExtractiveProvider


def open_pipe(tmp_path):
    return RAGPipeline.open(str(tmp_path / "index"), provider=ExtractiveProvider())


def test_edit_replaces_old_passages_and_refreshes_bm25(tmp_path):
    file = tmp_path / "policy.md"
    file.write_text("Refunds require an amber receipt.", encoding="utf-8")
    pipe = open_pipe(tmp_path)
    pipe.ingest_path(str(file))
    file.write_text("Refunds require a violet receipt.", encoding="utf-8")
    result = pipe.ingest_path(str(file))
    assert result["removed_chunks"] == result["new_chunks"] == 1
    assert len(pipe.store) == 1
    assert "violet" in pipe._bm25.search("violet", 1)[0][0].text
    assert all("amber" not in c.text for c in pipe.store.chunks)
    reloaded = open_pipe(tmp_path)
    assert "violet" in reloaded.retrieve("receipt")[0].chunk.text


def test_same_filename_and_content_in_different_directories(tmp_path):
    docs = tmp_path / "docs"
    for folder in ("a", "b"):
        target = docs / folder / "policy.md"
        target.parent.mkdir(parents=True)
        target.write_text("A shared policy document.", encoding="utf-8")
    pipe = open_pipe(tmp_path)
    result = pipe.ingest_path(str(docs))
    assert result["documents"] == 2
    assert len(pipe.store) == 2
    assert len({c.id for c in pipe.store.chunks}) == 2
    assert pipe.ingest_path(str(docs))["new_chunks"] == 0


def test_empty_file_clears_only_its_passages(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    first, second = docs / "a.md", docs / "b.md"
    first.write_text("Obsolete document content.", encoding="utf-8")
    second.write_text("Retained document content.", encoding="utf-8")
    pipe = open_pipe(tmp_path)
    pipe.ingest_path(str(docs))
    first.write_text("   \n", encoding="utf-8")
    result = pipe.ingest_path(str(first))
    assert result["removed_chunks"] == 1
    assert result["new_chunks"] == 0
    assert [Path(c.source).name for c in open_pipe(tmp_path).store.chunks] == ["b.md"]


def test_embedding_failure_preserves_existing_document(tmp_path, monkeypatch):
    file = tmp_path / "policy.md"
    file.write_text("Original policy.", encoding="utf-8")
    pipe = open_pipe(tmp_path)
    pipe.ingest_path(str(file))
    original = [c.to_dict() for c in pipe.store.chunks]
    file.write_text("Replacement policy.", encoding="utf-8")
    def fail(text):
        raise RuntimeError("embedding unavailable")
    monkeypatch.setattr(pipe.store.embedder, "embed", fail)
    with pytest.raises(RuntimeError):
        pipe.ingest_path(str(file))
    assert [c.to_dict() for c in pipe.store.chunks] == original
    assert [c.to_dict() for c in open_pipe(tmp_path).store.chunks] == original


def test_missing_path_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_pipe(tmp_path).ingest_path(str(tmp_path / "missing"))


def test_relative_and_absolute_paths_share_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file = tmp_path / "policy.md"
    file.write_text("A stable policy.", encoding="utf-8")
    pipe = open_pipe(tmp_path)
    pipe.ingest_path("policy.md")
    assert pipe.ingest_path(str(file))["new_chunks"] == 0
    assert len(pipe.store) == 1
