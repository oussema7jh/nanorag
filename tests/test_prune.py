"""Deletion reconciliation must be scoped, explicit, and failure-safe."""
from pathlib import Path

import pytest

from nanorag.cli import main
from nanorag.pipeline import RAGPipeline
from nanorag.providers import ExtractiveProvider
from nanorag.store import VectorStore


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs"
    sibling = tmp_path / "docs-other"
    root.mkdir()
    sibling.mkdir()
    files = [root / "removed.md", root / "kept.txt", sibling / "outside.md"]
    for file in files:
        file.write_text(f"Knowledge for {file.stem} document.", encoding="utf-8")
    pipe = RAGPipeline.open(str(tmp_path / "index"), provider=ExtractiveProvider())
    pipe.ingest_path(str(root))
    pipe.ingest_path(str(sibling))
    files[0].unlink()
    files[2].unlink()
    return pipe, root, files


def test_preview_does_not_modify_memory_or_disk(corpus):
    pipe, root, files = corpus
    before = Path(pipe.index_path).read_bytes()
    result = pipe.prune_path(str(root))
    assert result["dry_run"] is True
    assert result["sources"] == [str(files[0])]
    assert result["documents"] == result["removed_chunks"] == 1
    assert len(pipe.store) == 3
    assert Path(pipe.index_path).read_bytes() == before


def test_apply_preserves_other_corpora_and_refreshes_retrievers(corpus):
    pipe, root, files = corpus
    result = pipe.prune_path(str(root), dry_run=False)
    assert result["removed_chunks"] == 1
    assert {c.source for c in pipe.store.chunks} == {str(files[1]), str(files[2])}
    assert all(p.source != str(files[0]) for p in pipe.retrieve("removed"))
    assert all(c.source != str(files[0]) for c, _ in pipe._bm25.search("removed"))
    reloaded = VectorStore.load(pipe.index_path, pipe.embedder)
    assert len(reloaded) == 2
    assert pipe.prune_path(str(root), dry_run=False)["documents"] == 0


def test_missing_root_does_not_empty_the_index(corpus):
    pipe, root, files = corpus
    before = Path(pipe.index_path).read_bytes()
    with pytest.raises(FileNotFoundError):
        pipe.prune_path(str(root / "unavailable"), dry_run=False)
    assert len(pipe.store) == 3
    assert Path(pipe.index_path).read_bytes() == before


def test_file_scope_is_rejected(corpus):
    pipe, root, files = corpus
    with pytest.raises(NotADirectoryError):
        pipe.prune_path(str(files[1]), dry_run=False)


def test_stat_permission_error_preserves_index(corpus, monkeypatch):
    pipe, root, files = corpus
    original_stat = Path.stat
    def guarded_stat(path, *args, **kwargs):
        if path == files[1]:
            raise PermissionError("cannot inspect source")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(PermissionError):
        pipe.prune_path(str(root), dry_run=False)
    assert len(pipe.store) == 3
    assert len(VectorStore.load(pipe.index_path, pipe.embedder)) == 3


def test_persistence_failure_preserves_live_retrievers(corpus, monkeypatch):
    pipe, root, files = corpus
    before = Path(pipe.index_path).read_bytes()
    def fail_save(*args):
        raise OSError("disk full")
    monkeypatch.setattr(VectorStore, "save", fail_save)
    with pytest.raises(OSError):
        pipe.prune_path(str(root), dry_run=False)
    assert len(pipe.store) == len(pipe._bm25) == 3
    assert Path(pipe.index_path).read_bytes() == before


def test_legacy_relative_sources_are_preserved(corpus):
    pipe, root, files = corpus
    for chunk in pipe.store.chunks:
        if chunk.source == str(files[0]):
            chunk.metadata["source"] = "docs/removed.md"
    assert pipe.prune_path(str(root), dry_run=False)["documents"] == 0
    assert len(pipe.store) == 3


def test_nested_missing_sources_and_empty_index(tmp_path):
    root = tmp_path / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    file = nested / "a.md"
    file.write_text("A nested document.", encoding="utf-8")
    pipe = RAGPipeline.open(str(tmp_path / "index"), provider=ExtractiveProvider())
    pipe.ingest_path(str(root))
    file.unlink()
    nested.rmdir()
    assert pipe.prune_path(str(root), dry_run=False)["removed_chunks"] == 1
    assert pipe.retrieve("nested") == []
    assert pipe._bm25.search("nested") == []
    assert len(VectorStore.load(pipe.index_path, pipe.embedder)) == 0


def test_recreated_file_is_preserved(corpus):
    pipe, root, files = corpus
    assert pipe.prune_path(str(root))["documents"] == 1
    files[0].write_text("Recreated document; ingest to update it.", encoding="utf-8")
    assert pipe.prune_path(str(root), dry_run=False)["documents"] == 0
    assert len(pipe.store) == 3


def test_cli_preview_apply_and_error(corpus, capsys):
    pipe, root, files = corpus
    args = ["--provider", "extractive", "--data-dir", str(Path(pipe.index_path).parent), "prune"]
    assert main(args + [str(root)]) == 0
    assert "would remove 1 documents" in capsys.readouterr().out
    assert len(VectorStore.load(pipe.index_path, pipe.embedder)) == 3
    assert main(args + [str(root), "--apply"]) == 0
    assert "removed 1 documents" in capsys.readouterr().out
    assert len(VectorStore.load(pipe.index_path, pipe.embedder)) == 2
    assert main(args + [str(root / "absent"), "--apply"]) == 1
    assert "prune failed" in capsys.readouterr().err
