"""End-to-end pipeline tests (offline extractive provider)."""

import pytest

from nanorag.pipeline import RAGPipeline

DOCS = {
    "netrunning.md": (
        "# Netrunning\n\n"
        "Netrunners jack into the data grid using cyberdecks. On the other side "
        "they face ICE — Intrusion Countermeasures Electronics — defensive "
        "software that can fry an unprepared runner's nervous system. A daemon "
        "is a hostile program used to breach ICE.\n\n"
        "The Sandbreak protocol is the standard sequence for a quiet breach: "
        "scan the subnet, spoof the handshake, deploy the daemon, extract data "
        "before the Intrusion Countermeasures recalibrate.\n"
    ),
    "food.md": (
        "# Street Food\n\n"
        "The harbour district is known for sushi counters and noodle bars that "
        "stay open until dawn. Most street chefs accept credits only.\n"
    ),
}


@pytest.fixture()
def pipeline(tmp_path, monkeypatch):
    # force the offline provider regardless of the host environment
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_URL"):
        monkeypatch.delenv(var, raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text, encoding="utf-8")
    pipe = RAGPipeline.open(data_dir=str(tmp_path / "data"))
    pipe.ingest_path(str(docs))
    return pipe


def test_ingest_creates_persisted_index(pipeline, tmp_path):
    assert len(pipeline.store) > 0
    assert (tmp_path / "data" / "index.json").exists()


def test_retrieval_finds_expected_doc(pipeline):
    passages = pipeline.retrieve("What is ICE?", k=3)
    assert passages
    assert any(p.source.endswith("netrunning.md") for p in passages)
    top = passages[0]
    assert "ICE" in top.chunk.text or "Countermeasures" in top.chunk.text


def test_answer_is_cited_and_grounded(pipeline):
    answer = pipeline.ask("What is ICE?")
    assert answer.provider == "extractive"
    assert "[" in answer.text and "]" in answer.text  # citations present
    assert "Countermeasures" in answer.text or "ICE" in answer.text


def test_answer_off_topic_still_returns_passage(pipeline):
    answer = pipeline.ask("nonsense zzz qqq vocab")
    assert isinstance(answer.text, str) and answer.text


def test_index_roundtrip(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_URL"):
        monkeypatch.delenv(var, raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Alpha protocol documents. " * 40, encoding="utf-8")
    data = str(tmp_path / "data")

    first = RAGPipeline.open(data_dir=data)
    first.ingest_path(str(docs))
    n = len(first.store)

    second = RAGPipeline.open(data_dir=data)
    assert len(second.store) == n
    hits = second.retrieve("alpha protocol", k=2)
    assert hits[0].source.endswith("a.md")


def test_provider_resolution_prefers_env(monkeypatch):
    from nanorag.providers import AnthropicProvider, ExtractiveProvider, OpenAIProvider, resolve_provider

    assert isinstance(resolve_provider(), ExtractiveProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(resolve_provider(), OpenAIProvider)
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(resolve_provider(), AnthropicProvider)
    assert isinstance(resolve_provider("extractive"), ExtractiveProvider)
    try:
        resolve_provider("bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
