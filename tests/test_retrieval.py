"""Tests for BM25, the vector retriever and RRF hybrid fusion."""

from nanorag.chunking import Chunk
from nanorag.embeddings import HashedNGramEmbedder
from nanorag.retrieval import BM25, HybridRetriever, VectorRetriever
from nanorag.store import VectorStore

DOCS = [
    ("ice", "Intrusion Countermeasures Electronics, or ICE, protects corporate servers from netrunners."),
    ("districts", "The harbour district is famous for its sushi restaurants and street food markets."),
    ("implants", "Neural implants interface directly with the optic nerve to overlay augmented reality."),
    ("grid", "The data grid routes traffic through underground fibre optic trunks beneath the city."),
]


def _chunks():
    return [
        Chunk(id=f"c{i}", doc_id=d, seq=0, text=t, metadata={"source": f"{d}.md"})
        for i, (d, t) in enumerate(DOCS)
    ]


def test_bm25_ranks_matching_doc_first():
    bm25 = BM25()
    bm25.index(_chunks())
    hits = bm25.search("What is ICE?", k=2)
    assert hits[0][0].doc_id == "ice"


def test_bm25_handles_unseen_terms():
    bm25 = BM25()
    bm25.index(_chunks())
    assert bm25.search("zzz qqq", k=3) == []


def test_vector_retriever_finds_topical_doc():
    emb = HashedNGramEmbedder()
    store = VectorStore(emb)
    store.add(_chunks())
    hits = VectorRetriever(store, emb).search("sushi restaurants in the harbour", k=2)
    assert hits[0][0].doc_id == "districts"


def test_hybrid_beats_single_leg_on_fused_query():
    emb = HashedNGramEmbedder()
    store = VectorStore(emb)
    chunks = _chunks()
    store.add(chunks)
    bm25 = BM25()
    bm25.index(chunks)
    hybrid = HybridRetriever([bm25, VectorRetriever(store, emb)])
    hits = hybrid.search("neural implants and augmented reality overlays", k=1)
    assert hits[0][0].doc_id == "implants"


def test_rrf_promotes_docs_found_by_both_retrievers():
    emb = HashedNGramEmbedder()
    store = VectorStore(emb)
    chunks = _chunks()
    store.add(chunks)
    bm25 = BM25()
    bm25.index(chunks)
    hybrid = HybridRetriever([bm25, VectorRetriever(store, emb)], rrf_k=60)
    hits = hybrid.search("ICE", k=4)
    assert {c.doc_id for c, _ in hits} == {"ice", "grid", "districts", "implants"}
    assert hits[0][0].doc_id == "ice"


def test_store_roundtrip_preserves_search(tmp_path):
    emb = HashedNGramEmbedder(dim=128)
    store = VectorStore(emb)
    store.add(_chunks())
    path = str(tmp_path / "index.json")
    store.save(path)
    loaded = VectorStore.load(path, HashedNGramEmbedder(dim=128))
    q = emb.embed("underground fibre optic trunks")
    before = [c.id for c, _ in store.search(q, k=4)]
    after = [c.id for c, _ in loaded.search(q, k=4)]
    assert before == after
    assert loaded.stats()["chunks"] == 4


def test_store_add_is_idempotent():
    emb = HashedNGramEmbedder(dim=64)
    store = VectorStore(emb)
    chunks = _chunks()
    store.add(chunks)
    store.add(chunks)
    assert len(store) == 4
