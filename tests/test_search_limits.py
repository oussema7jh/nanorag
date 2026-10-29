"""Regression tests for result limits; runnable with unittest or pytest."""
import unittest
from unittest.mock import Mock

from nanorag.chunking import Chunk
from nanorag.embeddings import HashedNGramEmbedder
from nanorag.retrieval import BM25, HybridRetriever, VectorRetriever
from nanorag.store import VectorStore


class SearchLimitTests(unittest.TestCase):
    def setUp(self):
        self.embedder = HashedNGramEmbedder(dim=64)
        self.chunks = [
            Chunk(id=str(i), doc_id=str(i), seq=0, text="neural implants " * (i + 1), metadata={})
            for i in range(3)
        ]
        self.store = VectorStore(self.embedder)
        self.store.add(self.chunks)
        self.bm25 = BM25()
        self.bm25.index(self.chunks)
        self.vector = VectorRetriever(self.store, self.embedder)
        self.hybrid = HybridRetriever([self.bm25, self.vector])

    def test_nonpositive_limits_return_no_results(self):
        for k in (0, -1, -10):
            for populated in (True, False):
                store = self.store if populated else VectorStore(self.embedder)
                bm25 = self.bm25 if populated else BM25()
                vector = VectorRetriever(store, self.embedder)
                searches = [
                    lambda: bm25.search("neural", k),
                    lambda: store.search(self.embedder.embed("neural"), k),
                    lambda: vector.search("neural", k),
                    lambda: HybridRetriever([bm25, vector]).search("neural", k),
                ]
                for index, search in enumerate(searches):
                    with self.subTest(k=k, populated=populated, search=index):
                        self.assertEqual(search(), [])

    def test_nonpositive_limits_skip_embedding_and_child_retrievers(self):
        embedder = Mock()
        child = Mock()
        for k in (0, -1):
            self.assertEqual(VectorRetriever(self.store, embedder).search("neural", k), [])
            self.assertEqual(HybridRetriever([child]).search("neural", k), [])
        embedder.embed.assert_not_called()
        child.search.assert_not_called()

    def test_positive_limits_preserve_results_and_ranking(self):
        searches = [
            lambda k: self.bm25.search("neural", k),
            lambda k: self.store.search(self.embedder.embed("neural"), k),
            lambda k: self.vector.search("neural", k),
            lambda k: self.hybrid.search("neural", k),
        ]
        for index, search in enumerate(searches):
            with self.subTest(search=index):
                full = search(10)
                self.assertEqual(len(full), 3)
                self.assertEqual(search(1), full[:1])
                self.assertEqual(search(2), full[:2])


if __name__ == "__main__":
    unittest.main()
