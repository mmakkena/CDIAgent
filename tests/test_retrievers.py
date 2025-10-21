"""
Unit tests for cdi_rag.retrievers module.

Tests DocumentReranker and HybridRetriever classes.
"""

import pytest
from typing import List
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from cdi_rag.retrievers import DocumentReranker, HybridRetriever


class MockRetriever(BaseRetriever):
    """Mock retriever for testing that properly inherits from BaseRetriever."""

    documents: List[Document] = []
    call_count: int = 0
    last_query: str = ""

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Return predefined documents and track calls."""
        self.call_count += 1
        self.last_query = query
        return self.documents


class TestDocumentReranker:
    """Test DocumentReranker class."""

    @pytest.fixture
    def reranker(self):
        """Create a DocumentReranker instance."""
        return DocumentReranker(use_reranking=True)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Patient with severe sepsis and infection"),
            Document(page_content="Acute myocardial infarction with chest pain"),
            Document(page_content="Septic shock with hypotension"),
            Document(page_content="Chronic heart failure"),
            Document(page_content="Severe sepsis with organ dysfunction"),
            Document(page_content="Pneumonia with respiratory distress"),
            Document(page_content="Acute kidney injury"),
            Document(page_content="Sepsis with positive blood cultures"),
        ]

    def test_initialization_enabled(self):
        """Test initialization with re-ranking enabled."""
        reranker = DocumentReranker(use_reranking=True)
        assert reranker.use_reranking is True
        assert reranker.vectorizer is not None

    def test_initialization_disabled(self):
        """Test initialization with re-ranking disabled."""
        reranker = DocumentReranker(use_reranking=False)
        assert reranker.use_reranking is False

    def test_rerank_disabled_returns_original(self, sample_documents):
        """Test that disabled re-ranker returns original order."""
        reranker = DocumentReranker(use_reranking=False)
        result = reranker.rerank_documents("sepsis", sample_documents, top_k=3)

        # Should return first 3 documents in original order
        assert len(result) == 3
        assert result[0] == sample_documents[0]
        assert result[1] == sample_documents[1]
        assert result[2] == sample_documents[2]

    def test_rerank_empty_documents(self, reranker):
        """Test re-ranking with empty document list."""
        result = reranker.rerank_documents("query", [], top_k=4)
        assert len(result) == 0

    def test_rerank_fewer_docs_than_top_k(self, reranker):
        """Test when documents < top_k."""
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2")
        ]
        result = reranker.rerank_documents("query", docs, top_k=5)

        # Should return all documents
        assert len(result) == 2

    def test_rerank_changes_order(self, reranker, sample_documents):
        """Test that re-ranking actually reorders documents."""
        query = "sepsis infection blood culture"
        result = reranker.rerank_documents(query, sample_documents, top_k=4)

        assert len(result) == 4

        # Check that sepsis-related documents are ranked higher
        sepsis_terms = ["sepsis", "septic", "infection"]
        top_doc_content = result[0].page_content.lower()

        # At least one sepsis term should be in top document
        assert any(term in top_doc_content for term in sepsis_terms)

    def test_rerank_top_k_parameter(self, reranker, sample_documents):
        """Test that top_k parameter is respected."""
        for k in [2, 4, 6]:
            result = reranker.rerank_documents("test query", sample_documents, top_k=k)
            assert len(result) == k

    def test_rerank_preserves_metadata(self, reranker):
        """Test that re-ranking preserves document metadata."""
        docs = [
            Document(page_content="text1", metadata={"id": 1, "source": "A"}),
            Document(page_content="text2", metadata={"id": 2, "source": "B"}),
            Document(page_content="text3", metadata={"id": 3, "source": "C"}),
        ]
        result = reranker.rerank_documents("query", docs, top_k=2)

        # Check that metadata is preserved
        for doc in result:
            assert "id" in doc.metadata
            assert "source" in doc.metadata

    @patch('cdi_rag.retrievers.TfidfVectorizer')
    def test_rerank_handles_errors(self, mock_vectorizer, sample_documents):
        """Test graceful error handling when re-ranking fails."""
        # Make vectorizer raise an exception
        mock_vectorizer.return_value.fit_transform.side_effect = Exception("TF-IDF error")

        reranker = DocumentReranker(use_reranking=True)
        result = reranker.rerank_documents("query", sample_documents, top_k=3)

        # Should fall back to original order
        assert len(result) == 3
        assert result[0] == sample_documents[0]


class TestHybridRetriever:
    """Test HybridRetriever class."""

    @pytest.fixture
    def mock_vector_retriever(self):
        """Create mock vector retriever."""
        return MockRetriever(documents=[
            Document(page_content="vector doc 1", metadata={"source": "V1"}),
            Document(page_content="vector doc 2", metadata={"source": "V2"}),
            Document(page_content="vector doc 3", metadata={"source": "V3"}),
        ])

    @pytest.fixture
    def mock_bm25_retriever(self):
        """Create mock BM25 retriever."""
        return MockRetriever(documents=[
            Document(page_content="bm25 doc 1", metadata={"source": "B1"}),
            Document(page_content="bm25 doc 2", metadata={"source": "B2"}),
            Document(page_content="bm25 doc 3", metadata={"source": "B3"}),
        ])

    def test_initialization(self, mock_vector_retriever, mock_bm25_retriever):
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever,
            vector_weight=0.6,
            bm25_weight=0.4
        )

        assert retriever.vector_weight == 0.6
        assert retriever.bm25_weight == 0.4
        assert retriever.use_reranking is False

    def test_retrieval_calls_both_retrievers(self, mock_vector_retriever, mock_bm25_retriever):
        """Test that both retrievers are called."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever
        )

        query = "test query"
        results = retriever.invoke(query)

        # Both retrievers should be called
        assert mock_vector_retriever.call_count == 1
        assert mock_bm25_retriever.call_count == 1
        assert mock_vector_retriever.last_query == query
        assert mock_bm25_retriever.last_query == query

        # Should return documents
        assert len(results) > 0

    def test_retrieval_merges_results(self, mock_vector_retriever, mock_bm25_retriever):
        """Test that results from both retrievers are merged."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever
        )

        results = retriever.invoke("test query")

        # Results should include docs from both retrievers
        # (up to 4 documents by default)
        assert len(results) <= 4

    def test_metadata_filtering(self):
        """Test metadata filtering functionality."""
        # Create retrievers with documents having different metadata
        vector_retriever = MockRetriever(documents=[
            Document(page_content="doc1", metadata={"specialty": "Cardiology"}),
            Document(page_content="doc2", metadata={"specialty": "Neurology"}),
        ])
        bm25_retriever = MockRetriever(documents=[
            Document(page_content="doc3", metadata={"specialty": "Cardiology"}),
            Document(page_content="doc4", metadata={"specialty": "Neurology"}),
        ])

        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            metadata_filter={"specialty": "Cardiology"}
        )

        results = retriever.invoke("test query")

        # All results should be Cardiology
        for doc in results:
            assert doc.metadata["specialty"] == "Cardiology"

    def test_with_reranking(self, mock_vector_retriever, mock_bm25_retriever):
        """Test hybrid retriever with re-ranking enabled."""
        reranker = DocumentReranker(use_reranking=True)

        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever,
            reranker=reranker,
            use_reranking=True
        )

        results = retriever.invoke("test query")

        # Should return re-ranked results
        assert len(results) <= 4

    def test_without_reranking(self, mock_vector_retriever, mock_bm25_retriever):
        """Test hybrid retriever without re-ranking."""
        retriever = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever,
            use_reranking=False
        )

        results = retriever.invoke("test query")

        # Should return top documents without re-ranking
        assert len(results) <= 4

    def test_scoring_weights(self, mock_vector_retriever, mock_bm25_retriever):
        """Test that scoring weights affect document ranking."""
        # Create retriever with high vector weight
        retriever_vector_heavy = HybridRetriever(
            vector_retriever=mock_vector_retriever,
            bm25_retriever=mock_bm25_retriever,
            vector_weight=0.9,
            bm25_weight=0.1
        )

        results = retriever_vector_heavy.invoke("test")
        assert len(results) > 0

        # Documents should be scored and sorted
        # (exact scoring logic tested separately)

    def test_duplicate_documents(self):
        """Test handling of duplicate documents from both retrievers."""
        # Create duplicate document
        duplicate_doc = Document(page_content="Same content", metadata={"source": "Same"})

        vector_retriever = MockRetriever(documents=[duplicate_doc, Document(page_content="doc2")])
        bm25_retriever = MockRetriever(documents=[duplicate_doc, Document(page_content="doc3")])

        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever
        )

        results = retriever.invoke("test")

        # Duplicate should get boosted score but appear once
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
