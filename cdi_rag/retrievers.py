"""
Custom retrieval components for the CDI RAG system.

This module provides:
- DocumentReranker: TF-IDF-based document re-ranking for improved relevance
- HybridRetriever: Combines BM25 (keyword) and vector (semantic) search
"""

import logging
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

from .config import (
    RERANKING_MAX_FEATURES,
    RERANKING_NGRAM_RANGE,
    RERANKING_STOP_WORDS,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RETRIEVAL_K_INITIAL,
    RETRIEVAL_K_FINAL
)
from .utils import filter_documents_by_metadata


class DocumentReranker:
    """
    Re-ranks documents based on relevance to the query using TF-IDF similarity.

    This class provides a lightweight re-ranking mechanism using TF-IDF vectorization
    and cosine similarity. It's designed to improve retrieval precision by re-ordering
    documents based on lexical similarity to the query.

    Attributes:
        use_reranking: Whether re-ranking is enabled
        vectorizer: TfidfVectorizer for computing document similarities

    Example:
        >>> reranker = DocumentReranker(use_reranking=True)
        >>> docs = retriever.get_relevant_documents(query)  # 8 documents
        >>> reranked = reranker.rerank_documents(query, docs, top_k=4)  # Best 4
    """

    def __init__(self, use_reranking: bool = True):
        """
        Initialize the document re-ranker.

        Args:
            use_reranking: If False, re-ranking is skipped and original order is preserved
        """
        self.use_reranking = use_reranking
        self.vectorizer = TfidfVectorizer(
            stop_words=RERANKING_STOP_WORDS,
            max_features=RERANKING_MAX_FEATURES,
            ngram_range=RERANKING_NGRAM_RANGE
        )

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = RETRIEVAL_K_FINAL
    ) -> List[Document]:
        """
        Re-rank documents based on TF-IDF similarity to the query.

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents (or original order if re-ranking disabled/fails)

        Example:
            >>> docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]
            >>> reranked = reranker.rerank_documents("sepsis infection", docs, top_k=4)
            >>> len(reranked)
            4
        """
        # Skip re-ranking if disabled or not enough documents
        if not self.use_reranking or not documents:
            return documents[:top_k]

        if len(documents) <= top_k:
            return documents

        try:
            # Extract text from documents
            doc_texts = [doc.page_content for doc in documents]

            # Create TF-IDF vectors
            all_texts = [query] + doc_texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Calculate cosine similarity between query and documents
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors)[0]

            # Create scored documents
            scored_docs = list(zip(documents, similarities))

            # Sort by similarity score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Return top_k documents
            reranked_docs = [doc for doc, score in scored_docs[:top_k]]

            return reranked_docs

        except Exception as e:
            # If re-ranking fails, return original documents
            logger.warning(f"Re-ranking failed ({e}), returning original order")
            return documents[:top_k]


class HybridRetriever(BaseRetriever):
    """
    Custom retriever that combines BM25 (keyword) and vector (semantic) search.

    This retriever merges results from both keyword-based (BM25) and semantic
    (vector embedding) search, with configurable weighting. It also supports
    metadata filtering and optional TF-IDF re-ranking.

    Attributes:
        vector_retriever: Semantic search retriever
        bm25_retriever: Keyword-based BM25 retriever
        vector_weight: Weight for vector search results (default: 0.6)
        bm25_weight: Weight for BM25 results (default: 0.4)
        metadata_filter: Optional metadata filters to apply
        reranker: Optional DocumentReranker instance
        use_reranking: Whether to apply re-ranking

    Example:
        >>> from langchain_community.retrievers import BM25Retriever
        >>> bm25 = BM25Retriever.from_documents(documents)
        >>> vector = vectorstore.as_retriever(search_kwargs={"k": 3})
        >>> hybrid = HybridRetriever(
        ...     vector_retriever=vector,
        ...     bm25_retriever=bm25,
        ...     metadata_filter={"specialty": "Cardiology"}
        ... )
        >>> docs = hybrid.invoke("patient with chest pain")
    """

    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    vector_weight: float = VECTOR_WEIGHT
    bm25_weight: float = BM25_WEIGHT
    metadata_filter: Optional[dict] = None
    reranker: Optional[DocumentReranker] = None
    use_reranking: bool = False

    # Pydantic v2 configuration to allow arbitrary types (e.g., Mock objects in tests)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Get documents from both retrievers and merge results.

        This method:
        1. Retrieves documents from both BM25 and vector retrievers
        2. Applies metadata filtering if specified
        3. Merges results with weighted scoring
        4. Optionally applies re-ranking
        5. Returns top documents

        Args:
            query: Search query string
            run_manager: Optional callback manager (unused)

        Returns:
            List of top relevant documents (default: 4 documents)
        """
        # Get documents from both retrievers using invoke()
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # Apply metadata filtering if specified
        if self.metadata_filter:
            vector_docs = filter_documents_by_metadata(vector_docs, self.metadata_filter)
            bm25_docs = filter_documents_by_metadata(bm25_docs, self.metadata_filter)

        # Create a dict to track unique documents and their scores
        doc_scores = {}

        # Add vector search results with their weights
        for i, doc in enumerate(vector_docs):
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            # Score based on position (higher position = lower score)
            score = self.vector_weight * (1.0 / (i + 1))
            doc_scores[doc_key] = {'doc': doc, 'score': score}

        # Add BM25 results with their weights
        for i, doc in enumerate(bm25_docs):
            doc_key = doc.page_content[:100]
            score = self.bm25_weight * (1.0 / (i + 1))

            if doc_key in doc_scores:
                # Document found by both retrievers - boost score
                doc_scores[doc_key]['score'] += score
            else:
                doc_scores[doc_key] = {'doc': doc, 'score': score}

        # Sort by score and return top documents
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        # Get initial top documents (more than needed for re-ranking)
        initial_docs = [item['doc'] for item in sorted_docs[:RETRIEVAL_K_INITIAL]]

        # Apply re-ranking if enabled
        if self.use_reranking and self.reranker:
            reranked_docs = self.reranker.rerank_documents(
                query,
                initial_docs,
                top_k=RETRIEVAL_K_FINAL
            )
            return reranked_docs
        else:
            # Return top documents without re-ranking
            return initial_docs[:RETRIEVAL_K_FINAL]
