"""
CDI RAG System - Clinical Documentation Integrity Retrieval-Augmented Generation

A modular RAG system for generating CDI physician queries with support for:
- Hybrid search (BM25 + vector embeddings)
- TF-IDF document re-ranking
- Metadata filtering
- Query validation

Main Components:
    - get_llm_pipeline: Initialize language model
    - setup_chroma_db: Setup vector database
    - create_rag_chain: Create complete RAG chain

Example:
    >>> from cdi_rag import get_llm_pipeline, setup_chroma_db, create_rag_chain
    >>> llm = get_llm_pipeline()
    >>> vectorstore = setup_chroma_db()
    >>> qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True)
    >>> result = qa_chain.invoke({"query": "Patient with malnutrition..."})
"""

# Core functions (most commonly used)
from .models import get_llm_pipeline
from .database import setup_chroma_db
from .chains import create_rag_chain, PROMPT_TEMPLATE, SYSTEM_PROMPT

# Retrieval components
from .retrievers import DocumentReranker, HybridRetriever

# Utilities
from .utils import (
    convert_metadata_filter_to_chroma,
    extract_source_from_metadata,
    filter_documents_by_metadata
)

# Configuration (for advanced usage)
from . import config

# Data
from .data import MOCK_CDI_DOCUMENTS

__version__ = "2.0.0"

__all__ = [
    # Core functions
    "get_llm_pipeline",
    "setup_chroma_db",
    "create_rag_chain",

    # Prompts
    "PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",

    # Retrieval components
    "DocumentReranker",
    "HybridRetriever",

    # Utilities
    "convert_metadata_filter_to_chroma",
    "extract_source_from_metadata",
    "filter_documents_by_metadata",

    # Configuration module
    "config",

    # Data
    "MOCK_CDI_DOCUMENTS",
]
