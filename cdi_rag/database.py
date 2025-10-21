"""
Vector database initialization for the CDI RAG system.

This module handles ChromaDB setup with support for:
- Persistent storage of document embeddings
- Loading existing databases
- Creating new databases from document collections
"""

import logging
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
from .data import MOCK_CDI_DOCUMENTS

logger = logging.getLogger(__name__)


def setup_chroma_db():
    """
    Initialize or load the persistent ChromaDB vector store.

    This function:
    1. Checks if a persisted ChromaDB already exists
    2. If exists: Loads the existing database
    3. If not: Creates a new database from MOCK_CDI_DOCUMENTS
    4. Uses HuggingFace embeddings (all-MiniLM-L6-v2, 384 dimensions)

    Returns:
        Chroma: Initialized vector store ready for retrieval

    Example:
        >>> vectorstore = setup_chroma_db()
        >>> docs = vectorstore.similarity_search("malnutrition", k=2)

    Notes:
        - Database persisted at: ./chroma_db_cdi/
        - Collection name: cdi_guidelines
        - Embedding model: all-MiniLM-L6-v2 (384 dimensions)
        - Auto-persists with ChromaDB 0.4.x+
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if the database exists
    if os.path.exists(CHROMA_DB_PATH):
        logger.info(f"Loading existing ChromaDB from {CHROMA_DB_PATH}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
    else:
        logger.info(f"ChromaDB not found. Creating and persisting database at {CHROMA_DB_PATH}...")

        # Extract texts and metadata from MOCK_CDI_DOCUMENTS
        texts = [doc[0] for doc in MOCK_CDI_DOCUMENTS]
        metadatas = [doc[1] for doc in MOCK_CDI_DOCUMENTS]

        # Create and persist the database from texts
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # ChromaDB 0.4.x+ automatically persists, no need to call persist() manually
        logger.info(f"ChromaDB created and persisted with {vectorstore._collection.count()} documents.")

    return vectorstore
