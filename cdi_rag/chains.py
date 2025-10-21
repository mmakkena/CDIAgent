"""
RAG chain creation for the CDI RAG system.

This module orchestrates the retrieval and generation components to create
complete RAG chains with support for:
- Hybrid search (BM25 + vector)
- Metadata filtering
- Document re-ranking
- Customizable retrieval strategies
"""

import logging
from typing import Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever

from .config import (
    SYSTEM_PROMPT,
    RETRIEVAL_K_BM25,
    RETRIEVAL_K_VECTOR,
)
from .retrievers import DocumentReranker, HybridRetriever
from .utils import convert_metadata_filter_to_chroma

logger = logging.getLogger(__name__)


# Prompt template for CDI query generation
PROMPT_TEMPLATE = PromptTemplate(
    template=f"Context: {{context}}\n\nClinical Note: {{question}}\n\nCDI Specialist: {SYSTEM_PROMPT}",
    input_variables=["context", "question"]
)


def create_rag_chain(
    llm,
    vectorstore,
    use_hybrid_search: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_reranking: bool = True
):
    """
    Create a LangChain RetrievalQA chain with configurable retrieval strategy.

    This function orchestrates the complete RAG pipeline:
    1. Sets up retrieval (hybrid or vector-only)
    2. Applies metadata filtering if specified
    3. Configures re-ranking if enabled
    4. Creates the final QA chain with LLM

    Args:
        llm: Language model for generation (from get_llm_pipeline())
        vectorstore: ChromaDB vector store (from setup_chroma_db())
        use_hybrid_search: If True, uses BM25 + vector search (default: True)
        metadata_filter: Optional dict of metadata filters
                        e.g., {"specialty": "Cardiology", "icd10": ["I21.0", "I21.1"]}
        use_reranking: If True, applies TF-IDF re-ranking (default: True)

    Returns:
        RetrievalQA: Complete RAG chain ready for query generation

    Example:
        >>> llm = get_llm_pipeline()
        >>> vectorstore = setup_chroma_db()
        >>> qa_chain = create_rag_chain(
        ...     llm, vectorstore,
        ...     use_hybrid_search=True,
        ...     metadata_filter={"specialty": "Cardiology"},
        ...     use_reranking=True
        ... )
        >>> result = qa_chain.invoke({"query": "Patient with chest pain..."})

    Notes:
        - Hybrid search: 60% vector + 40% BM25
        - Re-ranking: Retrieves 8 docs, re-ranks to top 4
        - Fallback: Falls back to vector-only if hybrid search fails
    """

    if use_hybrid_search:
        logger.info("Setting up hybrid search (BM25 + Vector)...")

        # Get all documents from vectorstore for BM25 indexing
        collection = vectorstore._collection

        # Apply metadata filter if provided (for ChromaDB)
        if metadata_filter:
            chroma_filter = convert_metadata_filter_to_chroma(metadata_filter)
            results = collection.get(where=chroma_filter)
            logger.info(f"  Filtered to {len(results.get('documents', []))} documents matching: {metadata_filter}")
        else:
            results = collection.get()

        # Create Document objects from the collection
        documents = []
        if results and results.get('documents') and results.get('metadatas'):
            for doc_text, metadata in zip(results['documents'], results['metadatas']):
                documents.append(Document(page_content=doc_text, metadata=metadata))

        # Check if any documents were found
        if not documents:
            logger.warning("No documents found matching the filter criteria")
            logger.info("Falling back to vector-only search")

            # Fall back to vector-only search
            search_kwargs = {"k": RETRIEVAL_K_VECTOR}
            if metadata_filter:
                chroma_filter = convert_metadata_filter_to_chroma(metadata_filter)
                search_kwargs["filter"] = chroma_filter

            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        else:
            # Create BM25 retriever (keyword-based)
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = RETRIEVAL_K_BM25

            # Create vector retriever (semantic search)
            search_kwargs = {"k": RETRIEVAL_K_VECTOR}
            if metadata_filter:
                chroma_filter = convert_metadata_filter_to_chroma(metadata_filter)
                search_kwargs["filter"] = chroma_filter

            vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

            # Initialize reranker if enabled
            reranker = DocumentReranker(use_reranking=use_reranking) if use_reranking else None

            # Combine both retrievers with hybrid retriever
            hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
                metadata_filter=metadata_filter,
                reranker=reranker,
                use_reranking=use_reranking
            )

            retriever = hybrid_retriever

            # Build status message
            status_parts = ["✓ Hybrid search configured (60% semantic + 40% keyword)"]
            if use_reranking:
                status_parts.append("with TF-IDF re-ranking")
            if metadata_filter:
                status_parts.append(f"and filters: {metadata_filter}")
            logger.info(" ".join(status_parts))

    else:
        # Use only vector search
        logger.info("Using vector search only...")
        search_kwargs = {"k": 2}

        if metadata_filter:
            chroma_filter = convert_metadata_filter_to_chroma(metadata_filter)
            search_kwargs["filter"] = chroma_filter
            logger.info(f"  With metadata filters: {metadata_filter}")

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
    )

    return qa_chain
