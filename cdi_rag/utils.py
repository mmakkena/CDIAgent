"""
Utility functions for the CDI RAG system.

This module provides helper functions for metadata filtering, document processing,
and other common operations.
"""

from typing import Dict, Optional, List, Any


def convert_metadata_filter_to_chroma(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert metadata filter dictionary to ChromaDB format.

    ChromaDB uses special operators like $in for list values. This function
    standardizes the filter format across the system.

    Args:
        metadata_filter: Dictionary of metadata filters
                        e.g., {"specialty": "Cardiology", "icd10": ["I21.0", "I21.1"]}

    Returns:
        ChromaDB-formatted filter dictionary, or None if input is None
        e.g., {"specialty": "Cardiology", "icd10": {"$in": ["I21.0", "I21.1"]}}

    Example:
        >>> convert_metadata_filter_to_chroma({"specialty": "Cardiology"})
        {"specialty": "Cardiology"}
        >>> convert_metadata_filter_to_chroma({"icd10": ["I21.0", "I21.1"]})
        {"icd10": {"$in": ["I21.0", "I21.1"]}}
    """
    if metadata_filter is None:
        return None

    chroma_filter = {}
    for key, value in metadata_filter.items():
        if isinstance(value, list):
            # ChromaDB uses $in operator for list values
            chroma_filter[key] = {"$in": value}
        else:
            # Direct equality for single values
            chroma_filter[key] = value

    return chroma_filter


def extract_source_from_metadata(metadata: Dict[str, Any]) -> str:
    """
    Extract source information from document metadata.

    Tries multiple common metadata field names to find the source citation.

    Args:
        metadata: Document metadata dictionary

    Returns:
        Source string, or "Unknown Source" if not found

    Example:
        >>> extract_source_from_metadata({"source": "CDI Guideline 2024"})
        "CDI Guideline 2024"
        >>> extract_source_from_metadata({"sample_name": "Cardiology Note 1"})
        "Cardiology Note 1"
    """
    return (
        metadata.get('source') or
        metadata.get('sample_name') or
        metadata.get('specialty') or
        'Unknown Source'
    )


def filter_documents_by_metadata(
    documents: List[Any],
    metadata_filter: Dict[str, Any]
) -> List[Any]:
    """
    Filter a list of documents by metadata criteria.

    Supports exact matching, list membership, and case-insensitive partial matching
    for strings.

    Args:
        documents: List of Document objects with metadata attribute
        metadata_filter: Dictionary of filter criteria

    Returns:
        Filtered list of documents matching all criteria

    Example:
        >>> docs = [
        ...     Document(content="...", metadata={"specialty": "Cardiology"}),
        ...     Document(content="...", metadata={"specialty": "Neurology"})
        ... ]
        >>> filter_documents_by_metadata(docs, {"specialty": "Cardiology"})
        [Document(...)]  # Only Cardiology document
    """
    filtered_docs = []

    for doc in documents:
        match = True

        for key, value in metadata_filter.items():
            # Check if key exists in document metadata
            if key not in doc.metadata:
                match = False
                break

            doc_value = doc.metadata[key]

            # Handle different filter types
            if isinstance(value, list):
                # If filter value is a list, check if doc value is in the list
                if doc_value not in value:
                    match = False
                    break
            elif isinstance(value, str) and isinstance(doc_value, str):
                # Case-insensitive partial match for strings
                if value.lower() not in doc_value.lower():
                    match = False
                    break
            else:
                # Exact match for other types
                if doc_value != value:
                    match = False
                    break

        if match:
            filtered_docs.append(doc)

    return filtered_docs
