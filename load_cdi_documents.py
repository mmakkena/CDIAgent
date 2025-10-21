#!/usr/bin/env python3
"""
Script to load CDI guidelines and clinical documents into ChromaDB vector store.
Supports multiple formats: PDF, TXT, CSV, JSON
"""

import os
import json
from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

# Configuration
CHROMA_DB_PATH = "./chroma_db_cdi"
CHROMA_COLLECTION_NAME = "cdi_guidelines"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents_from_directory(directory_path: str, file_pattern: str = "**/*.txt") -> List[Document]:
    """
    Load documents from a directory using glob pattern.

    Args:
        directory_path: Path to directory containing documents
        file_pattern: Glob pattern (e.g., "**/*.txt", "**/*.pdf")

    Returns:
        List of Document objects
    """
    print(f"Loading documents from {directory_path} with pattern {file_pattern}")

    if file_pattern.endswith(".txt"):
        loader = DirectoryLoader(
            directory_path,
            glob=file_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
    elif file_pattern.endswith(".pdf"):
        loader = DirectoryLoader(
            directory_path,
            glob=file_pattern,
            loader_cls=PyPDFLoader
        )
    elif file_pattern.endswith(".csv"):
        loader = DirectoryLoader(
            directory_path,
            glob=file_pattern,
            loader_cls=CSVLoader
        )
    else:
        raise ValueError(f"Unsupported file pattern: {file_pattern}")

    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def load_from_text_list(text_list: List[tuple]) -> List[Document]:
    """
    Load documents from a list of (text, metadata) tuples.

    Args:
        text_list: List of (content, metadata_dict) tuples

    Returns:
        List of Document objects
    """
    documents = []
    for content, metadata in text_list:
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Created {len(documents)} documents from list")
    return documents


def load_from_json_file(json_file_path: str, content_key: str = "content", metadata_keys: List[str] = None) -> List[Document]:
    """
    Load documents from a JSON file.

    Args:
        json_file_path: Path to JSON file
        content_key: Key in JSON that contains the document content
        metadata_keys: Keys to include as metadata

    Returns:
        List of Document objects
    """
    print(f"Loading documents from JSON file: {json_file_path}")

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    if isinstance(data, list):
        for item in data:
            content = item.get(content_key, "")
            metadata = {}
            if metadata_keys:
                for key in metadata_keys:
                    if key in item:
                        metadata[key] = item[key]
            else:
                # Include all keys except content as metadata
                metadata = {k: v for k, v in item.items() if k != content_key}

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

    print(f"Loaded {len(documents)} documents from JSON")
    return documents


def create_or_update_vectorstore(documents: List[Document], clear_existing: bool = False):
    """
    Create or update the ChromaDB vector store with documents.

    Args:
        documents: List of Document objects to add
        clear_existing: If True, delete existing collection and create new one
    """
    print(f"\n{'Creating new' if clear_existing else 'Updating'} ChromaDB vector store...")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")

    # Filter complex metadata (lists, dicts) that ChromaDB doesn't support
    split_docs = filter_complex_metadata(split_docs)

    if clear_existing and os.path.exists(CHROMA_DB_PATH):
        print(f"Clearing existing database at {CHROMA_DB_PATH}")
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)

    # ChromaDB has a batch size limit, so we need to add documents in batches
    BATCH_SIZE = 1000  # Safe batch size (max is ~5461)

    if os.path.exists(CHROMA_DB_PATH) and not clear_existing:
        # Load existing vectorstore and add documents in batches
        print("Loading existing vector store and adding new documents...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )

        # Add documents in batches
        total_batches = (len(split_docs) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(split_docs), BATCH_SIZE):
            batch = split_docs[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            print(f"  Adding batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            vectorstore.add_documents(batch)

    else:
        # Create new vectorstore in batches
        print("Creating new vector store...")

        # Create with first batch
        first_batch = split_docs[:BATCH_SIZE]
        vectorstore = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION_NAME
        )
        print(f"  Created with first batch ({len(first_batch)} documents)")

        # Add remaining documents in batches
        if len(split_docs) > BATCH_SIZE:
            total_batches = (len(split_docs) + BATCH_SIZE - 1) // BATCH_SIZE
            for i in range(BATCH_SIZE, len(split_docs), BATCH_SIZE):
                batch = split_docs[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                print(f"  Adding batch {batch_num}/{total_batches} ({len(batch)} documents)...")
                vectorstore.add_documents(batch)

    # Get collection count
    total_docs = vectorstore._collection.count()
    print(f"✓ Vector store updated successfully!")
    print(f"  Total documents in collection: {total_docs}")

    return vectorstore


# Example usage functions

def example_load_cdi_guidelines():
    """Example: Load CDI guidelines from a predefined list."""

    CDI_GUIDELINES = [
        (
            "Severe malnutrition must be documented when BMI < 16 or albumin < 2.5 g/dL with clinical evidence. "
            "Query if 'malnutrition' is documented without severity (mild, moderate, severe).",
            {"source": "Malnutrition Guidelines 2024", "category": "Nutrition", "icd10": "E43"}
        ),
        (
            "Acute respiratory failure must be documented when patient requires mechanical ventilation, BiPAP, "
            "or CPAP with clinical indicators (PaO2 < 60 mmHg, PaCO2 > 50 mmHg, or pH < 7.35).",
            {"source": "Respiratory Guidelines 2024", "category": "Respiratory", "icd10": "J96.00"}
        ),
        (
            "Sepsis requires both infection documentation and organ dysfunction (SOFA score ≥2). "
            "Query if 'sepsis' is documented without organ dysfunction or if SIRS criteria only are met.",
            {"source": "Sepsis-3 Guidelines", "category": "Infectious Disease", "icd10": "A41.9"}
        ),
        (
            "Acute kidney injury (AKI) staging: Stage 1 (Cr 1.5-1.9x baseline), Stage 2 (Cr 2.0-2.9x baseline), "
            "Stage 3 (Cr ≥3x baseline or Cr ≥4.0). Document specific stage for accurate coding.",
            {"source": "KDIGO AKI Guidelines", "category": "Nephrology", "icd10": "N17.9"}
        ),
        (
            "Heart failure with reduced ejection fraction (HFrEF): LVEF ≤ 40%. "
            "Heart failure with preserved ejection fraction (HFpEF): LVEF ≥ 50%. "
            "Query for ejection fraction documentation if heart failure is documented.",
            {"source": "ACC/AHA Heart Failure Guidelines", "category": "Cardiology", "icd10": "I50.9"}
        ),
    ]

    documents = load_from_text_list(CDI_GUIDELINES)
    vectorstore = create_or_update_vectorstore(documents, clear_existing=True)
    return vectorstore


def example_load_from_files(data_directory: str):
    """Example: Load documents from a directory of files."""

    if not os.path.exists(data_directory):
        print(f"Directory not found: {data_directory}")
        return None

    # Load all text files
    documents = load_documents_from_directory(data_directory, "**/*.txt")

    # Optionally load PDFs
    # pdf_docs = load_documents_from_directory(data_directory, "**/*.pdf")
    # documents.extend(pdf_docs)

    vectorstore = create_or_update_vectorstore(documents, clear_existing=False)
    return vectorstore


def example_load_mtsamples_style():
    """
    Example: Load documents in MTSamples format.
    Format: JSON with fields like 'description', 'medical_specialty', 'sample_name', 'transcription'
    """

    # This is how you'd structure data from MTSamples or similar sources
    mtsamples_data = [
        {
            "content": "Patient presents with chest pain, dyspnea on exertion. EKG shows ST elevation. "
                      "Troponin elevated at 2.5. Diagnosed with acute myocardial infarction.",
            "medical_specialty": "Cardiology",
            "diagnosis": "Acute Myocardial Infarction",
            "source": "MTSamples",
            "category": "Discharge Summary"
        },
        {
            "content": "76-year-old male with confusion, fever, and elevated WBC. Blood cultures positive for "
                      "E. coli. Diagnosed with sepsis secondary to urinary tract infection.",
            "medical_specialty": "Internal Medicine",
            "diagnosis": "Sepsis due to UTI",
            "source": "MTSamples",
            "category": "Progress Note"
        },
    ]

    # Convert to Document objects
    documents = []
    for item in mtsamples_data:
        content = item.pop("content")
        doc = Document(page_content=content, metadata=item)
        documents.append(doc)

    vectorstore = create_or_update_vectorstore(documents, clear_existing=False)
    return vectorstore


def main():
    """Main function with menu options."""

    print("=" * 70)
    print("CDI Document Loader - Vector Database Population Tool")
    print("=" * 70)

    print("\nOptions:")
    print("1. Load example CDI guidelines (predefined list)")
    print("2. Load from directory of text files")
    print("3. Load from JSON file")
    print("4. Load MTSamples-style data")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        print("\n--- Loading Example CDI Guidelines ---")
        example_load_cdi_guidelines()

    elif choice == "2":
        data_dir = input("Enter directory path: ").strip()
        print(f"\n--- Loading from directory: {data_dir} ---")
        example_load_from_files(data_dir)

    elif choice == "3":
        json_path = input("Enter JSON file path: ").strip()
        content_key = input("Enter content key (default: 'content'): ").strip() or "content"
        print(f"\n--- Loading from JSON file: {json_path} ---")

        if os.path.exists(json_path):
            documents = load_from_json_file(json_path, content_key=content_key)
            create_or_update_vectorstore(documents, clear_existing=False)
        else:
            print(f"File not found: {json_path}")

    elif choice == "4":
        print("\n--- Loading MTSamples-style data ---")
        example_load_mtsamples_style()

    else:
        print("Invalid choice")

    print("\n" + "=" * 70)
    print("Done! Your ChromaDB vector store has been updated.")
    print(f"Location: {CHROMA_DB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
