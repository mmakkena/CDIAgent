#!/usr/bin/env python3
"""
Script to preview and explore documents in ChromaDB vector store.
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
CHROMA_DB_PATH = "./chroma_db_cdi"
CHROMA_COLLECTION_NAME = "cdi_guidelines"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_vectorstore():
    """Load the ChromaDB vector store."""
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: ChromaDB not found at {CHROMA_DB_PATH}")
        print("Run 'python download_mtsamples.py' or 'python load_cdi_documents.py' first.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )
    return vectorstore


def show_collection_info(vectorstore):
    """Display basic information about the collection."""
    print("\n" + "=" * 80)
    print("CHROMADB COLLECTION INFORMATION")
    print("=" * 80)

    total_docs = vectorstore._collection.count()
    print(f"\nTotal Documents: {total_docs}")
    print(f"Database Path: {CHROMA_DB_PATH}")
    print(f"Collection Name: {CHROMA_COLLECTION_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")


def list_all_documents(vectorstore, limit=10):
    """List all documents with their metadata."""
    print("\n" + "=" * 80)
    print(f"DOCUMENT LIST (showing first {limit} documents)")
    print("=" * 80)

    # Get all documents (using a generic query to retrieve them)
    # ChromaDB doesn't have a direct "get all" in LangChain, so we use peek
    collection = vectorstore._collection

    # Peek at the collection to see documents
    results = collection.peek(limit=limit)

    if not results or not results.get('ids'):
        print("No documents found.")
        return

    print(f"\nFound {len(results['ids'])} documents:\n")

    for i, (doc_id, metadata, document) in enumerate(zip(
        results['ids'],
        results['metadatas'],
        results['documents']
    ), 1):
        print(f"--- Document {i} ---")
        print(f"ID: {doc_id}")
        print(f"Metadata: {metadata}")
        print(f"Content Preview: {document[:200]}...")
        print()


def search_documents(vectorstore, query, k=3):
    """Search for similar documents."""
    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS FOR: '{query}'")
    print("=" * 80)

    results = vectorstore.similarity_search(query, k=k)

    if not results:
        print("\nNo results found.")
        return

    print(f"\nFound {len(results)} relevant documents:\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}")
        print()


def get_document_by_metadata(vectorstore, metadata_key, metadata_value):
    """Find documents by metadata field."""
    print("\n" + "=" * 80)
    print(f"DOCUMENTS WITH {metadata_key}='{metadata_value}'")
    print("=" * 80)

    collection = vectorstore._collection

    # Query with metadata filter
    try:
        results = collection.get(
            where={metadata_key: metadata_value}
        )

        if not results or not results.get('ids'):
            print(f"\nNo documents found with {metadata_key}='{metadata_value}'")
            return

        print(f"\nFound {len(results['ids'])} documents:\n")

        for i, (doc_id, metadata, document) in enumerate(zip(
            results['ids'],
            results['metadatas'],
            results['documents']
        ), 1):
            print(f"--- Document {i} ---")
            print(f"ID: {doc_id}")
            print(f"Metadata: {metadata}")
            print(f"Content:\n{document}")
            print()

    except Exception as e:
        print(f"Error filtering by metadata: {e}")


def show_unique_metadata_values(vectorstore):
    """Show unique values for common metadata fields."""
    print("\n" + "=" * 80)
    print("UNIQUE METADATA VALUES")
    print("=" * 80)

    collection = vectorstore._collection
    results = collection.peek(limit=collection.count())

    if not results or not results.get('metadatas'):
        print("No metadata found.")
        return

    # Collect all unique metadata keys and values
    metadata_summary = {}

    for metadata in results['metadatas']:
        for key, value in metadata.items():
            if key not in metadata_summary:
                metadata_summary[key] = set()
            metadata_summary[key].add(str(value))

    # Display summary
    for key, values in sorted(metadata_summary.items()):
        print(f"\n{key}:")
        for value in sorted(values):
            print(f"  - {value}")


def interactive_menu(vectorstore):
    """Interactive menu for exploring the database."""
    while True:
        print("\n" + "=" * 80)
        print("CHROMADB EXPLORER - MENU")
        print("=" * 80)
        print("\n1. Show collection information")
        print("2. List all documents")
        print("3. Search documents by content")
        print("4. Filter by metadata field")
        print("5. Show unique metadata values")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            show_collection_info(vectorstore)

        elif choice == "2":
            limit_input = input("How many documents to show? (default: 10): ").strip()
            limit = int(limit_input) if limit_input else 10
            list_all_documents(vectorstore, limit=limit)

        elif choice == "3":
            query = input("Enter search query: ").strip()
            k = input("How many results? (default: 3): ").strip()
            k = int(k) if k else 3
            search_documents(vectorstore, query, k=k)

        elif choice == "4":
            key = input("Enter metadata key (e.g., 'source', 'specialty'): ").strip()
            value = input(f"Enter value for '{key}': ").strip()
            get_document_by_metadata(vectorstore, key, value)

        elif choice == "5":
            show_unique_metadata_values(vectorstore)

        elif choice == "6":
            print("\nExiting...")
            break

        else:
            print("Invalid choice. Please try again.")


def main():
    """Main function."""
    print("\n" + "=" * 80)
    print("CHROMADB DOCUMENT VIEWER")
    print("=" * 80)

    # Load vector store
    print("\nLoading ChromaDB vector store...")
    vectorstore = load_vectorstore()

    if vectorstore is None:
        return

    print("✓ Vector store loaded successfully!")

    # Run interactive menu
    interactive_menu(vectorstore)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
