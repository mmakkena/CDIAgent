#!/usr/bin/env python3
"""Quick preview of ChromaDB contents - non-interactive."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db_cdi",
    embedding_function=embeddings,
    collection_name="cdi_guidelines"
)

collection = vectorstore._collection
total = collection.count()

print(f"\n{'='*80}")
print(f"CHROMADB QUICK PREVIEW")
print(f"{'='*80}")
print(f"\nTotal Documents: {total}")
print(f"Location: ./chroma_db_cdi")

# Peek at first 5 documents
print(f"\n{'='*80}")
print("FIRST 5 DOCUMENTS:")
print(f"{'='*80}\n")

results = collection.peek(limit=5)

if results and results.get('ids'):
    for i, (doc_id, metadata, document) in enumerate(zip(
        results['ids'],
        results['metadatas'],
        results['documents']
    ), 1):
        print(f"--- Document {i} ---")
        print(f"Metadata: {metadata}")
        print(f"Content: {document[:300]}...")
        print()

# Show unique metadata fields
print(f"{'='*80}")
print("METADATA SUMMARY:")
print(f"{'='*80}\n")

all_results = collection.peek(limit=total)
if all_results and all_results.get('metadatas'):
    metadata_summary = {}
    for metadata in all_results['metadatas']:
        for key, value in metadata.items():
            if key not in metadata_summary:
                metadata_summary[key] = set()
            metadata_summary[key].add(str(value))

    for key, values in sorted(metadata_summary.items()):
        print(f"{key}: {len(values)} unique values")
        # Show first few values
        for value in sorted(list(values))[:3]:
            print(f"  - {value}")
        if len(values) > 3:
            print(f"  ... and {len(values) - 3} more")
        print()

print(f"{'='*80}")
print("For full exploration, run: python view_chroma_db.py")
print(f"{'='*80}")
