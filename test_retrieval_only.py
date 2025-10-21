#!/usr/bin/env python3
"""
Fast test for metadata filtering - tests retrieval only, no LLM generation.
"""

from cdi_rag import setup_chroma_db, HybridRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

print("="*70)
print("METADATA FILTERING RETRIEVAL TEST")
print("="*70)

# Initialize vectorstore
vectorstore = setup_chroma_db()
collection = vectorstore._collection

# Test 1: No filter
print("\n--- Test 1: No Filter (All Documents) ---")
results = collection.get()
print(f"Total documents: {len(results.get('documents', []))}")

# Test 2: Filter by specialty="Cardiology"
print("\n\n--- Test 2: Filter by Specialty = 'Cardiology' ---")
filtered_results = collection.get(where={"specialty": "Cardiology"})
cardiology_docs = filtered_results.get('documents', [])
print(f"Cardiology documents: {len(cardiology_docs)}")

if cardiology_docs:
    print("\nDocument samples:")
    for i, (doc_text, metadata) in enumerate(zip(
        filtered_results['documents'][:3],
        filtered_results['metadatas'][:3]
    ), 1):
        specialty = metadata.get('specialty', 'N/A')
        sample_name = metadata.get('sample_name', 'Unknown')
        print(f"  {i}. {sample_name} (Specialty: {specialty})")

    # Verify all are Cardiology
    all_cardiology = all(
        meta.get('specialty') == 'Cardiology'
        for meta in filtered_results['metadatas']
    )
    if all_cardiology:
        print("\n✓ PASS: All documents are from Cardiology")
    else:
        print("\n✗ FAIL: Some documents are not from Cardiology")
else:
    print("  No Cardiology documents found")

# Test 3: Filter by specialty="Internal Medicine"
print("\n\n--- Test 3: Filter by Specialty = 'Internal Medicine' ---")
filtered_results2 = collection.get(where={"specialty": "Internal Medicine"})
im_docs = filtered_results2.get('documents', [])
print(f"Internal Medicine documents: {len(im_docs)}")

if im_docs:
    print("\nDocument samples:")
    for i, (doc_text, metadata) in enumerate(zip(
        filtered_results2['documents'][:3],
        filtered_results2['metadatas'][:3]
    ), 1):
        specialty = metadata.get('specialty', 'N/A')
        sample_name = metadata.get('sample_name', 'Unknown')
        print(f"  {i}. {sample_name} (Specialty: {specialty})")

    # Verify all are Internal Medicine
    all_im = all(
        meta.get('specialty') == 'Internal Medicine'
        for meta in filtered_results2['metadatas']
    )
    if all_im:
        print("\n✓ PASS: All documents are from Internal Medicine")
    else:
        print("\n✗ FAIL: Some documents are not from Internal Medicine")
else:
    print("  No Internal Medicine documents found")

# Test 4: Hybrid retriever with metadata filter
print("\n\n--- Test 4: Hybrid Retriever with Cardiology Filter ---")

# Get documents for BM25 indexing
filtered_results3 = collection.get(where={"specialty": "Cardiology"})
documents = []
if filtered_results3 and filtered_results3.get('documents') and filtered_results3.get('metadatas'):
    for doc_text, metadata in zip(filtered_results3['documents'], filtered_results3['metadatas']):
        documents.append(Document(page_content=doc_text, metadata=metadata))

if documents:
    print(f"Creating hybrid retriever with {len(documents)} Cardiology documents")

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    # Create vector retriever with filter
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"specialty": "Cardiology"}}
    )

    # Create hybrid retriever
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        vector_weight=0.6,
        bm25_weight=0.4,
        metadata_filter={"specialty": "Cardiology"}
    )

    # Test retrieval
    query = "Patient with chest pain and elevated cardiac enzymes"
    retrieved_docs = hybrid_retriever.invoke(query)

    print(f"\nRetrieved {len(retrieved_docs)} documents for query: '{query}'")
    for i, doc in enumerate(retrieved_docs, 1):
        specialty = doc.metadata.get('specialty', 'N/A')
        sample_name = doc.metadata.get('sample_name', 'Unknown')
        print(f"  {i}. {sample_name} (Specialty: {specialty})")

    # Verify all are Cardiology
    all_cardiology_retrieved = all(
        doc.metadata.get('specialty') == 'Cardiology'
        for doc in retrieved_docs
    )
    if all_cardiology_retrieved:
        print("\n✓ PASS: Hybrid retriever returned only Cardiology documents")
    else:
        print("\n✗ FAIL: Hybrid retriever returned non-Cardiology documents")
else:
    print("  No Cardiology documents to create hybrid retriever")

print("\n" + "="*70)
print("TESTS COMPLETED")
print("="*70)
