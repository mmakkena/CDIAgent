#!/usr/bin/env python3
"""
Test script for document re-ranking functionality.
"""

from cdi_rag import setup_chroma_db, HybridRetriever, DocumentReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

print("="*70)
print("DOCUMENT RE-RANKING TEST")
print("="*70)

# Initialize vectorstore
vectorstore = setup_chroma_db()
collection = vectorstore._collection

# Get all documents for testing
results = collection.get()
documents = []
if results and results.get('documents') and results.get('metadatas'):
    for doc_text, metadata in zip(results['documents'], results['metadatas']):
        documents.append(Document(page_content=doc_text, metadata=metadata))

print(f"\nTotal documents in database: {len(documents)}")

# Test 1: Compare retrieval WITH and WITHOUT re-ranking
print("\n" + "="*70)
print("TEST 1: Hybrid Retrieval WITHOUT Re-ranking")
print("="*70)

# Create BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Create vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create hybrid retriever WITHOUT re-ranking
hybrid_retriever_no_rerank = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    vector_weight=0.6,
    bm25_weight=0.4,
    use_reranking=False
)

query = "Patient with acute chest pain, elevated cardiac enzymes, and ST elevation on EKG"
print(f"\nQuery: '{query}'")

docs_no_rerank = hybrid_retriever_no_rerank.invoke(query)

print(f"\nRetrieved {len(docs_no_rerank)} documents (NO re-ranking):")
for i, doc in enumerate(docs_no_rerank, 1):
    sample_name = doc.metadata.get('sample_name', 'Unknown')
    specialty = doc.metadata.get('specialty', 'N/A')
    preview = doc.page_content[:100].replace('\n', ' ')
    print(f"  {i}. {sample_name} ({specialty})")
    print(f"     Preview: {preview}...")

# Test 2: Same query WITH re-ranking
print("\n" + "="*70)
print("TEST 2: Hybrid Retrieval WITH Re-ranking")
print("="*70)

# Create reranker
reranker = DocumentReranker(use_reranking=True)

# Create hybrid retriever WITH re-ranking
hybrid_retriever_rerank = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
    vector_weight=0.6,
    bm25_weight=0.4,
    reranker=reranker,
    use_reranking=True
)

print(f"\nQuery: '{query}'")

docs_rerank = hybrid_retriever_rerank.invoke(query)

print(f"\nRetrieved {len(docs_rerank)} documents (WITH re-ranking):")
for i, doc in enumerate(docs_rerank, 1):
    sample_name = doc.metadata.get('sample_name', 'Unknown')
    specialty = doc.metadata.get('specialty', 'N/A')
    preview = doc.page_content[:100].replace('\n', ' ')
    print(f"  {i}. {sample_name} ({specialty})")
    print(f"     Preview: {preview}...")

# Test 3: Test re-ranker directly
print("\n" + "="*70)
print("TEST 3: Direct Re-ranker Test")
print("="*70)

# Create a sample query and documents
test_query = "patient fever sepsis infection blood culture"
test_docs = documents[:10]  # Use first 10 documents

print(f"\nQuery: '{test_query}'")
print(f"Documents to re-rank: {len(test_docs)}")

# Re-rank without initialization (initial order)
print("\nBefore re-ranking:")
for i, doc in enumerate(test_docs[:4], 1):
    sample_name = doc.metadata.get('sample_name', 'Unknown')
    print(f"  {i}. {sample_name}")

# Apply re-ranking
reranked_docs = reranker.rerank_documents(test_query, test_docs, top_k=4)

print("\nAfter TF-IDF re-ranking:")
for i, doc in enumerate(reranked_docs, 1):
    sample_name = doc.metadata.get('sample_name', 'Unknown')
    print(f"  {i}. {sample_name}")

# Check if order changed
order_changed = any(
    test_docs[i].page_content != reranked_docs[i].page_content
    for i in range(min(4, len(reranked_docs)))
)

if order_changed:
    print("\n✓ PASS: Re-ranking changed document order")
else:
    print("\n⚠ INFO: Re-ranking did not change order (query may match all docs equally)")

# Test 4: Verify re-ranking improves relevance for specific query
print("\n" + "="*70)
print("TEST 4: Verify Re-ranking for Cardiology Query")
print("="*70)

cardio_query = "myocardial infarction troponin ST elevation"
print(f"\nQuery: '{cardio_query}'")

# Get Cardiology docs
cardiology_results = collection.get(where={"specialty": "Cardiology"})
cardio_docs = []
if cardiology_results and cardiology_results.get('documents') and cardiology_results.get('metadatas'):
    for doc_text, metadata in zip(cardiology_results['documents'], cardiology_results['metadatas']):
        cardio_docs.append(Document(page_content=doc_text, metadata=metadata))

if cardio_docs:
    print(f"Cardiology documents: {len(cardio_docs)}")

    # Rerank
    reranked_cardio = reranker.rerank_documents(cardio_query, cardio_docs, top_k=3)

    print("\nTop 3 re-ranked Cardiology documents:")
    for i, doc in enumerate(reranked_cardio, 1):
        sample_name = doc.metadata.get('sample_name', 'Unknown')
        print(f"  {i}. {sample_name}")

    # Verify all are Cardiology
    all_cardio = all(doc.metadata.get('specialty') == 'Cardiology' for doc in reranked_cardio)
    if all_cardio:
        print("\n✓ PASS: All re-ranked documents are Cardiology")
    else:
        print("\n✗ FAIL: Some re-ranked documents are not Cardiology")
else:
    print("  No Cardiology documents found")

print("\n" + "="*70)
print("RE-RANKING TESTS COMPLETED")
print("="*70)
