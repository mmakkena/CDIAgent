#!/usr/bin/env python3
"""
Simple test for metadata filtering using actual database metadata.
"""

from cdi_rag import get_llm_pipeline, setup_chroma_db, create_rag_chain

print("="*70)
print("METADATA FILTERING TEST")
print("="*70)

# Initialize components
llm = get_llm_pipeline()
vectorstore = setup_chroma_db()

# Test 1: No filter
print("\n--- Test 1: No Filter (All Documents) ---")
qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter=None)
result = qa_chain.invoke({"query": "Patient with chest pain and elevated troponin"})
print("\nRetrieved sources:")
for doc in result['source_documents']:
    print(f"  - {doc.metadata.get('sample_name', 'Unknown')} ({doc.metadata.get('specialty', 'N/A')})")

# Test 2: Filter by specialty="Cardiology"
print("\n\n--- Test 2: Filter by Specialty = 'Cardiology' ---")
qa_chain2 = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter={"specialty": "Cardiology"})
result2 = qa_chain2.invoke({"query": "Patient with chest pain and elevated troponin"})
print("\nRetrieved sources:")
for doc in result2['source_documents']:
    specialty = doc.metadata.get('specialty', 'N/A')
    print(f"  - {doc.metadata.get('sample_name', 'Unknown')} (Specialty: {specialty})")

# Verify all are Cardiology
all_cardiology = all(doc.metadata.get('specialty') == 'Cardiology' for doc in result2['source_documents'])
if all_cardiology:
    print("\n✓ PASS: All documents are from Cardiology specialty")
else:
    print("\n✗ FAIL: Some documents are not from Cardiology")

# Test 3: Filter by specialty="Internal Medicine"
print("\n\n--- Test 3: Filter by Specialty = 'Internal Medicine' ---")
qa_chain3 = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter={"specialty": "Internal Medicine"})
result3 = qa_chain3.invoke({"query": "Patient with fever and infection"})
print("\nRetrieved sources:")
for doc in result3['source_documents']:
    specialty = doc.metadata.get('specialty', 'N/A')
    print(f"  - {doc.metadata.get('sample_name', 'Unknown')} (Specialty: {specialty})")

# Verify all are Internal Medicine
all_im = all(doc.metadata.get('specialty') == 'Internal Medicine' for doc in result3['source_documents'])
if all_im:
    print("\n✓ PASS: All documents are from Internal Medicine specialty")
else:
    print("\n✗ FAIL: Some documents are not from Internal Medicine")

print("\n" + "="*70)
print("TESTS COMPLETED")
print("="*70)
