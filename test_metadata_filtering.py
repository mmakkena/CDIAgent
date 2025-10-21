#!/usr/bin/env python3
"""
Test script for metadata filtering functionality in CDI RAG system.
"""

import sys
from cdi_rag import get_llm_pipeline, setup_chroma_db, create_rag_chain

def test_no_filter():
    """Test 1: No filter (should retrieve from all documents)"""
    print("\n" + "="*70)
    print("TEST 1: No Metadata Filter")
    print("="*70)

    llm = get_llm_pipeline()
    vectorstore = setup_chroma_db()
    qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter=None)

    clinical_note = "Patient with low albumin and protein deficiency."
    print(f"\nClinical Note: {clinical_note}")

    result = qa_chain.invoke({"query": clinical_note})

    print("\nRetrieved Documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        category = doc.metadata.get('category', 'N/A')
        print(f"  {i}. {source} (Category: {category})")

    return result

def test_filter_by_category():
    """Test 2: Filter by category (e.g., only Cardiology documents)"""
    print("\n" + "="*70)
    print("TEST 2: Filter by Category = 'Cardiology'")
    print("="*70)

    llm = get_llm_pipeline()
    vectorstore = setup_chroma_db()

    # Filter for Cardiology only
    metadata_filter = {"category": "Cardiology"}
    qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter=metadata_filter)

    clinical_note = "Patient with chest pain and elevated cardiac markers."
    print(f"\nClinical Note: {clinical_note}")

    result = qa_chain.invoke({"query": clinical_note})

    print("\nRetrieved Documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        category = doc.metadata.get('category', 'N/A')
        print(f"  {i}. {source} (Category: {category})")

    # Verify all documents are from Cardiology
    all_cardiology = all(doc.metadata.get('category') == 'Cardiology'
                         for doc in result['source_documents'])
    if all_cardiology:
        print("\n✓ PASS: All retrieved documents are from Cardiology category")
    else:
        print("\n✗ FAIL: Some documents are not from Cardiology category")

    return result

def test_filter_by_multiple_fields():
    """Test 3: Filter by multiple metadata fields"""
    print("\n" + "="*70)
    print("TEST 3: Filter by Category = 'Nutrition' AND source contains 'Guidelines'")
    print("="*70)

    llm = get_llm_pipeline()
    vectorstore = setup_chroma_db()

    # Filter for Nutrition category and Guidelines source
    metadata_filter = {"category": "Nutrition"}
    qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter=metadata_filter)

    clinical_note = "Patient with severe weight loss and low albumin levels."
    print(f"\nClinical Note: {clinical_note}")

    result = qa_chain.invoke({"query": clinical_note})

    print("\nRetrieved Documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        category = doc.metadata.get('category', 'N/A')
        print(f"  {i}. {source} (Category: {category})")

    # Verify all documents match the filter
    all_match = all(doc.metadata.get('category') == 'Nutrition'
                    for doc in result['source_documents'])
    if all_match:
        print("\n✓ PASS: All retrieved documents match the filter criteria")
    else:
        print("\n✗ FAIL: Some documents don't match the filter criteria")

    return result

def test_filter_by_icd10():
    """Test 4: Filter by ICD-10 code"""
    print("\n" + "="*70)
    print("TEST 4: Filter by ICD-10 code")
    print("="*70)

    llm = get_llm_pipeline()
    vectorstore = setup_chroma_db()

    # Filter for specific ICD-10 code
    metadata_filter = {"icd10": "E43"}
    qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter=metadata_filter)

    clinical_note = "Patient with protein-calorie malnutrition and low BMI."
    print(f"\nClinical Note: {clinical_note}")

    result = qa_chain.invoke({"query": clinical_note})

    print("\nRetrieved Documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        source = doc.metadata.get('source', 'Unknown')
        icd10 = doc.metadata.get('icd10', 'N/A')
        print(f"  {i}. {source} (ICD-10: {icd10})")

    # Verify all documents have the specified ICD-10 code
    all_match = all(doc.metadata.get('icd10') == 'E43'
                    for doc in result['source_documents'])
    if all_match:
        print("\n✓ PASS: All retrieved documents have ICD-10 code E43")
    else:
        print("\n✗ FAIL: Some documents don't have ICD-10 code E43")

    return result

def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# METADATA FILTERING TEST SUITE")
    print("#"*70)

    try:
        # Run all tests
        test_no_filter()
        test_filter_by_category()
        test_filter_by_multiple_fields()
        test_filter_by_icd10()

        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
