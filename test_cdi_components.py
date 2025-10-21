#!/usr/bin/env python3
"""Test CDI RAG components individually."""

import os
import sys

print("=" * 60)
print("Testing CDI RAG Components")
print("=" * 60)

# Test 1: Imports
print("\n1. Testing imports...")
try:
    from cdi_rag import get_llm_pipeline, setup_chroma_db, create_rag_chain
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: ChromaDB Setup
print("\n2. Setting up ChromaDB vector store...")
try:
    vectorstore = setup_chroma_db()
    print(f"   ✓ ChromaDB initialized successfully")
    print(f"   - Collection contains {vectorstore._collection.count()} documents")
except Exception as e:
    print(f"   ✗ ChromaDB setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: LLM Pipeline (most time-consuming)
print("\n3. Loading LLM pipeline...")
print("   (This may take a few minutes for model download...)")
try:
    llm = get_llm_pipeline()
    print("   ✓ LLM pipeline loaded successfully")
except Exception as e:
    print(f"   ✗ LLM pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: RAG Chain Creation
print("\n4. Creating RAG chain...")
try:
    qa_chain = create_rag_chain(llm, vectorstore)
    print("   ✓ RAG chain created successfully")
except Exception as e:
    print(f"   ✗ RAG chain creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All components initialized successfully!")
print("=" * 60)
print("\nNote: The script would now be ready to process queries.")
print("Skipping actual inference to save time.")
