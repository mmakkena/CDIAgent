#!/usr/bin/env python3
"""Reset ChromaDB and reload with clean data."""

import os
import shutil

CHROMA_DB_PATH = "./chroma_db_cdi"

print("=" * 70)
print("CHROMADB RESET TOOL")
print("=" * 70)

if os.path.exists(CHROMA_DB_PATH):
    response = input(f"\nDelete existing database at {CHROMA_DB_PATH}? (yes/no): ").strip().lower()

    if response == 'yes':
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"✓ Deleted {CHROMA_DB_PATH}")
        print("\nNow run one of:")
        print("  1. python download_mtsamples.py  (for sample data)")
        print("  2. python load_cdi_documents.py  (for your own data)")
        print("\nIMPORTANT: Create a dedicated data directory first!")
        print("  mkdir -p data/cdi_guidelines")
        print("  # Add your .txt or .pdf files there")
        print("  # Then: python load_cdi_documents.py")
    else:
        print("Cancelled. Database not modified.")
else:
    print(f"\nNo database found at {CHROMA_DB_PATH}")
    print("Run 'python download_mtsamples.py' to create one.")

print("=" * 70)
