#!/usr/bin/env python3
"""Test script to identify import issues."""

print("Starting imports...")

print("1. Importing torch...")
import torch
print(f"   - torch imported. CUDA available: {torch.cuda.is_available()}")

print("2. Importing transformers...")
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
print("   - transformers imported successfully")

print("3. Importing langchain_community...")
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
print("   - langchain_community imported successfully")

print("4. Importing langchain_classic...")
from langchain_classic.chains import RetrievalQA
from langchain_classic.llms import HuggingFacePipeline
print("   - langchain_classic imported successfully")

print("\nAll imports successful!")
