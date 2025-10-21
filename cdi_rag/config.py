"""
Configuration constants for the CDI RAG system.

This module centralizes all configuration parameters for models, databases,
chunking, and retrieval settings.
"""

# --- Model Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_MODEL_NAME = "gpt2-medium"  # 355M params, ~1.5GB RAM

# --- Text Processing Configuration ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Database Configuration ---
CHROMA_DB_PATH = "./chroma_db_cdi"
CHROMA_COLLECTION_NAME = "cdi_guidelines"

# --- Retrieval Configuration ---
# Hybrid search weights
VECTOR_WEIGHT = 0.6  # 60% semantic search
BM25_WEIGHT = 0.4    # 40% keyword search

# Number of documents to retrieve
RETRIEVAL_K_BM25 = 3
RETRIEVAL_K_VECTOR = 3
RETRIEVAL_K_INITIAL = 8  # Documents to retrieve before re-ranking
RETRIEVAL_K_FINAL = 4    # Documents to return after re-ranking

# --- Re-ranking Configuration ---
RERANKING_MAX_FEATURES = 1000
RERANKING_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
RERANKING_STOP_WORDS = 'english'

# --- LLM Generation Configuration ---
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
DO_SAMPLE = True

# --- System Prompts ---
SYSTEM_PROMPT = """
You are an expert Clinical Documentation Integrity (CDI) specialist.
Your task is to analyze a clinical note and, using ONLY the provided CDI GUIDELINES,
generate a concise, professional, non-leading physician query to clarify the documentation gap.
If no relevant guidelines are found, state that no query is needed.

The output must strictly follow this format:
Query: [Your Generated Query]
Source Guidelines Retrieved: [List the 'source' of the guidelines you used]
"""
