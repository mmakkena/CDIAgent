# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Clinical Documentation Integrity (CDI) Large Language Model (LLM)** microservice that uses Retrieval-Augmented Generation (RAG) to analyze clinical notes and generate physician queries for documentation gaps. The system grounds responses in CDI guidelines to prevent hallucinations.

**Core Technologies:**
- **LLM**: Mistral-7B-Instruct-v0.2 with QLoRA (4-bit quantization via BitsAndBytes)
- **RAG Framework**: LangChain with ChromaDB vector store
- **Embeddings**: HuggingFace all-MiniLM-L6-v2
- **API**: FastAPI with async support
- **Hardware**: Requires NVIDIA GPU with 16GB+ VRAM

## Architecture

### Core Components

1. **cdi_rag_system.py** - RAG pipeline implementation
   - `get_llm_pipeline()`: Initializes Mistral model with QLoRA config
   - `setup_chroma_db()`: Creates/loads persistent ChromaDB vector store at `./chroma_db_cdi`
   - `create_rag_chain()`: Builds LangChain RetrievalQA chain with k=2 retrieval
   - Uses `DOCUMENTS` list as mock CDI knowledge base (to be replaced with production data)

2. **api_server.py** - FastAPI REST service
   - Global `qa_chain` initialized at startup (expensive operation, ~minutes)
   - POST `/generate_cdi_query`: Main endpoint accepting clinical notes
   - GET `/health`: Health check for RAG system status
   - Uses async `qa_chain.acall()` for non-blocking inference

### Data Flow

```
Clinical Note → Embedding → ChromaDB Retrieval (k=2 guidelines)
→ LLM with Context → Structured Query + Source Citations
```

## Common Commands

### Environment Setup

```bash
# Install dependencies
pip install langchain-community
pip install torch transformers accelerate langchain faiss-cpu pydantic fastapi uvicorn
pip install chromadb langchain-chroma
```

### Running the Application

```bash
# Standalone testing (runs test case in __main__)
python cdi_rag_system.py

# Start API server (model loads during startup)
uvicorn api_server:app --reload

# API will be available at:
# - Interactive docs: http://127.0.0.1:8000/docs
# - Health check: http://127.0.0.1:8000/health
```

### Testing the API

```bash
# Using curl
curl -X POST "http://127.0.0.1:8000/generate_cdi_query" \
  -H "Content-Type: application/json" \
  -d '{"clinical_note": "Patient presents with fever and cough. Chest X-ray positive for pneumonia. No organism specified."}'
```

## Key Configuration Constants

Located in `cdi_rag_system.py`:

- `MODEL_NAME`: LLM identifier (default: mistralai/Mistral-7B-Instruct-v0.2)
- `CHROMA_DB_PATH`: Vector store persistence location (default: ./chroma_db_cdi)
- `CHROMA_COLLECTION_NAME`: Collection identifier (default: cdi_guidelines)
- `CHUNK_SIZE/CHUNK_OVERLAP`: Text splitting params (500/50)
- `SYSTEM_PROMPT`: CDI specialist persona instructions

## Important Implementation Notes

### LangChain Package Structure
The codebase uses the newer modular LangChain architecture:
- `langchain-community`: Document loaders, embeddings, vector stores
- `langchain-text-splitters`: Text chunking utilities
- `langchain-core`: Core abstractions (prompts, etc.)
- `langchain`: Legacy imports like `RetrievalQA`, `HuggingFacePipeline`

### ChromaDB Persistence
- First run creates `./chroma_db_cdi/` directory with indexed DOCUMENTS
- Subsequent runs load existing database (much faster)
- Uses `vectorstore.persist()` for explicit disk writes
- Collection metadata includes source citations for traceability

### Model Loading (QLoRA)
```python
BitsAndBytesConfig(
    load_in_4bit=True,              # 4-bit quantization
    bnb_4bit_use_double_quant=True, # Double quantization
    bnb_4bit_quant_type="nf4",      # NormalFloat4 type
    bnb_4bit_compute_dtype=torch.bfloat16
)
```
This configuration enables 7B parameter model on single GPU with reduced VRAM.

### API Startup Behavior
- `@app.on_event("startup")` initializes global `qa_chain`
- Blocks server startup until model loads (intentional, prevents serving before ready)
- Failures raise `RuntimeError` to prevent silent degradation
- All requests check `qa_chain is None` before processing

## Production Readiness Considerations

### Known Limitations
1. **Mock Knowledge Base**: `DOCUMENTS` list in `cdi_rag_system.py` needs replacement with real CDI guidelines corpus
2. **Synchronous Pipeline**: Line 77 in `cdi_rag_system.py` uses `torch.utils.data.DataLoader` incorrectly as LangChain pipeline wrapper (non-standard pattern)
3. **No Error Recovery**: API startup failure is fatal; no graceful degradation
4. **In-Memory Limits**: No pagination or streaming for large result sets

### Recommended Enhancements
1. Replace mock `DOCUMENTS` with PDF/TXT corpus loading via `DirectoryLoader`
2. Migrate to scalable vector DB (Pinecone, Weaviate) if corpus exceeds 10K+ documents
3. Add fine-tuning with (Clinical Note, Query) pairs using PEFT library
4. Implement request queuing/throttling for production load

## Troubleshooting

### Model Loading Fails
- Verify CUDA/GPU availability: `torch.cuda.is_available()`
- Check VRAM: `nvidia-smi` (needs 16GB+ for 7B model even with 4-bit)
- Ensure transformers/accelerate versions compatible with QLoRA

### ChromaDB Errors
- Delete `./chroma_db_cdi/` to force rebuild
- Check file permissions on persistence directory
- Verify embedding model downloads: `~/.cache/huggingface/`

### API Returns 503
- Check startup logs for initialization errors
- Verify `qa_chain` is not None via `/health` endpoint
- Model download may timeout on first run (several GB)
