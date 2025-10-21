import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the necessary functions from the refactored CDI RAG package
from cdi_rag import get_llm_pipeline, setup_chroma_db, create_rag_chain, SYSTEM_PROMPT
from query_validator import CDIQueryValidator

# --- 1. FastAPI and State Initialization ---
app = FastAPI(
    title="CDI LLM Query Generator API",
    description="Microservice for Retrieval-Augmented Generation (RAG) to create CDI physician queries.",
    version="1.0.0"
)

# Global variables to store the initialized components
qa_chain = None
llm = None
vectorstore = None
validator = CDIQueryValidator(strict_mode=False)

# --- 2. Pydantic Models for API ---

class QueryRequest(BaseModel):
    """Defines the structure for the incoming request body."""
    clinical_note: str
    filters: dict = None  # Optional metadata filters (e.g., {"category": "Cardiology"})

class QueryResponse(BaseModel):
    """Defines the structure for the outgoing response body."""
    query: str
    source_documents: list[str]
    validation: dict = None  # Optional validation results

# --- 3. Startup Event: Initialize RAG Components ---

@app.on_event("startup")
async def startup_event():
    """Initializes the LLM and the persistent ChromaDB vector store on application startup."""
    global qa_chain, llm, vectorstore
    start_time = time.time()
    print("--- Starting CDI RAG System Initialization ---")

    try:
        # Initialize the persistent ChromaDB (loads or creates the index)
        vectorstore = setup_chroma_db()

        # Initialize the LLM (This is the most time-consuming step)
        llm = get_llm_pipeline()

        # Create the default RAG Chain with hybrid search (no filters)
        qa_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True)

        end_time = time.time()
        print(f"--- Initialization Complete in {end_time - start_time:.2f} seconds ---")

    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        qa_chain = None
        llm = None
        vectorstore = None
        raise RuntimeError("RAG components failed to initialize. Check model and database setup.") from e


# --- 4. API Endpoint ---

@app.post("/generate_cdi_query", response_model=QueryResponse)
async def generate_cdi_query(request: QueryRequest):
    """
    Accepts a clinical note and optional metadata filters, returns a CDI physician query.
    """
    if qa_chain is None or llm is None or vectorstore is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized or failed to load.")

    try:
        # Run the RAG Chain
        if request.filters:
            print(f"Processing note with filters {request.filters}: '{request.clinical_note[:50]}...'")
            # Create temporary chain with filters
            filtered_chain = create_rag_chain(llm, vectorstore, use_hybrid_search=True, metadata_filter=request.filters)
            result = await filtered_chain.ainvoke({"query": request.clinical_note})
        else:
            print(f"Processing note: '{request.clinical_note[:50]}...'")
            # Use cached default chain (no filters)
            result = await qa_chain.ainvoke({"query": request.clinical_note})

        # Extract and format the results
        generated_query = result['result'].strip()

        # Extract source information from metadata (handle different metadata formats)
        source_docs = []
        for doc in result['source_documents']:
            source = (
                doc.metadata.get('source') or
                doc.metadata.get('sample_name') or
                doc.metadata.get('specialty') or
                'Unknown Source'
            )
            source_docs.append(source)

        # Validate the generated query
        validation_result = validator.validate(generated_query)

        # Create validation summary for response
        validation_summary = {
            "is_valid": validation_result.is_valid,
            "score": validation_result.score,
            "checks": validation_result.checks,
            "warnings": validation_result.warnings,
            "errors": validation_result.errors
        }

        # Log validation results
        if not validation_result.is_valid:
            print(f"⚠ Warning: Generated query failed validation (score: {validation_result.score:.0%})")
            for error in validation_result.errors:
                print(f"  ✗ {error}")

        return QueryResponse(
            query=generated_query,
            source_documents=source_docs,
            validation=validation_summary
        )

    except Exception as e:
        print(f"Error during query generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during LLM generation.")


# --- 5. Health Check Endpoint ---

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    status = "healthy" if (qa_chain is not None and llm is not None and vectorstore is not None) else "unhealthy"
    return {"status": status}
