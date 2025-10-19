from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cdi_rag_system import initialize_llm_for_rag, setup_rag_pipeline, generate_cdi_query
import asyncio

# --- 1. FastAPI App Initialization & Pydantic Schemas ---
app = FastAPI(
    title="CDI LLM Query Generator API",
    description="Microservice for generating Clinical Documentation Integrity (CDI) queries using a RAG-based LLM."
)

class ClinicalNoteInput(BaseModel):
    """Input model for the clinical note."""
    clinical_note: str

class CDIQueryOutput(BaseModel):
    """Output model for the generated CDI query and sources."""
    query: str
    source_documents: list[str]

# --- 2. Load Model Once on Startup ---
# Global variables to hold the loaded LLM and RAG chain
llm_pipeline = None
cdi_qa_chain = None

@app.on_event("startup")
async def startup_event():
    """Load the LLM and RAG components when the application starts."""
    global llm_pipeline, cdi_qa_chain
    print("Starting up CDI LLM... This may take a few minutes for model loading.")
    try:
        # Since model loading is heavy, we'll run it synchronously
        llm_pipeline = await asyncio.to_thread(initialize_llm_for_rag)
        cdi_qa_chain = await asyncio.to_thread(setup_rag_pipeline, llm_pipeline)
        print("CDI LLM is ready!")
    except Exception as e:
        print(f"Failed to load LLM or RAG components: {e}")
        # In a real app, you might want a more graceful failure

# --- 3. API Endpoint ---

@app.post(
    "/generate_cdi_query", 
    response_model=CDIQueryOutput,
    summary="Generate a CDI query from a clinical note"
)
async def generate_query_endpoint(note_input: ClinicalNoteInput):
    """
    Analyzes a clinical note against CDI guidelines and generates a physician query 
    to improve documentation specificity.
    """
    if cdi_qa_chain is None:
        raise HTTPException(status_code=503, detail="CDI LLM is still loading or failed to initialize.")
    
    clinical_note = note_input.clinical_note
    
    try:
        # Run the synchronous RAG chain call in an executor thread
        query, sources = await asyncio.to_thread(
            generate_cdi_query, 
            clinical_note, 
            cdi_qa_chain
        )
        
        return CDIQueryOutput(query=query, source_documents=sources)
        
    except Exception as e:
        print(f"Error during query generation: {e}")
        raise HTTPException(status_code=500, detail="Internal LLM processing error.")

# --- 4. How to Run the API ---

# To run this API, save the code above as `api_server.py` and run:
# uvicorn api_server:app --reload

# The API will be available at http://127.0.0.1:8000/
# You can test it via the interactive documentation at http://127.0.0.1:8000/docs
