import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration Constants ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHROMA_DB_PATH = "./chroma_db_cdi"
CHROMA_COLLECTION_NAME = "cdi_guidelines"

# --- 1. Mock CDI Knowledge Base ---
# In a real-world scenario, this list would be replaced by loaded documents (e.g., PDFs, TXT files)
DOCUMENTS = [
    ("A documentation gap exists when a patient has a clinical indicator of severe malnutrition (BMI < 16, protein-calorie intake deficiency) but the final diagnosis is only 'malnutrition' without severity specified. A query is required to clarify the degree (severe, moderate, mild) of malnutrition.", {"source": "Malnutrition Guideline 2023"}),
    ("For pneumonia cases, the physician must specify the suspected or confirmed causative organism (e.g., Bacterial, Viral, Aspiration) to ensure accurate ICD-10 coding. If unspecified, a query is mandatory.", {"source": "Infectious Disease Coding 2024"}),
    ("When a patient presents with respiratory distress requiring mechanical ventilation for over 96 hours, the physician must link the underlying cause (e.g., Acute Respiratory Failure) to the intervention to support medical necessity.", {"source": "Respiratory Failure Policy 2022"}),
    ("Clinical indicators of Acute Kidney Injury (AKI) stage 2 or 3 (e.g., specific creatinine and output values) must be documented as 'Acute Kidney Failure' for higher specificity, or queried for clarification.", {"source": "Nephrology Documentation Rules"}),
    ("The clinical documentation must clearly link all secondary diagnoses to the treatment provided or impact on the length of stay for proper medical necessity review.", {"source": "General Coding Rule 1.4"}),
]

# --- 2. System and Prompt Setup ---
SYSTEM_PROMPT = """
You are an expert Clinical Documentation Integrity (CDI) specialist.
Your task is to analyze a clinical note and, using ONLY the provided CDI GUIDELINES,
generate a concise, professional, non-leading physician query to clarify the documentation gap.
If no relevant guidelines are found, state that no query is needed.

The output must strictly follow this format:
Query: [Your Generated Query]
Source Guidelines Retrieved: [List the 'source' of the guidelines you used]
"""
PROMPT_TEMPLATE = PromptTemplate(
    template="Context: {context}\n\nClinical Note: {question}\n\nCDI Specialist: {system_prompt}",
    input_variables=["context", "question", "system_prompt"]
)

# --- 3. Model and Pipeline Initialization ---

def get_llm_pipeline():
    """Initializes and returns the HuggingFace LLM pipeline using QLoRA."""
    print(f"Loading tokenizer and model: {MODEL_NAME}...")

    # QLoRA configuration for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Wrap the model in a LangChain LLM object
    llm = HuggingFacePipeline(
        pipeline=torch.utils.data.DataLoader(  # Using DataLoader to satisfy the LangChain constructor
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    )
    return llm

def setup_chroma_db():
    """Initializes or loads the persistent ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if the database exists
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Loading existing ChromaDB from {CHROMA_DB_PATH}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
    else:
        print(f"ChromaDB not found. Creating and persisting database at {CHROMA_DB_PATH}...")
        
        # Convert documents into LangChain Documents
        documents = []
        for content, metadata in DOCUMENTS:
            documents.append(TextLoader(content, metadata=metadata))
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        # Note: We manually create Documents here for simplicity with the list of tuples.
        # In a real application, you'd load files with DirectoryLoader and then split.
        texts = [doc[0] for doc in DOCUMENTS]
        metadatas = [doc[1] for doc in DOCUMENTS]

        # Use the LangChain utility to create and persist the DB from texts
        vectorstore = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=CHROMA_COLLECTION_NAME
        )
        vectorstore.persist() # Explicitly persist the data to disk
        print(f"ChromaDB created and persisted with {vectorstore._collection.count()} documents.")

    return vectorstore

def create_rag_chain(llm, vectorstore):
    """Creates the LangChain RetrievalQA chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
    )
    return qa_chain

# --- 4. Main Execution ---

if __name__ == "__main__":
    
    # Initialize all components
    llm = get_llm_pipeline()
    vectorstore = setup_chroma_db()
    qa_chain = create_rag_chain(llm, vectorstore)

    print("\nCDI RAG setup complete.")
    print("-" * 30)

    # Test Case: Documentation Gap in Malnutrition
    clinical_note_1 = "Patient admitted with severe protein-calorie malnutrition, BMI 15, requiring supplemental feeding. Physician note documents 'Malnutrition'."
    
    print("\n--- Generating CDI Query ---")
    
    # Run the RAG chain
    result = qa_chain({"query": clinical_note_1, "system_prompt": SYSTEM_PROMPT})
    
    # Process and display the output
    generated_query = result['result'].strip()
    source_docs = [doc.metadata['source'] for doc in result['source_documents']]
    
    print(f"Generated Query:\n{generated_query}")
    print("-" * 30)
    print(f"Source Guidelines Retrieved:\n* " + "\n* ".join(source_docs))

    # Clean up the tokenizer to free resources (optional, but good practice)
    del llm
