import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# --- 1. CDI Knowledge Base (Mock Documents) ---
# In a real-world system, this would be thousands of clinical guidelines,
# official coding books (ICD, CPT), and hospital-specific policies.
DOCUMENTS = [
    "A documentation gap exists when a patient has a clinical indicator of severe malnutrition (BMI < 16, protein-calorie intake deficiency) but the final diagnosis is only 'malnutrition' without severity specified. A query is required to clarify the degree (severe, moderate, mild) of malnutrition.",
    "For chronic obstructive pulmonary disease (COPD) exacerbation, the clinical note must specify if the patient has 'Acute exacerbation of chronic obstructive pulmonary disease' to support the highest level of coding specificity.",
    "Sepsis is documented when there is evidence of infection (e.g., positive blood culture) plus organ dysfunction (e.g., AKI, hypotension). If a physician documents 'sepsis' but the organ dysfunction is missing, a query should be generated to clarify organ involvement or an alternative diagnosis.",
    "If a patient's chest X-ray shows 'pneumonia' and the physician only documents 'pneumonia', a query must be issued to specify the organism (e.g., 'Aspiration pneumonia', 'Pneumococcal pneumonia') or state 'unspecified organism' for maximum specificity and to avoid an audit risk.",
    "The clinical documentation must clearly link all secondary diagnoses to the treatment provided or impact on the length of stay for proper medical necessity review."
]

# --- 2. Model & Quantization Setup (QLoRA style for efficiency) ---
# This is a good way to run a smaller LLM on consumer-grade hardware.
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" # A high-performing, medium-sized model

def initialize_llm_for_rag():
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    
    # 4-bit Quantization (QLoRA style setup for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Configure HuggingFace pipeline for text generation
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        return_full_text=False, # Only return the generated response
    )
    
    return pipeline

# --- 3. RAG Pipeline Setup (Vector Store) ---

def setup_rag_pipeline(llm_pipeline):
    # a. Split the CDI documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    texts = text_splitter.create_documents(DOCUMENTS)
    
    # b. Create Embeddings for the CDI documents
    # A robust embedding model is crucial for good retrieval
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} # Use 'cuda' if available
    )
    
    # c. Create a FAISS Vector Store (In-Memory for this example)
    vectorstore = FAISS.from_documents(texts, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant docs

    # d. Define the CDI Prompt Template
    CDI_PROMPT_TEMPLATE = """
    You are an expert Clinical Documentation Integrity (CDI) specialist.
    Your task is to analyze the provided clinical note and the CDI guidelines.
    Your final output must be a concise, professionally phrased CDI query for the physician.
    
    CDI Guidelines (Context):
    {context}
    
    Clinical Note:
    {question}
    
    CDI Query (must ask for clarification or missing specificity, do not answer for the doctor):
    """
    
    CDI_PROMPT = PromptTemplate(
        template=CDI_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )

    # e. Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(pipeline=llm_pipeline),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": CDI_PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# --- 4. Main Function to Run the CDI Query Generation ---

def generate_cdi_query(clinical_note: str, qa_chain):
    """Generates a CDI query from a clinical note using the RAG chain."""
    
    print("\n--- Generating CDI Query ---")
    
    # Pass the clinical note as the "question" to the RAG chain
    result = qa_chain.invoke({"query": clinical_note})
    
    cdi_query = result['result'].strip()
    source_documents = [doc.page_content for doc in result['source_documents']]
    
    print(f"Generated Query:\n{cdi_query}")
    print("\nSource CDI Guidelines Retrieved:")
    for doc in source_documents:
        print(f"- {doc}")
        
    return cdi_query, source_documents

# --- Example Usage ---
# You would run this part to test the system before the API.

if __name__ == "__main__":
    # Example Clinical Note with a CDI gap
    sample_clinical_note = """
    Patient is a 68-year-old male admitted with a severe exacerbation of his
    longstanding COPD. He is requiring BiPAP and his O2 saturation remains 
    low. His primary physician notes on the chart that he also has severe 
    protein-calorie malnutrition due to poor intake, with a BMI of 15.
    """
    
    llm_pipeline = initialize_llm_for_rag()
    cdi_qa_chain = setup_rag_pipeline(llm_pipeline)

    generate_cdi_query(sample_clinical_note, cdi_qa_chain)
