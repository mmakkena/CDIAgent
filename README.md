That's a great next step\! A well-structured `README.md` is essential for documenting, sharing, and deploying the CDI LLM.

Here is the `README.md` file, tailored to the two Python files (`cdi_rag_system.py` and `api_server.py`) and the technologies you used (RAG, Mistral, QLoRA, FastAPI).

-----

# 🏥 CDI LLM Query Generator

## 📝 Overview

This repository contains the code for a **Clinical Documentation Integrity (CDI) Large Language Model (LLM)** microservice. The system is designed to automatically analyze clinical notes, identify potential documentation gaps based on CDI best practices, and generate concise, professional **physician queries** for clarification.

It utilizes a **Retrieval-Augmented Generation (RAG)** framework, grounding the LLM's response in a curated knowledge base of CDI guidelines, which significantly enhances the accuracy and clinical relevance of the generated queries.

### Key Technologies

| Feature | Technology | Purpose |
| :--- | :--- | :--- |
| **LLM Core** | `Mistral-7B-Instruct-v0.2` | High-performance base model for generation. |
| **Efficient Loading** | **QLoRA** (via `BitsAndBytes`) | Loads the 7B parameter model in 4-bit precision to run on a single GPU. |
| **Knowledge Grounding** | **RAG** (via `LangChain`) | Uses external CDI guidelines (context) to prevent LLM hallucinations. |
| **Vector Store** | `FAISS` | Efficient in-memory indexing and retrieval of relevant CDI guidelines. |
| **Deployment** | `FastAPI` | Provides a high-performance, asynchronous REST API for serving the model. |

-----

## 💻 Setup and Installation

### 1\. Prerequisites

  * **Python 3.10+**
  * A **NVIDIA GPU** with at least 16GB of VRAM is highly recommended to run the 7B parameter model efficiently, even with 4-bit quantization.
  * The project consists of two core files: `cdi_rag_system.py` and `api_server.py`.

### 2\. Install Dependencies

Install the required packages using `pip`. Note that `torch` and `accelerate` are crucial for the model loading.

```bash
pip install torch transformers accelerate langchain faiss-cpu pydantic fastapi uvicorn
```

-----

## 🚀 Usage

There are two ways to use the CDI LLM: as a standalone Python script for testing, or as a full API service for integration.

### A. Standalone Testing (Direct Python Script)

The `cdi_rag_system.py` file contains the logic for the RAG pipeline and includes a basic test case.

1.  **Run the script:**

    ```bash
    python cdi_rag_system.py
    ```

2.  **Expected Output:** The script will load the model, set up the RAG chain with the mock CDI documents, process the sample clinical note, and print the resulting physician query along with the guidelines it used.

    ```
    Loading tokenizer and model: mistralai/Mistral-7B-Instruct-v0.2...
    ... (Model loading messages) ...
    CDI RAG setup complete.

    --- Generating CDI Query ---
    Generated Query:
    Query: The clinical documentation identifies a BMI of 15, which meets the clinical criteria for severe malnutrition. Please clarify the degree of protein-calorie malnutrition: Severe, Moderate, Mild, or unable to determine?

    Source CDI Guidelines Retrieved:
    - A documentation gap exists when a patient has a clinical indicator of severe malnutrition (BMI < 16, protein-calorie intake deficiency) but the final diagnosis is only 'malnutrition' without severity specified. A query is required to clarify the degree (severe, moderate, mild) of malnutrition.
    - The clinical documentation must clearly link all secondary diagnoses to the treatment provided or impact on the length of stay for proper medical necessity review.
    ...
    ```

### B. Deploying as a REST API (FastAPI)

For production use and integration with an Electronic Health Record (EHR) system or CDI workflow, deploy the model as an API.

1.  **Start the API Server:**

    ```bash
    uvicorn api_server:app --reload
    ```

    *The model loading will occur during the startup phase, which may take a few minutes.*

2.  **Access the API:**
    The service will be available at: `http://127.0.0.1:8000/`

3.  **Test the Endpoint:**
    Open the interactive API documentation (Swagger UI) at: `http://127.0.0.1:8000/docs`

4.  **Endpoint Details:**

    | Method | Path | Description |
    | :--- | :--- | :--- |
    | `POST` | `/generate_cdi_query` | Submits a clinical note and receives a generated CDI query. |

    **Request Body (JSON):**

    ```json
    {
      "clinical_note": "Patient presents with fever and cough. Chest X-ray positive for pneumonia. No organism specified."
    }
    ```

    **Response Body (JSON):**

    ```json
    {
      "query": "Query: The chest X-ray confirms pneumonia. Please specify the suspected or confirmed causative organism (e.g., Aspiration, Bacterial, Viral, or Unspecified Organism) to ensure accurate coding.",
      "source_documents": [
        "If a patient's chest X-ray shows 'pneumonia' and the physician only documents 'pneumonia', a query must be issued to specify the organism...",
        "..."
      ]
    }
    ```

-----

## 🛠️ Customization and Development

### 1\. Expanding the Knowledge Base

The most critical step for production-readiness is replacing the mock `DOCUMENTS` list in `cdi_rag_system.py` with a robust, indexed knowledge base.

  * **Process:** Load hundreds or thousands of **clinical practice guidelines**, **ICD/CPT coding manuals**, and **hospital-specific policies** into the RAG system.
  * **Recommendation:** Migrate from the in-memory `FAISS` to a persistent vector database like **Chroma**, **Weaviate**, or **Pinecone** for scalability.

### 2\. Fine-Tuning the LLM

While the RAG context is powerful, **fine-tuning** the base LLM on a dataset of **(Clinical Note, CDI Query) pairs** will dramatically improve the tone, format, and clinical reasoning of the output.

  * **Technique:** Use the **QLoRA** setup as a starting point, but integrate the `PEFT` library for low-rank adapter training.
  * **Data:** The training data must be professionally curated by certified CDI specialists.

### 3\. Model Selection

The current model is `Mistral-7B-Instruct-v0.2`. You may substitute this with other instruction-tuned models, especially **medical domain-specific LLMs** (e.g., specialized Llama or Mistral variants) for better performance.

```python
MODEL_NAME = "path/to/your/finetuned/model"
```
