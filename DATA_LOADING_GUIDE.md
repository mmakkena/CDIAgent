# CDI Data Loading Guide

This guide explains how to load clinical documentation and CDI guidelines into your RAG system.

## Overview

Your system uses **Retrieval-Augmented Generation (RAG)**, not traditional model training. You populate a vector database (ChromaDB) with CDI guidelines and clinical examples, which the model retrieves at query time.

## Available Scripts

### 1. `load_cdi_documents.py`
Multi-purpose document loader supporting various formats:
- Text files (.txt)
- PDF files (.pdf)
- CSV files (.csv)
- JSON files (.json)
- Python lists of documents

### 2. `download_mtsamples.py`
Downloads sample clinical notes (free, public domain) and loads them into ChromaDB.

## Quick Start

### Option 1: Load Sample CDI Guidelines (Fastest)

```bash
python load_cdi_documents.py
# Select option 1
```

This loads 5 pre-configured CDI guidelines covering common scenarios.

### Option 2: Load MTSamples Clinical Notes

```bash
python download_mtsamples.py
```

This creates sample clinical notes and loads them into your vector database.

### Option 3: Load Your Own Documents

```bash
# Create a directory with your documents
mkdir -p data/cdi_guidelines
# Add .txt, .pdf, or .csv files to the directory

# Run the loader
python load_cdi_documents.py
# Select option 2 and enter: data/cdi_guidelines
```

## Recommended Public Datasets

### 1. **MIMIC-III/IV** (Most Comprehensive)
- **Best for**: Production CDI systems
- **Content**: 2+ million de-identified clinical notes
- **Access**: Free with credentialing
- **Setup Time**: 1-2 hours for CITI training + DUA

**Steps to Access:**
1. Go to https://physionet.org/
2. Create account
3. Complete CITI "Data or Specimens Only Research" course
4. Sign Data Use Agreement
5. Download NOTEEVENTS.csv or NOTEEVENTS.csv.gz

**Loading MIMIC-III Notes:**
```python
# Example code to load MIMIC-III discharge summaries
import pandas as pd
from load_cdi_documents import load_from_text_list, create_or_update_vectorstore

# Read MIMIC-III notes
notes_df = pd.read_csv('NOTEEVENTS.csv')

# Filter for discharge summaries (most relevant for CDI)
discharge_notes = notes_df[notes_df['CATEGORY'] == 'Discharge summary']

# Convert to format for loading
documents = []
for _, row in discharge_notes.iterrows():
    documents.append((
        row['TEXT'],
        {
            'source': 'MIMIC-III',
            'category': row['CATEGORY'],
            'subject_id': str(row['SUBJECT_ID']),
            'hadm_id': str(row['HADM_ID'])
        }
    ))

# Load into vector store
from langchain_core.documents import Document
docs = [Document(page_content=text, metadata=meta) for text, meta in documents[:1000]]  # Start with 1000

from load_cdi_documents import create_or_update_vectorstore
create_or_update_vectorstore(docs, clear_existing=False)
```

### 2. **MTSamples** (Easiest)
- **Best for**: Quick testing
- **Content**: ~4,500 medical transcription samples
- **Access**: Public domain, no registration
- **Setup Time**: Immediate

**Manual Download:**
Visit https://www.mtsamples.com/ and browse by specialty.

### 3. **i2b2 NLP Datasets**
- **Best for**: Specific NLP tasks
- **Content**: De-identified notes from challenges
- **Access**: Free with DUA
- **Website**: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

### 4. **Synthea (Synthetic Data)**
- **Best for**: Testing without PHI concerns
- **Content**: Synthetic patient records
- **Access**: Open source
- **Website**: https://github.com/synthetichealth/synthea

## Data Format Guidelines

### For CDI Guidelines
Structure your CDI guidelines as text with metadata:

```python
CDI_GUIDELINES = [
    (
        "Guideline text explaining when to query...",
        {
            "source": "Source document name",
            "category": "Clinical category",
            "icd10": "Relevant ICD-10 code",
            "effective_date": "2024-01-01"
        }
    ),
]
```

### For Clinical Notes
Include relevant metadata for better retrieval:

```python
CLINICAL_NOTES = [
    (
        "Full clinical note text...",
        {
            "specialty": "Cardiology",
            "note_type": "Discharge Summary",
            "diagnosis": "Primary diagnosis",
            "source": "Data source name"
        }
    ),
]
```

## Creating Custom JSON Datasets

Create a JSON file with this structure:

```json
[
  {
    "content": "Your CDI guideline or clinical note text here...",
    "source": "CDC Guidelines 2024",
    "category": "Cardiology",
    "icd10": "I21.9",
    "keywords": ["MI", "cardiac", "chest pain"]
  },
  {
    "content": "Another guideline...",
    "source": "AHA Guidelines",
    "category": "Cardiology",
    "icd10": "I50.9",
    "keywords": ["heart failure", "HF", "edema"]
  }
]
```

Then load it:
```bash
python load_cdi_documents.py
# Select option 3
# Enter file path: your_data.json
# Enter content key: content
```

## Verifying Your Data

After loading data, test it:

```bash
python cdi_rag_system.py
```

The script will:
1. Load your vector database
2. Run a test query
3. Show retrieved guidelines

## Data Size Recommendations

- **Development**: 50-100 guidelines sufficient
- **Testing**: 500-1,000 clinical notes
- **Production**: 10,000+ notes + comprehensive guidelines

## Vector Database Management

### View Current Database Size
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db_cdi",
    embedding_function=embeddings,
    collection_name="cdi_guidelines"
)

print(f"Total documents: {vectorstore._collection.count()}")
```

### Clear Database
```bash
rm -rf ./chroma_db_cdi
```

### Backup Database
```bash
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db_cdi
```

## Best Practices

1. **Start Small**: Load 50-100 documents first to test
2. **Chunk Size**: Default 500 characters works well for CDI
3. **Metadata**: Include rich metadata for better filtering
4. **Update Regularly**: Add new guidelines as they're published
5. **Version Control**: Track which guidelines are in your DB

## Troubleshooting

### "No relevant guidelines found"
- Ensure documents are loaded: Check `vectorstore._collection.count()`
- Try different query phrasing
- Add more diverse examples

### Slow Loading
- Reduce batch size
- Use smaller embedding model
- Process in chunks

### Memory Issues
- Load documents in batches
- Use smaller chunk size
- Increase system RAM or use cloud instance

## Next Steps

After loading data:

1. **Test Queries**: Run `python cdi_rag_system.py`
2. **Evaluate Results**: Check if retrieved guidelines are relevant
3. **Fine-tune**: Adjust chunk size, retrieval count (k parameter)
4. **Scale Up**: Add more documents as needed

## Resources

- **MIMIC-III**: https://physionet.org/content/mimiciii/
- **MTSamples**: https://www.mtsamples.com/
- **i2b2**: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Synthea**: https://github.com/synthetichealth/synthea
- **LangChain Docs**: https://python.langchain.com/docs/

## Support

For issues with:
- **Data loading**: Check `load_cdi_documents.py` documentation
- **Vector DB**: See ChromaDB docs at https://docs.trychroma.com/
- **Model issues**: See `cdi_rag_system.py`
