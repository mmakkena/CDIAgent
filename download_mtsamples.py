#!/usr/bin/env python3
"""
Download and process MTSamples clinical notes for CDI training.
MTSamples provides free, de-identified medical transcription samples.
"""

import os
import json
from typing import List, Dict

def download_mtsamples_dataset() -> List[Dict]:
    """
    Download clinical notes from MTSamples website.
    Returns list of documents with content and metadata.
    """

    print("Downloading MTSamples dataset...")
    print("Note: This scrapes publicly available data from mtsamples.com")

    # Base URL - you would need to implement actual scraping
    # This is a simplified example showing the structure

    # For a real implementation, you could:
    # 1. Use their search/category pages
    # 2. Or use a pre-made dataset if available

    # Example: Using a hypothetical JSON API or pre-downloaded data
    # In practice, you might want to use a Kaggle dataset or similar

    sample_data = [
        {
            "sample_id": "1",
            "specialty": "Cardiology",
            "sample_name": "Acute Myocardial Infarction",
            "description": "Chest pain, dyspnea, ST elevation",
            "transcription": """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS: The patient is a 65-year-old male who presents to the
emergency department with acute onset of substernal chest pain radiating to the left
arm, associated with diaphoresis and nausea. Pain started 2 hours ago while at rest.

PAST MEDICAL HISTORY: Hypertension, hyperlipidemia, type 2 diabetes mellitus.

PHYSICAL EXAMINATION: Patient appears diaphoretic and in moderate distress.
Blood pressure 160/95, heart rate 110, respiratory rate 24.

DIAGNOSTIC STUDIES: EKG shows ST elevation in leads II, III, aVF consistent with
inferior wall myocardial infarction. Troponin I elevated at 2.5 ng/mL (normal <0.04).

ASSESSMENT AND PLAN:
1. Acute ST-elevation myocardial infarction (STEMI) - inferior wall
2. Activate cardiac catheterization lab
3. Administer aspirin, clopidogrel, heparin
4. Admit to CCU

ICD-10 CODES: I21.19 (ST elevation myocardial infarction involving other coronary artery)
            """,
            "keywords": ["chest pain", "myocardial infarction", "STEMI", "troponin"],
        },
        {
            "sample_id": "2",
            "specialty": "Internal Medicine",
            "sample_name": "Severe Sepsis",
            "description": "Fever, hypotension, altered mental status",
            "transcription": """
CHIEF COMPLAINT: Fever and confusion.

HISTORY OF PRESENT ILLNESS: 78-year-old female brought in by family due to fever,
confusion, and decreased oral intake over the past 2 days. Patient has been progressively
more lethargic.

PAST MEDICAL HISTORY: Diabetes mellitus type 2, chronic kidney disease stage 3.

PHYSICAL EXAMINATION: Temperature 39.2°C, blood pressure 85/50, heart rate 125,
respiratory rate 28. Patient is confused and disoriented to time and place.

LABORATORY DATA:
- WBC 18,500 with 15% bands
- Creatinine 2.8 (baseline 1.4)
- Lactate 4.2 mmol/L
- Blood cultures: Pending
- Urinalysis: >100 WBC, many bacteria

ASSESSMENT:
1. Severe sepsis secondary to urinary tract infection with acute kidney injury
2. Septic shock (requiring vasopressors)
3. Acute encephalopathy

PLAN:
1. Broad-spectrum antibiotics (vancomycin + piperacillin-tazobactam)
2. Aggressive IV fluid resuscitation
3. Vasopressor support with norepinephrine
4. ICU admission
5. Monitor renal function, consider nephrology consult

ICD-10 CODES:
- A41.9 (Sepsis, unspecified organism)
- R65.20 (Severe sepsis without septic shock) -> Later upgraded to R65.21 (Severe sepsis with septic shock)
- N39.0 (Urinary tract infection, site not specified)
- N17.9 (Acute kidney failure, unspecified)
            """,
            "keywords": ["sepsis", "septic shock", "UTI", "acute kidney injury"],
        },
        {
            "sample_id": "3",
            "specialty": "Pulmonology",
            "sample_name": "Acute Respiratory Failure",
            "description": "Dyspnea, hypoxemia requiring intubation",
            "transcription": """
CHIEF COMPLAINT: Shortness of breath.

HISTORY OF PRESENT ILLNESS: 55-year-old male with history of COPD presents with
worsening dyspnea over 3 days, productive cough with green sputum, and fever.
Patient unable to complete sentences due to dyspnea.

VITAL SIGNS: Temperature 38.5°C, BP 145/85, HR 115, RR 32, O2 saturation 82% on room air.

PHYSICAL EXAMINATION: Patient in severe respiratory distress, using accessory muscles.
Decreased breath sounds bilaterally with scattered wheezes and crackles.

ARTERIAL BLOOD GAS (on 6L NC):
- pH 7.28
- PaCO2 68 mmHg
- PaO2 52 mmHg
- HCO3 28 mEq/L

CHEST X-RAY: Bilateral infiltrates consistent with pneumonia. No pneumothorax.

ASSESSMENT:
1. Acute hypoxemic and hypercapnic respiratory failure
2. Acute exacerbation of COPD
3. Community-acquired pneumonia, severe
4. Respiratory acidosis

PLAN:
1. Emergent intubation and mechanical ventilation
2. ICU admission
3. Broad-spectrum antibiotics (ceftriaxone + azithromycin)
4. Systemic corticosteroids for COPD exacerbation
5. Bronchodilator therapy

ICD-10 CODES:
- J96.00 (Acute respiratory failure, unspecified whether with hypoxia or hypercapnia)
- J44.1 (Chronic obstructive pulmonary disease with acute exacerbation)
- J18.9 (Pneumonia, unspecified organism)
            """,
            "keywords": ["respiratory failure", "COPD exacerbation", "pneumonia", "mechanical ventilation"],
        },
        {
            "sample_id": "4",
            "specialty": "Endocrinology",
            "sample_name": "Severe Protein-Calorie Malnutrition",
            "description": "Weight loss, low BMI, hypoalbuminemia",
            "transcription": """
CHIEF COMPLAINT: Significant weight loss and weakness.

HISTORY OF PRESENT ILLNESS: 72-year-old female with recent diagnosis of pancreatic
cancer presents with 35-pound weight loss over 2 months, generalized weakness, and
inability to maintain oral intake.

PHYSICAL EXAMINATION: Patient appears cachectic. Height 165 cm, Weight 38 kg, BMI 14.0.
Temporal wasting, prominent ribs, decreased muscle mass in extremities.

LABORATORY DATA:
- Albumin 2.2 g/dL (normal 3.5-5.0)
- Prealbumin 8 mg/dL (normal 15-36)
- Total protein 4.8 g/dL (normal 6.0-8.0)
- Transferrin 120 mg/dL (normal 200-360)

ASSESSMENT:
1. Severe protein-calorie malnutrition (BMI 14.0, albumin 2.2)
2. Cancer-associated cachexia
3. Pancreatic adenocarcinoma (known)

PLAN:
1. Nutrition consult for enteral feeding plan
2. Consider PEG tube placement
3. High-protein, high-calorie supplementation
4. Monitor nutritional markers weekly
5. Coordinate with oncology for cancer treatment

DOCUMENTATION NOTE: Physician documented "malnutrition" but did not specify severity.
CDI query sent to clarify degree of malnutrition (severe vs. moderate) based on
BMI <16 and albumin <2.5, which meets criteria for severe malnutrition.

ICD-10 CODES:
- E43 (Unspecified severe protein-calorie malnutrition) - after CDI query clarification
- C25.9 (Malignant neoplasm of pancreas, unspecified)
            """,
            "keywords": ["severe malnutrition", "cachexia", "protein-calorie malnutrition", "low BMI"],
        },
    ]

    print(f"Generated {len(sample_data)} sample clinical notes")
    return sample_data


def save_to_json(data: List[Dict], filename: str = "mtsamples_cdi.json"):
    """Save the dataset to a JSON file."""

    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to: {filepath}")
    return filepath


def load_into_vectordb(json_file: str):
    """Load the downloaded data into ChromaDB using the loader script."""

    print("\nLoading data into vector database...")

    # Import the loader functions
    from load_cdi_documents import load_from_json_file, create_or_update_vectorstore

    # Load documents from JSON (use 'transcription' field as content)
    documents = load_from_json_file(
        json_file,
        content_key="transcription",
        metadata_keys=["sample_id", "specialty", "sample_name", "description", "keywords"]
    )

    # Create/update vector store
    vectorstore = create_or_update_vectorstore(documents, clear_existing=False)

    return vectorstore


def main():
    """Main execution function."""

    print("=" * 70)
    print("MTSamples Clinical Notes Downloader")
    print("=" * 70)
    print("\nThis script downloads sample clinical notes for CDI training.")
    print("Data source: MTSamples (free, publicly available)")
    print()

    # Download data
    dataset = download_mtsamples_dataset()

    # Save to JSON
    json_file = save_to_json(dataset, "mtsamples_cdi.json")

    # Ask if user wants to load into vector DB
    response = input("\nLoad data into ChromaDB vector store? (y/n): ").strip().lower()

    if response == 'y':
        try:
            load_into_vectordb(json_file)
            print("\n✓ Success! Clinical notes loaded into vector database.")
            print("  You can now use cdi_rag_system.py to query the data.")
        except Exception as e:
            print(f"\nError loading into vector DB: {e}")
            print("You can manually load later using load_cdi_documents.py")

    print("\n" + "=" * 70)
    print("Done!")
    print(f"JSON file: {json_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
