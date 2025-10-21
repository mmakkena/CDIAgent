"""
Mock CDI guidelines for testing and development.

In production, these would be loaded from a document corpus (PDFs, text files, etc.)
using DirectoryLoader or similar tools.
"""

# Mock CDI Knowledge Base
# Each tuple contains: (document_text, metadata_dict)
MOCK_CDI_DOCUMENTS = [
    (
        "A documentation gap exists when a patient has a clinical indicator of severe "
        "malnutrition (BMI < 16, protein-calorie intake deficiency) but the final diagnosis "
        "is only 'malnutrition' without severity specified. A query is required to clarify "
        "the degree (severe, moderate, mild) of malnutrition.",
        {"source": "Malnutrition Guideline 2023"}
    ),
    (
        "For pneumonia cases, the physician must specify the suspected or confirmed causative "
        "organism (e.g., Bacterial, Viral, Aspiration) to ensure accurate ICD-10 coding. "
        "If unspecified, a query is mandatory.",
        {"source": "Infectious Disease Coding 2024"}
    ),
    (
        "When a patient presents with respiratory distress requiring mechanical ventilation "
        "for over 96 hours, the physician must link the underlying cause (e.g., Acute "
        "Respiratory Failure) to the intervention to support medical necessity.",
        {"source": "Respiratory Failure Policy 2022"}
    ),
    (
        "Clinical indicators of Acute Kidney Injury (AKI) stage 2 or 3 (e.g., specific "
        "creatinine and output values) must be documented as 'Acute Kidney Failure' for "
        "higher specificity, or queried for clarification.",
        {"source": "Nephrology Documentation Rules"}
    ),
    (
        "The clinical documentation must clearly link all secondary diagnoses to the treatment "
        "provided or impact on the length of stay for proper medical necessity review.",
        {"source": "General Coding Rule 1.4"}
    ),
]
