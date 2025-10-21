"""
Unit tests for cdi_rag.utils module.

Tests utility functions for metadata filtering and document processing.
"""

import pytest
from langchain_core.documents import Document
from cdi_rag.utils import (
    convert_metadata_filter_to_chroma,
    extract_source_from_metadata,
    filter_documents_by_metadata
)


class TestConvertMetadataFilterToChroma:
    """Test convert_metadata_filter_to_chroma function."""

    def test_none_filter(self):
        """Test that None input returns None."""
        result = convert_metadata_filter_to_chroma(None)
        assert result is None

    def test_single_value_filter(self):
        """Test filter with single string value."""
        filter_dict = {"specialty": "Cardiology"}
        result = convert_metadata_filter_to_chroma(filter_dict)
        assert result == {"specialty": "Cardiology"}

    def test_list_value_filter(self):
        """Test filter with list value gets $in operator."""
        filter_dict = {"icd10": ["I21.0", "I21.1", "I21.9"]}
        result = convert_metadata_filter_to_chroma(filter_dict)
        assert result == {"icd10": {"$in": ["I21.0", "I21.1", "I21.9"]}}

    def test_mixed_filter(self):
        """Test filter with both single and list values."""
        filter_dict = {
            "specialty": "Cardiology",
            "icd10": ["I21.0", "I21.1"],
            "category": "Acute"
        }
        result = convert_metadata_filter_to_chroma(filter_dict)

        assert result["specialty"] == "Cardiology"
        assert result["icd10"] == {"$in": ["I21.0", "I21.1"]}
        assert result["category"] == "Acute"

    def test_empty_filter(self):
        """Test empty filter dict."""
        result = convert_metadata_filter_to_chroma({})
        assert result == {}

    def test_numeric_value(self):
        """Test filter with numeric value."""
        filter_dict = {"year": 2024}
        result = convert_metadata_filter_to_chroma(filter_dict)
        assert result == {"year": 2024}


class TestExtractSourceFromMetadata:
    """Test extract_source_from_metadata function."""

    def test_source_field(self):
        """Test extraction when 'source' field exists."""
        metadata = {"source": "CDI Guideline 2024", "other": "data"}
        result = extract_source_from_metadata(metadata)
        assert result == "CDI Guideline 2024"

    def test_sample_name_fallback(self):
        """Test fallback to 'sample_name' when 'source' missing."""
        metadata = {"sample_name": "Cardiology Case 1", "other": "data"}
        result = extract_source_from_metadata(metadata)
        assert result == "Cardiology Case 1"

    def test_specialty_fallback(self):
        """Test fallback to 'specialty' when others missing."""
        metadata = {"specialty": "Neurology", "other": "data"}
        result = extract_source_from_metadata(metadata)
        assert result == "Neurology"

    def test_unknown_source(self):
        """Test 'Unknown Source' when no fields present."""
        metadata = {"other": "data", "random": "value"}
        result = extract_source_from_metadata(metadata)
        assert result == "Unknown Source"

    def test_empty_metadata(self):
        """Test empty metadata dict."""
        metadata = {}
        result = extract_source_from_metadata(metadata)
        assert result == "Unknown Source"

    def test_priority_order(self):
        """Test that 'source' has priority over others."""
        metadata = {
            "source": "Official Source",
            "sample_name": "Sample Name",
            "specialty": "Specialty"
        }
        result = extract_source_from_metadata(metadata)
        assert result == "Official Source"


class TestFilterDocumentsByMetadata:
    """Test filter_documents_by_metadata function."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Cardiology case 1",
                metadata={"specialty": "Cardiology", "icd10": "I21.0", "year": 2024}
            ),
            Document(
                page_content="Cardiology case 2",
                metadata={"specialty": "Cardiology", "icd10": "I21.1", "year": 2023}
            ),
            Document(
                page_content="Neurology case 1",
                metadata={"specialty": "Neurology", "icd10": "G40.0", "year": 2024}
            ),
            Document(
                page_content="Internal Medicine case",
                metadata={"specialty": "Internal Medicine", "icd10": "E11.9", "year": 2024}
            ),
        ]

    def test_filter_by_single_field(self, sample_documents):
        """Test filtering by single metadata field."""
        filter_dict = {"specialty": "Cardiology"}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 2
        assert all(doc.metadata["specialty"] == "Cardiology" for doc in result)

    def test_filter_by_multiple_fields(self, sample_documents):
        """Test filtering by multiple metadata fields (AND logic)."""
        filter_dict = {"specialty": "Cardiology", "year": 2024}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 1
        assert result[0].metadata["specialty"] == "Cardiology"
        assert result[0].metadata["year"] == 2024

    def test_filter_with_list_value(self, sample_documents):
        """Test filtering with list of allowed values."""
        filter_dict = {"icd10": ["I21.0", "I21.1"]}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 2
        assert all(doc.metadata["specialty"] == "Cardiology" for doc in result)

    def test_filter_case_insensitive_string(self, sample_documents):
        """Test case-insensitive partial string matching."""
        filter_dict = {"specialty": "cardio"}  # Lowercase partial match
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 2
        assert all("Cardio" in doc.metadata["specialty"] for doc in result)

    def test_filter_no_matches(self, sample_documents):
        """Test filter that returns no matches."""
        filter_dict = {"specialty": "Oncology"}  # Doesn't exist
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 0

    def test_filter_all_match(self, sample_documents):
        """Test filter where all documents match."""
        filter_dict = {"year": 2024}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 3

    def test_filter_missing_field(self, sample_documents):
        """Test filter for field that doesn't exist in metadata."""
        filter_dict = {"nonexistent": "value"}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 0

    def test_empty_filter(self, sample_documents):
        """Test with empty filter dict (should return all docs)."""
        filter_dict = {}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        # Empty filter means no restrictions, but our implementation
        # only adds docs that match, so empty filter returns all
        # Actually, looking at the code, empty filter will match all
        # because the loop doesn't execute
        assert len(result) == len(sample_documents)

    def test_numeric_exact_match(self, sample_documents):
        """Test exact numeric value matching."""
        filter_dict = {"year": 2023}
        result = filter_documents_by_metadata(sample_documents, filter_dict)

        assert len(result) == 1
        assert result[0].metadata["year"] == 2023


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
