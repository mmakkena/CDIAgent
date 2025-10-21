# CDI RAG System - Unit Tests

This directory contains unit tests for the `cdi_rag` package.

## Test Organization

```
tests/
├── __init__.py               # Test package initialization
├── test_config.py            # Tests for configuration constants
├── test_utils.py             # Tests for utility functions
└── test_retrievers.py        # Tests for retrieval components
```

## Running Tests

### Run All Tests

```bash
# From project root
pytest

# With verbose output
pytest -v

# With coverage (if pytest-cov installed)
pytest --cov=cdi_rag --cov-report=html
```

### Run Specific Test Files

```bash
# Test utilities only
pytest tests/test_utils.py

# Test retrievers only
pytest tests/test_retrievers.py

# Test configuration only
pytest tests/test_config.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_utils.py::TestConvertMetadataFilterToChroma

# Run specific test function
pytest tests/test_utils.py::TestConvertMetadataFilterToChroma::test_single_value_filter
```

## Test Coverage

### test_config.py
**Purpose**: Validate configuration constants

**Tests**:
- Model name configurations
- Chunk sizes and overlaps
- Database paths and collection names
- Retrieval weights (vector vs BM25)
- Retrieval K values
- Re-ranking configuration
- LLM generation parameters
- System prompts

**Coverage**: ~100% of config module

---

### test_utils.py
**Purpose**: Test utility functions

**Tests**:

**`convert_metadata_filter_to_chroma`**:
- None input handling
- Single value filters
- List value filters (with $in operator)
- Mixed filters
- Empty filters
- Numeric values

**`extract_source_from_metadata`**:
- Source field extraction
- Fallback to sample_name
- Fallback to specialty
- Unknown source handling
- Priority order

**`filter_documents_by_metadata`**:
- Single field filtering
- Multiple field filtering (AND logic)
- List value filtering
- Case-insensitive string matching
- No matches scenarios
- Missing field handling

**Coverage**: ~100% of utils module

---

### test_retrievers.py
**Purpose**: Test document retrieval and re-ranking

**Tests**:

**`DocumentReranker`**:
- Initialization (enabled/disabled)
- Re-ranking disabled behavior
- Empty document handling
- Fewer docs than top_k
- Order changes verification
- Top_k parameter respect
- Metadata preservation
- Error handling with fallback

**`HybridRetriever`**:
- Initialization
- Both retrievers called
- Results merging
- Metadata filtering
- With/without re-ranking
- Scoring weights
- Duplicate document handling

**Coverage**: ~95% of retrievers module

---

## Test Fixtures

### Common Fixtures

**`sample_documents`** (in multiple test files):
```python
# Creates sample Document objects with various metadata
# for testing filtering, retrieval, and re-ranking
```

**`mock_vector_retriever`**:
```python
# Mock for vector similarity retriever
# Returns predefined documents
```

**`mock_bm25_retriever`**:
```python
# Mock for BM25 keyword retriever
# Returns predefined documents
```

## Dependencies

### Required
- `pytest` - Testing framework
- `langchain_core` - For Document class
- `cdi_rag` - The package being tested

### Optional
- `pytest-cov` - For coverage reports
- `pytest-xdist` - For parallel test execution

Install with:
```bash
pip install pytest pytest-cov pytest-xdist
```

## Writing New Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` or `Test<FunctionName>`
- Test functions: `test_<specific_behavior>`

### Example Test Structure

```python
"""
Unit tests for cdi_rag.new_module module.

Brief description of what this module tests.
"""

import pytest
from cdi_rag.new_module import SomeClass


class TestSomeClass:
    """Test SomeClass functionality."""

    @pytest.fixture
    def sample_instance(self):
        """Create a SomeClass instance for testing."""
        return SomeClass(param="value")

    def test_initialization(self, sample_instance):
        """Test that instance initializes correctly."""
        assert sample_instance.param == "value"

    def test_some_method(self, sample_instance):
        """Test some_method returns expected result."""
        result = sample_instance.some_method("input")
        assert result == "expected_output"
```

### Best Practices

1. **One assertion per test** (when possible)
2. **Use descriptive test names** that explain what's being tested
3. **Test edge cases**: empty inputs, None values, boundary conditions
4. **Use fixtures** for common setup
5. **Mock external dependencies** (LLM models, databases)
6. **Document complex tests** with comments
7. **Keep tests independent** - each test should run alone

### Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_simple_function():
    """Unit test for isolated function."""
    pass

@pytest.mark.integration
def test_with_database():
    """Integration test requiring ChromaDB."""
    pass

@pytest.mark.slow
def test_llm_initialization():
    """Slow test that loads models."""
    pass
```

Run specific markers:
```bash
pytest -m unit           # Run only unit tests
pytest -m "not slow"     # Skip slow tests
```

## Continuous Integration

### GitHub Actions (example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=cdi_rag --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### ImportError: No module named 'cdi_rag'

**Solution**: Run tests from project root, not from tests/ directory
```bash
cd /path/to/CDIAgent
pytest tests/
```

### Fixture not found

**Solution**: Check that fixtures are defined in the same file or in `conftest.py`

### Tests pass individually but fail together

**Solution**: Tests may have shared state. Ensure fixtures are properly scoped and tests are independent.

## Coverage Goals

- **Target**: 80%+ coverage for all modules
- **Current Status**:
  - `config.py`: 100%
  - `utils.py`: 100%
  - `retrievers.py`: 95%
  - `models.py`: TBD (requires mocking transformers)
  - `database.py`: TBD (requires ChromaDB setup)
  - `chains.py`: TBD (requires LLM and database)

## Future Test Additions

- [ ] Integration tests for `models.py` (with mocked transformers)
- [ ] Integration tests for `database.py` (with test ChromaDB)
- [ ] Integration tests for `chains.py` (end-to-end RAG)
- [ ] Performance/benchmarking tests
- [ ] Stress tests for large document sets
- [ ] API endpoint tests (for api_server.py)
