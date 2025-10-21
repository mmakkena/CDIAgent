"""
Unit tests for cdi_rag.config module.

Tests configuration constants and their validity.
"""

import pytest
from cdi_rag import config


class TestConfigConstants:
    """Test configuration constants."""

    def test_model_names_exist(self):
        """Test that model name constants are defined."""
        assert hasattr(config, 'MODEL_NAME')
        assert hasattr(config, 'FALLBACK_MODEL_NAME')
        assert isinstance(config.MODEL_NAME, str)
        assert isinstance(config.FALLBACK_MODEL_NAME, str)

    def test_chunk_sizes(self):
        """Test chunk size constants."""
        assert hasattr(config, 'CHUNK_SIZE')
        assert hasattr(config, 'CHUNK_OVERLAP')
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_database_config(self):
        """Test database configuration constants."""
        assert hasattr(config, 'CHROMA_DB_PATH')
        assert hasattr(config, 'CHROMA_COLLECTION_NAME')
        assert isinstance(config.CHROMA_DB_PATH, str)
        assert isinstance(config.CHROMA_COLLECTION_NAME, str)

    def test_retrieval_weights(self):
        """Test retrieval weight constants."""
        assert hasattr(config, 'VECTOR_WEIGHT')
        assert hasattr(config, 'BM25_WEIGHT')

        # Weights should be between 0 and 1
        assert 0 <= config.VECTOR_WEIGHT <= 1
        assert 0 <= config.BM25_WEIGHT <= 1

        # Weights should sum to 1.0 (or close to it)
        total_weight = config.VECTOR_WEIGHT + config.BM25_WEIGHT
        assert abs(total_weight - 1.0) < 0.01

    def test_retrieval_k_values(self):
        """Test retrieval K constants."""
        assert hasattr(config, 'RETRIEVAL_K_BM25')
        assert hasattr(config, 'RETRIEVAL_K_VECTOR')
        assert hasattr(config, 'RETRIEVAL_K_INITIAL')
        assert hasattr(config, 'RETRIEVAL_K_FINAL')

        # All should be positive integers
        assert config.RETRIEVAL_K_BM25 > 0
        assert config.RETRIEVAL_K_VECTOR > 0
        assert config.RETRIEVAL_K_INITIAL > 0
        assert config.RETRIEVAL_K_FINAL > 0

        # Initial should be >= Final (retrieve more, then re-rank to fewer)
        assert config.RETRIEVAL_K_INITIAL >= config.RETRIEVAL_K_FINAL

    def test_reranking_config(self):
        """Test re-ranking configuration constants."""
        assert hasattr(config, 'RERANKING_MAX_FEATURES')
        assert hasattr(config, 'RERANKING_NGRAM_RANGE')
        assert hasattr(config, 'RERANKING_STOP_WORDS')

        assert config.RERANKING_MAX_FEATURES > 0
        assert isinstance(config.RERANKING_NGRAM_RANGE, tuple)
        assert len(config.RERANKING_NGRAM_RANGE) == 2
        assert config.RERANKING_NGRAM_RANGE[0] <= config.RERANKING_NGRAM_RANGE[1]

    def test_llm_generation_params(self):
        """Test LLM generation parameter constants."""
        assert hasattr(config, 'MAX_NEW_TOKENS')
        assert hasattr(config, 'TEMPERATURE')
        assert hasattr(config, 'TOP_K')
        assert hasattr(config, 'TOP_P')
        assert hasattr(config, 'DO_SAMPLE')

        # Value ranges
        assert config.MAX_NEW_TOKENS > 0
        assert 0 <= config.TEMPERATURE <= 2.0
        assert config.TOP_K > 0
        assert 0 <= config.TOP_P <= 1.0
        assert isinstance(config.DO_SAMPLE, bool)

    def test_system_prompt_exists(self):
        """Test that SYSTEM_PROMPT is defined and not empty."""
        assert hasattr(config, 'SYSTEM_PROMPT')
        assert isinstance(config.SYSTEM_PROMPT, str)
        assert len(config.SYSTEM_PROMPT) > 0
        assert 'CDI' in config.SYSTEM_PROMPT


class TestConfigValues:
    """Test specific configuration values."""

    def test_default_model_name(self):
        """Test default model name."""
        assert "mistral" in config.MODEL_NAME.lower() or "gpt" in config.FALLBACK_MODEL_NAME.lower()

    def test_default_weights_sum_to_one(self):
        """Test that default weights sum to 1.0."""
        assert config.VECTOR_WEIGHT + config.BM25_WEIGHT == 1.0

    def test_default_retrieval_pipeline(self):
        """Test default retrieval pipeline configuration."""
        # Should retrieve 8, re-rank to 4
        assert config.RETRIEVAL_K_INITIAL == 8
        assert config.RETRIEVAL_K_FINAL == 4

    def test_ngram_range_includes_unigrams_and_bigrams(self):
        """Test that ngram range includes both unigrams and bigrams."""
        assert config.RERANKING_NGRAM_RANGE == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
