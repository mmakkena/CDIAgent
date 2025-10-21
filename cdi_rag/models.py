"""
LLM model initialization for the CDI RAG system.

This module handles loading and configuration of language models with support for:
- GPU acceleration with 4-bit quantization (QLoRA)
- CPU fallback for systems without CUDA
- Configurable generation parameters
"""

import logging
import warnings
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from langchain_huggingface import HuggingFacePipeline

logger = logging.getLogger(__name__)

from .config import (
    MODEL_NAME,
    FALLBACK_MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    DO_SAMPLE
)

# Suppress internal transformers warnings
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


def get_llm_pipeline():
    """
    Initialize and return the HuggingFace LLM pipeline.

    This function:
    1. Checks for CUDA availability
    2. Loads appropriate model (Mistral-7B on GPU, GPT2-medium on CPU)
    3. Configures quantization for GPU models (4-bit QLoRA)
    4. Creates a text generation pipeline
    5. Wraps pipeline in LangChain-compatible interface

    Returns:
        HuggingFacePipeline: LangChain-wrapped LLM pipeline ready for use

    Example:
        >>> llm = get_llm_pipeline()
        >>> result = llm.invoke("What is CDI?")

    Notes:
        - GPU mode: Uses Mistral-7B with 4-bit quantization (~7GB VRAM)
        - CPU mode: Uses GPT2-medium for testing (~1.5GB RAM)
        - Production systems should use GPU with Mistral-7B for best quality
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info(f"CUDA detected. Loading model with 4-bit quantization: {MODEL_NAME}")
        model_to_load = MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)

        # QLoRA configuration for 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        logger.warning("CUDA not available. Using smaller CPU-friendly model for testing...")
        logger.info(f"Loading fallback model: {FALLBACK_MODEL_NAME}")
        logger.warning("(Production systems should use GPU with Mistral-7B)")

        model_to_load = FALLBACK_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_to_load)

        # Set pad_token if not already set (required for GPT-2)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_to_load,
            dtype=torch.float32
        )

    # Create a transformers pipeline for text generation
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P
    )

    # Wrap the pipeline in a LangChain LLM object
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm
