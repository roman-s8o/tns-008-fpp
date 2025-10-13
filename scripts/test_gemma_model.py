#!/usr/bin/env python3
"""
Test script to verify Gemma model loading on Mac M3.

This script tests:
- HuggingFace authentication
- Model loading with quantization settings
- MPS device availability
- Forward pass with sample text

Usage:
    python scripts/test_gemma_model.py
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.gemma import (
    load_gemma_model,
    load_gemma_tokenizer,
    GemmaConfig,
)
from src.models.gemma.model_loader import test_model_forward_pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test Gemma model loading and basic functionality."""
    
    logger.info("=" * 70)
    logger.info("Gemma Model Loading Test")
    logger.info("=" * 70)
    
    # Check for HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("⚠ HUGGINGFACE_TOKEN not found in environment")
        logger.warning("Please set it using: export HUGGINGFACE_TOKEN='your_token'")
        logger.warning("Or: huggingface-cli login")
        logger.info("")
        logger.info("Attempting to load without token...")
    else:
        logger.info("✓ HuggingFace token found")
    
    # Create config
    config = GemmaConfig(
        model_name="google/gemma-3-270m",
        batch_size=4,
        use_quantization=True,
        quantization_bits=4,
    )
    
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Quantization: {config.quantization_bits}-bit")
    logger.info(f"  - Device: {config.device}")
    logger.info("")
    
    try:
        # Load tokenizer
        logger.info("Step 1/3: Loading tokenizer...")
        tokenizer = load_gemma_tokenizer(config)
        logger.info("")
        
        # Load model
        logger.info("Step 2/3: Loading model...")
        model, device = load_gemma_model(config)
        logger.info("")
        
        # Test forward pass
        logger.info("Step 3/3: Testing forward pass...")
        success = test_model_forward_pass(
            model=model,
            tokenizer=tokenizer,
            device=device,
            test_text="The stock market today showed significant movement in tech stocks. "
                      "Apple and Microsoft both gained over 2%."
        )
        logger.info("")
        
        if success:
            logger.info("=" * 70)
            logger.info("✓ ALL TESTS PASSED!")
            logger.info("=" * 70)
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Run preprocessing: python scripts/preprocess_data.py")
            logger.info("  2. Build dataset: python scripts/build_dataset.py")
            logger.info("  3. Start training: python scripts/train_gemma_ssl.py")
            logger.info("")
            return 0
        else:
            logger.error("=" * 70)
            logger.error("✗ TESTS FAILED")
            logger.error("=" * 70)
            return 1
            
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"✗ ERROR: {e}")
        logger.error("=" * 70)
        logger.error("")
        logger.error("Troubleshooting:")
        logger.error("  1. Make sure you have accepted the Gemma license:")
        logger.error("     https://huggingface.co/google/gemma-3-270m")
        logger.error("  2. Authenticate with HuggingFace:")
        logger.error("     huggingface-cli login")
        logger.error("  3. Set environment variable:")
        logger.error("     export HUGGINGFACE_TOKEN='your_token'")
        logger.error("  4. Ensure sufficient memory (16GB RAM minimum)")
        logger.error("")
        
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

