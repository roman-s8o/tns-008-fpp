#!/usr/bin/env python3
"""
Test script to verify FinBERT model loading on Mac M3.

This script tests:
- FinBERT model loading
- MPS device availability
- Forward pass with MLM
- Masked token prediction

Usage:
    python scripts/test_finbert_model.py
"""

import os
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.finbert import (
    load_finbert_model,
    load_finbert_tokenizer,
    FinBERTConfig,
)
from src.models.finbert.model_loader import test_model_forward_pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Test FinBERT model loading and basic functionality."""
    
    logger.info("=" * 70)
    logger.info("FinBERT Model Loading Test")
    logger.info("=" * 70)
    
    # Create config
    config = FinBERTConfig(
        model_name="ProsusAI/finbert",
        batch_size=16,
        mlm_probability=0.15,
    )
    
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - Model: {config.model_name}")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - MLM probability: {config.mlm_probability}")
    logger.info(f"  - Device: {config.device}")
    logger.info("")
    
    try:
        # Load tokenizer
        logger.info("Step 1/3: Loading tokenizer...")
        tokenizer = load_finbert_tokenizer(config)
        logger.info("")
        
        # Load model
        logger.info("Step 2/3: Loading model...")
        model, device = load_finbert_model(config)
        logger.info("")
        
        # Test forward pass and MLM
        logger.info("Step 3/3: Testing MLM forward pass...")
        success = test_model_forward_pass(
            model=model,
            tokenizer=tokenizer,
            device=device,
            test_text=(
                "Apple Inc. reported strong quarterly earnings today. "
                "The stock price surged 5% in after-hours trading. "
                "Analysts upgraded their price targets following the results."
            )
        )
        logger.info("")
        
        if success:
            logger.info("=" * 70)
            logger.info("✓ ALL TESTS PASSED!")
            logger.info("=" * 70)
            logger.info("")
            logger.info("FinBERT is ready for MLM training!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Verify dataset exists: ls data/processed/")
            logger.info("  2. Start training: python scripts/train_finbert_mlm.py")
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
        logger.error("  1. Check internet connection (model downloads from HuggingFace)")
        logger.error("  2. Ensure sufficient memory (16GB RAM minimum)")
        logger.error("  3. Verify PyTorch installation: pip install torch")
        logger.error("")
        
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

