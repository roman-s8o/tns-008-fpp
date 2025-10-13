#!/usr/bin/env python3
"""
Train Gemma model with SSL (Self-Supervised Learning) using MLM.

This script trains the Gemma-3-270m model on the preprocessed financial
news + price sequences using Masked Language Modeling.

Usage:
    python scripts/train_gemma_ssl.py [--epochs 2] [--batch-size 4]
"""

import os
import sys
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training import train_gemma_mlm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Gemma model with SSL (MLM)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-3-270m",
        help="HuggingFace model name"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/processed",
        help="Path to preprocessed dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/gemma",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("Gemma SSL Pre-training (MLM)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - Model: {args.model_name}")
    logger.info(f"  - Dataset: {args.dataset_path}")
    logger.info(f"  - Output: {args.output_dir}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info("")
    
    # Check dataset exists
    train_file = os.path.join(args.dataset_path, "train.parquet")
    val_file = os.path.join(args.dataset_path, "validation.parquet")
    
    if not os.path.exists(train_file):
        logger.error(f"Training dataset not found: {train_file}")
        logger.error("Please run: python scripts/build_dataset.py")
        return 1
    
    if not os.path.exists(val_file):
        logger.warning(f"Validation dataset not found: {val_file}")
        logger.warning("Training will proceed without validation")
    
    # Check HuggingFace token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("⚠ HUGGINGFACE_TOKEN not found")
        logger.warning("Attempting without token (may fail for gated models)")
    
    try:
        # Train model
        metrics = train_gemma_mlm(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_auth_token=hf_token,
        )
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Training Complete!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Final Metrics:")
        for key, value in metrics.items():
            logger.info(f"  - {key}: {value}")
        logger.info("")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"✗ Training failed: {e}")
        logger.error("=" * 70)
        
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

