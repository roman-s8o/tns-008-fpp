#!/usr/bin/env python3
"""
Train FinBERT with SSL Masked Language Modeling (MLM).

This script trains the FinBERT model on the preprocessed financial
news + price sequences using Masked Language Modeling.

Usage:
    python scripts/train_finbert_mlm.py [--epochs 3] [--batch-size 16]
"""

import os
import sys
import argparse
import logging

# Disable TensorFlow imports in transformers (we only use PyTorch)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.finbert_mlm_trainer import train_finbert_mlm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FinBERT with SSL Masked Language Modeling (MLM)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="ProsusAI/finbert",
        help="HuggingFace model name (default: ProsusAI/finbert)"
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/processed",
        help="Path to preprocessed dataset (default: data/processed)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/finbert",
        help="Directory to save model checkpoints (default: data/models/finbert)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size per device (default: 16)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("FinBERT SSL Pre-training with MLM")
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
        logger.error(f"✗ Training dataset not found: {train_file}")
        logger.error("Please run: python scripts/build_dataset.py")
        return 1
    
    if not os.path.exists(val_file):
        logger.warning(f"⚠ Validation dataset not found: {val_file}")
        logger.warning("Training will proceed without validation")
    
    logger.info("✓ Dataset files found")
    logger.info("")
    
    try:
        # Train model
        logger.info("Starting FinBERT MLM training...")
        logger.info("")
        
        metrics = train_finbert_mlm(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Training Complete!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Final Metrics:")
        logger.info(f"  - Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"  - Eval loss: {metrics.get('eval_eval_loss', metrics.get('eval_loss', 'N/A')):.4f}")
        logger.info(f"  - Training time: {metrics.get('train_runtime', 0):.2f}s")
        logger.info(f"  - Samples/second: {metrics.get('train_samples_per_second', 0):.2f}")
        logger.info("")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info(f"Tokenized samples saved to: {args.output_dir}/tokenized_sample.txt")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - View training logs: tensorboard --logdir {}/logs".format(args.output_dir))
        logger.info("  - Inspect tokenized samples: cat {}/tokenized_sample.txt".format(args.output_dir))
        logger.info("  - Continue to Milestone 8: Contrastive Learning")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"✗ Training failed: {e}")
        logger.error("=" * 70)
        logger.error("")
        
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

