#!/usr/bin/env python3
"""
Train FinBERT with Contrastive Learning (NT-Xent/InfoNCE).

This script trains FinBERT with a projection head using contrastive learning
on financial news + price sequences.

Usage:
    python scripts/train_finbert_contrastive.py [--epochs 10] [--batch-size 8]
"""

import os
import sys
import argparse
import logging

# Disable TensorFlow imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.finbert import (
    load_finbert_tokenizer,
    FinBERTConfig,
    FinBERTWithProjection,
)
from src.models.finbert.model_loader import load_finbert_model
from src.training.contrastive_trainer import ContrastiveTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FinBERT with Contrastive Learning"
    )
    
    parser.add_argument(
        "--finbert-checkpoint",
        type=str,
        default="data/models/finbert",
        help="Path to trained FinBERT checkpoint from Milestone 7"
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
        default="data/models/finbert_contrastive",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for NT-Xent loss"
    )
    
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=128,
        help="Projection dimension for embeddings"
    )
    
    parser.add_argument(
        "--freeze-bert",
        action="store_true",
        help="Freeze FinBERT weights (only train projection head)"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("=" * 70)
    logger.info("FinBERT Contrastive Learning Training")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - FinBERT checkpoint: {args.finbert_checkpoint}")
    logger.info(f"  - Dataset: {args.dataset_path}")
    logger.info(f"  - Output: {args.output_dir}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Projection dim: {args.projection_dim}")
    logger.info(f"  - Freeze BERT: {args.freeze_bert}")
    logger.info("")
    
    # Check dataset exists
    train_file = os.path.join(args.dataset_path, "train.parquet")
    val_file = os.path.join(args.dataset_path, "validation.parquet")
    
    if not os.path.exists(train_file):
        logger.error(f"✗ Training dataset not found: {train_file}")
        return 1
    
    if not os.path.exists(val_file):
        logger.warning(f"⚠ Validation dataset not found: {val_file}")
    
    logger.info("✓ Dataset files found")
    logger.info("")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        config = FinBERTConfig()
        tokenizer = load_finbert_tokenizer(config)
        logger.info("✓ Tokenizer loaded")
        
        # Load pre-trained FinBERT from Milestone 7
        logger.info(f"Loading pre-trained FinBERT from {args.finbert_checkpoint}...")
        finbert_model, device = load_finbert_model(config)
        
        # Load checkpoint weights if available
        checkpoint_path = os.path.join(args.finbert_checkpoint, "model.safetensors")
        if os.path.exists(checkpoint_path):
            logger.info("✓ Loading MLM-trained weights...")
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
            finbert_model.load_state_dict(state_dict, strict=False)
            logger.info("✓ MLM-trained weights loaded")
        else:
            logger.warning("⚠ No checkpoint found, using base FinBERT weights")
        
        # Wrap with projection head
        logger.info("Adding projection head...")
        model = FinBERTWithProjection(
            finbert_model=finbert_model,
            projection_dim=args.projection_dim,
            freeze_bert=args.freeze_bert,
        )
        model = model.to(device)
        logger.info(f"✓ Model with projection head ready on {device}")
        logger.info("")
        
        # Create trainer
        trainer_config = {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'temperature': args.temperature,
            'max_seq_length': 512,
        }
        
        trainer = ContrastiveTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            config=trainer_config,
        )
        
        # Load datasets
        train_dataset = trainer.load_dataset(args.dataset_path, split="train")
        eval_dataset = trainer.load_dataset(args.dataset_path, split="validation")
        
        # Train
        logger.info("Starting training...")
        logger.info("")
        
        metrics = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✓ Training Complete!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Training Summary:")
        logger.info(f"  - Final train loss: {metrics['train_losses'][-1]:.4f}")
        if metrics['eval_losses']:
            logger.info(f"  - Final eval loss: {metrics['eval_losses'][-1]:.4f}")
            logger.info(f"  - Best eval loss: {min(metrics['eval_losses']):.4f}")
        logger.info("")
        logger.info(f"Model saved to: {args.output_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  - Evaluate embeddings quality")
        logger.info("  - Use for downstream tasks (classification, regression)")
        logger.info("  - Continue to Milestone 9: SSL Pre-training (full dataset)")
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

