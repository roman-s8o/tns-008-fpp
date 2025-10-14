#!/usr/bin/env python3
"""
Train FinBERT with Multi-Task SSL Pre-training

This script implements Milestone 9: SSL Pre-training (FinBERT) with multi-task learning.
Combines MLM and contrastive learning objectives.

Usage:
    python scripts/train_finbert_multitask.py

Features:
    - Multi-task learning (MLM + Contrastive)
    - Configurable loss weighting (default: 50/50)
    - Cosine learning rate scheduler with warmup
    - Perplexity tracking
    - Best checkpoint saving
    - 20 epochs training with batch size 8
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.models.finbert.multitask_model import FinBERTMultiTask
from src.training.finbert_multitask_trainer import MultiTaskTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/multitask_training.log'),
    ]
)
logger = logging.getLogger(__name__)


def load_mlm_checkpoint(checkpoint_path: str, device: str):
    """
    Load pre-trained FinBERT MLM checkpoint from Milestone 7.
    
    Args:
        checkpoint_path: Path to MLM checkpoint directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("=" * 80)
    logger.info(f"Loading MLM checkpoint from: {checkpoint_path}")
    logger.info("=" * 80)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MLM checkpoint not found: {checkpoint_path}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    logger.info(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load MLM model
    logger.info("Loading MLM model...")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    
    # Move to device
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ MLM model loaded successfully!")
    logger.info(f"  - Parameters: {total_params:,}")
    logger.info(f"  - Device: {device}")
    logger.info("=" * 80 + "\n")
    
    return model, tokenizer


def create_multitask_model(mlm_model, device: str, projection_dim: int = 128):
    """
    Create multi-task model from MLM checkpoint.
    
    Args:
        mlm_model: Pre-trained MLM model
        device: Device
        projection_dim: Projection dimension for contrastive learning
        
    Returns:
        Multi-task model
    """
    logger.info("Creating multi-task model...")
    
    # Create multi-task model
    model = FinBERTMultiTask(
        finbert_mlm_model=mlm_model,
        projection_dim=projection_dim,
        freeze_bert=False,  # Train all parameters
    )
    
    # Move entire model to device (ensures projection head is also on device)
    model = model.to(device)
    
    # Model info
    param_info = model.get_num_parameters()
    logger.info(f"✓ Multi-task model created!")
    logger.info(f"  - Total parameters: {param_info['total']:,}")
    logger.info(f"  - Trainable parameters: {param_info['trainable']:,}")
    logger.info(f"  - Projection dimension: {projection_dim}")
    logger.info(f"  - Device: {device}")
    logger.info("")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Multi-Task SSL Pre-training for FinBERT (Milestone 9)"
    )
    
    parser.add_argument(
        "--mlm_checkpoint",
        type=str,
        default="data/models/finbert",
        help="Path to MLM checkpoint from Milestone 7"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed",
        help="Path to processed dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/models/finbert_multitask",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--mlm_weight",
        type=float,
        default=0.5,
        help="Weight for MLM loss (0-1)"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.5,
        help="Weight for contrastive loss (0-1)"
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128,
        help="Projection dimension for contrastive learning"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to train on"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("=" * 80)
    logger.info("Milestone 9: Multi-Task SSL Pre-training (FinBERT)")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(f"  - MLM checkpoint: {args.mlm_checkpoint}")
    logger.info(f"  - Dataset: {args.dataset_path}")
    logger.info(f"  - Output: {args.output_dir}")
    logger.info(f"  - Epochs: {args.num_epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - MLM weight: {args.mlm_weight}")
    logger.info(f"  - Contrastive weight: {args.contrastive_weight}")
    logger.info(f"  - Projection dim: {args.projection_dim}")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Device: {args.device}")
    logger.info("")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Determine device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
        logger.info("✓ Using MPS (Metal Performance Shaders) - Mac GPU")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        logger.info("✓ Using CUDA - NVIDIA GPU")
    else:
        device = "cpu"
        logger.info("⚠ Using CPU (training will be slow)")
    logger.info("")
    
    # Load MLM checkpoint
    mlm_model, tokenizer = load_mlm_checkpoint(args.mlm_checkpoint, device)
    
    # Create multi-task model
    model = create_multitask_model(mlm_model, device, args.projection_dim)
    
    # Training configuration
    config = {
        # Model settings
        "max_seq_length": 512,
        "projection_dim": args.projection_dim,
        
        # Training hyperparameters
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        
        # MLM settings
        "mlm_probability": 0.15,
        
        # Contrastive settings
        "temperature": args.temperature,
        
        # Loss weighting
        "mlm_weight": args.mlm_weight,
        "contrastive_weight": args.contrastive_weight,
        
        # Logging
        "logging_steps": 10,
        "save_steps": 100,
    }
    
    # Create trainer
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = trainer.load_dataset(args.dataset_path, split="train")
    eval_dataset = trainer.load_dataset(args.dataset_path, split="validation")
    logger.info("")
    
    # Train
    try:
        metrics = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"✓ Training completed successfully!")
        logger.info(f"  - Final train loss: {metrics['train_losses'][-1]:.4f}")
        logger.info(f"  - Final eval loss: {metrics['eval_losses'][-1]:.4f}")
        logger.info(f"  - Final perplexity: {metrics['perplexities'][-1]:.4f}")
        logger.info(f"  - Best perplexity: {min(metrics['perplexities']):.4f}")
        logger.info(f"  - Model saved to: {args.output_dir}")
        logger.info("=" * 80)
        
        # Check if perplexity target met
        best_perplexity = min(metrics['perplexities'])
        if best_perplexity < 2.0:
            logger.info(f"\n🎉 SUCCESS! Achieved target perplexity < 2.0: {best_perplexity:.4f}")
        else:
            logger.info(f"\n⚠ Target perplexity not met. Best: {best_perplexity:.4f}, Target: <2.0")
            logger.info("Consider training for more epochs or adjusting hyperparameters.")
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

