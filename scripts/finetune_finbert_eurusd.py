#!/usr/bin/env python3
"""
Fine-Tune FinBERT for EUR/USD Forex Prediction (Milestone 16)

Multi-task fine-tuning with LoRA:
- Direction prediction (up/down)
- Bucket prediction (5 classes)

Configuration:
- LoRA: r=16, alpha=32, all attention layers
- Training: 30 epochs, batch=16, lr=2e-5, warmup=10%
- Loss weighting: 50/50 (direction/bucket)
- Target: >70% direction accuracy, MAE <2% for buckets

Usage:
    python scripts/finetune_finbert_eurusd.py [--epochs 30] [--batch-size 16]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import torch
import pandas as pd
from transformers import AutoTokenizer

from src.models.finbert.finetune_head import FinBERTForexPredictor
from src.preprocessing.forex_preprocessor import ForexDataPreprocessor
from src.training.forex_finetune_trainer import (
    ForexDataset,
    ForexFineTuneTrainer,
    setup_lora_model
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/finbert_eurusd_finetuning.log'),
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main fine-tuning script."""
    parser = argparse.ArgumentParser(
        description='Fine-tune FinBERT for EUR/USD forex prediction'
    )
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--output-dir', type=str, default='data/models/finbert_forex_finetuned',
                       help='Output directory')
    parser.add_argument('--test-run', action='store_true', help='Run with small subset for testing')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("EUR/USD Forex Fine-Tuning (Milestone 16)")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - LoRA rank: {args.lora_r}")
    logger.info(f"  - LoRA alpha: {args.lora_alpha}")
    logger.info(f"  - Output: {args.output_dir}")
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        logger.info(f"  - Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info(f"  - Device: CUDA")
    else:
        device = "cpu"
        logger.info(f"  - Device: CPU")
    
    try:
        # Load datasets
        logger.info("\n📊 Loading EUR/USD datasets...")
        if args.test_run:
            train_df = pd.read_parquet('data/processed/train.parquet')[:100]
            val_df = pd.read_parquet('data/processed/validation.parquet')[:20]
            logger.info("  TEST RUN: Using small subset")
        else:
            train_df = pd.read_parquet('data/processed/train.parquet')
            val_df = pd.read_parquet('data/processed/validation.parquet')
        
        logger.info(f"  ✓ Train: {len(train_df)} sequences")
        logger.info(f"  ✓ Val: {len(val_df)} sequences")
        
        # Preprocess data
        logger.info("\n🔧 Preprocessing data...")
        preprocessor = ForexDataPreprocessor(
            normalize_features=True,
            fill_nan_strategy='mean'
        )
        
        train_processed, train_texts = preprocessor.prepare_dataset(train_df, fit=True)
        val_processed, val_texts = preprocessor.prepare_dataset(val_df, fit=False)
        logger.info("  ✓ Data preprocessed")
        
        # Create datasets
        logger.info("\n📝 Creating PyTorch datasets...")
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        train_dataset = ForexDataset(
            texts=train_texts,
            direction_labels=train_processed['direction_label'].values,
            bucket_labels=train_processed['bucket_label'].values,
            tokenizer=tokenizer,
        )
        
        val_dataset = ForexDataset(
            texts=val_texts,
            direction_labels=val_processed['direction_label'].values,
            bucket_labels=val_processed['bucket_label'].values,
            tokenizer=tokenizer,
        )
        logger.info(f"  ✓ Datasets created")
        
        # Create model
        logger.info("\n🤖 Initializing FinBERT Forex Predictor...")
        model = FinBERTForexPredictor(
            finbert_model_path="data/models/finbert_contrastive",
            freeze_bert=False,
        )
        
        # Apply LoRA
        model = setup_lora_model(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        
        # Create trainer
        logger.info("\n🎓 Creating trainer...")
        trainer = ForexFineTuneTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=args.output_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            direction_weight=0.5,
            bucket_weight=0.5,
            use_lora=False,  # Already applied manually
        )
        
        # Train
        logger.info("\n🚀 Starting training...")
        start_time = datetime.now()
        
        history = trainer.train()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("FINE-TUNING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Training time: {elapsed/60:.1f} minutes")
        logger.info(f"Best direction accuracy: {max(history['direction_acc']):.1%}")
        logger.info(f"Best bucket accuracy: {max(history['bucket_acc']):.1%}")
        logger.info(f"Best bucket MAE: {min(history['bucket_mae']):.3f}")
        logger.info(f"\nModel saved to: {args.output_dir}")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

