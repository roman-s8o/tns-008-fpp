#!/usr/bin/env python3
"""
Pre-process and cache EUR/USD data for fast training.

This script pre-processes all data once and saves formatted texts,
so fine-tuning can start immediately without slow preprocessing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import logging

from src.preprocessing.forex_preprocessor import ForexDataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Pre-processing EUR/USD data for caching...")
    
    # Load datasets
    train_df = pd.read_parquet('data/processed/train.parquet')
    val_df = pd.read_parquet('data/processed/validation.parquet')
    
    logger.info(f"Loaded: {len(train_df)} train, {len(val_df)} val")
    
    # Process
    preprocessor = ForexDataPreprocessor()
    
    logger.info("Processing train set...")
    train_processed, train_texts = preprocessor.prepare_dataset(train_df, fit=True)
    
    logger.info("Processing val set...")
    val_processed, val_texts = preprocessor.prepare_dataset(val_df, fit=False)
    
    # Save
    logger.info("Saving cached data...")
    
    cache = {
        'train_texts': train_texts,
        'train_direction': train_processed['direction_label'].values,
        'train_bucket': train_processed['bucket_label'].values,
        'val_texts': val_texts,
        'val_direction': val_processed['direction_label'].values,
        'val_bucket': val_processed['bucket_label'].values,
        'preprocessor': preprocessor,
    }
    
    with open('data/processed/cached_preprocessed.pkl', 'wb') as f:
        pickle.dump(cache, f)
    
    logger.info(f"✓ Cached to data/processed/cached_preprocessed.pkl")
    logger.info(f"  Train: {len(train_texts)} sequences")
    logger.info(f"  Val: {len(val_texts)} sequences")

if __name__ == "__main__":
    main()

