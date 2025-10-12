#!/usr/bin/env python3
"""
Dataset construction script for Milestone 5.

Builds complete dataset with train/validation/fine-tuning splits.

Usage:
    python scripts/build_dataset.py [--max-sequences 10000] [--incremental]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.dataset_builder import DatasetBuilder
from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build full dataset with temporal splits"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=100000,
        help="Maximum number of sequences to generate (default: 100000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental update of existing dataset"
    )
    parser.add_argument(
        "--no-weekend-mapping",
        action="store_true",
        help="Disable weekend news mapping to trading days"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--finetune-ratio",
        type=float,
        default=0.1,
        help="Fine-tuning set ratio (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/dataset_construction.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    # Initialize builder
    logger.info("="*60)
    logger.info("MILESTONE 5: DATASET CONSTRUCTION")
    logger.info("="*60)
    logger.info(f"  Target sequences: {args.max_sequences}")
    logger.info(f"  Split ratios: {args.train_ratio}/{args.val_ratio}/{args.finetune_ratio}")
    logger.info(f"  Weekend mapping: {'Disabled' if args.no_weekend_mapping else 'Enabled'}")
    logger.info(f"  Mode: {'Incremental' if args.incremental else 'Full rebuild'}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("")
    
    builder = DatasetBuilder()
    
    # Load price data
    logger.info("STEP 1: Loading price data...")
    builder.sequence_builder.load_and_process_prices()
    
    if not builder.sequence_builder.price_cache:
        logger.error("No price data loaded. Run price ingestion first.")
        return 1
    
    logger.info(f"✓ Loaded {len(builder.sequence_builder.price_cache)} tickers")
    logger.info("")
    
    start_time = datetime.now()
    
    # Build or update dataset
    if args.incremental:
        logger.info("STEP 2: Incremental dataset update...")
        existing_path = Path(args.output_dir) / "full_dataset.parquet"
        
        if not existing_path.exists():
            logger.error(f"Existing dataset not found: {existing_path}")
            logger.info("Run without --incremental to create initial dataset")
            return 1
        
        df = builder.incremental_update(str(existing_path))
    else:
        logger.info("STEP 2: Building full dataset...")
        df = builder.build_full_dataset(
            include_weekends=not args.no_weekend_mapping,
            max_sequences=args.max_sequences
        )
    
    if df.empty:
        logger.error("No sequences generated. Check news and price data availability.")
        return 1
    
    logger.info(f"✓ Generated {len(df)} sequences")
    logger.info("")
    
    # Create temporal splits
    logger.info("STEP 3: Creating temporal splits...")
    train_df, val_df, finetune_df = builder.create_temporal_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        finetune_ratio=args.finetune_ratio
    )
    logger.info("")
    
    # Save datasets
    logger.info("STEP 4: Saving datasets...")
    builder.save_splits(train_df, val_df, finetune_df, output_dir=args.output_dir)
    logger.info("")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate statistics
    logger.info("="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    
    stats = builder.get_dataset_statistics(df)
    
    logger.info(f"Total sequences: {stats['total_sequences']}")
    logger.info(f"Unique tickers: {stats['unique_tickers']}")
    logger.info(f"Unique dates: {stats['unique_dates']}")
    logger.info(f"Date range: {stats['date_range']}")
    logger.info(f"Avg news/sequence: {stats['avg_news_per_sequence']:.1f}")
    logger.info(f"Processing time: {duration:.1f} seconds")
    logger.info("")
    
    # Show top tickers
    top_tickers = sorted(
        stats['sequences_per_ticker'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    logger.info("Top 10 tickers by sequence count:")
    for ticker, count in top_tickers:
        logger.info(f"  {ticker}: {count}")
    logger.info("")
    
    # Sample sequence
    logger.info("-"*60)
    logger.info("Sample sequence from training set:")
    logger.info("-"*60)
    sample = train_df.iloc[0]
    logger.info(f"Ticker: {sample['ticker']}")
    logger.info(f"Date: {sample['date']}")
    logger.info(f"News count: {sample['news_count']}")
    logger.info(f"News (first 150 chars): {sample['news_text'][:150]}...")
    logger.info(f"Price features: {sample['price_features']}")
    logger.info("-"*60)
    logger.info("")
    
    # Success criteria
    print("="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    
    # Check size
    target_min = min(args.max_sequences, 1000)  # At least 1k or target
    if len(df) >= target_min:
        print(f"✅ DATASET SIZE: {len(df)} sequences (target: {target_min}+)")
    else:
        print(f"⚠️  DATASET SIZE: {len(df)} sequences (target: {target_min}+)")
        print("   Consider collecting more news data to reach target")
    
    # Check splits
    if len(train_df) > 0 and len(val_df) > 0 and len(finetune_df) > 0:
        print("✅ SPLITS: All splits have data")
    else:
        print("❌ SPLITS: One or more splits are empty")
    
    # Check processing time
    if duration < 1800:  # < 30 minutes
        print(f"✅ PERFORMANCE: Completed in {duration:.1f}s (< 30 min)")
    else:
        print(f"⚠️  PERFORMANCE: Took {duration:.1f}s (> 30 min target)")
    
    # Check data quality
    has_news = (df['news_count'] > 0).all()
    has_prices = df['price_features'].notna().all()
    
    if has_news and has_prices:
        print("✅ DATA QUALITY: All sequences have news and prices")
    else:
        print("❌ DATA QUALITY: Some sequences missing data")
    
    print("")
    print(f"📊 Dataset saved to: {args.output_dir}/")
    print(f"   - train.parquet ({len(train_df)} sequences)")
    print(f"   - validation.parquet ({len(val_df)} sequences)")
    print(f"   - finetune.parquet ({len(finetune_df)} sequences)")
    print(f"   - full_dataset.parquet ({len(df)} sequences)")
    print(f"   - metadata.json")
    
    return 0 if len(df) >= target_min else 1


if __name__ == "__main__":
    sys.exit(main())

