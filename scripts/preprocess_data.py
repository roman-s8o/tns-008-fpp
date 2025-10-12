#!/usr/bin/env python3
"""
Data preprocessing script for Milestone 4.

Creates multimodal sequences combining news articles and price data.

Usage:
    python scripts/preprocess_data.py [--limit 10000] [--days 365] [--test]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.sequence_builder import SequenceBuilder
from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess financial news and price data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum number of sequences to generate (default: 10000)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Use data from last N days (default: 365, 0=all data)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/sequences.parquet",
        help="Output file path (default: data/processed/sequences.parquet)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process 100 sequences"
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
        "logs/preprocessing.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    
    # Set limit for test mode
    limit = 100 if args.test else args.limit
    
    if args.test:
        logger.warning(f"TEST MODE: Only processing {limit} sequences")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = None
    if args.days > 0:
        start_date = end_date - timedelta(days=args.days)
    
    # Initialize builder
    logger.info("Initializing sequence builder...")
    logger.info(f"  Target sequences: {limit}")
    if start_date:
        logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
    else:
        logger.info(f"  Date range: All available data")
    logger.info(f"  Output: {args.output}")
    
    builder = SequenceBuilder()
    
    # Load price data
    logger.info("\n" + "="*60)
    logger.info("STEP 1: LOADING PRICE DATA")
    logger.info("="*60)
    
    builder.load_and_process_prices(tickers=tickers)
    
    if not builder.price_cache:
        logger.error("No price data loaded. Run price ingestion first.")
        return 1
    
    logger.info(f"✓ Loaded {len(builder.price_cache)} tickers")
    
    # Build dataset
    logger.info("\n" + "="*60)
    logger.info("STEP 2: BUILDING SEQUENCES")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    sequences = builder.build_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        max_sequences=limit
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Show statistics
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING RESULTS")
    logger.info("="*60)
    logger.info(f"Total sequences: {len(sequences)}")
    logger.info(f"Processing time: {duration:.1f} seconds")
    
    if sequences:
        # Calculate statistics
        tickers_used = set([s['ticker'] for s in sequences])
        dates_used = set([s['date'] for s in sequences])
        avg_news = sum([s['news_count'] for s in sequences]) / len(sequences)
        
        logger.info(f"Unique tickers: {len(tickers_used)}")
        logger.info(f"Unique dates: {len(dates_used)}")
        logger.info(f"Avg news per sequence: {avg_news:.1f}")
        
        # Show sample
        logger.info("\n" + "-"*60)
        logger.info("Sample Sequence:")
        logger.info("-"*60)
        sample = sequences[0]
        logger.info(f"Ticker: {sample['ticker']}")
        logger.info(f"Date: {sample['date']}")
        logger.info(f"News count: {sample['news_count']}")
        logger.info(f"News (first 150 chars): {sample['news_text'][:150]}...")
        logger.info(f"Price features: {sample['price_features']}")
        logger.info("-"*60)
    
    # Save dataset
    if sequences:
        logger.info("\n" + "="*60)
        logger.info("STEP 3: SAVING DATASET")
        logger.info("="*60)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        builder.save_dataset(sequences, str(output_path))
        
        logger.info(f"✓ Saved to {args.output}")
        
        # Show file size
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"  File size: {file_size:.2f} MB")
    
    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    
    if len(sequences) >= limit * 0.8:  # At least 80% of target
        print(f"✅ SUCCESS: Generated {len(sequences)} sequences")
        
        # Check data quality
        if sequences:
            has_news = all([s['news_text'] for s in sequences])
            has_prices = all([s['price_features'] for s in sequences])
            
            if has_news and has_prices:
                print("✅ DATA QUALITY: All sequences have both news and prices")
            else:
                print("⚠️  DATA QUALITY: Some sequences missing data")
        
        if duration < 1800:  # < 30 minutes
            print(f"✅ PERFORMANCE: Completed in {duration:.1f}s (< 30 min)")
        else:
            print(f"⚠️  PERFORMANCE: Took {duration:.1f}s (> 30 min target)")
        
        return 0
    else:
        print(f"❌ FAILURE: Only {len(sequences)} sequences (target: {limit})")
        print("   Possible causes:")
        print("   - Not enough news articles in database")
        print("   - Date range too narrow")
        print("   - Ticker-news alignment issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())

