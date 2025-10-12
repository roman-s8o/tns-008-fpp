#!/usr/bin/env python3
"""
Script to fetch Nasdaq-100 historical prices.

Usage:
    python scripts/fetch_nasdaq_prices.py [--years 5] [--no-web] [--test]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.fetch_prices import PriceDataFetcher
from loguru import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch Nasdaq-100 historical price data"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of historical data to fetch (default: 5)"
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Use hardcoded ticker list instead of fetching from web"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only fetch data for 5 sample tickers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/prices",
        help="Output directory for CSV files (default: data/raw/prices)"
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
        "logs/price_ingestion.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    # Initialize fetcher
    logger.info(f"Initializing price data fetcher...")
    logger.info(f"  Lookback period: {args.years} years")
    logger.info(f"  Use web tickers: {not args.no_web}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    fetcher = PriceDataFetcher(
        output_dir=args.output_dir,
        lookback_years=args.years,
        use_web_tickers=not args.no_web
    )
    
    # Get ticker list
    if args.test:
        # Test mode: only fetch a few tickers
        test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        logger.warning(f"TEST MODE: Only fetching {len(test_tickers)} tickers")
        results = fetcher.fetch_all_tickers(
            tickers=test_tickers,
            save_individual=True,
            save_combined=True
        )
    else:
        # Full mode: fetch all Nasdaq-100 tickers
        results = fetcher.fetch_all_tickers(
            save_individual=True,
            save_combined=True
        )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total tickers: {results['total']}")
    print(f"Successful: {results['successful']} ({results['success_rate']:.1f}%)")
    print(f"Failed: {results['failed']}")
    print(f"Output directory: {results['output_dir']}")
    
    if results['failed_tickers']:
        print(f"\nFailed tickers: {', '.join(results['failed_tickers'])}")
    
    # Success criteria
    print("\n" + "="*60)
    if results['success_rate'] >= 100:
        print("✅ SUCCESS: All tickers downloaded!")
        return 0
    elif results['success_rate'] >= 95:
        print(f"⚠️  PARTIAL SUCCESS: {results['success_rate']:.1f}% downloaded")
        return 0
    else:
        print(f"❌ FAILURE: Only {results['success_rate']:.1f}% downloaded")
        return 1


if __name__ == "__main__":
    sys.exit(main())

