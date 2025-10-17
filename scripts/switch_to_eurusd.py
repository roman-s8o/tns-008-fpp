#!/usr/bin/env python3
"""
Switch from Nasdaq-100 to EUR/USD Forex Trading System

This script:
1. Backs up current data (optional archive)
2. Clears old Nasdaq price data
3. Fetches 10 years of EUR/USD data
4. Updates configuration for forex trading
5. Clears news database (will be refilled with forex news)
6. Resets processed data for rebuild

Usage:
    python scripts/switch_to_eurusd.py [--archive]
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.forex_prices import fetch_eurusd_prices, calculate_forex_returns, add_basic_indicators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/eurusd_conversion.log'),
    ]
)
logger = logging.getLogger(__name__)


def archive_current_data(archive_dir: str = "data/archive"):
    """
    Archive current Nasdaq data before replacement.
    
    Args:
        archive_dir: Directory to store archived data
    """
    logger.info("Archiving current Nasdaq-100 data...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = Path(archive_dir) / f"nasdaq_backup_{timestamp}"
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Archive price data
    prices_dir = Path("data/raw/prices")
    if prices_dir.exists():
        shutil.copytree(prices_dir, archive_path / "prices", dirs_exist_ok=True)
        logger.info(f"  ✓ Archived price data to {archive_path / 'prices'}")
    
    # Archive processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        shutil.copytree(processed_dir, archive_path / "processed", dirs_exist_ok=True)
        logger.info(f"  ✓ Archived processed data to {archive_path / 'processed'}")
    
    # Archive news database
    news_db = Path("data/raw/news.db")
    if news_db.exists():
        shutil.copy2(news_db, archive_path / "news.db")
        logger.info(f"  ✓ Archived news database to {archive_path / 'news.db'}")
    
    logger.info(f"✓ Archive complete: {archive_path}")
    return archive_path


def clear_old_data():
    """Clear old Nasdaq-100 data."""
    logger.info("Clearing old Nasdaq-100 data...")
    
    # Clear price data
    prices_dir = Path("data/raw/prices")
    if prices_dir.exists():
        for file in prices_dir.glob("*.csv"):
            if file.name != "eurusd_prices.csv":  # Keep EUR/USD if already exists
                file.unlink()
                logger.info(f"  ✓ Deleted {file.name}")
    
    # Clear processed data
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for file in processed_dir.glob("*.parquet"):
            file.unlink()
            logger.info(f"  ✓ Deleted {file.name}")
        
        # Also clear metadata
        metadata_file = processed_dir / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info(f"  ✓ Deleted metadata.json")
    
    logger.info("✓ Old data cleared")


def clear_news_database():
    """
    Clear news database for forex news refill.
    Keeps schema but removes articles.
    """
    logger.info("Clearing news database...")
    
    import sqlite3
    db_path = "data/raw/news.db"
    
    if Path(db_path).exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get count before
        cursor.execute("SELECT COUNT(*) FROM articles")
        count_before = cursor.fetchone()[0]
        
        # Clear articles
        cursor.execute("DELETE FROM articles")
        cursor.execute("DELETE FROM fetch_history")
        cursor.execute("DELETE FROM sources")
        
        conn.commit()
        conn.close()
        
        logger.info(f"  ✓ Cleared {count_before} articles from database")
        logger.info(f"  ✓ Database schema preserved for forex news")
    else:
        logger.info("  ℹ No news database found (will be created)")
    
    logger.info("✓ News database cleared")


def fetch_eurusd_data():
    """Fetch and prepare EUR/USD price data."""
    logger.info("Fetching 10 years of EUR/USD data...")
    
    # Fetch data
    df = fetch_eurusd_prices(years=10)
    
    if df.empty:
        logger.error("Failed to fetch EUR/USD data")
        return None
    
    # Calculate returns and indicators
    logger.info("Calculating returns and technical indicators...")
    df = calculate_forex_returns(df)
    df = add_basic_indicators(df)
    
    # Save processed version
    output_file = Path("data/raw/prices/eurusd_processed.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved processed EUR/USD data to {output_file}")
    
    return df


def update_configuration():
    """Update configuration files for EUR/USD forex trading."""
    logger.info("Updating configuration for EUR/USD forex trading...")
    
    config_file = Path("config/config.yaml")
    
    if config_file.exists():
        import yaml
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configuration
        config['asset'] = {
            'type': 'forex',
            'symbol': 'EURUSD',
            'name': 'EUR/USD',
            'base_currency': 'EUR',
            'quote_currency': 'USD',
        }
        
        config['data']['tickers'] = ['EURUSD']
        config['data']['years'] = 10
        
        # Update prediction buckets
        config['prediction'] = {
            'targets': ['direction', 'bucket'],
            'horizon': '1d',
            'buckets': {
                1: {'label': 'large_down', 'threshold': '<-0.5%'},
                2: {'label': 'small_down', 'threshold': '-0.5% to -0.2%'},
                3: {'label': 'flat', 'threshold': '-0.2% to +0.2%'},
                4: {'label': 'small_up', 'threshold': '+0.2% to +0.5%'},
                5: {'label': 'large_up', 'threshold': '>+0.5%'},
            }
        }
        
        # Update technical indicators
        config['indicators'] = {
            'enabled': ['SMA', 'RSI', 'MACD', 'BollingerBands', 'ATR', 
                       'Fibonacci', 'PivotPoints', 'Stochastic', 'CCI'],
            'fibonacci_period': 20,  # Swing high/low lookback
            'pivot_types': ['standard', 'fibonacci'],
        }
        
        # Save updated config
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"  ✓ Updated {config_file}")
    else:
        logger.warning(f"  ⚠ Config file not found: {config_file}")
    
    logger.info("✓ Configuration updated")


def display_summary(df):
    """Display summary of EUR/USD data."""
    logger.info("\n" + "="*60)
    logger.info("EUR/USD FOREX TRADING SYSTEM - DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"\n📊 Price Data:")
    logger.info(f"  - Total days: {len(df)}")
    logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  - Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
    
    logger.info(f"\n📈 Returns Statistics:")
    logger.info(f"  - Mean daily return: {df['returns'].mean():.4f}% ({df['pips_change'].mean():.2f} pips)")
    logger.info(f"  - Std daily return: {df['returns'].std():.4f}% ({df['pips_change'].std():.2f} pips)")
    logger.info(f"  - Max gain: {df['returns'].max():.4f}% ({df['pips_change'].max():.2f} pips)")
    logger.info(f"  - Max loss: {df['returns'].min():.4f}% ({df['pips_change'].min():.2f} pips)")
    
    logger.info(f"\n🎯 Prediction Targets (Next-day buckets):")
    bucket_dist = df['bucket_1d'].value_counts().sort_index()
    total = bucket_dist.sum()
    for bucket, count in bucket_dist.items():
        pct = count / total * 100
        logger.info(f"  - Bucket {int(bucket)}: {count} days ({pct:.1f}%)")
    
    logger.info(f"\n📉 Direction Balance:")
    direction_dist = df['direction_1d'].value_counts()
    if 0 in direction_dist.index and 1 in direction_dist.index:
        down_days = direction_dist[0]
        up_days = direction_dist[1]
        total_dir = down_days + up_days
        logger.info(f"  - Down days: {down_days} ({down_days/total_dir*100:.1f}%)")
        logger.info(f"  - Up days: {up_days} ({up_days/total_dir*100:.1f}%)")
    
    logger.info("\n" + "="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Switch from Nasdaq-100 to EUR/USD forex trading system'
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive current Nasdaq data before replacement'
    )
    parser.add_argument(
        '--skip-news-clear',
        action='store_true',
        help='Skip clearing news database (useful for testing)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("SWITCHING TO EUR/USD FOREX TRADING SYSTEM")
    logger.info("="*60)
    
    try:
        # Step 1: Archive if requested
        if args.archive:
            archive_path = archive_current_data()
            logger.info(f"\n✓ Step 1: Data archived to {archive_path}")
        else:
            logger.info("\n⚠ Step 1: Skipping archive (use --archive to save Nasdaq data)")
        
        # Step 2: Clear old data
        logger.info("\n📦 Step 2: Clearing old Nasdaq data...")
        clear_old_data()
        
        # Step 3: Fetch EUR/USD data
        logger.info("\n💱 Step 3: Fetching EUR/USD data...")
        df = fetch_eurusd_data()
        
        if df is None:
            logger.error("Failed to fetch EUR/USD data. Aborting.")
            return 1
        
        # Step 4: Clear news database
        if not args.skip_news_clear:
            logger.info("\n📰 Step 4: Clearing news database...")
            clear_news_database()
        else:
            logger.info("\n⚠ Step 4: Skipping news database clear")
        
        # Step 5: Update configuration
        logger.info("\n⚙️  Step 5: Updating configuration...")
        update_configuration()
        
        # Step 6: Display summary
        display_summary(df)
        
        logger.info("\n" + "="*60)
        logger.info("✅ CONVERSION COMPLETE!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Run forex news fetchers (Milestone continues...)")
        logger.info("2. Implement technical indicators")
        logger.info("3. Add economic calendar scraping")
        logger.info("4. Rebuild dataset with forex features")
        logger.info("5. Retrain models on EUR/USD data")
        logger.info("\nLog saved to: logs/eurusd_conversion.log")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error during conversion: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

