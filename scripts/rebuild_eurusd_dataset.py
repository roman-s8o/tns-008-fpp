#!/usr/bin/env python3
"""
Rebuild Dataset for EUR/USD Forex Trading

Combines:
- EUR/USD price data with 50 technical indicators
- Forex-relevant news with sentiment, NER, topics
- Economic calendar events and features

Creates train/validation/finetune splits for model training.

Usage:
    python scripts/rebuild_eurusd_dataset.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data_ingestion.forex_prices import fetch_eurusd_prices, calculate_forex_returns
from src.preprocessing.technical_indicators import TechnicalIndicators
from src.data_ingestion.economic_calendar import EconomicCalendar
from src.data_ingestion.news_database import NewsDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/dataset_rebuild.log'),
    ]
)
logger = logging.getLogger(__name__)


class EURUSDDatasetBuilder:
    """Build EUR/USD forex trading dataset."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.db = NewsDatabase()
        self.calendar = EconomicCalendar()
    
    def build_complete_dataset(self):
        """Build complete EUR/USD dataset with all features."""
        logger.info("="*60)
        logger.info("Building EUR/USD Forex Trading Dataset")
        logger.info("="*60)
        
        # Step 1: Load EUR/USD price data with indicators
        logger.info("\n📊 Step 1: Loading EUR/USD price data...")
        df = fetch_eurusd_prices(years=10)
        df = calculate_forex_returns(df)
        df = TechnicalIndicators.add_all_indicators(df, fibonacci_lookback=20)
        logger.info(f"  ✓ Loaded {len(df)} days with {len(df.columns)} price/indicator columns")
        
        # Step 2: Add economic calendar features
        logger.info("\n📅 Step 2: Adding economic calendar features...")
        start_date = pd.to_datetime(df['date']).min()
        end_date = pd.to_datetime(df['date']).max()
        events = self.calendar.fetch_calendar_data(start_date, end_date, mock_data=True)
        df = self.calendar.create_calendar_features(df, events)
        logger.info(f"  ✓ Added calendar features. Total columns: {len(df.columns)}")
        
        # Step 3: Get news data
        logger.info("\n📰 Step 3: Loading news data...")
        all_articles = self.db.get_articles(limit=10000)
        logger.info(f"  ✓ Loaded {len(all_articles)} news articles")
        
        # Step 4: Build sequences (align news to trading days)
        logger.info("\n🔗 Step 4: Building news-price sequences...")
        sequences = self._build_sequences(df, all_articles)
        logger.info(f"  ✓ Built {len(sequences)} sequences")
        
        # Step 5: Create splits
        logger.info("\n✂️  Step 5: Creating train/val/finetune splits...")
        train, val, finetune = self._create_splits(sequences)
        logger.info(f"  ✓ Train: {len(train)}, Val: {len(val)}, Finetune: {len(finetune)}")
        
        # Step 6: Save datasets
        logger.info("\n💾 Step 6: Saving datasets...")
        self._save_datasets(train, val, finetune, sequences)
        
        logger.info("\n" + "="*60)
        logger.info("✅ EUR/USD Dataset Build Complete!")
        logger.info("="*60)
        
        return {
            'train': train,
            'validation': val,
            'finetune': finetune,
            'total_sequences': len(sequences),
        }
    
    def _build_sequences(self, price_df: pd.DataFrame, articles: list) -> pd.DataFrame:
        """
        Build sequences by aligning news to each trading day.
        
        Each sequence contains:
        - Price data for the day
        - All technical indicators
        - Calendar features
        - News from that day (aggregated)
        - Sentiment/NER/topic features
        - Prediction targets (next-day movement)
        """
        sequences = []
        
        # Convert articles to DataFrame for easier filtering
        if articles:
            news_df = pd.DataFrame(articles)
            news_df['published_at'] = pd.to_datetime(news_df['published_at'])
        else:
            news_df = pd.DataFrame()
        
        for _, row in tqdm(price_df.iterrows(), total=len(price_df), desc="Building sequences"):
            date = pd.to_datetime(row['date'])
            
            # Get news for this day
            if not news_df.empty:
                day_news = news_df[news_df['published_at'].dt.date == date.date()]
            else:
                day_news = pd.DataFrame()
            
            # Aggregate news text
            if len(day_news) > 0:
                news_text = " ".join(day_news['title'].fillna('').astype(str).tolist()[:5])  # Top 5 headlines
                news_count = len(day_news)
                
                # Aggregate sentiment (average)
                if 'finbert_sentiment_score' in day_news.columns:
                    avg_sentiment = day_news['finbert_sentiment_score'].mean()
                else:
                    avg_sentiment = 0.0
                
                # Aggregate entities
                if 'entity_tickers' in day_news.columns:
                    all_tickers = []
                    for tickers in day_news['entity_tickers'].fillna(''):
                        if tickers:
                            all_tickers.extend(tickers.split(','))
                    unique_tickers = list(set(all_tickers))
                    ticker_mentions = len(unique_tickers)
                else:
                    ticker_mentions = 0
                
                # Primary topic (most common)
                if 'primary_topic_name' in day_news.columns:
                    topics = day_news['primary_topic_name'].value_counts()
                    primary_topic = topics.index[0] if len(topics) > 0 else "Unknown"
                else:
                    primary_topic = "Unknown"
            else:
                news_text = ""
                news_count = 0
                avg_sentiment = 0.0
                ticker_mentions = 0
                primary_topic = "No news"
            
            # Build sequence
            sequence = {
                'date': date,
                'ticker': 'EURUSD',
                
                # Price data
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'returns': row['returns'],
                'pips_change': row['pips_change'],
                
                # Technical indicators (key ones)
                'sma_20': row.get('sma_20', np.nan),
                'sma_50': row.get('sma_50', np.nan),
                'rsi_14': row.get('rsi_14', np.nan),
                'macd': row.get('macd', np.nan),
                'bb_percent': row.get('bb_percent', np.nan),
                'atr_14': row.get('atr_14', np.nan),
                'stoch_k': row.get('stoch_k', np.nan),
                'cci_20': row.get('cci_20', np.nan),
                
                # Calendar features
                'major_events_today': row.get('major_events_today', 0),
                'events_this_week': row.get('events_this_week', 0),
                'fed_today': row.get('fed_today', 0),
                'ecb_today': row.get('ecb_today', 0),
                'nfp_today': row.get('nfp_today', 0),
                'days_to_fed_rate': row.get('days_to_fed_rate', 999),
                'days_to_ecb_rate': row.get('days_to_ecb_rate', 999),
                'days_to_nfp': row.get('days_to_nfp', 999),
                
                # News features
                'news_text': news_text,
                'news_count': news_count,
                'avg_sentiment': avg_sentiment,
                'ticker_mentions': ticker_mentions,
                'primary_topic': primary_topic,
                
                # Prediction targets
                'direction_1d': row.get('direction_1d', np.nan),
                'returns_1d': row.get('returns_1d', np.nan),
                'bucket_1d': row.get('bucket_1d', np.nan),
            }
            
            sequences.append(sequence)
        
        return pd.DataFrame(sequences)
    
    def _create_splits(self, df: pd.DataFrame) -> tuple:
        """
        Create temporal train/validation/finetune splits.
        
        - Train: 80% (oldest data)
        - Validation: 10% (middle data)
        - Finetune: 10% (most recent data)
        """
        # Remove rows with NaN targets
        df = df.dropna(subset=['direction_1d', 'bucket_1d'])
        
        # Sort by date
        df = df.sort_values('date')
        
        n = len(df)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        finetune = df.iloc[val_end:].copy()
        
        # Add split column
        train['split'] = 'train'
        val['split'] = 'validation'
        finetune['split'] = 'finetune'
        
        return train, val, finetune
    
    def _save_datasets(self, train, val, finetune, full_df):
        """Save datasets to parquet files."""
        train.to_parquet(self.output_dir / 'train.parquet', index=False)
        val.to_parquet(self.output_dir / 'validation.parquet', index=False)
        finetune.to_parquet(self.output_dir / 'finetune.parquet', index=False)
        
        # Save combined dataset
        full_df.to_parquet(self.output_dir / 'full_dataset.parquet', index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'asset': 'EURUSD',
            'asset_type': 'forex',
            'total_sequences': len(full_df),
            'train_size': len(train),
            'validation_size': len(val),
            'finetune_size': len(finetune),
            'date_range': {
                'start': str(full_df['date'].min()),
                'end': str(full_df['date'].max()),
            },
            'features': {
                'price_columns': 7,
                'technical_indicators': 8,
                'calendar_features': 8,
                'news_features': 5,
                'total_columns': len(full_df.columns),
            },
            'prediction_targets': ['direction_1d', 'returns_1d', 'bucket_1d'],
            'bucket_distribution': full_df['bucket_1d'].value_counts().to_dict(),
        }
        
        import json
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"  ✓ Saved train.parquet ({len(train)} sequences)")
        logger.info(f"  ✓ Saved validation.parquet ({len(val)} sequences)")
        logger.info(f"  ✓ Saved finetune.parquet ({len(finetune)} sequences)")
        logger.info(f"  ✓ Saved full_dataset.parquet ({len(full_df)} sequences)")
        logger.info(f"  ✓ Saved metadata.json")


def main():
    """Main entry point."""
    try:
        builder = EURUSDDatasetBuilder()
        results = builder.build_complete_dataset()
        
        print("\n" + "="*60)
        print("Dataset Summary")
        print("="*60)
        print(f"Total sequences: {results['total_sequences']}")
        print(f"Training: {len(results['train'])}")
        print(f"Validation: {len(results['validation'])}")
        print(f"Fine-tuning: {len(results['finetune'])}")
        print("\nFiles saved to: data/processed/")
        print("="*60)
        
        return 0
    except Exception as e:
        logger.error(f"❌ Dataset build failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

