"""
Multimodal sequence builder combining news and price data.

This module creates training sequences that combine financial news
articles with corresponding price movements.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.news_database import NewsDatabase
from src.data_ingestion.data_utils import load_ticker_data, get_available_tickers
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.price_processor import PriceProcessor
from loguru import logger
import yaml


class SequenceBuilder:
    """Builds multimodal sequences combining news and prices."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize sequence builder.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preproc_config = self.config.get('preprocessing', {})
        
        # Initialize components
        self.db = NewsDatabase()
        self.text_cleaner = TextCleaner(
            max_length=self.preproc_config['news'].get('max_article_length', 2000),
            include_metadata=self.preproc_config['news'].get('include_metadata', True),
            use_summary_fallback=self.preproc_config['news'].get('use_summary_fallback', True)
        )
        self.price_processor = PriceProcessor(
            calc_log_returns=self.preproc_config['prices'].get('calc_log_returns', True),
            calc_ohlc_returns=self.preproc_config['prices'].get('calc_ohlc_returns', True),
            calc_volume_change=self.preproc_config['prices'].get('calc_volume_change', True)
        )
        
        # Cache processed price data
        self.price_cache = {}
        
        logger.info("Initialized SequenceBuilder")
    
    def load_and_process_prices(
        self,
        tickers: Optional[List[str]] = None,
        price_dir: str = "data/raw/prices"
    ):
        """
        Load and process price data for tickers.
        
        Args:
            tickers: List of tickers to load (None = all available)
            price_dir: Directory containing price CSV files
        """
        if tickers is None:
            tickers = get_available_tickers(price_dir)
        
        logger.info(f"Loading price data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            df = load_ticker_data(ticker, price_dir)
            if df is not None:
                processed_df = self.price_processor.process_ticker_data(df)
                self.price_cache[ticker] = processed_df
        
        logger.info(f"Loaded and processed {len(self.price_cache)} tickers")
    
    def get_news_for_date(
        self,
        date: datetime,
        ticker: Optional[str] = None
    ) -> List[Dict]:
        """
        Get news articles for a specific date and ticker.
        
        Args:
            date: Target date
            ticker: Optional ticker symbol
            
        Returns:
            List of article dictionaries
        """
        # Query database for articles on this date
        start_date = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_date = datetime(date.year, date.month, date.day, 23, 59, 59)
        
        articles = self.db.get_articles(
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        # Filter by ticker if specified
        if ticker and self.preproc_config['ticker_matching'].get('use_ticker_specific', True):
            ticker_articles = [
                a for a in articles 
                if a.get('tickers') and ticker in a.get('tickers', [])
            ]
            
            # Use ticker-specific news if available
            if ticker_articles:
                return ticker_articles
            
            # Fall back to general market news
            if self.preproc_config['ticker_matching'].get('use_market_fallback', True):
                return articles
            
            return []
        
        return articles
    
    def create_sequence(
        self,
        ticker: str,
        date: datetime,
        articles: List[Dict],
        price_features: Dict
    ) -> Optional[Dict]:
        """
        Create a multimodal sequence from news and price data.
        
        Args:
            ticker: Stock ticker
            date: Date for the sequence
            articles: List of news articles
            price_features: Price feature dictionary
            
        Returns:
            Sequence dictionary, or None if invalid
        """
        if not articles or not price_features:
            return None
        
        # Clean and concatenate articles
        if self.preproc_config['news'].get('concat_articles', True):
            separator = self.preproc_config['news'].get('article_separator', ' [SEP] ')
            text = self.text_cleaner.clean_articles_batch(articles, separator=separator)
        else:
            # Use only the first article
            article = articles[0]
            text = self.text_cleaner.clean_article(
                title=article.get('title', ''),
                content=article.get('content'),
                summary=article.get('summary'),
                source=article.get('source'),
                published_at=str(article.get('published_at', ''))
            )
        
        if not text or len(text) < self.preproc_config['news'].get('min_length', 50):
            return None
        
        # Format the sequence according to config
        sequence_format = self.preproc_config['sequences'].get('format', '')
        
        # Build the complete sequence
        sequence = {
            'ticker': ticker,
            'date': date.strftime('%Y-%m-%d'),
            'news_text': text,
            'news_count': len(articles),
            'price_features': price_features,
            # Create formatted text sequence
            'sequence_text': self._format_sequence(text, price_features),
            'metadata': {
                'sources': list(set([a.get('source', '') for a in articles])),
                'article_count': len(articles)
            }
        }
        
        return sequence
    
    def _format_sequence(self, news_text: str, price_features: Dict) -> str:
        """
        Format a sequence according to configuration.
        
        Args:
            news_text: Cleaned news text
            price_features: Price feature dictionary
            
        Returns:
            Formatted sequence string
        """
        # Build returns string
        returns_parts = []
        if 'log_return' in price_features:
            returns_parts.append(f"log={price_features['log_return']:.2f}%")
        if 'open_close_return' in price_features:
            returns_parts.append(f"oc={price_features['open_close_return']:.2f}%")
        if 'high_low_range' in price_features:
            returns_parts.append(f"hl={price_features['high_low_range']:.2f}%")
        
        returns_str = ", ".join(returns_parts)
        
        # Format sequence
        sequence = (
            f"News: {news_text}; "
            f"Price: open={price_features['open']:.2f}, "
            f"high={price_features['high']:.2f}, "
            f"low={price_features['low']:.2f}, "
            f"close={price_features['close']:.2f}, "
            f"volume={int(price_features['volume'])}, "
            f"returns=({returns_str})"
        )
        
        return sequence
    
    def build_dataset(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_sequences: int = 10000
    ) -> List[Dict]:
        """
        Build a dataset of sequences.
        
        Args:
            tickers: List of tickers to include
            start_date: Start date for sequences
            end_date: End date for sequences  
            max_sequences: Maximum number of sequences to generate
            
        Returns:
            List of sequence dictionaries
        """
        if not self.price_cache:
            logger.warning("Price cache is empty. Loading all available tickers...")
            self.load_and_process_prices()
        
        if tickers is None:
            tickers = list(self.price_cache.keys())
        
        # Filter tickers based on minimum news requirement
        min_news = self.preproc_config['ticker_matching'].get('min_news_per_ticker', 5)
        
        sequences = []
        ticker_counts = {}
        
        logger.info(f"Building dataset from {len(tickers)} tickers...")
        
        for ticker in tickers:
            if ticker not in self.price_cache:
                continue
            
            price_df = self.price_cache[ticker]
            
            # Filter by date range if specified
            if start_date:
                price_df = price_df[price_df['Date'] >= start_date]
            if end_date:
                price_df = price_df[price_df['Date'] <= end_date]
            
            ticker_sequences = 0
            
            # Iterate through dates
            for _, row in price_df.iterrows():
                if len(sequences) >= max_sequences:
                    break
                
                date = pd.to_datetime(row['Date'])
                
                # Get news for this date
                articles = self.get_news_for_date(date, ticker)
                
                if not articles:
                    continue
                
                # Get price features
                price_features = self.price_processor.get_price_features_for_date(
                    price_df, date, ticker
                )
                
                if not price_features:
                    continue
                
                # Create sequence
                sequence = self.create_sequence(ticker, date, articles, price_features)
                
                if sequence:
                    sequences.append(sequence)
                    ticker_sequences += 1
            
            ticker_counts[ticker] = ticker_sequences
            
            if len(sequences) >= max_sequences:
                break
        
        logger.info(f"Built {len(sequences)} sequences from {len(ticker_counts)} tickers")
        logger.debug(f"Sequences per ticker: {ticker_counts}")
        
        return sequences
    
    def save_dataset(self, sequences: List[Dict], output_path: str):
        """
        Save dataset to disk.
        
        Args:
            sequences: List of sequences
            output_path: Output file path
        """
        df = pd.DataFrame(sequences)
        
        # Save as CSV
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        # Save as JSON
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        # Save as parquet (more efficient)
        elif output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            df.to_csv(output_path + '.csv', index=False)
        
        logger.info(f"Saved {len(sequences)} sequences to {output_path}")


if __name__ == "__main__":
    # Test sequence builder
    builder = SequenceBuilder()
    
    # Load price data
    builder.load_and_process_prices(tickers=['AAPL', 'MSFT', 'GOOGL'])
    
    # Build sample dataset
    sequences = builder.build_dataset(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        max_sequences=10
    )
    
    print(f"\nBuilt {len(sequences)} sequences")
    
    if sequences:
        print(f"\nSample sequence:")
        sample = sequences[0]
        print(f"Ticker: {sample['ticker']}")
        print(f"Date: {sample['date']}")
        print(f"News count: {sample['news_count']}")
        print(f"News (first 200 chars): {sample['news_text'][:200]}...")
        print(f"Price features: {sample['price_features']}")

