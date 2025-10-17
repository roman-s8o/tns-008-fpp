#!/usr/bin/env python3
"""
Fetch Forex-Relevant News (EUR/USD Focus)

Since dedicated forex RSS feeds are often protected, this script:
1. Uses existing general financial news sources (RSS feeds, NewsAPI, Alpha Vantage)
2. Filters for forex-relevant content (EUR, USD, Fed, ECB, currency, exchange rate keywords)
3. Stores in news database

This provides broader market context that affects EUR/USD movements.

Usage:
    python scripts/fetch_forex_news.py [--max-articles 200]
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

from src.data_ingestion.news_fetchers import (
    RSSFeedFetcher,
    AlphaVantageFetcher,
    NewsAPIFetcher
)
from src.data_ingestion.news_database import NewsDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/forex_news_ingestion.log'),
    ]
)
logger = logging.getLogger(__name__)


# Forex-relevant keywords for filtering
FOREX_KEYWORDS = [
    # Currencies
    'EUR', 'USD', 'euro', 'dollar', 'currency', 'forex', 'FX',
    'exchange rate', 'currency pair', 'EURUSD', 'EUR/USD',
    
    # Central banks
    'Fed', 'Federal Reserve', 'ECB', 'European Central Bank',
    'central bank', 'monetary policy', 'interest rate', 'rate decision',
    
    # Economic indicators
    'inflation', 'CPI', 'GDP', 'employment', 'NFP', 'nonfarm payrolls',
    'PMI', 'retail sales', 'consumer confidence', 'trade balance',
    
    # Market terms
    'currency market', 'forex trading', 'dollar strength', 'euro weakness',
    'safe haven', 'risk sentiment', 'volatility',
]


def is_forex_relevant(title: str, content: str) -> bool:
    """
    Check if article is relevant to forex/EUR-USD trading.
    
    Args:
        title: Article title
        content: Article content
        
    Returns:
        True if article contains forex-relevant keywords
    """
    text = (title + ' ' + content).lower()
    
    # Check for any forex keyword
    for keyword in FOREX_KEYWORDS:
        if keyword.lower() in text:
            return True
    
    return False


def fetch_forex_news(max_articles: int = 200) -> None:
    """
    Fetch forex-relevant news from multiple sources.
    
    Args:
        max_articles: Maximum number of articles to fetch
    """
    logger.info("="*60)
    logger.info("Fetching Forex-Relevant News for EUR/USD Trading")
    logger.info("="*60)
    
    # Initialize database
    db = NewsDatabase()
    
    # Initialize fetchers (uses built-in financial RSS feeds)
    rss_fetcher = RSSFeedFetcher()
    
    all_articles = []
    forex_articles = []
    
    # Fetch from RSS sources
    logger.info(f"\n📰 Fetching from RSS feeds...")
    try:
        logger.info(f"Fetching financial news from multiple sources...")
        articles = rss_fetcher.fetch_articles(limit=max_articles * 2)  # Fetch more to filter
        
        # Filter for forex relevance
        forex_filtered = [
            a for a in articles 
            if is_forex_relevant(a.get('title', ''), a.get('content', ''))
        ]
        
        logger.info(f"  Total: {len(articles)}, Forex-relevant: {len(forex_filtered)}")
        
        all_articles = articles
        forex_articles = forex_filtered
        
    except Exception as e:
        logger.error(f"  Error fetching RSS: {e}")
        all_articles = []
        forex_articles = []
    
    # Use filtered forex articles if we got enough, otherwise use all
    if len(forex_articles) >= 20:
        articles_to_store = forex_articles[:max_articles]
        logger.info(f"\n✓ Using {len(articles_to_store)} forex-filtered articles")
    else:
        articles_to_store = all_articles[:max_articles]
        logger.info(f"\n⚠ Only {len(forex_articles)} forex articles found, using all {len(articles_to_store)} articles")
    
    # Store in database
    if articles_to_store:
        logger.info(f"\n💾 Storing articles in database...")
        inserted, duplicates = db.bulk_insert_articles(articles_to_store)
        
        logger.info(f"\n✓ Storage complete:")
        logger.info(f"  - New articles: {inserted}")
        logger.info(f"  - Duplicates skipped: {duplicates}")
        logger.info(f"  - Total in database: {db.get_article_count()}")
    
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("FOREX NEWS FETCHING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total articles fetched: {len(all_articles)}")
    if len(all_articles) > 0:
        logger.info(f"Forex-relevant articles: {len(forex_articles)} ({len(forex_articles)/len(all_articles)*100:.1f}%)")
    else:
        logger.info(f"Forex-relevant articles: 0")
    logger.info(f"Articles stored: {len(articles_to_store)}")
    logger.info(f"\nForex keyword matches:")
    
    # Count keyword frequency
    keyword_counts = {}
    for article in forex_articles:
        text = (article.get('title', '') + ' ' + article.get('content', '')).lower()
        for keyword in FOREX_KEYWORDS:
            if keyword.lower() in text:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Show top 10 keywords
    top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for keyword, count in top_keywords:
        print(f"  {keyword}: {count}")
    
    logger.info("\n" + "="*60)
    
    # Note about dedicated forex feeds
    logger.info("\n📝 NOTE: Dedicated forex RSS feeds (ForexFactory, FXStreet, DailyFX)")
    logger.info("   often have anti-scraping protections. Using general financial")
    logger.info("   news with forex filtering provides good coverage of factors")
    logger.info("   affecting EUR/USD (Fed/ECB policy, economic data, market sentiment).")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch forex-relevant news for EUR/USD trading system'
    )
    parser.add_argument(
        '--max-articles',
        type=int,
        default=200,
        help='Maximum number of articles to fetch (default: 200)'
    )
    
    args = parser.parse_args()
    
    try:
        fetch_forex_news(max_articles=args.max_articles)
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

