#!/usr/bin/env python3
"""
Script to fetch financial news from multiple sources and store in SQLite.

Usage:
    python scripts/fetch_news.py [--limit 2000] [--sources rss,newsapi] [--days 1]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.news_database import NewsDatabase
from src.data_ingestion.news_fetchers import (
    RSSFeedFetcher,
    NewsAPIFetcher,
    AlphaVantageFetcher
)
from loguru import logger


class NewsAggregator:
    """Aggregates news from multiple sources."""
    
    def __init__(self, db_path: str = "data/raw/news.db"):
        """
        Initialize news aggregator.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db = NewsDatabase(db_path)
        self.fetchers = {}
        
        # Initialize available fetchers
        self._init_fetchers()
    
    def _init_fetchers(self):
        """Initialize news fetchers."""
        # RSS feeds (always available)
        self.fetchers['rss'] = RSSFeedFetcher()
        
        # NewsAPI (requires API key)
        newsapi = NewsAPIFetcher()
        if newsapi.api_key:
            self.fetchers['newsapi'] = newsapi
            logger.info("NewsAPI fetcher initialized")
        else:
            logger.warning("NewsAPI not available (no API key)")
        
        # Alpha Vantage (requires API key)
        alphavantage = AlphaVantageFetcher()
        if alphavantage.api_key:
            self.fetchers['alphavantage'] = alphavantage
            logger.info("Alpha Vantage fetcher initialized")
        else:
            logger.warning("Alpha Vantage not available (no API key)")
        
        logger.info(f"Initialized {len(self.fetchers)} news fetchers")
    
    def fetch_all_sources(
        self,
        target_count: int = 2000,
        from_date: Optional[datetime] = None,
        sources: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Fetch news from all available sources.
        
        Args:
            target_count: Target number of articles to fetch
            from_date: Fetch articles from this date
            sources: List of specific sources to use (None = all)
            
        Returns:
            Dictionary with fetch statistics
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=1)
        
        # Filter sources if specified
        active_fetchers = self.fetchers
        if sources:
            active_fetchers = {
                k: v for k, v in self.fetchers.items() 
                if k in sources
            }
        
        if not active_fetchers:
            logger.error("No active fetchers available")
            return {}
        
        total_fetched = 0
        total_inserted = 0
        total_duplicates = 0
        stats = {}
        
        # Calculate articles per source
        articles_per_source = target_count // len(active_fetchers)
        
        for source_name, fetcher in active_fetchers.items():
            try:
                logger.info(f"Fetching from {source_name}...")
                
                # Fetch articles
                articles = fetcher.fetch_articles(
                    from_date=from_date,
                    limit=articles_per_source
                )
                
                if not articles:
                    logger.warning(f"No articles fetched from {source_name}")
                    continue
                
                # Insert into database
                inserted, duplicates = self.db.bulk_insert_articles(articles)
                
                # Record fetch
                self.db.record_fetch(
                    source=source_name,
                    articles_fetched=len(articles),
                    success=True
                )
                
                # Update statistics
                total_fetched += len(articles)
                total_inserted += inserted
                total_duplicates += duplicates
                
                stats[source_name] = {
                    'fetched': len(articles),
                    'inserted': inserted,
                    'duplicates': duplicates
                }
                
                logger.info(
                    f"{source_name}: {inserted} inserted, "
                    f"{duplicates} duplicates"
                )
                
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
                self.db.record_fetch(
                    source=source_name,
                    articles_fetched=0,
                    success=False,
                    error_message=str(e)
                )
                continue
        
        # Overall statistics
        stats['total'] = {
            'fetched': total_fetched,
            'inserted': total_inserted,
            'duplicates': total_duplicates
        }
        
        return stats
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return self.db.get_statistics()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch financial news from multiple sources"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Target number of articles to fetch (default: 200)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Fetch articles from last N days (default: 1)"
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated list of sources (rss,newsapi,alphavantage)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/raw/news.db",
        help="Path to SQLite database (default: data/raw/news.db)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only fetch 10 articles"
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
        "logs/news_ingestion.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    # Parse sources
    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(',')]
    
    # Set limit for test mode
    limit = 10 if args.test else args.limit
    
    if args.test:
        logger.warning(f"TEST MODE: Only fetching {limit} articles")
    
    # Calculate from_date
    from_date = datetime.now() - timedelta(days=args.days)
    
    # Initialize aggregator
    logger.info(f"Initializing news aggregator...")
    logger.info(f"  Target articles: {limit}")
    logger.info(f"  From date: {from_date.date()}")
    logger.info(f"  Database: {args.db_path}")
    
    aggregator = NewsAggregator(db_path=args.db_path)
    
    # Show initial stats
    initial_stats = aggregator.get_database_stats()
    logger.info(f"Initial database: {initial_stats['total_articles']} articles")
    
    # Fetch news
    logger.info("\n" + "="*60)
    logger.info("STARTING NEWS FETCH")
    logger.info("="*60 + "\n")
    
    start_time = datetime.now()
    
    fetch_stats = aggregator.fetch_all_sources(
        target_count=limit,
        from_date=from_date,
        sources=sources
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Show results
    logger.info("\n" + "="*60)
    logger.info("FETCH RESULTS")
    logger.info("="*60)
    
    for source, stats in fetch_stats.items():
        if source == 'total':
            continue
        logger.info(
            f"{source}: {stats['inserted']} new, "
            f"{stats['duplicates']} duplicates"
        )
    
    total = fetch_stats.get('total', {})
    logger.info("\n" + "-"*60)
    logger.info(f"Total fetched: {total.get('fetched', 0)}")
    logger.info(f"Total inserted: {total.get('inserted', 0)}")
    logger.info(f"Total duplicates: {total.get('duplicates', 0)}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("-"*60)
    
    # Show final stats
    final_stats = aggregator.get_database_stats()
    logger.info(f"\nFinal database: {final_stats['total_articles']} articles")
    logger.info(f"Articles added: {final_stats['total_articles'] - initial_stats['total_articles']}")
    
    # Success criteria for Milestone 3
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    
    inserted = total.get('inserted', 0)
    
    if inserted >= limit * 0.5:  # At least 50% of target
        print(f"✅ SUCCESS: Fetched {inserted} new articles")
        if duration < 900:  # < 15 minutes
            print(f"✅ PERFORMANCE: Completed in {duration:.1f}s (< 15 min)")
        else:
            print(f"⚠️  PERFORMANCE: Took {duration:.1f}s (> 15 min target)")
        return 0
    else:
        print(f"❌ FAILURE: Only {inserted} new articles (target: {limit})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

