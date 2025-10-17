"""
Forex News Fetchers Module

This module fetches forex-specific news from multiple free sources:
- ForexFactory RSS
- FXStreet RSS  
- DailyFX RSS
- Investing.com Forex section

Designed to fetch 50-200 forex articles per day.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
import re

logger = logging.getLogger(__name__)


class ForexNewsFetcher:
    """Base class for forex news fetchers."""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def fetch_news(self) -> List[Dict]:
        """Fetch news from source. To be implemented by subclasses."""
        raise NotImplementedError
    
    def _clean_text(self, text: str) -> str:
        """Clean HTML and whitespace from text."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        
        try:
            # Try common RSS date formats
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str)
        except:
            try:
                # Try ISO format
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                logger.warning(f"Could not parse date: {date_str}")
                return None


class ForexFactoryFetcher(ForexNewsFetcher):
    """
    Fetch news from ForexFactory RSS feed.
    
    ForexFactory is a popular forex community with news and economic calendar.
    """
    
    RSS_URL = "https://www.forexfactory.com/news.xml"
    
    def __init__(self):
        super().__init__("ForexFactory")
    
    def fetch_news(self) -> List[Dict]:
        """Fetch news from ForexFactory RSS."""
        logger.info(f"Fetching news from {self.source_name}...")
        
        try:
            feed = feedparser.parse(self.RSS_URL)
            
            articles = []
            for entry in feed.entries:
                article = {
                    'title': self._clean_text(entry.get('title', '')),
                    'content': self._clean_text(entry.get('summary', '') or entry.get('description', '')),
                    'url': entry.get('link', ''),
                    'published_at': self._parse_date(entry.get('published', '')),
                    'source': self.source_name,
                    'author': entry.get('author', None),
                }
                
                # Only add if we have title and date
                if article['title'] and article['published_at']:
                    articles.append(article)
            
            logger.info(f"  ✓ Fetched {len(articles)} articles from {self.source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {self.source_name}: {e}")
            return []


class FXStreetFetcher(ForexNewsFetcher):
    """
    Fetch news from FXStreet RSS feed.
    
    FXStreet provides forex news, analysis, and charts.
    """
    
    RSS_URL = "https://www.fxstreet.com/news/feed"
    
    def __init__(self):
        super().__init__("FXStreet")
    
    def fetch_news(self) -> List[Dict]:
        """Fetch news from FXStreet RSS."""
        logger.info(f"Fetching news from {self.source_name}...")
        
        try:
            feed = feedparser.parse(self.RSS_URL)
            
            articles = []
            for entry in feed.entries:
                article = {
                    'title': self._clean_text(entry.get('title', '')),
                    'content': self._clean_text(entry.get('summary', '') or entry.get('description', '')),
                    'url': entry.get('link', ''),
                    'published_at': self._parse_date(entry.get('published', '')),
                    'source': self.source_name,
                    'author': entry.get('author', None),
                }
                
                if article['title'] and article['published_at']:
                    articles.append(article)
            
            logger.info(f"  ✓ Fetched {len(articles)} articles from {self.source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {self.source_name}: {e}")
            return []


class DailyFXFetcher(ForexNewsFetcher):
    """
    Fetch news from DailyFX RSS feed.
    
    DailyFX is IG Group's forex and CFD website providing news and analysis.
    """
    
    RSS_URLS = [
        "https://www.dailyfx.com/feeds/market-news",
        "https://www.dailyfx.com/feeds/forex-news",
    ]
    
    def __init__(self):
        super().__init__("DailyFX")
    
    def fetch_news(self) -> List[Dict]:
        """Fetch news from DailyFX RSS feeds."""
        logger.info(f"Fetching news from {self.source_name}...")
        
        all_articles = []
        
        for rss_url in self.RSS_URLS:
            try:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    article = {
                        'title': self._clean_text(entry.get('title', '')),
                        'content': self._clean_text(entry.get('summary', '') or entry.get('description', '')),
                        'url': entry.get('link', ''),
                        'published_at': self._parse_date(entry.get('published', '')),
                        'source': self.source_name,
                        'author': entry.get('author', None),
                    }
                    
                    if article['title'] and article['published_at']:
                        all_articles.append(article)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching from {rss_url}: {e}")
                continue
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        logger.info(f"  ✓ Fetched {len(unique_articles)} articles from {self.source_name}")
        return unique_articles


class InvestingDotComForexFetcher(ForexNewsFetcher):
    """
    Fetch forex news from Investing.com.
    
    Investing.com has comprehensive forex news coverage.
    Note: This uses RSS feed which may have limited articles.
    """
    
    RSS_URL = "https://www.investing.com/rss/forex_news.rss"
    
    def __init__(self):
        super().__init__("Investing.com Forex")
    
    def fetch_news(self) -> List[Dict]:
        """Fetch news from Investing.com forex RSS."""
        logger.info(f"Fetching news from {self.source_name}...")
        
        try:
            feed = feedparser.parse(self.RSS_URL)
            
            articles = []
            for entry in feed.entries:
                article = {
                    'title': self._clean_text(entry.get('title', '')),
                    'content': self._clean_text(entry.get('summary', '') or entry.get('description', '')),
                    'url': entry.get('link', ''),
                    'published_at': self._parse_date(entry.get('published', '')),
                    'source': self.source_name,
                    'author': entry.get('author', None),
                }
                
                if article['title'] and article['published_at']:
                    articles.append(article)
            
            logger.info(f"  ✓ Fetched {len(articles)} articles from {self.source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching from {self.source_name}: {e}")
            return []


class ForexNewsAggregator:
    """
    Aggregates news from multiple forex sources.
    """
    
    def __init__(self):
        self.fetchers = [
            ForexFactoryFetcher(),
            FXStreetFetcher(),
            DailyFXFetcher(),
            InvestingDotComForexFetcher(),
        ]
    
    def fetch_all_news(self, days_back: int = 1) -> List[Dict]:
        """
        Fetch news from all sources.
        
        Args:
            days_back: Number of days to look back for news
            
        Returns:
            List of article dictionaries
        """
        logger.info(f"Fetching forex news from {len(self.fetchers)} sources...")
        
        all_articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for fetcher in self.fetchers:
            try:
                articles = fetcher.fetch_news()
                
                # Filter by date
                recent_articles = [
                    a for a in articles 
                    if a.get('published_at') and a['published_at'] >= cutoff_date
                ]
                
                all_articles.extend(recent_articles)
                
                # Rate limiting between sources
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error with {fetcher.source_name}: {e}")
                continue
        
        # Remove duplicates by URL and title
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Add forex-specific tags
        for article in unique_articles:
            article['tickers'] = self._extract_forex_pairs(article['title'] + ' ' + article.get('content', ''))
        
        logger.info(f"✓ Total unique forex articles: {len(unique_articles)}")
        return unique_articles
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by URL and similar titles."""
        seen_urls = set()
        seen_titles = set()
        unique = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            # Check URL
            if url and url in seen_urls:
                continue
            
            # Check title similarity (simple exact match)
            if title in seen_titles:
                continue
            
            seen_urls.add(url)
            seen_titles.add(title)
            unique.append(article)
        
        return unique
    
    def _extract_forex_pairs(self, text: str) -> List[str]:
        """
        Extract forex pairs from text.
        
        Looks for patterns like EUR/USD, EURUSD, etc.
        """
        if not text:
            return []
        
        pairs = []
        
        # Common currency codes
        currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 
                     'CNY', 'HKD', 'SGD', 'INR', 'KRW', 'BRL', 'MXN', 'RUB']
        
        # Pattern 1: EUR/USD
        pattern1 = r'\b(' + '|'.join(currencies) + r')/(' + '|'.join(currencies) + r')\b'
        matches1 = re.findall(pattern1, text.upper())
        for base, quote in matches1:
            if base != quote:
                pairs.append(f"{base}/USD" if quote == 'USD' else f"{base}/{quote}")
        
        # Pattern 2: EURUSD (6 letters, 3+3)
        pattern2 = r'\b(' + '|'.join(currencies) + ')(' + '|'.join(currencies) + r')\b'
        matches2 = re.findall(pattern2, text.upper())
        for base, quote in matches2:
            if base != quote and len(base) == 3 and len(quote) == 3:
                pairs.append(f"{base}/{quote}")
        
        # Always include EURUSD if we're focused on it
        if 'EUR' in text.upper() or 'EURO' in text.upper() or 'DOLLAR' in text.upper():
            pairs.append('EURUSD')
        
        return list(set(pairs))


def test_forex_fetchers():
    """Test function to verify forex news fetching."""
    logging.basicConfig(level=logging.INFO)
    
    aggregator = ForexNewsAggregator()
    articles = aggregator.fetch_all_news(days_back=7)
    
    print("\n" + "="*60)
    print(f"Forex News Fetching Test")
    print("="*60)
    print(f"\nTotal articles fetched: {len(articles)}")
    
    # Count by source
    by_source = {}
    for article in articles:
        source = article.get('source', 'Unknown')
        by_source[source] = by_source.get(source, 0) + 1
    
    print(f"\nArticles by source:")
    for source, count in sorted(by_source.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")
    
    # Show sample articles
    print(f"\nSample articles:")
    for i, article in enumerate(articles[:5], 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Date: {article['published_at']}")
        print(f"   Pairs: {', '.join(article.get('tickers', []))}")
        print(f"   URL: {article['url'][:60]}...")


if __name__ == "__main__":
    test_forex_fetchers()

