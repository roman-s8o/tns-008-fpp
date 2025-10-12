"""
News fetchers for various sources.

This module implements fetchers for different news APIs and RSS feeds.
"""

import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import feedparser
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class NewsFetcher(ABC):
    """Abstract base class for news fetchers."""
    
    def __init__(self, source_name: str):
        """
        Initialize news fetcher.
        
        Args:
            source_name: Name of the news source
        """
        self.source_name = source_name
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        return session
    
    @abstractmethod
    def fetch_articles(
        self,
        query: Optional[str] = None,
        from_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch articles from the news source.
        
        Args:
            query: Search query
            from_date: Fetch articles from this date
            limit: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
        pass


class NewsAPIFetcher(NewsFetcher):
    """Fetcher for NewsAPI.org (free tier: 100 requests/day)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI fetcher.
        
        Args:
            api_key: NewsAPI API key (or set NEWSAPI_KEY env var)
        """
        super().__init__("newsapi")
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        
        if not self.api_key:
            logger.warning("NewsAPI key not found. Set NEWSAPI_KEY environment variable.")
    
    def fetch_articles(
        self,
        query: Optional[str] = "stock OR market OR trading OR nasdaq OR finance",
        from_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch financial news from NewsAPI.
        
        Args:
            query: Search query (default: financial keywords)
            from_date: Fetch articles from this date
            limit: Maximum number of articles (max 100 per request)
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            logger.error("NewsAPI key required")
            return []
        
        # Default to last 24 hours
        if from_date is None:
            from_date = datetime.now() - timedelta(days=1)
        
        try:
            url = f"{self.base_url}/everything"
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': min(limit, 100),  # Max 100 per request
                'apiKey': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = []
            for item in data.get('articles', []):
                # Parse published date
                published_at = datetime.fromisoformat(
                    item['publishedAt'].replace('Z', '+00:00')
                )
                
                article = {
                    'title': item.get('title', ''),
                    'content': item.get('content') or item.get('description', ''),
                    'summary': item.get('description', ''),
                    'source': f"newsapi_{item['source']['name']}",
                    'author': item.get('author'),
                    'published_at': published_at,
                    'url': item.get('url'),
                    'tickers': self._extract_tickers(item.get('title', '') + ' ' + item.get('description', ''))
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []
    
    @staticmethod
    def _extract_tickers(text: str) -> List[str]:
        """Extract potential stock tickers from text."""
        # Simple ticker extraction (can be improved with NLP)
        import re
        tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        # Filter common false positives
        excluded = {'THE', 'AND', 'FOR', 'NOT', 'BUT', 'CEO', 'CFO', 'IPO', 'ETF', 'USA', 'USD'}
        return [t for t in tickers if t not in excluded][:10]  # Max 10 tickers


class RSSFeedFetcher(NewsFetcher):
    """Fetcher for RSS feeds (unlimited, free)."""
    
    # Popular financial news RSS feeds
    FINANCIAL_FEEDS = {
        'reuters_business': 'https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best',
        'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
        'marketwatch': 'https://www.marketwatch.com/rss/topstories',
        'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
        'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
        'bloomberg': 'https://www.bloomberg.com/feed/podcast/bloomberg-markets.xml',
        'ft': 'https://www.ft.com/?format=rss',
        'wsj_markets': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    }
    
    def __init__(self, feed_urls: Optional[Dict[str, str]] = None):
        """
        Initialize RSS feed fetcher.
        
        Args:
            feed_urls: Dictionary of feed names to URLs (uses defaults if None)
        """
        super().__init__("rss_feeds")
        self.feed_urls = feed_urls or self.FINANCIAL_FEEDS
    
    def fetch_articles(
        self,
        query: Optional[str] = None,
        from_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch articles from RSS feeds.
        
        Args:
            query: Not used for RSS (filtering done after fetch)
            from_date: Fetch articles from this date
            limit: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=1)
        
        all_articles = []
        
        for feed_name, feed_url in self.feed_urls.items():
            try:
                logger.debug(f"Fetching RSS feed: {feed_name}")
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Parse published date
                    if hasattr(entry, 'published_parsed'):
                        published_at = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed'):
                        published_at = datetime(*entry.updated_parsed[:6])
                    else:
                        published_at = datetime.now()
                    
                    # Filter by date
                    if published_at < from_date:
                        continue
                    
                    # Extract content
                    content = ''
                    if hasattr(entry, 'content'):
                        content = entry.content[0].value
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    
                    # Clean HTML from content
                    if content:
                        soup = BeautifulSoup(content, 'html.parser')
                        content = soup.get_text(strip=True)
                    
                    article = {
                        'title': entry.get('title', ''),
                        'content': content,
                        'summary': entry.get('summary', '')[:500],
                        'source': f"rss_{feed_name}",
                        'author': entry.get('author'),
                        'published_at': published_at,
                        'url': entry.get('link'),
                        'tickers': NewsAPIFetcher._extract_tickers(
                            entry.get('title', '') + ' ' + content
                        )
                    }
                    all_articles.append(article)
                
                logger.info(f"Fetched {len(feed.entries)} articles from {feed_name}")
                
                # Small delay to be respectful
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_name}: {e}")
                continue
            
            # Stop if we have enough articles
            if len(all_articles) >= limit:
                break
        
        # Sort by published date and limit
        all_articles.sort(key=lambda x: x['published_at'], reverse=True)
        articles = all_articles[:limit]
        
        logger.info(f"Total articles fetched from RSS: {len(articles)}")
        return articles


class AlphaVantageFetcher(NewsFetcher):
    """Fetcher for Alpha Vantage News Sentiment API (free tier: 25 requests/day)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage fetcher.
        
        Args:
            api_key: Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env var)
        """
        super().__init__("alpha_vantage")
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        
        if not self.api_key:
            logger.warning("Alpha Vantage key not found. Set ALPHA_VANTAGE_API_KEY environment variable.")
    
    def fetch_articles(
        self,
        query: Optional[str] = None,
        from_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Fetch financial news from Alpha Vantage.
        
        Args:
            query: Search query (tickers or topics)
            from_date: Fetch articles from this date (not supported by API, recent only)
            limit: Maximum number of articles (max 1000 per request)
            
        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            logger.error("Alpha Vantage API key required")
            return []
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_key,
                'limit': min(limit, 1000),  # Max 1000
                'sort': 'LATEST'
            }
            
            if query:
                params['topics'] = query
            
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' not in data:
                logger.error(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")
                return []
            
            articles = []
            for item in data.get('feed', []):
                # Parse published date
                published_at = datetime.strptime(
                    item['time_published'],
                    '%Y%m%dT%H%M%S'
                )
                
                # Extract tickers from ticker sentiment
                tickers = [
                    t['ticker'] 
                    for t in item.get('ticker_sentiment', [])
                ]
                
                article = {
                    'title': item.get('title', ''),
                    'content': item.get('summary', ''),
                    'summary': item.get('summary', '')[:500],
                    'source': f"alphavantage_{item.get('source', 'unknown')}",
                    'author': item.get('authors', [None])[0] if item.get('authors') else None,
                    'published_at': published_at,
                    'url': item.get('url'),
                    'tickers': tickers
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from Alpha Vantage")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return []


if __name__ == "__main__":
    # Test fetchers
    import sys
    
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    print("Testing RSS Feed Fetcher...")
    rss_fetcher = RSSFeedFetcher()
    articles = rss_fetcher.fetch_articles(limit=10)
    print(f"Fetched {len(articles)} articles from RSS")
    if articles:
        print(f"Sample article: {articles[0]['title']}")
    
    print("\n" + "="*60 + "\n")
    
    print("Testing NewsAPI Fetcher...")
    newsapi_fetcher = NewsAPIFetcher()
    if newsapi_fetcher.api_key:
        articles = newsapi_fetcher.fetch_articles(limit=10)
        print(f"Fetched {len(articles)} articles from NewsAPI")
        if articles:
            print(f"Sample article: {articles[0]['title']}")
    else:
        print("NewsAPI key not configured")
    
    print("\n" + "="*60 + "\n")
    
    print("Testing Alpha Vantage Fetcher...")
    av_fetcher = AlphaVantageFetcher()
    if av_fetcher.api_key:
        articles = av_fetcher.fetch_articles(limit=10)
        print(f"Fetched {len(articles)} articles from Alpha Vantage")
        if articles:
            print(f"Sample article: {articles[0]['title']}")
    else:
        print("Alpha Vantage key not configured")

