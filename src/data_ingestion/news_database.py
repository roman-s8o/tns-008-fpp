"""
Database schema and management for news articles.

This module handles SQLite database operations for storing and retrieving
financial news articles.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NewsDatabase:
    """Manages SQLite database for news articles."""
    
    def __init__(self, db_path: str = "data/raw/news.db"):
        """
        Initialize the news database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        logger.info(f"Initialized NewsDatabase at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_hash TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    source TEXT NOT NULL,
                    author TEXT,
                    published_at TIMESTAMP NOT NULL,
                    url TEXT,
                    tickers TEXT,
                    sentiment_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_published_at 
                ON articles(published_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source 
                ON articles(source)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_article_hash 
                ON articles(article_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tickers 
                ON articles(tickers)
            """)
            
            # Sources table (track API usage)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    api_type TEXT,
                    last_fetch TIMESTAMP,
                    total_articles INTEGER DEFAULT 0,
                    daily_limit INTEGER,
                    rate_limit_reset TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Fetch history table (track API calls)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fetch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    fetch_date DATE NOT NULL,
                    articles_fetched INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT 1,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("Database tables initialized")
    
    @staticmethod
    def _generate_hash(title: str, content: str, published_at: str) -> str:
        """
        Generate a unique hash for an article to detect duplicates.
        
        Args:
            title: Article title
            content: Article content
            published_at: Publication timestamp
            
        Returns:
            SHA-256 hash string
        """
        # Combine title, content snippet, and date for hash
        content_snippet = content[:500] if content else ""
        hash_input = f"{title}{content_snippet}{published_at}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def insert_article(
        self,
        title: str,
        content: Optional[str],
        source: str,
        published_at: datetime,
        url: Optional[str] = None,
        author: Optional[str] = None,
        summary: Optional[str] = None,
        tickers: Optional[List[str]] = None
    ) -> Optional[int]:
        """
        Insert a news article into the database.
        
        Args:
            title: Article title
            content: Full article content
            source: News source name
            published_at: Publication datetime
            url: Article URL
            author: Article author
            summary: Article summary
            tickers: List of related stock tickers
            
        Returns:
            Article ID if inserted, None if duplicate
        """
        # Generate hash for duplicate detection
        article_hash = self._generate_hash(
            title,
            content or "",
            published_at.isoformat()
        )
        
        # Convert tickers list to comma-separated string
        tickers_str = ",".join(tickers) if tickers else None
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO articles (
                        article_hash, title, content, summary, source, 
                        author, published_at, url, tickers
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    article_hash, title, content, summary, source,
                    author, published_at, url, tickers_str
                ))
                
                article_id = cursor.lastrowid
                logger.debug(f"Inserted article {article_id}: {title[:50]}...")
                return article_id
                
        except sqlite3.IntegrityError:
            # Duplicate article
            logger.debug(f"Duplicate article skipped: {title[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Error inserting article: {e}")
            return None
    
    def bulk_insert_articles(self, articles: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple articles efficiently.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (inserted_count, duplicate_count)
        """
        inserted = 0
        duplicates = 0
        
        for article in articles:
            article_id = self.insert_article(
                title=article.get('title', ''),
                content=article.get('content'),
                source=article.get('source', ''),
                published_at=article.get('published_at', datetime.now()),
                url=article.get('url'),
                author=article.get('author'),
                summary=article.get('summary'),
                tickers=article.get('tickers')
            )
            
            if article_id:
                inserted += 1
            else:
                duplicates += 1
        
        logger.info(f"Bulk insert: {inserted} new, {duplicates} duplicates")
        return inserted, duplicates
    
    def get_articles(
        self,
        source: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Retrieve articles from the database.
        
        Args:
            source: Filter by source
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of articles to return
            
        Returns:
            List of article dictionaries
        """
        query = "SELECT * FROM articles WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if start_date:
            query += " AND published_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND published_at <= ?"
            params.append(end_date)
        
        query += " ORDER BY published_at DESC LIMIT ?"
        params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            articles = []
            for row in rows:
                article = dict(row)
                # Convert tickers string back to list
                if article.get('tickers'):
                    article['tickers'] = article['tickers'].split(',')
                articles.append(article)
            
            return articles
    
    def get_article_count(
        self,
        source: Optional[str] = None,
        date: Optional[datetime] = None
    ) -> int:
        """
        Get count of articles.
        
        Args:
            source: Filter by source
            date: Filter by specific date
            
        Returns:
            Number of articles
        """
        query = "SELECT COUNT(*) FROM articles WHERE 1=1"
        params = []
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if date:
            query += " AND DATE(published_at) = DATE(?)"
            params.append(date)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            return count
    
    def record_fetch(
        self,
        source: str,
        articles_fetched: int,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Record a fetch operation in history.
        
        Args:
            source: News source name
            articles_fetched: Number of articles fetched
            success: Whether fetch was successful
            error_message: Error message if failed
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fetch_history (
                    source, fetch_date, articles_fetched, success, error_message
                ) VALUES (?, DATE('now'), ?, ?, ?)
            """, (source, articles_fetched, success, error_message))
            
            logger.debug(f"Recorded fetch: {source} - {articles_fetched} articles")
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total articles
            cursor.execute("SELECT COUNT(*) FROM articles")
            total_articles = cursor.fetchone()[0]
            
            # Articles by source
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM articles 
                GROUP BY source 
                ORDER BY count DESC
            """)
            by_source = dict(cursor.fetchall())
            
            # Date range
            cursor.execute("""
                SELECT MIN(published_at), MAX(published_at) 
                FROM articles
            """)
            date_range = cursor.fetchone()
            
            # Recent activity
            cursor.execute("""
                SELECT DATE(published_at) as date, COUNT(*) as count 
                FROM articles 
                WHERE DATE(published_at) >= DATE('now', '-7 days')
                GROUP BY DATE(published_at)
                ORDER BY date DESC
            """)
            recent_activity = dict(cursor.fetchall())
            
            return {
                "total_articles": total_articles,
                "by_source": by_source,
                "date_range": date_range,
                "recent_activity": recent_activity
            }
    
    def cleanup_old_articles(self, days_to_keep: int = 365):
        """
        Remove articles older than specified days.
        
        Args:
            days_to_keep: Number of days of articles to keep
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM articles 
                WHERE published_at < DATE('now', '-' || ? || ' days')
            """, (days_to_keep,))
            
            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} old articles")
            return deleted


if __name__ == "__main__":
    # Test database
    logging.basicConfig(level=logging.INFO)
    
    db = NewsDatabase("data/raw/news_test.db")
    
    # Test insert
    article_id = db.insert_article(
        title="Test Article",
        content="This is a test article about financial markets.",
        source="test_source",
        published_at=datetime.now(),
        url="https://example.com/test",
        tickers=["AAPL", "MSFT"]
    )
    
    print(f"Inserted article ID: {article_id}")
    
    # Test statistics
    stats = db.get_statistics()
    print(f"\nDatabase statistics:")
    print(f"Total articles: {stats['total_articles']}")
    print(f"By source: {stats['by_source']}")

