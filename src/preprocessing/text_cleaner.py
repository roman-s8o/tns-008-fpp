"""
Text preprocessing utilities for news articles.

This module handles HTML cleaning, text normalization, and other
preprocessing tasks for news articles.
"""

import re
from typing import Optional, List
from bs4 import BeautifulSoup
import unicodedata
from loguru import logger


class TextCleaner:
    """Cleans and normalizes news article text."""
    
    def __init__(
        self,
        max_length: int = 2000,
        include_metadata: bool = True,
        use_summary_fallback: bool = True
    ):
        """
        Initialize text cleaner.
        
        Args:
            max_length: Maximum character length per article
            include_metadata: Whether to include article metadata
            use_summary_fallback: Use summary when full content unavailable
        """
        self.max_length = max_length
        self.include_metadata = include_metadata
        self.use_summary_fallback = use_summary_fallback
    
    @staticmethod
    def remove_html(text: str) -> str:
        """
        Remove HTML tags and entities from text.
        
        Args:
            text: Text with potential HTML
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean whitespace
        text = soup.get_text(separator=' ')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Text with potential URLs
            
        Returns:
            Text without URLs
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters.
        
        Args:
            text: Text with unicode characters
            
        Returns:
            Normalized text
        """
        # Normalize unicode (e.g., convert é to e)
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep_basic_punct: bool = True) -> str:
        """
        Remove special characters.
        
        Args:
            text: Text to clean
            keep_basic_punct: Keep basic punctuation (.,!?;:)
            
        Returns:
            Cleaned text
        """
        if keep_basic_punct:
            # Keep alphanumeric, spaces, and basic punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'\"$%()]', '', text)
        else:
            # Keep only alphanumeric and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        return text
    
    def truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length (uses self.max_length if None)
            
        Returns:
            Truncated text
        """
        max_len = max_length or self.max_length
        
        if len(text) <= max_len:
            return text
        
        # Truncate at word boundary
        truncated = text[:max_len]
        last_space = truncated.rfind(' ')
        
        if last_space > 0:
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def clean_article(
        self,
        title: str,
        content: Optional[str],
        summary: Optional[str] = None,
        source: Optional[str] = None,
        published_at: Optional[str] = None,
        remove_urls: bool = True,
        normalize_unicode: bool = False
    ) -> str:
        """
        Clean and process a news article.
        
        Args:
            title: Article title
            content: Article content
            summary: Article summary
            source: News source
            published_at: Publication timestamp
            remove_urls: Whether to remove URLs
            normalize_unicode: Whether to normalize unicode
            
        Returns:
            Cleaned article text
        """
        # Start with title
        text_parts = []
        
        # Add metadata if enabled
        if self.include_metadata and source:
            text_parts.append(f"[SOURCE: {source}]")
        
        if self.include_metadata and published_at:
            text_parts.append(f"[DATE: {published_at}]")
        
        # Add title
        clean_title = self.remove_html(title)
        clean_title = self.normalize_whitespace(clean_title)
        text_parts.append(f"TITLE: {clean_title}")
        
        # Add content or fallback to summary
        text_content = content
        if not text_content and self.use_summary_fallback and summary:
            text_content = summary
        
        if text_content:
            # Clean content
            text_content = self.remove_html(text_content)
            text_content = self.normalize_whitespace(text_content)
            
            if remove_urls:
                text_content = self.remove_urls(text_content)
            
            if normalize_unicode:
                text_content = self.normalize_unicode(text_content)
            
            text_parts.append(f"CONTENT: {text_content}")
        
        # Combine all parts
        full_text = " ".join(text_parts)
        
        # Truncate if necessary
        full_text = self.truncate_text(full_text)
        
        return full_text
    
    def clean_articles_batch(
        self,
        articles: List[dict],
        separator: str = " [SEP] "
    ) -> str:
        """
        Clean and concatenate multiple articles.
        
        Args:
            articles: List of article dictionaries
            separator: Separator between articles
            
        Returns:
            Concatenated cleaned text
        """
        cleaned_articles = []
        
        for article in articles:
            cleaned = self.clean_article(
                title=article.get('title', ''),
                content=article.get('content'),
                summary=article.get('summary'),
                source=article.get('source'),
                published_at=str(article.get('published_at', ''))
            )
            
            if cleaned:
                cleaned_articles.append(cleaned)
        
        # Concatenate with separator
        result = separator.join(cleaned_articles)
        
        # Truncate if total length exceeds maximum
        result = self.truncate_text(result, max_length=self.max_length * len(articles))
        
        return result


def remove_duplicates(text: str) -> str:
    """
    Remove duplicate sentences from text.
    
    Args:
        text: Text with potential duplicates
        
    Returns:
        Deduplicated text
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    return '. '.join(unique_sentences) + '.'


if __name__ == "__main__":
    # Test text cleaner
    cleaner = TextCleaner(max_length=500)
    
    # Sample article with HTML
    title = "Apple Stock Rises on <b>Strong</b> Earnings"
    content = """
    <p>Apple Inc. <a href="http://example.com">reported</a> strong quarterly earnings...</p>
    <script>alert('test');</script>
    <p>The company's revenue grew by 15%   in Q4.</p>
    """
    
    cleaned = cleaner.clean_article(
        title=title,
        content=content,
        source="Financial Times",
        published_at="2025-10-12"
    )
    
    print("Original title:", title)
    print("Original content:", content[:100])
    print("\nCleaned article:")
    print(cleaned)

