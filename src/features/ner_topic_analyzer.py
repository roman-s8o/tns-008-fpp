"""
Named Entity Recognition (NER) and Topic Modeling Module for Financial News

This module provides:
1. NER using BERT-based model (dslim/bert-base-NER)
2. Financial entity extraction (tickers, currencies, amounts)
3. Topic modeling using LDA (Latent Dirichlet Allocation)

All features are designed for financial news analysis with ticker-specific insights.
"""

import os
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter
import json

# Disable TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from gensim import corpora, models
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


class FinancialNERExtractor:
    """
    Named Entity Recognition extractor for financial news.
    
    Combines BERT-based NER with financial-specific entity extraction
    (tickers, currencies, amounts, percentages).
    """
    
    # Major currency codes
    CURRENCY_CODES = {
        'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD',
        'CNY', 'HKD', 'SGD', 'INR', 'KRW', 'BRL', 'MXN', 'RUB',
        'ZAR', 'TRY', 'SEK', 'NOK', 'DKK', 'PLN', 'THB', 'IDR',
        'MYR', 'PHP', 'CZK', 'ILS', 'CLP', 'TWD', 'AED', 'SAR'
    }
    
    # Common financial instruments
    INSTRUMENTS = {
        'bond', 'bonds', 'stock', 'stocks', 'equity', 'equities',
        'future', 'futures', 'option', 'options', 'ETF', 'ETFs',
        'fund', 'funds', 'derivative', 'derivatives', 'swap', 'swaps',
        'treasury', 'treasuries', 'security', 'securities'
    }
    
    def __init__(
        self,
        ner_model: str = "dslim/bert-base-NER",
        device: str = "mps",
        nasdaq_tickers: Optional[List[str]] = None,
    ):
        """
        Initialize financial NER extractor.
        
        Args:
            ner_model: HuggingFace NER model name
            device: Device to run on
            nasdaq_tickers: List of valid Nasdaq tickers for validation
        """
        self.device = device
        self.nasdaq_tickers = set(nasdaq_tickers) if nasdaq_tickers else set()
        
        logger.info(f"Loading NER model: {ner_model}")
        
        # Load NER pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model)
        
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "mps" else -1,  # Use CPU for pipeline (MPS has issues)
            aggregation_strategy="simple"
        )
        
        logger.info(f"✓ NER model loaded")
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for financial entity extraction."""
        # Stock ticker pattern (1-5 uppercase letters, possibly with numbers)
        self.ticker_pattern = re.compile(r'\b[A-Z]{1,5}\b')
        
        # Currency amount pattern ($1.5M, €500K, £2.3B, etc.)
        self.amount_pattern = re.compile(
            r'[\$€£¥]\s*\d+(?:\.\d+)?(?:\s*[KMBTkmbt])?|'
            r'\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand|dollars|euros|pounds)',
            re.IGNORECASE
        )
        
        # Percentage pattern (5%, 3.5 percent, etc.)
        self.percentage_pattern = re.compile(
            r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:percent|percentage)',
            re.IGNORECASE
        )
        
        # Currency code pattern (USD, EUR/USD, etc.)
        self.currency_code_pattern = re.compile(
            r'\b(?:' + '|'.join(self.CURRENCY_CODES) + r')\b|'
            r'\b(?:' + '|'.join(self.CURRENCY_CODES) + r')\/(?:' + '|'.join(self.CURRENCY_CODES) + r')\b'
        )
        
        # Date pattern (Q1 2024, FY2024, 2024-Q3, etc.)
        self.date_pattern = re.compile(
            r'\b(?:Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4}-Q[1-4])\b',
            re.IGNORECASE
        )
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing:
                - entities_all: List of all entity dictionaries
                - entities_by_type: Dict grouping entities by type
                - entity_counts: Dict with count of each entity type
                - tickers: List of extracted stock tickers
                - currencies: List of extracted currency codes
                - amounts: List of extracted monetary amounts
                - percentages: List of extracted percentages
        """
        # Extract using BERT NER
        ner_entities = self._extract_bert_ner(text)
        
        # Extract financial entities
        financial_entities = self._extract_financial_entities(text)
        
        # Combine and organize
        all_entities = ner_entities + financial_entities
        
        # Group by type
        entities_by_type = defaultdict(list)
        for entity in all_entities:
            entities_by_type[entity['type']].append(entity['text'])
        
        # Count by type
        entity_counts = {k: len(v) for k, v in entities_by_type.items()}
        
        # Extract specific financial entities
        tickers = self._extract_tickers(text)
        currencies = self._extract_currencies(text)
        amounts = self._extract_amounts(text)
        percentages = self._extract_percentages(text)
        
        return {
            'entities_all': all_entities,
            'entities_by_type': dict(entities_by_type),
            'entity_counts': entity_counts,
            'tickers': tickers,
            'currencies': currencies,
            'amounts': amounts,
            'percentages': percentages,
        }
    
    def _extract_bert_ner(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using BERT NER model."""
        try:
            # Run NER pipeline
            results = self.ner_pipeline(text)
            
            # Convert to standard format
            entities = []
            for entity in results:
                entities.append({
                    'text': entity['word'],
                    'type': entity['entity_group'],
                    'score': float(entity['score']),  # Convert numpy float32 to Python float
                    'start': entity.get('start'),
                    'end': entity.get('end'),
                })
            
            return entities
        except Exception as e:
            logger.warning(f"BERT NER extraction failed: {e}")
            return []
    
    def _extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial-specific entities using patterns."""
        entities = []
        
        # Extract amounts
        for match in self.amount_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'type': 'MONETARY_AMOUNT',
                'score': 1.0,
                'start': match.start(),
                'end': match.end(),
            })
        
        # Extract percentages
        for match in self.percentage_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'type': 'PERCENTAGE',
                'score': 1.0,
                'start': match.start(),
                'end': match.end(),
            })
        
        # Extract currency codes
        for match in self.currency_code_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'type': 'CURRENCY',
                'score': 1.0,
                'start': match.start(),
                'end': match.end(),
            })
        
        # Extract fiscal periods
        for match in self.date_pattern.finditer(text):
            entities.append({
                'text': match.group(),
                'type': 'FISCAL_PERIOD',
                'score': 1.0,
                'start': match.start(),
                'end': match.end(),
            })
        
        return entities
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract and validate stock tickers."""
        # Find potential tickers
        potential_tickers = self.ticker_pattern.findall(text)
        
        # Filter valid tickers
        valid_tickers = []
        for ticker in potential_tickers:
            # Check if in Nasdaq list (if available)
            if self.nasdaq_tickers and ticker in self.nasdaq_tickers:
                valid_tickers.append(ticker)
            # Or check if appears in typical ticker context
            elif self._is_likely_ticker(ticker, text):
                valid_tickers.append(ticker)
        
        return list(set(valid_tickers))  # Remove duplicates
    
    def _is_likely_ticker(self, ticker: str, text: str) -> bool:
        """Check if a word is likely a stock ticker based on context."""
        # Skip common words that match ticker pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 
                       'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
                       'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
                       'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'LET',
                       'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        if ticker in common_words:
            return False
        
        # Check for ticker-related context words nearby
        ticker_contexts = ['stock', 'shares', 'ticker', 'NYSE', 'NASDAQ', 
                          'trading', 'equity', 'listed', 'quoted']
        
        # Look for context within 50 characters
        ticker_pos = text.find(ticker)
        if ticker_pos >= 0:
            context = text[max(0, ticker_pos-50):min(len(text), ticker_pos+50)].lower()
            if any(ctx in context for ctx in ticker_contexts):
                return True
        
        # If ticker appears multiple times, likely valid
        if text.count(ticker) >= 2:
            return True
        
        return False
    
    def _extract_currencies(self, text: str) -> List[str]:
        """Extract currency codes."""
        matches = self.currency_code_pattern.findall(text)
        return list(set(matches))  # Remove duplicates
    
    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts."""
        matches = self.amount_pattern.findall(text)
        return matches
    
    def _extract_percentages(self, text: str) -> List[str]:
        """Extract percentages."""
        matches = self.percentage_pattern.findall(text)
        return matches
    
    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of entity extraction results
        """
        results = []
        for text in texts:
            results.append(self.extract_entities(text))
        return results


class TopicModeler:
    """
    Topic modeling using Latent Dirichlet Allocation (LDA).
    
    Identifies major themes in financial news articles.
    """
    
    # Financial stopwords (in addition to standard stopwords)
    FINANCIAL_STOPWORDS = {
        'said', 'says', 'according', 'also', 'would', 'could', 'may',
        'will', 'can', 'new', 'year', 'time', 'company', 'companies',
        'market', 'markets', 'stock', 'stocks', 'price', 'prices'
    }
    
    def __init__(
        self,
        num_topics: int = 10,
        min_word_freq: int = 2,
        passes: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            min_word_freq: Minimum word frequency to include
            passes: Number of training passes
            random_state: Random seed for reproducibility
        """
        self.num_topics = num_topics
        self.min_word_freq = min_word_freq
        self.passes = passes
        self.random_state = random_state
        
        # Load stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
            self.stopwords.update(self.FINANCIAL_STOPWORDS)
        except:
            logger.warning("Failed to load stopwords, using minimal set")
            self.stopwords = self.FINANCIAL_STOPWORDS
        
        self.dictionary = None
        self.lda_model = None
        self.corpus = None
        self.topic_labels = None
        
        logger.info(f"Initialized TopicModeler with {num_topics} topics")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split
            tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [
            token for token in tokens 
            if token not in self.stopwords and len(token) > 3
        ]
        
        return tokens
    
    def fit(self, documents: List[str]) -> Dict[str, Any]:
        """
        Fit LDA model on documents.
        
        Args:
            documents: List of text documents
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training LDA model on {len(documents)} documents...")
        
        # Preprocess documents
        processed_docs = [self._preprocess_text(doc) for doc in documents]
        
        # Remove empty documents
        processed_docs = [doc for doc in processed_docs if len(doc) > 0]
        
        if len(processed_docs) < self.num_topics:
            logger.warning(
                f"Number of documents ({len(processed_docs)}) is less than "
                f"number of topics ({self.num_topics}). Reducing topics."
            )
            self.num_topics = max(2, len(processed_docs) // 2)
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_docs)
        
        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=self.min_word_freq,
            no_above=0.7,
            keep_n=1000
        )
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        # Train LDA model
        self.lda_model = models.LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=self.passes,
            workers=2,  # Use 2 workers for Mac
            per_word_topics=True
        )
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=processed_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Generate topic labels
        self._generate_topic_labels()
        
        logger.info(f"✓ LDA model trained")
        logger.info(f"  - Coherence score: {coherence_score:.4f}")
        logger.info(f"  - Vocabulary size: {len(self.dictionary)}")
        
        return {
            'num_topics': self.num_topics,
            'coherence_score': coherence_score,
            'vocab_size': len(self.dictionary),
            'num_documents': len(processed_docs),
        }
    
    def _generate_topic_labels(self):
        """Generate human-readable labels for topics."""
        self.topic_labels = {}
        
        for topic_id in range(self.num_topics):
            # Get top words for topic
            top_words = self.lda_model.show_topic(topic_id, topn=5)
            
            # Create label from top 3 words
            label_words = [word for word, _ in top_words[:3]]
            label = " / ".join(label_words)
            
            self.topic_labels[topic_id] = label
        
        logger.info(f"Generated topic labels:")
        for topic_id, label in self.topic_labels.items():
            logger.info(f"  Topic {topic_id}: {label}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict topic distribution for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing:
                - topic_distribution: List of (topic_id, probability) tuples
                - primary_topic_id: ID of most probable topic
                - primary_topic_name: Name of most probable topic
                - primary_topic_prob: Probability of primary topic
        """
        if self.lda_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Preprocess text
        tokens = self._preprocess_text(text)
        
        # Convert to BOW
        bow = self.dictionary.doc2bow(tokens)
        
        # Get topic distribution
        topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=0.01)
        
        # Sort by probability
        topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)
        
        # Get primary topic
        if len(topic_dist) > 0:
            primary_topic_id = topic_dist[0][0]
            primary_topic_prob = topic_dist[0][1]
            primary_topic_name = self.topic_labels.get(primary_topic_id, f"Topic {primary_topic_id}")
        else:
            primary_topic_id = -1
            primary_topic_prob = 0.0
            primary_topic_name = "Unknown"
        
        return {
            'topic_distribution': topic_dist,
            'primary_topic_id': primary_topic_id,
            'primary_topic_name': primary_topic_name,
            'primary_topic_prob': primary_topic_prob,
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict topic distributions for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of topic prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def get_topic_words(self, topic_id: int, num_words: int = 10) -> List[Tuple[str, float]]:
        """
        Get top words for a topic.
        
        Args:
            topic_id: Topic ID
            num_words: Number of words to return
            
        Returns:
            List of (word, probability) tuples
        """
        if self.lda_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.lda_model.show_topic(topic_id, topn=num_words)
    
    def get_all_topics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all topics.
        
        Returns:
            Dictionary mapping topic_id to topic info
        """
        if self.lda_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        topics = {}
        for topic_id in range(self.num_topics):
            topics[topic_id] = {
                'label': self.topic_labels.get(topic_id, f"Topic {topic_id}"),
                'top_words': self.get_topic_words(topic_id, num_words=10),
            }
        
        return topics


class NERTopicAnalyzer:
    """
    Combined NER and topic analysis for financial news.
    """
    
    def __init__(
        self,
        ner_model: str = "dslim/bert-base-NER",
        num_topics: int = 10,
        device: str = "mps",
        nasdaq_tickers: Optional[List[str]] = None,
    ):
        """
        Initialize combined NER and topic analyzer.
        
        Args:
            ner_model: NER model name
            num_topics: Number of topics for LDA
            device: Device to run on
            nasdaq_tickers: List of valid Nasdaq tickers
        """
        logger.info("Initializing NER & Topic Analyzer")
        
        self.ner_extractor = FinancialNERExtractor(
            ner_model=ner_model,
            device=device,
            nasdaq_tickers=nasdaq_tickers,
        )
        
        self.topic_modeler = TopicModeler(
            num_topics=num_topics,
        )
        
        self.is_fitted = False
        
        logger.info("✓ NER & Topic Analyzer ready")
    
    def fit_topics(self, documents: List[str]) -> Dict[str, Any]:
        """
        Fit topic model on documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Training metrics
        """
        metrics = self.topic_modeler.fit(documents)
        self.is_fitted = True
        return metrics
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze single text for both NER and topics.
        
        Args:
            text: Input text
            
        Returns:
            Combined analysis results
        """
        # Extract entities
        ner_results = self.ner_extractor.extract_entities(text)
        
        # Predict topics (if model is fitted)
        if self.is_fitted:
            topic_results = self.topic_modeler.predict(text)
        else:
            topic_results = {
                'topic_distribution': [],
                'primary_topic_id': -1,
                'primary_topic_name': 'Not fitted',
                'primary_topic_prob': 0.0,
            }
        
        # Combine results
        return {
            'ner': ner_results,
            'topics': topic_results,
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of analysis results
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results
    
    def get_summary_statistics(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from multiple analyses.
        
        Args:
            analyses: List of analysis results
            
        Returns:
            Summary statistics
        """
        # Count entity types
        entity_type_counts = Counter()
        all_tickers = set()
        all_currencies = set()
        
        for analysis in analyses:
            ner = analysis['ner']
            for entity_type, count in ner['entity_counts'].items():
                entity_type_counts[entity_type] += count
            all_tickers.update(ner['tickers'])
            all_currencies.update(ner['currencies'])
        
        # Count topics
        topic_counts = Counter()
        for analysis in analyses:
            if self.is_fitted:
                topic_id = analysis['topics']['primary_topic_id']
                if topic_id >= 0:
                    topic_counts[topic_id] += 1
        
        return {
            'num_documents': len(analyses),
            'entity_type_counts': dict(entity_type_counts),
            'unique_tickers': sorted(list(all_tickers)),
            'num_unique_tickers': len(all_tickers),
            'unique_currencies': sorted(list(all_currencies)),
            'num_unique_currencies': len(all_currencies),
            'topic_distribution': dict(topic_counts),
            'most_common_topic': topic_counts.most_common(1)[0] if topic_counts else None,
        }


def get_ner_topic_analyzer(
    ner_model: str = "dslim/bert-base-NER",
    num_topics: int = 10,
    device: str = "mps",
    nasdaq_tickers: Optional[List[str]] = None,
) -> NERTopicAnalyzer:
    """
    Factory function to get NER and topic analyzer.
    
    Args:
        ner_model: NER model name
        num_topics: Number of topics
        device: Device to run on
        nasdaq_tickers: List of valid Nasdaq tickers
        
    Returns:
        NERTopicAnalyzer instance
    """
    return NERTopicAnalyzer(
        ner_model=ner_model,
        num_topics=num_topics,
        device=device,
        nasdaq_tickers=nasdaq_tickers,
    )

