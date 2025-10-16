"""
Sentiment Analysis Module for Financial News

This module provides dual sentiment analysis approaches:
1. Pre-trained FinBERT-sentiment (domain-specific)
2. SSL-trained embeddings with sentiment classifier

Both methods provide sentiment labels, scores, and confidence metrics.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod

# Disable TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""
    
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing:
                - label: str (positive/negative/neutral)
                - scores: dict (probability distribution)
                - sentiment_score: float (-1 to 1)
                - confidence: float (0 to 1)
        """
        pass
    
    @abstractmethod
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of sentiment dictionaries
        """
        pass


class FinBERTSentimentAnalyzer(SentimentAnalyzer):
    """
    Sentiment analyzer using pre-trained FinBERT-sentiment model.
    
    Uses ProsusAI/finbert-tone or yiyanghkust/finbert-tone for
    domain-specific financial sentiment analysis.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "mps",
    ):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading FinBERT-sentiment model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Label mapping (FinBERT outputs: positive, negative, neutral)
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        logger.info(f"✓ FinBERT-sentiment loaded on {device}")
        logger.info(f"  - Labels: {list(self.id2label.values())}")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of single text."""
        results = self.analyze_batch([text])
        return results[0]
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Process each sample in batch
                for j in range(len(batch_texts)):
                    sample_probs = probs[j].cpu().numpy()
                    predicted_class = int(torch.argmax(probs[j]).cpu())
                    
                    # Get label
                    label = self.id2label[predicted_class].lower()
                    
                    # Create scores dict
                    scores = {
                        self.id2label[k].lower(): float(sample_probs[k])
                        for k in range(len(sample_probs))
                    }
                    
                    # Calculate sentiment score (-1 to 1)
                    sentiment_score = self._calculate_sentiment_score(scores)
                    
                    # Confidence is max probability
                    confidence = float(np.max(sample_probs))
                    
                    results.append({
                        'label': label,
                        'scores': scores,
                        'sentiment_score': sentiment_score,
                        'confidence': confidence,
                        'method': 'finbert-sentiment',
                    })
        
        return results
    
    def _calculate_sentiment_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate continuous sentiment score from probability distribution.
        
        Args:
            scores: Dictionary of sentiment probabilities
            
        Returns:
            Sentiment score from -1 (negative) to +1 (positive)
        """
        positive = scores.get('positive', 0.0)
        negative = scores.get('negative', 0.0)
        neutral = scores.get('neutral', 0.0)
        
        # Weighted score: positive contributes +1, negative -1, neutral 0
        sentiment_score = positive * 1.0 + negative * (-1.0) + neutral * 0.0
        
        return sentiment_score


class SSLSentimentAnalyzer(SentimentAnalyzer):
    """
    Sentiment analyzer using SSL-trained embeddings.
    
    Uses embeddings from FinBERT Contrastive model with a simple
    sentiment classification head trained on financial keywords.
    """
    
    def __init__(
        self,
        ssl_model_path: str = "data/models/finbert_contrastive",
        device: str = "mps",
    ):
        """
        Initialize SSL-based sentiment analyzer.
        
        Args:
            ssl_model_path: Path to SSL-trained model
            device: Device to run on
        """
        self.ssl_model_path = ssl_model_path
        self.device = device
        
        logger.info(f"Loading SSL model for sentiment: {ssl_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ssl_model_path)
        
        # Load SSL model (FinBERT Contrastive)
        from transformers import AutoModelForMaskedLM
        from ..models.finbert.projection_head import FinBERTWithProjection
        
        base_model = AutoModelForMaskedLM.from_pretrained("ProsusAI/finbert")
        self.model = FinBERTWithProjection(
            finbert_model=base_model,
            projection_dim=128,
            freeze_bert=False,
        )
        
        # Load trained weights
        checkpoint = torch.load(
            os.path.join(ssl_model_path, 'contrastive_model.pt'),
            map_location=device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Simple sentiment classifier based on embedding patterns
        # Use heuristic: positive keywords have higher similarity to positive context
        self._init_sentiment_keywords()
        
        logger.info(f"✓ SSL sentiment analyzer loaded on {device}")
    
    def _init_sentiment_keywords(self):
        """Initialize sentiment keyword embeddings for comparison."""
        # Financial sentiment keywords
        self.positive_keywords = [
            "profit", "growth", "surge", "rally", "gain", "bullish", "upgrade",
            "beat expectations", "strong earnings", "record high", "outperform"
        ]
        
        self.negative_keywords = [
            "loss", "decline", "plunge", "crash", "bearish", "downgrade",
            "miss expectations", "weak earnings", "record low", "underperform"
        ]
        
        # Get embeddings for keywords
        with torch.no_grad():
            pos_embeddings = self._get_text_embeddings(self.positive_keywords)
            neg_embeddings = self._get_text_embeddings(self.negative_keywords)
        
        # Average embeddings
        self.positive_prototype = torch.mean(pos_embeddings, dim=0, keepdim=True)
        self.negative_prototype = torch.mean(neg_embeddings, dim=0, keepdim=True)
        
        logger.info("✓ Sentiment keyword prototypes initialized")
    
    def _get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        embeddings = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
        
        return embeddings
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of single text."""
        results = self.analyze_batch([text])
        return results[0]
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Analyze sentiment using SSL embeddings.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Get embeddings
                embeddings = self._get_text_embeddings(batch_texts)
                
                # Compute similarity to positive and negative prototypes
                pos_sim = torch.cosine_similarity(
                    embeddings,
                    self.positive_prototype.expand(len(batch_texts), -1),
                    dim=1
                )
                neg_sim = torch.cosine_similarity(
                    embeddings,
                    self.negative_prototype.expand(len(batch_texts), -1),
                    dim=1
                )
                
                # Process each sample
                for j in range(len(batch_texts)):
                    pos_score = float(pos_sim[j].cpu())
                    neg_score = float(neg_sim[j].cpu())
                    
                    # Normalize to probabilities
                    # Use softmax-like transformation
                    total = np.exp(pos_score) + np.exp(neg_score) + 1.0  # +1 for neutral
                    
                    scores = {
                        'positive': float(np.exp(pos_score) / total),
                        'negative': float(np.exp(neg_score) / total),
                        'neutral': float(1.0 / total),
                    }
                    
                    # Determine label
                    label = max(scores, key=scores.get)
                    
                    # Calculate sentiment score (-1 to 1)
                    sentiment_score = pos_score - neg_score
                    # Normalize to [-1, 1]
                    sentiment_score = np.tanh(sentiment_score)
                    
                    # Confidence is difference between top two scores
                    sorted_scores = sorted(scores.values(), reverse=True)
                    confidence = float(sorted_scores[0] - sorted_scores[1])
                    
                    results.append({
                        'label': label,
                        'scores': scores,
                        'sentiment_score': float(sentiment_score),
                        'confidence': confidence,
                        'method': 'ssl-embeddings',
                    })
        
        return results


class DualSentimentAnalyzer:
    """
    Combines both FinBERT-sentiment and SSL methods for comparison.
    """
    
    def __init__(
        self,
        finbert_model: str = "ProsusAI/finbert",
        ssl_model_path: str = "data/models/finbert_contrastive",
        device: str = "mps",
    ):
        """
        Initialize dual sentiment analyzer.
        
        Args:
            finbert_model: FinBERT-sentiment model name
            ssl_model_path: Path to SSL model
            device: Device to run on
        """
        logger.info("Initializing Dual Sentiment Analyzer")
        
        self.finbert_analyzer = FinBERTSentimentAnalyzer(
            model_name=finbert_model,
            device=device,
        )
        
        self.ssl_analyzer = SSLSentimentAnalyzer(
            ssl_model_path=ssl_model_path,
            device=device,
        )
        
        logger.info("✓ Dual Sentiment Analyzer ready")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze with both methods."""
        results = self.analyze_batch([text])
        return results[0]
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze with both methods and combine results.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of combined results dictionaries
        """
        logger.info(f"Analyzing {len(texts)} texts with both methods...")
        
        # Get results from both methods
        finbert_results = self.finbert_analyzer.analyze_batch(texts)
        ssl_results = self.ssl_analyzer.analyze_batch(texts)
        
        # Combine results
        combined_results = []
        for i, text in enumerate(texts):
            combined = {
                'text': text[:200],  # Truncate for display
                'finbert': finbert_results[i],
                'ssl': ssl_results[i],
                'agreement': finbert_results[i]['label'] == ssl_results[i]['label'],
                'avg_sentiment_score': (
                    finbert_results[i]['sentiment_score'] + 
                    ssl_results[i]['sentiment_score']
                ) / 2.0,
            }
            combined_results.append(combined)
        
        # Calculate agreement rate
        agreement_rate = sum(1 for r in combined_results if r['agreement']) / len(combined_results)
        logger.info(f"✓ Analysis complete. Agreement rate: {agreement_rate:.1%}")
        
        return combined_results


def get_sentiment_analyzer(
    method: str = "dual",
    finbert_model: str = "ProsusAI/finbert",
    ssl_model_path: str = "data/models/finbert_contrastive",
    device: str = "mps",
) -> SentimentAnalyzer:
    """
    Factory function to get sentiment analyzer.
    
    Args:
        method: "finbert", "ssl", or "dual"
        finbert_model: FinBERT model name
        ssl_model_path: Path to SSL model
        device: Device to run on
        
    Returns:
        Sentiment analyzer instance
    """
    if method == "finbert":
        return FinBERTSentimentAnalyzer(finbert_model, device)
    elif method == "ssl":
        return SSLSentimentAnalyzer(ssl_model_path, device)
    elif method == "dual":
        return DualSentimentAnalyzer(finbert_model, ssl_model_path, device)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'finbert', 'ssl', or 'dual'")

