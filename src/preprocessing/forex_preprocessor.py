"""
EUR/USD Forex Data Preprocessor

Prepares EUR/USD sequences for model training:
1. Formats text with structured tags
2. Normalizes numerical features
3. Handles missing values
4. Creates labels for multi-task learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ForexDataPreprocessor:
    """
    Preprocess EUR/USD forex data for model training.
    """
    
    def __init__(
        self,
        normalize_features: bool = True,
        fill_nan_strategy: str = 'mean',  # 'mean', 'median', 'zero', 'forward_fill'
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalize_features: Whether to normalize numerical features
            fill_nan_strategy: Strategy for handling NaN values
        """
        self.normalize_features = normalize_features
        self.fill_nan_strategy = fill_nan_strategy
        
        # Feature statistics (computed during fit)
        self.feature_means = {}
        self.feature_stds = {}
        self.is_fitted = False
        
        logger.info("ForexDataPreprocessor initialized")
    
    def format_text_with_tags(self, row: pd.Series) -> str:
        """
        Format sequence as structured text with tags.
        
        Format:
        [NEWS] headline text [PRICE] EUR/USD: 1.1234 +15 pips 
        [IND] RSI:52 MACD:-0.002 BB:0.34 [CAL] Fed today
        
        Args:
            row: DataFrame row with sequence data
            
        Returns:
            Formatted text string
        """
        parts = []
        
        # NEWS section
        news_text = row.get('news_text', '')
        if news_text and news_text.strip():
            news_snippet = news_text[:200]  # Limit length
            parts.append(f"[NEWS] {news_snippet}")
        else:
            parts.append("[NEWS] No news today")
        
        # PRICE section
        close = row.get('close', 0)
        pips = row.get('pips_change', 0)
        direction_word = "up" if pips > 0 else "down"
        parts.append(f"[PRICE] EUR/USD: {close:.4f} {direction_word} {abs(pips):.1f} pips")
        
        # INDICATORS section
        indicators = []
        if not pd.isna(row.get('rsi_14')):
            indicators.append(f"RSI:{row['rsi_14']:.1f}")
        if not pd.isna(row.get('macd')):
            indicators.append(f"MACD:{row['macd']:.4f}")
        if not pd.isna(row.get('bb_percent')):
            indicators.append(f"BB:{row['bb_percent']:.2f}")
        if not pd.isna(row.get('atr_14')):
            indicators.append(f"ATR:{row['atr_14']:.4f}")
        if not pd.isna(row.get('stoch_k')):
            indicators.append(f"STOCH:{row['stoch_k']:.1f}")
        
        if indicators:
            parts.append(f"[IND] {' '.join(indicators)}")
        
        # CALENDAR section
        calendar_events = []
        if row.get('fed_today', 0) == 1:
            calendar_events.append("Fed meeting")
        if row.get('ecb_today', 0) == 1:
            calendar_events.append("ECB meeting")
        if row.get('nfp_today', 0) == 1:
            calendar_events.append("NFP release")
        
        days_to_fed = row.get('days_to_fed_rate', 999)
        if days_to_fed < 7:
            calendar_events.append(f"Fed in {int(days_to_fed)}d")
        
        if calendar_events:
            parts.append(f"[CAL] {', '.join(calendar_events)}")
        else:
            parts.append("[CAL] No major events")
        
        return " ".join(parts)
    
    def _batch_format_text(self, df: pd.DataFrame) -> List[str]:
        """
        Fast batch text formatting (vectorized).
        
        Args:
            df: DataFrame with sequences
            
        Returns:
            List of formatted texts
        """
        texts = []
        
        # Pre-compute common values
        news_texts = df['news_text'].fillna('').values
        closes = df['close'].fillna(0).values
        pips = df['pips_change'].fillna(0).values
        
        # Indicators
        rsi = df['rsi_14'].fillna(0).values
        macd = df['macd'].fillna(0).values
        bb = df['bb_percent'].fillna(0).values
        
        # Calendar
        fed_today = df['fed_today'].fillna(0).values
        ecb_today = df['ecb_today'].fillna(0).values
        nfp_today = df['nfp_today'].fillna(0).values
        
        for i in range(len(df)):
            # NEWS
            news = news_texts[i][:200] if news_texts[i] else "No news today"
            news_part = f"[NEWS] {news}"
            
            # PRICE
            direction = "up" if pips[i] > 0 else "down"
            price_part = f"[PRICE] EUR/USD: {closes[i]:.4f} {direction} {abs(pips[i]):.1f} pips"
            
            # INDICATORS
            ind_part = f"[IND] RSI:{rsi[i]:.1f} MACD:{macd[i]:.4f} BB:{bb[i]:.2f}"
            
            # CALENDAR
            events = []
            if fed_today[i] == 1:
                events.append("Fed meeting")
            if ecb_today[i] == 1:
                events.append("ECB meeting")
            if nfp_today[i] == 1:
                events.append("NFP release")
            
            cal_part = f"[CAL] {', '.join(events)}" if events else "[CAL] No major events"
            
            texts.append(f"{news_part} {price_part} {ind_part} {cal_part}")
        
        return texts
    
    def normalize_numerical_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Normalize numerical features using z-score normalization.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to normalize
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        for col in feature_columns:
            if col in df.columns:
                if self.is_fitted:
                    # Use stored statistics
                    mean = self.feature_means.get(col, 0)
                    std = self.feature_stds.get(col, 1)
                else:
                    # Compute and store statistics
                    mean = df[col].mean()
                    std = df[col].std()
                    self.feature_means[col] = mean
                    self.feature_stds[col] = std if std > 0 else 1.0
                
                # Normalize
                df[col] = (df[col] - mean) / (std if std > 0 else 1.0)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            df: DataFrame with features
            feature_columns: List of columns to check
            
        Returns:
            DataFrame with NaN values handled
        """
        df = df.copy()
        
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            if df[col].isna().any():
                if self.fill_nan_strategy == 'mean':
                    fill_value = df[col].mean()
                elif self.fill_nan_strategy == 'median':
                    fill_value = df[col].median()
                elif self.fill_nan_strategy == 'zero':
                    fill_value = 0
                elif self.fill_nan_strategy == 'forward_fill':
                    df[col] = df[col].fillna(method='ffill')
                    continue
                else:
                    fill_value = 0
                
                df[col] = df[col].fillna(fill_value)
        
        return df
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        fit: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame with sequences
            fit: Whether to fit normalization parameters (True for train, False for val/test)
            
        Returns:
            Tuple of (processed_df, formatted_texts)
        """
        logger.info(f"Preprocessing {len(df)} sequences...")
        
        df = df.copy()
        
        # Define feature columns to normalize
        feature_cols = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_percent', 'bb_width', 'atr_14',
            'stoch_k', 'stoch_d', 'cci_20',
            'sma_20', 'sma_50', 'sma_200',
        ]
        
        # Handle missing values
        df = self.handle_missing_values(df, feature_cols)
        logger.info("  ✓ Handled missing values")
        
        # Normalize features
        if self.normalize_features:
            if fit:
                self.is_fitted = True
            df = self.normalize_numerical_features(df, feature_cols)
            logger.info("  ✓ Normalized features")
        
        # Format text (vectorized for speed)
        logger.info("  Formatting text sequences...")
        formatted_texts = self._batch_format_text(df)
        
        logger.info("  ✓ Formatted text sequences")
        
        # Prepare labels
        df['direction_label'] = df['direction_1d'].astype(int)
        df['bucket_label'] = (df['bucket_1d'] - 1).astype(int)  # Convert 1-5 to 0-4 for 0-indexed
        
        logger.info("  ✓ Prepared labels")
        
        return df, formatted_texts
    
    def create_training_batch(
        self,
        df: pd.DataFrame,
        texts: List[str],
        indices: List[int]
    ) -> Dict:
        """
        Create a training batch from indices.
        
        Args:
            df: Preprocessed DataFrame
            texts: List of formatted text strings
            indices: Indices to include in batch
            
        Returns:
            Dictionary with texts and labels
        """
        batch_texts = [texts[i] for i in indices]
        batch_directions = df.iloc[indices]['direction_label'].values
        batch_buckets = df.iloc[indices]['bucket_label'].values
        
        return {
            'texts': batch_texts,
            'direction_labels': batch_directions,
            'bucket_labels': batch_buckets,
        }


def test_preprocessor():
    """Test the forex data preprocessor."""
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    df = pd.read_parquet('data/processed/train.parquet')
    logger.info(f"Loaded {len(df)} training sequences")
    
    # Initialize preprocessor
    preprocessor = ForexDataPreprocessor(normalize_features=True, fill_nan_strategy='mean')
    
    # Preprocess
    df_processed, texts = preprocessor.prepare_dataset(df[:100], fit=True)
    
    print("\n" + "="*60)
    print("Forex Data Preprocessor Test")
    print("="*60)
    print(f"\nProcessed {len(df_processed)} sequences")
    print(f"\nSample formatted text:")
    print(texts[0])
    print(f"\nLabels:")
    print(f"  Direction: {df_processed.iloc[0]['direction_label']}")
    print(f"  Bucket: {df_processed.iloc[0]['bucket_label']}")
    
    print(f"\nNormalized features (sample RSI):")
    print(df_processed['rsi_14'].describe())
    
    print("\n✓ Preprocessor test successful!")


if __name__ == "__main__":
    test_preprocessor()

