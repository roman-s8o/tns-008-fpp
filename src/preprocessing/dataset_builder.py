"""
Full dataset construction with train/validation/fine-tuning splits.

This module handles building complete datasets with temporal splits
and incremental updates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.sequence_builder import SequenceBuilder
from loguru import logger


class DatasetBuilder:
    """Builds complete datasets with train/val/test splits."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize dataset builder.
        
        Args:
            config_path: Path to configuration file
        """
        self.sequence_builder = SequenceBuilder(config_path)
        self.config = self.sequence_builder.config
        logger.info("Initialized DatasetBuilder")
    
    @staticmethod
    def is_trading_day(date: datetime) -> bool:
        """
        Check if a date is a trading day (Monday-Friday, excluding holidays).
        
        Args:
            date: Date to check
            
        Returns:
            True if trading day, False otherwise
        """
        # Check if weekend
        if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # TODO: Add major US market holidays (NYSE)
        # For now, just check weekends
        return True
    
    @staticmethod
    def map_to_next_trading_day(date: datetime) -> datetime:
        """
        Map a date to the next trading day.
        
        If the date falls on a weekend, map to next Monday.
        
        Args:
            date: Input date
            
        Returns:
            Next trading day
        """
        while not DatasetBuilder.is_trading_day(date):
            date = date + timedelta(days=1)
        return date
    
    def build_full_dataset(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_weekends: bool = True,
        max_sequences: int = 100000
    ) -> pd.DataFrame:
        """
        Build full dataset with weekend news mapping.
        
        Args:
            tickers: List of tickers to include
            start_date: Start date for dataset
            end_date: End date for dataset
            include_weekends: Map weekend news to next trading day
            max_sequences: Maximum sequences to generate
            
        Returns:
            DataFrame with all sequences
        """
        logger.info("Building full dataset...")
        
        # Build sequences
        sequences = self.sequence_builder.build_dataset(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            max_sequences=max_sequences
        )
        
        if not sequences:
            logger.error("No sequences generated")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(sequences)
        
        # Handle weekend news mapping
        if include_weekends:
            df = self._map_weekend_news(df)
        
        # Add metadata
        df['created_at'] = datetime.now()
        
        logger.info(f"Built dataset with {len(df)} sequences")
        return df
    
    def _map_weekend_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map weekend news to next trading day.
        
        Args:
            df: DataFrame with sequences
            
        Returns:
            DataFrame with mapped dates
        """
        # Convert date strings to datetime
        df['date_dt'] = pd.to_datetime(df['date'])
        
        # Identify weekend dates
        df['is_weekend'] = df['date_dt'].dt.dayofweek >= 5
        
        # Map to next trading day
        df['trading_date'] = df['date_dt'].apply(
            lambda d: self.map_to_next_trading_day(d)
        )
        
        # Update date column
        df['original_date'] = df['date']
        df['date'] = df['trading_date'].dt.strftime('%Y-%m-%d')
        
        weekend_count = df['is_weekend'].sum()
        if weekend_count > 0:
            logger.info(f"Mapped {weekend_count} weekend news items to next trading day")
        
        # Drop temporary columns
        df = df.drop(columns=['date_dt', 'is_weekend', 'trading_date'])
        
        return df
    
    def create_temporal_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        finetune_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/fine-tuning splits.
        
        Most recent data goes to fine-tuning set (for incremental learning).
        
        Args:
            df: Full dataset DataFrame
            train_ratio: Ratio for training set (default: 0.8)
            val_ratio: Ratio for validation set (default: 0.1)
            finetune_ratio: Ratio for fine-tuning set (default: 0.1)
            
        Returns:
            Tuple of (train_df, val_df, finetune_df)
        """
        if abs(train_ratio + val_ratio + finetune_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Sort by date (temporal order)
        df = df.sort_values('date').reset_index(drop=True)
        
        total = len(df)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Split temporally
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        finetune_df = df.iloc[train_size+val_size:].copy()
        
        # Add split labels
        train_df['split'] = 'train'
        val_df['split'] = 'validation'
        finetune_df['split'] = 'finetune'
        
        logger.info(f"Temporal splits created:")
        logger.info(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"    Date range: {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"  Validation: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
        logger.info(f"    Date range: {val_df['date'].min()} to {val_df['date'].max()}")
        logger.info(f"  Fine-tune: {len(finetune_df)} ({len(finetune_df)/total*100:.1f}%)")
        logger.info(f"    Date range: {finetune_df['date'].min()} to {finetune_df['date'].max()}")
        
        return train_df, val_df, finetune_df
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        finetune_df: pd.DataFrame,
        output_dir: str = "data/processed"
    ):
        """
        Save train/val/finetune splits to disk.
        
        Args:
            train_df: Training set
            val_df: Validation set
            finetune_df: Fine-tuning set
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        train_df.to_parquet(output_path / "train.parquet", index=False)
        val_df.to_parquet(output_path / "validation.parquet", index=False)
        finetune_df.to_parquet(output_path / "finetune.parquet", index=False)
        
        # Also save combined dataset
        full_df = pd.concat([train_df, val_df, finetune_df], ignore_index=True)
        full_df.to_parquet(output_path / "full_dataset.parquet", index=False)
        
        logger.info(f"Saved datasets to {output_dir}:")
        logger.info(f"  train.parquet: {len(train_df)} sequences")
        logger.info(f"  validation.parquet: {len(val_df)} sequences")
        logger.info(f"  finetune.parquet: {len(finetune_df)} sequences")
        logger.info(f"  full_dataset.parquet: {len(full_df)} sequences")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_sequences': len(full_df),
            'train_size': len(train_df),
            'val_size': len(val_df),
            'finetune_size': len(finetune_df),
            'tickers': sorted(full_df['ticker'].unique().tolist()),
            'date_range': f"{full_df['date'].min()} to {full_df['date'].max()}",
            'split_ratios': {
                'train': len(train_df) / len(full_df),
                'validation': len(val_df) / len(full_df),
                'finetune': len(finetune_df) / len(full_df)
            }
        }
        
        import json
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata.json")
    
    def incremental_update(
        self,
        existing_dataset_path: str,
        new_start_date: Optional[datetime] = None,
        new_end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Incrementally update an existing dataset with new sequences.
        
        Args:
            existing_dataset_path: Path to existing full_dataset.parquet
            new_start_date: Start date for new data
            new_end_date: End date for new data
            
        Returns:
            Updated full dataset
        """
        logger.info("Performing incremental dataset update...")
        
        # Load existing dataset
        existing_df = pd.read_parquet(existing_dataset_path)
        logger.info(f"Loaded existing dataset: {len(existing_df)} sequences")
        
        # Determine date range for new data
        if new_start_date is None:
            # Start from day after last date in existing dataset
            last_date = pd.to_datetime(existing_df['date']).max()
            new_start_date = last_date + timedelta(days=1)
        
        # Build new sequences
        new_sequences = self.sequence_builder.build_dataset(
            start_date=new_start_date,
            end_date=new_end_date,
            max_sequences=100000
        )
        
        if not new_sequences:
            logger.warning("No new sequences generated")
            return existing_df
        
        # Convert to DataFrame
        new_df = pd.DataFrame(new_sequences)
        new_df['created_at'] = datetime.now()
        
        # Map weekend news
        new_df = self._map_weekend_news(new_df)
        
        # Combine with existing
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates (same ticker + date)
        updated_df = updated_df.drop_duplicates(
            subset=['ticker', 'date'],
            keep='last'  # Keep newest version
        )
        
        logger.info(f"Added {len(new_df)} new sequences")
        logger.info(f"Updated dataset: {len(updated_df)} total sequences")
        
        return updated_df
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate dataset statistics.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_sequences': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'unique_dates': df['date'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'avg_news_per_sequence': df['news_count'].mean(),
            'tickers': sorted(df['ticker'].unique().tolist()),
            'sequences_per_ticker': df['ticker'].value_counts().to_dict(),
        }
        
        if 'split' in df.columns:
            stats['split_distribution'] = df['split'].value_counts().to_dict()
        
        return stats


if __name__ == "__main__":
    # Test dataset builder
    builder = DatasetBuilder()
    
    # Load price data
    builder.sequence_builder.load_and_process_prices()
    
    # Build dataset
    df = builder.build_full_dataset(max_sequences=1000)
    
    if not df.empty:
        print(f"\nBuilt dataset: {len(df)} sequences")
        
        # Create splits
        train, val, finetune = builder.create_temporal_splits(df)
        
        # Show statistics
        stats = builder.get_dataset_statistics(df)
        print(f"\nDataset statistics:")
        print(f"  Total sequences: {stats['total_sequences']}")
        print(f"  Unique tickers: {stats['unique_tickers']}")
        print(f"  Date range: {stats['date_range']}")

