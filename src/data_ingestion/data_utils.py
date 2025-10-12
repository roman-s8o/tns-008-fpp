"""
Utility functions for data management and analysis.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_ticker_data(ticker: str, data_dir: str = "data/raw/prices") -> Optional[pd.DataFrame]:
    """
    Load price data for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing CSV files
        
    Returns:
        DataFrame with price data, or None if not found
    """
    csv_file = Path(data_dir) / f"{ticker}.csv"
    
    if not csv_file.exists():
        logger.error(f"Data file not found for {ticker}")
        return None
    
    try:
        df = pd.read_csv(csv_file, parse_dates=['Date'])
        return df
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {e}")
        return None


def load_combined_data(data_dir: str = "data/raw/prices") -> Optional[pd.DataFrame]:
    """
    Load combined dataset with all tickers.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        DataFrame with all price data, or None if not found
    """
    csv_file = Path(data_dir) / "nasdaq100_combined.csv"
    
    if not csv_file.exists():
        logger.error("Combined data file not found")
        return None
    
    try:
        df = pd.read_csv(csv_file, parse_dates=['Date'])
        return df
    except Exception as e:
        logger.error(f"Error loading combined data: {e}")
        return None


def get_available_tickers(data_dir: str = "data/raw/prices") -> List[str]:
    """
    Get list of tickers that have been downloaded.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of available ticker symbols
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_dir}")
        return []
    
    csv_files = list(data_path.glob("*.csv"))
    
    # Exclude combined file
    tickers = [
        f.stem for f in csv_files 
        if f.name != "nasdaq100_combined.csv"
    ]
    
    return sorted(tickers)


def get_data_statistics(data_dir: str = "data/raw/prices") -> Dict[str, any]:
    """
    Get statistics about downloaded data.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Dictionary with statistics
    """
    tickers = get_available_tickers(data_dir)
    
    if not tickers:
        return {
            "ticker_count": 0,
            "total_rows": 0,
            "date_range": None
        }
    
    # Load a sample to get date range
    sample_df = load_ticker_data(tickers[0], data_dir)
    
    if sample_df is None:
        return {
            "ticker_count": len(tickers),
            "total_rows": 0,
            "date_range": None
        }
    
    # Get overall statistics
    min_date = sample_df['Date'].min()
    max_date = sample_df['Date'].max()
    
    # Count total rows across all tickers
    total_rows = 0
    for ticker in tickers:
        df = load_ticker_data(ticker, data_dir)
        if df is not None:
            total_rows += len(df)
    
    return {
        "ticker_count": len(tickers),
        "tickers": tickers,
        "total_rows": total_rows,
        "date_range": f"{min_date.date()} to {max_date.date()}",
        "avg_rows_per_ticker": total_rows // len(tickers) if tickers else 0
    }


def validate_price_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate price data quality.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
    
    # Check for negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            negative = (df[col] < 0).sum()
            if negative > 0:
                issues.append(f"Negative values in {col}: {negative} rows")
    
    # Check High >= Low
    if 'High' in df.columns and 'Low' in df.columns:
        invalid = (df['High'] < df['Low']).sum()
        if invalid > 0:
            issues.append(f"High < Low: {invalid} rows")
    
    # Check for duplicate dates
    if 'Date' in df.columns:
        duplicates = df['Date'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Duplicate dates: {duplicates} rows")
    
    # Check volume
    if 'Volume' in df.columns:
        zero_volume = (df['Volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"Zero volume: {zero_volume} rows")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_rows": len(df)
    }


def get_price_summary(ticker: str, data_dir: str = "data/raw/prices") -> Optional[Dict[str, any]]:
    """
    Get summary statistics for a ticker's price data.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Directory containing CSV files
        
    Returns:
        Dictionary with summary statistics
    """
    df = load_ticker_data(ticker, data_dir)
    
    if df is None:
        return None
    
    return {
        "ticker": ticker,
        "rows": len(df),
        "date_range": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
        "close_stats": {
            "min": df['Close'].min(),
            "max": df['Close'].max(),
            "mean": df['Close'].mean(),
            "std": df['Close'].std()
        },
        "volume_stats": {
            "min": df['Volume'].min(),
            "max": df['Volume'].max(),
            "mean": df['Volume'].mean()
        }
    }


if __name__ == "__main__":
    # Test utilities
    logging.basicConfig(level=logging.INFO)
    
    print("Getting data statistics...")
    stats = get_data_statistics()
    print(f"\nData Statistics:")
    print(f"  Tickers: {stats['ticker_count']}")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Date range: {stats['date_range']}")
    
    if stats['ticker_count'] > 0:
        # Test loading a sample ticker
        sample_ticker = stats['tickers'][0]
        print(f"\nLoading sample ticker: {sample_ticker}")
        df = load_ticker_data(sample_ticker)
        if df is not None:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"\n  First 3 rows:")
            print(df.head(3))

