"""
Forex Price Data Fetching Module

This module fetches historical EUR/USD price data from Yahoo Finance.
Designed for forex trading prediction system.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def fetch_eurusd_prices(
    start_date: str = None,
    end_date: str = None,
    years: int = 10,
    output_dir: str = "data/raw/prices"
) -> pd.DataFrame:
    """
    Fetch EUR/USD historical price data from Yahoo Finance.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        years: Number of years to fetch if start_date not provided
        output_dir: Directory to save price data
        
    Returns:
        DataFrame with EUR/USD price data
    """
    # Calculate date range
    if end_date is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date is None:
        start_date = end_date - timedelta(days=years * 365)
    else:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    logger.info(f"Fetching EUR/USD data from {start_date.date()} to {end_date.date()}")
    
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker("EURUSD=X")
        df = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        
        if df.empty:
            logger.error("No data fetched from Yahoo Finance")
            return pd.DataFrame()
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add ticker column
        df['ticker'] = 'EURUSD'
        
        # Select and reorder columns
        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        # Remove timezone info if present
        if pd.api.types.is_datetime64tz_dtype(df['date']):
            df['date'] = df['date'].dt.tz_localize(None)
        
        logger.info(f"✓ Fetched {len(df)} days of EUR/USD data")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Price range: {df['close'].min():.4f} to {df['close'].max():.4f}")
        
        # Save to CSV
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / "eurusd_prices.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"✓ Saved to {csv_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching EUR/USD data: {e}")
        raise


def get_eurusd_info() -> dict:
    """
    Get EUR/USD pair information.
    
    Returns:
        Dictionary with pair information
    """
    return {
        'symbol': 'EURUSD=X',
        'name': 'EUR/USD',
        'type': 'FOREX',
        'base_currency': 'EUR',
        'quote_currency': 'USD',
        'description': 'Euro vs US Dollar exchange rate',
        'market_hours': '24/5 (Sunday 5pm ET - Friday 5pm ET)',
        'pip_value': 0.0001,
        'typical_spread': '0.0001 - 0.0003',
    }


def calculate_forex_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various return metrics for forex data.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with added return columns
    """
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate intraday range
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    df['open_close_change'] = (df['close'] - df['open']) / df['open']
    
    # Calculate pips (for forex: 0.0001 = 1 pip for EUR/USD)
    df['pips_change'] = (df['close'] - df['close'].shift(1)) * 10000
    df['pips_high_low'] = (df['high'] - df['low']) * 10000
    
    # Forward returns (for prediction targets)
    df['returns_1d'] = df['returns'].shift(-1)
    df['log_returns_1d'] = df['log_returns'].shift(-1)
    df['pips_1d'] = df['pips_change'].shift(-1)
    
    # Direction (for classification)
    df['direction_1d'] = (df['returns_1d'] > 0).astype(int)
    
    # Buckets (EUR/USD specific thresholds)
    def assign_bucket(ret):
        if pd.isna(ret):
            return None
        elif ret < -0.005:  # < -0.5%
            return 1  # Large down
        elif ret < -0.002:  # -0.5% to -0.2%
            return 2  # Small down
        elif ret <= 0.002:  # -0.2% to +0.2%
            return 3  # Flat
        elif ret <= 0.005:  # +0.2% to +0.5%
            return 4  # Small up
        else:  # > +0.5%
            return 5  # Large up
    
    df['bucket_1d'] = df['returns_1d'].apply(assign_bucket)
    
    return df


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to forex data.
    
    This is a simplified version. Full indicators will be in Milestone 15.
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    df = df.sort_values('date')
    
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Volatility (ATR approximation)
    df['range'] = df['high'] - df['low']
    df['atr_14'] = df['range'].rolling(window=14).mean()
    
    return df


if __name__ == "__main__":
    # Test the module
    import numpy as np
    
    logging.basicConfig(level=logging.INFO)
    
    # Fetch EUR/USD data
    df = fetch_eurusd_prices(years=10)
    
    print("\n" + "="*60)
    print("EUR/USD Price Data Summary")
    print("="*60)
    print(f"Total days: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nPrice statistics:")
    print(df[['open', 'high', 'low', 'close']].describe())
    
    # Calculate returns
    df = calculate_forex_returns(df)
    print(f"\nReturn statistics:")
    print(df[['returns', 'log_returns', 'pips_change']].describe())
    
    # Add indicators
    df = add_basic_indicators(df)
    print(f"\nIndicators added: {len(df.columns)} total columns")
    
    # Show bucket distribution
    print(f"\nBucket distribution (next-day prediction targets):")
    print(df['bucket_1d'].value_counts().sort_index())
    
    print("\nSample data:")
    print(df[['date', 'close', 'returns', 'pips_change', 'direction_1d', 'bucket_1d']].tail(10))

