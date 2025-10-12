"""
Price data preprocessing utilities.

This module handles price normalization, log-returns calculation,
and other preprocessing tasks for stock price data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from loguru import logger


class PriceProcessor:
    """Processes and normalizes price data."""
    
    def __init__(
        self,
        calc_log_returns: bool = True,
        calc_ohlc_returns: bool = True,
        calc_volume_change: bool = True
    ):
        """
        Initialize price processor.
        
        Args:
            calc_log_returns: Calculate standard log returns
            calc_ohlc_returns: Calculate OHLC-based returns
            calc_volume_change: Calculate volume change percentage
        """
        self.calc_log_returns = calc_log_returns
        self.calc_ohlc_returns = calc_ohlc_returns
        self.calc_volume_change = calc_volume_change
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """
        Calculate log returns: log(Price_t / Price_{t-1}).
        
        Args:
            prices: Series of prices
            
        Returns:
            Series of log returns
        """
        # Avoid log(0) or division by zero
        returns = np.log(prices / prices.shift(1))
        
        # Replace inf and nan with 0
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
        
        return returns
    
    @staticmethod
    def calculate_open_to_close_return(open_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
        """
        Calculate intraday return: (Close - Open) / Open.
        
        Args:
            open_prices: Series of opening prices
            close_prices: Series of closing prices
            
        Returns:
            Series of intraday returns
        """
        returns = (close_prices - open_prices) / open_prices
        
        # Replace inf and nan with 0
        returns = returns.replace([np.inf, -np.inf], 0).fillna(0)
        
        return returns
    
    @staticmethod
    def calculate_high_low_range(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
        """
        Calculate high-low range: (High - Low) / Close.
        
        Args:
            high_prices: Series of high prices
            low_prices: Series of low prices
            close_prices: Series of closing prices
            
        Returns:
            Series of high-low ranges
        """
        range_pct = (high_prices - low_prices) / close_prices
        
        # Replace inf and nan with 0
        range_pct = range_pct.replace([np.inf, -np.inf], 0).fillna(0)
        
        return range_pct
    
    @staticmethod
    def calculate_volume_change(volumes: pd.Series) -> pd.Series:
        """
        Calculate volume change percentage.
        
        Args:
            volumes: Series of trading volumes
            
        Returns:
            Series of volume change percentages
        """
        volume_change = (volumes - volumes.shift(1)) / volumes.shift(1)
        
        # Replace inf and nan with 0
        volume_change = volume_change.replace([np.inf, -np.inf], 0).fillna(0)
        
        return volume_change
    
    def process_ticker_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process price data for a single ticker.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional calculated features
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure Date column is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            # Convert to timezone-naive for easier processing
            if hasattr(df['Date'].dtype, 'tz') and df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            df = df.sort_values('Date')
        
        # Calculate standard log returns
        if self.calc_log_returns and 'Close' in df.columns:
            df['log_return'] = self.calculate_log_returns(df['Close'])
            df['log_return_pct'] = df['log_return'] * 100  # Convert to percentage
        
        # Calculate OHLC-based returns
        if self.calc_ohlc_returns:
            if 'Open' in df.columns and 'Close' in df.columns:
                df['open_close_return'] = self.calculate_open_to_close_return(
                    df['Open'], df['Close']
                )
                df['open_close_return_pct'] = df['open_close_return'] * 100
            
            if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
                df['high_low_range'] = self.calculate_high_low_range(
                    df['High'], df['Low'], df['Close']
                )
                df['high_low_range_pct'] = df['high_low_range'] * 100
        
        # Calculate volume change
        if self.calc_volume_change and 'Volume' in df.columns:
            df['volume_change'] = self.calculate_volume_change(df['Volume'])
            df['volume_change_pct'] = df['volume_change'] * 100
        
        return df
    
    def process_multiple_tickers(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Process price data for multiple tickers.
        
        Args:
            data_dict: Dictionary of ticker -> DataFrame
            
        Returns:
            Dictionary of processed DataFrames
        """
        processed = {}
        
        for ticker, df in data_dict.items():
            try:
                processed[ticker] = self.process_ticker_data(df)
                logger.debug(f"Processed {ticker}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
        
        logger.info(f"Processed {len(processed)} tickers")
        return processed
    
    @staticmethod
    def get_price_features_for_date(
        df: pd.DataFrame,
        date: pd.Timestamp,
        ticker: str
    ) -> Optional[Dict]:
        """
        Get price features for a specific date.
        
        Args:
            df: Processed price DataFrame
            date: Target date
            ticker: Stock ticker
            
        Returns:
            Dictionary of price features, or None if not found
        """
        # Filter by date (handle timezone-aware and naive datetimes)
        try:
            if hasattr(date, 'tz') and date.tz is not None:
                # Date is timezone-aware
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                date = date.tz_localize(None)
            
            # Convert to date only for comparison
            df['DateOnly'] = pd.to_datetime(df['Date']).dt.date
            target_date = date.date() if hasattr(date, 'date') else date
            
            row = df[df['DateOnly'] == target_date]
            
            if row.empty:
                return None
            
            row = row.iloc[0]
            
            # Build feature dictionary
            features = {
                'ticker': ticker,
                'date': str(date),
                'open': float(row.get('Open', 0)),
                'high': float(row.get('High', 0)),
                'low': float(row.get('Low', 0)),
                'close': float(row.get('Close', 0)),
                'volume': float(row.get('Volume', 0)),
            }
            
            # Add calculated features if available
            if 'log_return_pct' in row:
                features['log_return'] = float(row['log_return_pct'])
            
            if 'open_close_return_pct' in row:
                features['open_close_return'] = float(row['open_close_return_pct'])
            
            if 'high_low_range_pct' in row:
                features['high_low_range'] = float(row['high_low_range_pct'])
            
            if 'volume_change_pct' in row:
                features['volume_change'] = float(row['volume_change_pct'])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {ticker} on {date}: {e}")
            return None
    
    @staticmethod
    def validate_price_data(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate price data quality.
        
        Args:
            df: Price DataFrame
            
        Returns:
            Validation results dictionary
        """
        issues = []
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for missing values
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            issues.append(f"Null values: {null_counts[null_counts > 0].to_dict()}")
        
        # Check High >= Low
        if 'High' in df.columns and 'Low' in df.columns:
            invalid = (df['High'] < df['Low']).sum()
            if invalid > 0:
                issues.append(f"High < Low: {invalid} rows")
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    issues.append(f"Negative {col}: {negative} rows")
        
        # Check for extreme returns (potential errors)
        if 'log_return' in df.columns:
            extreme = (np.abs(df['log_return']) > 0.5).sum()  # >50% daily change
            if extreme > 0:
                issues.append(f"Extreme returns (>50%): {extreme} rows")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_rows': len(df),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else None
        }


if __name__ == "__main__":
    # Test price processor
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data_ingestion.data_utils import load_ticker_data
    from pathlib import Path
    
    # Load sample data
    df = load_ticker_data('AAPL', 'data/raw/prices')
    
    if df is not None:
        print(f"Loaded AAPL data: {len(df)} rows")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Process
        processor = PriceProcessor()
        processed = processor.process_ticker_data(df)
        
        print(f"\nProcessed columns: {list(processed.columns)}")
        print(f"\nSample processed data:")
        print(processed[['Date', 'Close', 'log_return_pct', 'open_close_return_pct']].tail())
        
        # Validate
        validation = processor.validate_price_data(processed)
        print(f"\nValidation: {'✓ Valid' if validation['valid'] else '✗ Invalid'}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")
    else:
        print("Could not load AAPL data. Run price ingestion first.")

