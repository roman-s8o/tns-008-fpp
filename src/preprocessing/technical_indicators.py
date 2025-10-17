"""
Technical Indicators for Forex Trading

Implements 9 technical indicators optimized for EUR/USD:
1. SMA (Simple Moving Average)
2. RSI (Relative Strength Index)
3. MACD (Moving Average Convergence Divergence)
4. Bollinger Bands
5. ATR (Average True Range)
6. Fibonacci Retracements
7. Pivot Points
8. Stochastic Oscillator
9. CCI (Commodity Channel Index)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for forex price data.
    """
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Add Simple Moving Averages.
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods for SMA
            
        Returns:
            DataFrame with SMA columns added
        """
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period (default 14)
            
        Returns:
            DataFrame with RSI column
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        
        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal
        
        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            
        Returns:
            DataFrame with MACD columns
        """
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Middle Band = SMA(period)
        Upper Band = Middle + (std_dev * std)
        Lower Band = Middle - (std_dev * std)
        
        Args:
            df: DataFrame with 'close' column
            period: Period for SMA and std calculation
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands columns
        """
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (std_dev * std)
        df['bb_lower'] = df['bb_middle'] - (std_dev * std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR).
        
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = SMA(TR, period)
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default 14)
            
        Returns:
            DataFrame with ATR column
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_fibonacci_retracements(
        df: pd.DataFrame, 
        lookback: int = 20,
        levels: List[float] = [0.236, 0.382, 0.5, 0.618, 0.786]
    ) -> pd.DataFrame:
        """
        Add Fibonacci Retracement levels.
        
        Calculates Fibonacci levels based on swing high/low over lookback period.
        
        Args:
            df: DataFrame with 'high', 'low' columns
            lookback: Period to find swing high/low (default 20)
            levels: Fibonacci retracement levels
            
        Returns:
            DataFrame with Fibonacci columns
        """
        # Find swing high and low over lookback period
        swing_high = df['high'].rolling(window=lookback).max()
        swing_low = df['low'].rolling(window=lookback).min()
        
        diff = swing_high - swing_low
        
        # Calculate Fibonacci levels
        for level in levels:
            df[f'fib_{int(level*1000)}'] = swing_high - (diff * level)
        
        # Also add the extremes
        df['fib_high'] = swing_high
        df['fib_low'] = swing_low
        
        return df
    
    @staticmethod
    def add_pivot_points(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Add Pivot Points.
        
        Standard: P = (H + L + C) / 3
        R1 = 2P - L, S1 = 2P - H
        R2 = P + (H - L), S2 = P - (H - L)
        
        Fibonacci: Uses Fibonacci ratios for support/resistance
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            method: 'standard' or 'fibonacci'
            
        Returns:
            DataFrame with pivot point columns
        """
        # Use previous day's high, low, close
        high = df['high'].shift(1)
        low = df['low'].shift(1)
        close = df['close'].shift(1)
        
        # Pivot point
        pivot = (high + low + close) / 3
        df['pivot'] = pivot
        
        if method == 'standard':
            # Standard pivot points
            df['pivot_r1'] = 2 * pivot - low
            df['pivot_s1'] = 2 * pivot - high
            df['pivot_r2'] = pivot + (high - low)
            df['pivot_s2'] = pivot - (high - low)
            df['pivot_r3'] = high + 2 * (pivot - low)
            df['pivot_s3'] = low - 2 * (high - pivot)
            
        elif method == 'fibonacci':
            # Fibonacci pivot points
            range_hl = high - low
            df['pivot_r1'] = pivot + (range_hl * 0.382)
            df['pivot_s1'] = pivot - (range_hl * 0.382)
            df['pivot_r2'] = pivot + (range_hl * 0.618)
            df['pivot_s2'] = pivot - (range_hl * 0.618)
            df['pivot_r3'] = pivot + (range_hl * 1.000)
            df['pivot_s3'] = pivot - (range_hl * 1.000)
        
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.
        
        %K = 100 * (Close - Low_n) / (High_n - Low_n)
        %D = SMA(%K, d_period)
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            k_period: Period for %K (default 14)
            d_period: Period for %D (default 3)
            
        Returns:
            DataFrame with Stochastic columns
        """
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add Commodity Channel Index (CCI).
        
        TP = (High + Low + Close) / 3
        CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: CCI period (default 20)
            
        Returns:
            DataFrame with CCI column
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        
        # Mean deviation
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mean_dev)
        
        return df
    
    @staticmethod
    def add_all_indicators(
        df: pd.DataFrame,
        fibonacci_lookback: int = 20,
        pivot_method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Add all technical indicators at once.
        
        Args:
            df: DataFrame with OHLC data
            fibonacci_lookback: Lookback period for Fibonacci
            pivot_method: Method for pivot points ('standard' or 'fibonacci')
            
        Returns:
            DataFrame with all indicators
        """
        logger.info("Adding all technical indicators...")
        
        # Ensure data is sorted by date
        df = df.sort_values('date').copy()
        
        # 1. SMA
        df = TechnicalIndicators.add_sma(df)
        logger.info("  ✓ Added SMA")
        
        # 2. RSI
        df = TechnicalIndicators.add_rsi(df)
        logger.info("  ✓ Added RSI")
        
        # 3. MACD
        df = TechnicalIndicators.add_macd(df)
        logger.info("  ✓ Added MACD")
        
        # 4. Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df)
        logger.info("  ✓ Added Bollinger Bands")
        
        # 5. ATR
        df = TechnicalIndicators.add_atr(df)
        logger.info("  ✓ Added ATR")
        
        # 6. Fibonacci Retracements
        df = TechnicalIndicators.add_fibonacci_retracements(df, lookback=fibonacci_lookback)
        logger.info("  ✓ Added Fibonacci Retracements")
        
        # 7. Pivot Points
        df = TechnicalIndicators.add_pivot_points(df, method=pivot_method)
        logger.info("  ✓ Added Pivot Points")
        
        # 8. Stochastic
        df = TechnicalIndicators.add_stochastic(df)
        logger.info("  ✓ Added Stochastic Oscillator")
        
        # 9. CCI
        df = TechnicalIndicators.add_cci(df)
        logger.info("  ✓ Added CCI")
        
        logger.info(f"✓ All indicators added. Total columns: {len(df.columns)}")
        
        return df


def test_indicators():
    """Test technical indicators on EUR/USD data."""
    logging.basicConfig(level=logging.INFO)
    
    # Load EUR/USD data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data_ingestion.forex_prices import fetch_eurusd_prices, calculate_forex_returns
    
    logger.info("Fetching EUR/USD data...")
    df = fetch_eurusd_prices(years=1)
    df = calculate_forex_returns(df)
    
    logger.info(f"Loaded {len(df)} days of data")
    
    # Add all indicators
    df = TechnicalIndicators.add_all_indicators(df, fibonacci_lookback=20)
    
    # Display sample
    print("\n" + "="*60)
    print("Technical Indicators Test - EUR/USD")
    print("="*60)
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"\nIndicator columns added:")
    
    indicator_cols = [col for col in df.columns if col not in 
                     ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
    for col in sorted(indicator_cols):
        print(f"  - {col}")
    
    print(f"\nSample data (last 5 rows):")
    display_cols = ['date', 'close', 'rsi_14', 'macd', 'bb_percent', 'atr_14', 'stoch_k']
    print(df[display_cols].tail())
    
    # Statistics
    print(f"\nIndicator Statistics:")
    print(df[['rsi_14', 'macd', 'cci_20', 'stoch_k']].describe())


if __name__ == "__main__":
    test_indicators()

