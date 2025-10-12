"""
Tests for data ingestion module.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.nasdaq_tickers import get_nasdaq100_tickers, get_nasdaq100_tickers_from_web
from src.data_ingestion.fetch_prices import PriceDataFetcher
from src.data_ingestion.data_utils import validate_price_data


class TestNasdaqTickers:
    """Tests for ticker list management."""
    
    def test_get_nasdaq100_tickers(self):
        """Test getting hardcoded ticker list."""
        tickers = get_nasdaq100_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers
        
        # Check no duplicates
        assert len(tickers) == len(set(tickers))
    
    def test_get_nasdaq100_tickers_from_web(self):
        """Test fetching tickers from web."""
        tickers = get_nasdaq100_tickers_from_web()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 80  # Nasdaq-100 should have close to 100 stocks
        
        # Should contain major tech stocks
        major_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        for stock in major_stocks:
            assert stock in tickers or stock.replace("L", "") in tickers


class TestPriceDataFetcher:
    """Tests for price data fetching."""
    
    def test_initialization(self, tmp_path):
        """Test fetcher initialization."""
        fetcher = PriceDataFetcher(
            output_dir=str(tmp_path),
            lookback_years=1
        )
        
        assert fetcher.output_dir == tmp_path
        assert fetcher.lookback_years == 1
        assert fetcher.start_date < fetcher.end_date
    
    def test_fetch_ticker_data(self, tmp_path):
        """Test fetching data for a single ticker."""
        fetcher = PriceDataFetcher(
            output_dir=str(tmp_path),
            lookback_years=1
        )
        
        # Test with a known good ticker
        df = fetcher.fetch_ticker_data("AAPL")
        
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Check required columns
        required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in df.columns
        
        # Check ticker is correct
        assert df['Ticker'].iloc[0] == "AAPL"
    
    def test_fetch_invalid_ticker(self, tmp_path):
        """Test fetching data for an invalid ticker."""
        fetcher = PriceDataFetcher(
            output_dir=str(tmp_path),
            lookback_years=1
        )
        
        # This should return None or empty DataFrame
        df = fetcher.fetch_ticker_data("INVALID_TICKER_XYZ")
        
        assert df is None or df.empty
    
    def test_save_ticker_data(self, tmp_path):
        """Test saving ticker data to CSV."""
        fetcher = PriceDataFetcher(
            output_dir=str(tmp_path),
            lookback_years=1
        )
        
        # Create sample data
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Ticker': ['AAPL'] * 5,
            'Open': [150.0, 151.0, 152.0, 153.0, 154.0],
            'High': [151.0, 152.0, 153.0, 154.0, 155.0],
            'Low': [149.0, 150.0, 151.0, 152.0, 153.0],
            'Close': [150.5, 151.5, 152.5, 153.5, 154.5],
            'Volume': [1000000] * 5
        })
        
        # Save data
        success = fetcher.save_ticker_data("AAPL", df)
        assert success
        
        # Check file exists
        csv_file = tmp_path / "AAPL.csv"
        assert csv_file.exists()
        
        # Load and verify
        loaded_df = pd.read_csv(csv_file)
        assert len(loaded_df) == 5
        assert list(loaded_df.columns) == list(df.columns)
    
    def test_validate_data(self, tmp_path):
        """Test data validation."""
        fetcher = PriceDataFetcher(
            output_dir=str(tmp_path),
            lookback_years=1
        )
        
        # Create and save sample data
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Ticker': ['AAPL'] * 10,
            'Open': [150.0] * 10,
            'High': [151.0] * 10,
            'Low': [149.0] * 10,
            'Close': [150.5] * 10,
            'Volume': [1000000] * 10
        })
        fetcher.save_ticker_data("AAPL", df)
        
        # Validate
        validation = fetcher.validate_data("AAPL")
        
        assert validation["valid"]
        assert validation["rows"] == 10
        assert "date_range" in validation


class TestDataValidation:
    """Tests for data validation utilities."""
    
    def test_validate_good_data(self):
        """Test validation of good quality data."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=10),
            'Ticker': ['AAPL'] * 10,
            'Open': [150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0],
            'High': [151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0],
            'Low': [149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0],
            'Close': [150.5, 151.5, 152.5, 153.5, 154.5, 155.5, 156.5, 157.5, 158.5, 159.5],
            'Volume': [1000000] * 10
        })
        
        validation = validate_price_data(df)
        
        assert validation["valid"]
        assert len(validation["issues"]) == 0
        assert validation["total_rows"] == 10
    
    def test_validate_data_with_issues(self):
        """Test validation of data with issues."""
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=5),
            'Ticker': ['AAPL'] * 5,
            'Open': [150.0, -1.0, 152.0, 153.0, 154.0],  # Negative price
            'High': [151.0, 152.0, 153.0, 152.0, 155.0],  # High < Low in one row
            'Low': [149.0, 150.0, 151.0, 154.0, 153.0],
            'Close': [150.5, None, 152.5, 153.5, 154.5],  # Missing value
            'Volume': [1000000, 0, 1000000, 1000000, 1000000]  # Zero volume
        })
        
        validation = validate_price_data(df)
        
        assert not validation["valid"]
        assert len(validation["issues"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

