"""
Price data ingestion for Nasdaq-100 stocks.

This module fetches historical OHLCV (Open, High, Low, Close, Volume) data
from Yahoo Finance for all Nasdaq-100 component stocks.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import logging
from loguru import logger
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion.nasdaq_tickers import get_nasdaq100_tickers, get_nasdaq100_tickers_from_web


class PriceDataFetcher:
    """Fetches and stores historical price data for stocks."""
    
    def __init__(
        self,
        output_dir: str = "data/raw/prices",
        lookback_years: int = 5,
        use_web_tickers: bool = True
    ):
        """
        Initialize the price data fetcher.
        
        Args:
            output_dir: Directory to store CSV files
            lookback_years: Number of years of historical data to fetch
            use_web_tickers: Whether to fetch tickers from web (more current)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lookback_years = lookback_years
        self.use_web_tickers = use_web_tickers
        
        # Calculate date range
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_years * 365)
        
        logger.info(f"Initialized PriceDataFetcher")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def fetch_ticker_data(
        self,
        ticker: str,
        retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            retries: Number of retry attempts on failure
            
        Returns:
            DataFrame with OHLCV data, or None if failed
        """
        import time
        
        for attempt in range(retries):
            try:
                # Add a small delay to avoid rate limiting
                if attempt > 0:
                    time.sleep(2)
                
                # Create a ticker object (let yfinance handle the session)
                stock = yf.Ticker(ticker)
                
                # Try to fetch data with period parameter (more reliable)
                df = stock.history(
                    period=f"{self.lookback_years}y",
                    interval="1d",
                    auto_adjust=True,
                    actions=False
                )
                
                # If that fails, try with start/end dates
                if df.empty:
                    df = stock.history(
                        start=self.start_date.strftime("%Y-%m-%d"),
                        end=self.end_date.strftime("%Y-%m-%d"),
                        interval="1d",
                        auto_adjust=True,
                        actions=False
                    )
                
                if df.empty:
                    # One more attempt with download method
                    df = yf.download(
                        ticker,
                        start=self.start_date.strftime("%Y-%m-%d"),
                        end=self.end_date.strftime("%Y-%m-%d"),
                        progress=False
                    )
                
                if df.empty:
                    if attempt == retries - 1:
                        logger.warning(f"{ticker}: No data returned after all retries")
                    continue
                
                # Ensure we have the required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.warning(f"{ticker}: Missing columns: {missing_cols}")
                    continue
                
                # Select only OHLCV columns
                df = df[required_cols].copy()
                
                # Add ticker column
                df['Ticker'] = ticker
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                
                # Ensure Date column exists
                if 'Date' not in df.columns:
                    logger.warning(f"{ticker}: No Date column found")
                    continue
                
                # Reorder columns
                df = df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Remove any NaN rows
                df = df.dropna()
                
                if len(df) == 0:
                    logger.warning(f"{ticker}: No valid data after cleaning")
                    continue
                
                logger.debug(f"{ticker}: Fetched {len(df)} rows")
                return df
                
            except Exception as e:
                error_msg = str(e)
                if "40" in error_msg:  # HTTP errors
                    logger.warning(f"{ticker}: HTTP error, might be rate limited")
                    time.sleep(5)
                logger.warning(f"{ticker}: Attempt {attempt + 1}/{retries} failed: {error_msg[:100]}")
                if attempt < retries - 1:
                    continue
                else:
                    logger.error(f"{ticker}: All attempts failed")
                    return None
        
        return None
    
    def save_ticker_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save ticker data to CSV file.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with price data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            output_file = self.output_dir / f"{ticker}.csv"
            df.to_csv(output_file, index=False)
            logger.debug(f"{ticker}: Saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"{ticker}: Failed to save: {e}")
            return False
    
    def fetch_all_tickers(
        self,
        tickers: Optional[List[str]] = None,
        save_individual: bool = True,
        save_combined: bool = True
    ) -> Dict[str, any]:
        """
        Fetch data for all tickers.
        
        Args:
            tickers: List of tickers to fetch (if None, uses Nasdaq-100)
            save_individual: Whether to save individual CSV files per ticker
            save_combined: Whether to save a combined CSV with all tickers
            
        Returns:
            Dictionary with summary statistics
        """
        # Get ticker list
        if tickers is None:
            if self.use_web_tickers:
                tickers = get_nasdaq100_tickers_from_web()
            else:
                tickers = get_nasdaq100_tickers()
        
        logger.info(f"Starting data fetch for {len(tickers)} tickers")
        
        # Track results
        successful = []
        failed = []
        all_data = []
        
        # Fetch data for each ticker with progress bar
        for ticker in tqdm(tickers, desc="Fetching prices"):
            df = self.fetch_ticker_data(ticker)
            
            if df is not None and not df.empty:
                if save_individual:
                    self.save_ticker_data(ticker, df)
                
                all_data.append(df)
                successful.append(ticker)
            else:
                failed.append(ticker)
        
        # Save combined dataset
        if save_combined and all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_file = self.output_dir / "nasdaq100_combined.csv"
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined dataset to {combined_file}")
            logger.info(f"Combined dataset size: {len(combined_df)} rows")
        
        # Calculate statistics
        total = len(tickers)
        success_count = len(successful)
        fail_count = len(failed)
        success_rate = (success_count / total * 100) if total > 0 else 0
        
        # Log summary
        logger.info("="*60)
        logger.info("DATA FETCH SUMMARY")
        logger.info("="*60)
        logger.info(f"Total tickers: {total}")
        logger.info(f"Successful: {success_count} ({success_rate:.1f}%)")
        logger.info(f"Failed: {fail_count}")
        
        if failed:
            logger.warning(f"Failed tickers: {', '.join(failed)}")
        
        return {
            "total": total,
            "successful": success_count,
            "failed": fail_count,
            "success_rate": success_rate,
            "successful_tickers": successful,
            "failed_tickers": failed,
            "output_dir": str(self.output_dir)
        }
    
    def validate_data(self, ticker: str) -> Dict[str, any]:
        """
        Validate downloaded data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with validation results
        """
        csv_file = self.output_dir / f"{ticker}.csv"
        
        if not csv_file.exists():
            return {"valid": False, "error": "File not found"}
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check required columns
            required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {"valid": False, "error": f"Missing columns: {missing_cols}"}
            
            # Check for data
            if df.empty:
                return {"valid": False, "error": "Empty dataframe"}
            
            # Check date range
            df['Date'] = pd.to_datetime(df['Date'])
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            days_of_data = (max_date - min_date).days
            
            # Check for reasonable amount of data (at least 70% of expected trading days)
            expected_days = self.lookback_years * 252  # ~252 trading days per year
            if len(df) < expected_days * 0.7:
                logger.warning(
                    f"{ticker}: Only {len(df)} days of data "
                    f"(expected ~{expected_days})"
                )
            
            return {
                "valid": True,
                "rows": len(df),
                "date_range": f"{min_date.date()} to {max_date.date()}",
                "days_span": days_of_data,
                "columns": list(df.columns)
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}


def main():
    """Main function to fetch Nasdaq-100 price data."""
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/price_ingestion.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG"
    )
    
    # Create output directory for logs
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize fetcher
    fetcher = PriceDataFetcher(
        output_dir="data/raw/prices",
        lookback_years=5,
        use_web_tickers=True
    )
    
    # Fetch all data
    results = fetcher.fetch_all_tickers(
        save_individual=True,
        save_combined=True
    )
    
    # Validate a sample of tickers
    if results["successful_tickers"]:
        logger.info("\nValidating sample tickers...")
        sample_tickers = results["successful_tickers"][:5]
        
        for ticker in sample_tickers:
            validation = fetcher.validate_data(ticker)
            if validation["valid"]:
                logger.info(
                    f"{ticker}: ✓ Valid - {validation['rows']} rows, "
                    f"{validation['date_range']}"
                )
            else:
                logger.error(f"{ticker}: ✗ Invalid - {validation['error']}")
    
    # Check success criteria
    if results["success_rate"] >= 100:
        logger.success("✅ SUCCESS: 100% of tickers downloaded!")
    elif results["success_rate"] >= 95:
        logger.warning(f"⚠️  PARTIAL SUCCESS: {results['success_rate']:.1f}% downloaded")
    else:
        logger.error(f"❌ FAILURE: Only {results['success_rate']:.1f}% downloaded")
    
    return results


if __name__ == "__main__":
    results = main()

