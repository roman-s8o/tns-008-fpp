"""
Nasdaq-100 ticker list management.

This module provides functions to get and manage the list of Nasdaq-100 tickers.
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


def get_nasdaq100_tickers() -> List[str]:
    """
    Get the list of current Nasdaq-100 component stocks.
    
    Note: The Nasdaq-100 composition changes periodically. This list is accurate
    as of October 2025. For production use, consider fetching this dynamically
    from a reliable data source.
    
    Returns:
        List of ticker symbols
    """
    # Current Nasdaq-100 components (major ones)
    # In production, this should be fetched from a reliable API or updated regularly
    tickers = [
        # Technology - Mega Cap
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
        "AVGO", "ADBE", "CSCO", "INTC", "AMD", "QCOM", "TXN", "INTU",
        "AMAT", "MU", "ADI", "LRCX", "KLAC", "SNPS", "CDNS", "MCHP",
        "NXPI", "MRVL", "FTNT", "PANW", "CRWD", "WDAY", "TEAM", "DDOG",
        "SNOW", "ZS", "OKTA", "NET", "CFLT", "DKNG",
        
        # Consumer Discretionary
        "AMZN", "TSLA", "SBUX", "BKNG", "MAR", "ABNB", "EBAY", "DASH",
        "LCID", "RIVN",
        
        # Communication Services
        "META", "GOOGL", "GOOG", "NFLX", "CMCSA", "CHTR", "ATVI", "EA",
        "TTWO", "ZM", "MTCH",
        
        # Healthcare & Biotech
        "AMGN", "GILD", "VRTX", "REGN", "BIIB", "ILMN", "MRNA", "ISRG",
        "ALGN", "IDXX", "DXCM", "EXAS",
        
        # Consumer Staples
        "COST", "PEP", "MDLZ", "KDP", "MNST",
        
        # Industrials
        "HON", "ADP", "PAYX", "FAST", "VRSK",
        
        # Other
        "ASML", "PYPL", "ADSK", "ROST", "CTAS", "PCAR", "ODFL", "LULU",
        "SGEN", "WBA", "XEL", "CEG", "AEP", "EXC"
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    logger.info(f"Loaded {len(unique_tickers)} Nasdaq-100 tickers")
    return unique_tickers


def get_nasdaq100_tickers_from_web() -> List[str]:
    """
    Fetch current Nasdaq-100 tickers from Wikipedia (more reliable for production).
    
    Returns:
        List of ticker symbols
        
    Note:
        This is a more robust approach for production use. Falls back to
        hardcoded list if web scraping fails.
    """
    try:
        import pandas as pd
        
        # Wikipedia maintains an up-to-date list
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        
        # The ticker list is usually in one of the first few tables
        for table in tables[:5]:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                tickers = table[ticker_col].tolist()
                
                # Clean up tickers (remove NaN, whitespace)
                tickers = [str(t).strip() for t in tickers if pd.notna(t)]
                
                if len(tickers) > 80:  # Nasdaq-100 should have ~100 stocks
                    logger.info(f"Fetched {len(tickers)} tickers from Wikipedia")
                    return tickers
        
        logger.warning("Could not parse tickers from Wikipedia, using fallback list")
        return get_nasdaq100_tickers()
        
    except Exception as e:
        logger.error(f"Error fetching tickers from web: {e}")
        logger.info("Using fallback hardcoded ticker list")
        return get_nasdaq100_tickers()


if __name__ == "__main__":
    # Test the ticker fetching
    logging.basicConfig(level=logging.INFO)
    
    print("Hardcoded ticker list:")
    tickers = get_nasdaq100_tickers()
    print(f"Count: {len(tickers)}")
    print(f"Sample: {tickers[:10]}")
    
    print("\n" + "="*50 + "\n")
    
    print("Web-fetched ticker list:")
    web_tickers = get_nasdaq100_tickers_from_web()
    print(f"Count: {len(web_tickers)}")
    print(f"Sample: {web_tickers[:10]}")

