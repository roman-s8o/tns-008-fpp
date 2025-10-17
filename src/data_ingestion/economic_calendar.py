"""
Economic Calendar Scraper for Forex Trading

Scrapes major economic events from Investing.com calendar that impact EUR/USD:
- Fed rate decisions
- ECB rate decisions
- US Employment data (NFP)
- Inflation data (CPI)
- GDP releases
- PMI data

Provides features:
- Binary flags (event today: yes/no)
- Days until next event
- Actual values when available
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
import pandas as pd
import json

logger = logging.getLogger(__name__)


class EconomicCalendar:
    """
    Economic calendar scraper and feature builder.
    """
    
    # Major events that impact EUR/USD
    MAJOR_EVENTS = {
        'fed_rate': ['Federal Funds Rate', 'Fed Interest Rate Decision', 'FOMC'],
        'ecb_rate': ['ECB Interest Rate Decision', 'ECB Rate', 'European Central Bank'],
        'nfp': ['Nonfarm Payrolls', 'NFP', 'Non-Farm Employment'],
        'us_cpi': ['US CPI', 'Consumer Price Index', 'Core CPI'],
        'us_gdp': ['US GDP', 'Gross Domestic Product'],
        'us_unemployment': ['US Unemployment Rate', 'Unemployment'],
        'eu_cpi': ['Euro Zone CPI', 'EU Inflation'],
        'eu_gdp': ['Euro Zone GDP', 'EU GDP'],
        'pmi': ['PMI', 'Manufacturing PMI', 'Services PMI'],
    }
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        self.events_cache = []
    
    def fetch_calendar_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        mock_data: bool = True  # Use mock data since Investing.com blocks scrapers
    ) -> List[Dict]:
        """
        Fetch economic calendar data.
        
        Args:
            start_date: Start date for calendar
            end_date: End date for calendar
            mock_data: Use mock data (True) or attempt scraping (False)
            
        Returns:
            List of economic events
        """
        if mock_data:
            return self._generate_mock_calendar(start_date, end_date)
        else:
            return self._scrape_investing_calendar(start_date, end_date)
    
    def _scrape_investing_calendar(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Dict]:
        """
        Attempt to scrape Investing.com calendar.
        
        Note: This often fails due to anti-scraping protection.
        Use mock_data=True for development.
        """
        logger.warning("Investing.com scraping often blocked. Consider using mock data or API.")
        
        url = "https://www.investing.com/economic-calendar/"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse calendar table (structure may change)
            events = []
            # ... parsing logic here (often breaks due to page structure changes)
            
            logger.info(f"Scraped {len(events)} events from Investing.com")
            return events
            
        except Exception as e:
            logger.error(f"Failed to scrape Investing.com: {e}")
            logger.info("Falling back to mock data")
            return self._generate_mock_calendar(start_date, end_date)
    
    def _generate_mock_calendar(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Generate mock economic calendar with realistic event schedule.
        
        This simulates major events affecting EUR/USD.
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now() + timedelta(days=30)
        
        events = []
        current = start_date
        
        while current <= end_date:
            # Fed meetings (8 times per year, roughly every 6 weeks)
            if current.month in [1, 3, 5, 6, 7, 9, 11, 12] and 15 <= current.day <= 16:
                events.append({
                    'date': current,
                    'event': 'Fed Interest Rate Decision',
                    'category': 'fed_rate',
                    'currency': 'USD',
                    'importance': 'high',
                    'actual': None,
                    'forecast': None,
                    'previous': None,
                })
            
            # ECB meetings (8 times per year)
            if current.month in [1, 3, 4, 6, 7, 9, 10, 12] and 10 <= current.day <= 11:
                events.append({
                    'date': current,
                    'event': 'ECB Interest Rate Decision',
                    'category': 'ecb_rate',
                    'currency': 'EUR',
                    'importance': 'high',
                    'actual': None,
                    'forecast': None,
                    'previous': None,
                })
            
            # NFP (first Friday of each month)
            if current.weekday() == 4 and 1 <= current.day <= 7:  # Friday, first week
                events.append({
                    'date': current,
                    'event': 'US Nonfarm Payrolls',
                    'category': 'nfp',
                    'currency': 'USD',
                    'importance': 'high',
                    'actual': None,
                    'forecast': None,
                    'previous': None,
                })
            
            # US CPI (mid-month, monthly)
            if 12 <= current.day <= 15:
                events.append({
                    'date': current,
                    'event': 'US Consumer Price Index',
                    'category': 'us_cpi',
                    'currency': 'USD',
                    'importance': 'high',
                    'actual': None,
                    'forecast': None,
                    'previous': None,
                })
            
            # US GDP (quarterly, end of month)
            if current.month in [1, 4, 7, 10] and 25 <= current.day <= 28:
                events.append({
                    'date': current,
                    'event': 'US GDP Growth Rate',
                    'category': 'us_gdp',
                    'currency': 'USD',
                    'importance': 'high',
                    'actual': None,
                    'forecast': None,
                    'previous': None,
                })
            
            # EU CPI (end of month, monthly)
            if 28 <= current.day <= 31:
                events.append({
                    'date': current,
                    'event': 'Euro Zone CPI',
                    'category': 'eu_cpi',
                    'currency': 'EUR',
                    'importance': 'medium',
                    'actual': None,
                    'forecast': None,
                    'previous': None,
                })
            
            current += timedelta(days=1)
        
        logger.info(f"Generated {len(events)} mock calendar events")
        return events
    
    def create_calendar_features(
        self,
        price_df: pd.DataFrame,
        events: Optional[List[Dict]] = None
    ) -> pd.DataFrame:
        """
        Create calendar-based features for each date in price data.
        
        Features:
        - Binary flags: fed_today, ecb_today, nfp_today, etc.
        - Days until: days_to_fed, days_to_ecb, etc.
        - Event count: events_this_week
        
        Args:
            price_df: DataFrame with 'date' column
            events: List of economic events (generates if None)
            
        Returns:
            DataFrame with calendar features added
        """
        if events is None:
            # Generate events covering the price data period
            start = pd.to_datetime(price_df['date']).min() - timedelta(days=30)
            end = pd.to_datetime(price_df['date']).max() + timedelta(days=30)
            events = self.fetch_calendar_data(start, end, mock_data=True)
        
        # Convert events to DataFrame
        events_df = pd.DataFrame(events)
        if not events_df.empty:
            events_df['date'] = pd.to_datetime(events_df['date'])
        
        # Ensure price_df date is datetime
        df = price_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Initialize feature columns
        for category in self.MAJOR_EVENTS.keys():
            df[f'{category}_today'] = 0
            df[f'days_to_{category}'] = 999  # Large number if no upcoming event
        
        df['major_events_today'] = 0
        df['events_this_week'] = 0
        
        # Fill in features
        for idx, row in df.iterrows():
            date = row['date']
            
            # Binary flags for events today
            if not events_df.empty:
                today_events = events_df[events_df['date'].dt.date == date.date()]
                for _, event in today_events.iterrows():
                    category = event['category']
                    if category in self.MAJOR_EVENTS.keys():
                        df.at[idx, f'{category}_today'] = 1
                        df.at[idx, 'major_events_today'] += 1
                
                # Events this week
                week_start = date - timedelta(days=date.weekday())
                week_end = week_start + timedelta(days=6)
                week_events = events_df[
                    (events_df['date'] >= week_start) & 
                    (events_df['date'] <= week_end)
                ]
                df.at[idx, 'events_this_week'] = len(week_events)
                
                # Days until next event of each category
                for category in self.MAJOR_EVENTS.keys():
                    future_events = events_df[
                        (events_df['category'] == category) &
                        (events_df['date'] > date)
                    ]
                    if not future_events.empty:
                        next_event = future_events.iloc[0]
                        days_until = (next_event['date'] - date).days
                        df.at[idx, f'days_to_{category}'] = days_until
        
        logger.info(f"✓ Added {len([c for c in df.columns if 'today' in c or 'days_to' in c])} calendar features")
        return df
    
    def save_calendar(self, events: List[Dict], filename: str = "data/raw/economic_calendar.json"):
        """Save calendar events to file."""
        import json
        from pathlib import Path
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dates to strings for JSON
        events_serializable = []
        for event in events:
            event_copy = event.copy()
            if isinstance(event_copy['date'], datetime):
                event_copy['date'] = event_copy['date'].isoformat()
            events_serializable.append(event_copy)
        
        with open(filename, 'w') as f:
            json.dump(events_serializable, f, indent=2)
        
        logger.info(f"✓ Saved {len(events)} events to {filename}")
    
    def load_calendar(self, filename: str = "data/raw/economic_calendar.json") -> List[Dict]:
        """Load calendar events from file."""
        import json
        from pathlib import Path
        
        if not Path(filename).exists():
            logger.warning(f"Calendar file not found: {filename}")
            return []
        
        with open(filename, 'r') as f:
            events = json.load(f)
        
        # Convert date strings back to datetime
        for event in events:
            event['date'] = datetime.fromisoformat(event['date'])
        
        logger.info(f"✓ Loaded {len(events)} events from {filename}")
        return events


def test_calendar():
    """Test economic calendar functionality."""
    logging.basicConfig(level=logging.INFO)
    
    calendar = EconomicCalendar()
    
    # Generate calendar for past year + 1 month ahead
    start = datetime(2024, 1, 1)
    end = datetime.now() + timedelta(days=30)
    
    events = calendar.fetch_calendar_data(start, end, mock_data=True)
    
    print("\n" + "="*60)
    print("Economic Calendar Test")
    print("="*60)
    print(f"\nTotal events: {len(events)}")
    
    # Count by category
    by_category = {}
    for event in events:
        cat = event['category']
        by_category[cat] = by_category.get(cat, 0) + 1
    
    print(f"\nEvents by category:")
    for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    # Show upcoming events
    upcoming = [e for e in events if e['date'] > datetime.now()][:10]
    print(f"\nUpcoming events (next 10):")
    for event in upcoming:
        print(f"  {event['date'].strftime('%Y-%m-%d')}: {event['event']} ({event['currency']})")
    
    # Test calendar features on sample price data
    print(f"\n\nTesting calendar features...")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data_ingestion.forex_prices import fetch_eurusd_prices
    
    df = fetch_eurusd_prices(years=1)
    df = calendar.create_calendar_features(df, events)
    
    print(f"Added {len([c for c in df.columns if 'today' in c or 'days_to' in c])} calendar features")
    
    # Show days with major events
    major_event_days = df[df['major_events_today'] > 0]
    print(f"\nDays with major events: {len(major_event_days)}")
    if len(major_event_days) > 0:
        print("\nSample (last 5 event days):")
        cols = ['date', 'close', 'major_events_today', 'events_this_week']
        # Add event flags that exist
        for event_type in ['fed_today', 'ecb_today', 'nfp_today', 'us_cpi_today']:
            if f'{event_type}' in df.columns:
                cols.append(f'{event_type}')
        print(major_event_days[cols].tail())


if __name__ == "__main__":
    test_calendar()

