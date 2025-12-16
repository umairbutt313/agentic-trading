#!/usr/bin/env python3
"""
Market Status Checker - Verify if US stock market is currently open
Uses multiple methods to determine market status including NYSE/NASDAQ hours
"""

import requests
import json
from datetime import datetime, timezone
import pytz
from typing import Dict, Optional, Tuple

class MarketStatusChecker:
    def __init__(self):
        self.eastern = pytz.timezone('US/Eastern')
        
    def get_current_eastern_time(self) -> datetime:
        """Get current time in Eastern timezone"""
        return datetime.now(self.eastern)
        
    def is_market_day(self, dt: datetime) -> bool:
        """Check if given date is a trading day (Mon-Fri, excluding holidays)"""
        # Monday=0, Sunday=6
        weekday = dt.weekday()
        if weekday >= 5:  # Saturday or Sunday
            return False
            
        # Basic holiday check (extend as needed)
        market_holidays_2025 = [
            # New Year's Day, MLK Day, Presidents Day, Good Friday, 
            # Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas
            (1, 1), (1, 20), (2, 17), (4, 18), (5, 26), (7, 4), (9, 1), (11, 27), (12, 25)
        ]
        
        date_tuple = (dt.month, dt.day)
        return date_tuple not in market_holidays_2025
        
    def get_market_hours(self) -> Dict:
        """Get market hours for US exchanges"""
        return {
            "regular": {
                "open": "09:30",
                "close": "16:00",
                "timezone": "US/Eastern"
            },
            "extended": {
                "pre_market": "04:00",
                "after_hours_close": "20:00",
                "timezone": "US/Eastern"
            }
        }
        
    def check_market_status_local(self) -> Dict:
        """Check market status using local time calculations"""
        now = self.get_current_eastern_time()
        hours = self.get_market_hours()
        
        # Check if it's a trading day
        is_trading_day = self.is_market_day(now)
        
        # Parse regular hours
        regular_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        regular_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Parse extended hours
        pre_market_open = now.replace(hour=4, minute=0, second=0, microsecond=0)
        after_hours_close = now.replace(hour=20, minute=0, second=0, microsecond=0)
        
        status = {
            "timestamp": now.isoformat(),
            "eastern_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_trading_day": is_trading_day,
            "market_status": "CLOSED",
            "session_type": "CLOSED",
            "hours": hours
        }
        
        if not is_trading_day:
            status["reason"] = "Weekend or Holiday"
            return status
            
        # Determine market session
        if pre_market_open <= now < regular_open:
            status["market_status"] = "EXTENDED"
            status["session_type"] = "PRE_MARKET"
        elif regular_open <= now < regular_close:
            status["market_status"] = "OPEN"
            status["session_type"] = "REGULAR"
        elif regular_close <= now < after_hours_close:
            status["market_status"] = "EXTENDED"
            status["session_type"] = "AFTER_HOURS"
        else:
            status["market_status"] = "CLOSED"
            status["session_type"] = "CLOSED"
            
        return status
        
    def check_market_status_api(self) -> Optional[Dict]:
        """Check market status using financial APIs (backup method)"""
        try:
            # Using Alpha Vantage free API (no key required for basic market status)
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": "AAPL",  # Use Apple as indicator
                "apikey": "demo"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data:
                    return {
                        "source": "alphavantage",
                        "symbol": "AAPL",
                        "last_updated": data["Global Quote"].get("07. latest trading day", ""),
                        "status": "API_AVAILABLE"
                    }
        except Exception as e:
            print(f"API check failed: {e}")
            return None
            
    def check_market_status_yahoo(self) -> Optional[Dict]:
        """Check market status using Yahoo Finance (alternative method)"""
        try:
            # Yahoo Finance doesn't require API key for basic quotes
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "chart" in data and data["chart"]["result"]:
                    result = data["chart"]["result"][0]
                    meta = result.get("meta", {})
                    
                    return {
                        "source": "yahoo_finance",
                        "symbol": "AAPL",
                        "exchange_name": meta.get("exchangeName", ""),
                        "market_state": meta.get("marketState", ""),
                        "regular_market_time": meta.get("regularMarketTime", ""),
                        "status": "API_AVAILABLE"
                    }
        except Exception as e:
            print(f"Yahoo Finance check failed: {e}")
            return None
            
    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive market status from multiple sources"""
        results = {
            "timestamp": datetime.now(self.eastern).isoformat(),
            "local_calculation": self.check_market_status_local(),
            "api_sources": {}
        }
        
        # Try API sources
        yahoo_status = self.check_market_status_yahoo()
        if yahoo_status:
            results["api_sources"]["yahoo"] = yahoo_status
            
        api_status = self.check_market_status_api()
        if api_status:
            results["api_sources"]["alphavantage"] = api_status
            
        return results
        
    def is_market_open(self) -> Tuple[bool, str]:
        """Simple boolean check if market is open with reason"""
        status = self.check_market_status_local()
        
        if status["market_status"] in ["OPEN", "EXTENDED"]:
            return True, f"{status['session_type']} session active"
        else:
            reason = status.get("reason", "Outside trading hours")
            return False, reason

def main():
    """Main function to check and display market status"""
    checker = MarketStatusChecker()
    
    print("🔍 US Stock Market Status Checker")
    print("=" * 50)
    
    # Quick status check
    is_open, reason = checker.is_market_open()
    print(f"Market Open: {'✅ YES' if is_open else '❌ NO'}")
    print(f"Status: {reason}")
    print()
    
    # Detailed status
    print("📊 Detailed Market Information:")
    status = checker.get_comprehensive_status()
    local = status["local_calculation"]
    
    print(f"Current Time (ET): {local['eastern_time']}")
    print(f"Trading Day: {'Yes' if local['is_trading_day'] else 'No'}")
    print(f"Market Status: {local['market_status']}")
    print(f"Session Type: {local['session_type']}")
    
    print("\n⏰ Market Hours:")
    hours = local["hours"]
    print(f"Regular Hours: {hours['regular']['open']} - {hours['regular']['close']} ET")
    print(f"Extended Hours: {hours['extended']['pre_market']} - {hours['extended']['after_hours_close']} ET")
    
    # API verification
    if status["api_sources"]:
        print("\n🌐 API Verification:")
        for source, data in status["api_sources"].items():
            print(f"{source.title()}: {data.get('status', 'Available')}")
            if 'market_state' in data:
                print(f"  Market State: {data['market_state']}")

if __name__ == "__main__":
    main()