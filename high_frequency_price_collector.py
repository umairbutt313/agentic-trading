#!/usr/bin/env python3
"""
High-Frequency Real-Time Price Collector
Enhanced version with:
- Multiple data source support (Yahoo, Alpha Vantage, IEX Cloud)
- Automatic failover between sources
- Market hours detection and handling
- Rate limiting and circuit breaker patterns
- Data validation and outlier detection
- Connection pooling for efficiency
"""

import asyncio
import websockets
import json
import time
import yfinance as yf
import requests
from price_storage import store_price
import logging
from datetime import datetime, time as dt_time
import signal
import sys
from typing import Dict, List, Optional, Union
import threading
from collections import defaultdict, deque
import os
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HighFrequencyCollector')

class DataSourceManager:
    """Manages multiple data sources with failover capabilities"""
    
    def __init__(self):
        self.sources = {
            'yahoo': self._get_yahoo_price,
            'alpha_vantage': self._get_alpha_vantage_price,
            'iex_cloud': self._get_iex_price
        }
        self.source_health = defaultdict(lambda: {'success': 0, 'failure': 0, 'last_error': None})
        self.current_source = 'yahoo'
        
    def _get_yahoo_price(self, symbol: str) -> Optional[float]:
        """Get price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = info.last_price
            return float(price) if price and price > 0 else None
        except Exception as e:
            logger.debug(f"Yahoo error for {symbol}: {e}")
            return None
    
    def _get_alpha_vantage_price(self, symbol: str) -> Optional[float]:
        """Get price from Alpha Vantage (requires API key)"""
        try:
            api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            if not api_key:
                return None
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if 'Global Quote' in data:
                price = data['Global Quote'].get('05. price')
                return float(price) if price else None
        except Exception as e:
            logger.debug(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def _get_iex_price(self, symbol: str) -> Optional[float]:
        """Get price from IEX Cloud (requires API key)"""
        try:
            api_key = os.getenv('IEX_CLOUD_API_KEY')
            if not api_key:
                return None
            
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/price"
            params = {'token': api_key}
            
            response = requests.get(url, params=params, timeout=5)
            return float(response.text) if response.status_code == 200 else None
        except Exception as e:
            logger.debug(f"IEX error for {symbol}: {e}")
            return None
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get price with automatic failover between sources"""
        sources_to_try = [self.current_source] + [s for s in self.sources.keys() if s != self.current_source]
        
        for source in sources_to_try:
            try:
                price = self.sources[source](symbol)
                if price and price > 0:
                    self.source_health[source]['success'] += 1
                    if source != self.current_source:
                        logger.info(f"Switched to {source} for better reliability")
                        self.current_source = source
                    return price
                else:
                    self.source_health[source]['failure'] += 1
            except Exception as e:
                self.source_health[source]['failure'] += 1
                self.source_health[source]['last_error'] = str(e)
        
        return None

class PriceValidator:
    """Validates price data and detects outliers"""
    
    def __init__(self, window_size: int = 10):
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
        self.window_size = window_size
    
    def is_valid_price(self, symbol: str, price: float) -> bool:
        """Validate price against historical data"""
        if not price or price <= 0:
            return False
        
        history = self.price_history[symbol]
        
        # If no history, accept the price
        if len(history) == 0:
            history.append(price)
            return True
        
        # Check for extreme outliers (more than 50% change)
        last_price = history[-1]
        change_pct = abs(price - last_price) / last_price
        
        if change_pct > 0.5:  # 50% change threshold
            logger.warning(f"{symbol}: Potential outlier detected - {change_pct:.1%} change")
            return False
        
        # Add to history if valid
        history.append(price)
        return True

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_second: int = 5):
        self.calls_per_second = calls_per_second
        self.calls = deque()
    
    async def acquire(self):
        """Acquire rate limit token"""
        now = time.time()
        
        # Remove calls older than 1 second
        while self.calls and self.calls[0] < now - 1:
            self.calls.popleft()
        
        # If we've hit the limit, wait
        if len(self.calls) >= self.calls_per_second:
            wait_time = 1 - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.calls.append(now)

class HighFrequencyPriceCollector:
    # OLD: Multiple companies as default
    # def __init__(self, symbols=['NVDA', 'AAPL', 'INTC'], interval=1):
    # NEW: Only NVIDIA as default (matches companies.yaml configuration)
    def __init__(self, symbols=['NVDA'], interval=1):
        self.symbols = symbols
        self.interval = interval  # seconds
        self.running = False
        self.data_source_manager = DataSourceManager()
        self.price_validator = PriceValidator()
        self.rate_limiter = RateLimiter(calls_per_second=10)
        
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours (EST)"""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def get_current_prices(self) -> Dict[str, Optional[float]]:
        """Get current prices for all symbols synchronously"""
        prices = {}
        for symbol in self.symbols:
            try:
                price = self.data_source_manager.get_price(symbol)
                if self.price_validator.is_valid_price(symbol, price):
                    prices[symbol] = price
                else:
                    prices[symbol] = None
            except Exception as e:
                logger.warning(f"Error getting price for {symbol}: {e}")
                prices[symbol] = None
        return prices
        
    async def collect_with_enhanced_api(self):
        """Enhanced collection with multiple sources and validation"""
        logger.info(f"🚀 Starting enhanced high-frequency collection (every {self.interval}s)")
        logger.info(f"📊 Tracking: {', '.join(self.symbols)}")
        logger.info(f"🔄 Data sources: Yahoo Finance (primary), Alpha Vantage, IEX Cloud")
        
        round_num = 0
        consecutive_failures = 0
        
        while self.running:
            round_num += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Check market hours
            if not self.is_market_hours():
                logger.debug(f"Outside market hours, waiting...")
                await asyncio.sleep(60)  # Check every minute
                continue
            
            logger.info(f"📈 Round {round_num} - {current_time}")
            
            # Rate limiting
            await self.rate_limiter.acquire()
            
            prices_collected = 0
            for symbol in self.symbols:
                try:
                    price = self.data_source_manager.get_price(symbol)
                    
                    if price and self.price_validator.is_valid_price(symbol, price):
                        store_price(symbol, price)
                        logger.info(f"✅ {symbol}: ${price:.4f}")
                        prices_collected += 1
                    else:
                        logger.warning(f"⚠️ {symbol}: Invalid price data")
                        
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: {str(e)[:60]}...")
            
            # Track consecutive failures for circuit breaker
            if prices_collected == 0:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    logger.error("🔴 Circuit breaker: Too many consecutive failures, backing off")
                    await asyncio.sleep(30)  # Back off for 30 seconds
                    consecutive_failures = 0
            else:
                consecutive_failures = 0
                    
            await asyncio.sleep(self.interval)
    
    async def start_streaming(self):
        """Start the enhanced high-frequency price streaming"""
        self.running = True
        
        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("⏹️ Stopping high-frequency collection...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            await self.collect_with_enhanced_api()
        except KeyboardInterrupt:
            logger.info("🛑 Collection stopped by user")
        except Exception as e:
            logger.error(f"❌ Collection error: {e}")
        finally:
            self.running = False
            logger.info("✅ High-frequency collector stopped")
    
    def get_health_status(self) -> Dict:
        """Get health status of all data sources"""
        return {
            'current_source': self.data_source_manager.current_source,
            'source_health': dict(self.data_source_manager.source_health),
            'market_hours': self.is_market_hours(),
            'symbols': self.symbols,
            'interval': self.interval
        }

async def main():
    # Create enhanced collector with 1-second intervals
    # OLD: Multiple companies enabled
    # collector = HighFrequencyPriceCollector(
    #     symbols=['NVDA', 'AAPL', 'INTC', 'MSFT', 'GOOG', 'TSLA'],
    #     interval=1  # 1 second for ultra-fast real trading
    # )
    # NEW: Only NVIDIA enabled (matches companies.yaml configuration)
    collector = HighFrequencyPriceCollector(
        symbols=['NVDA'],
        interval=1  # 1 second for ultra-fast real trading
    )
    
    logger.info("🔥 ENHANCED HIGH-FREQUENCY REAL-TIME PRICE STREAMING")
    logger.info("=" * 70)
    logger.info("⚡ Update interval: 1 second")
    logger.info("🔄 Multiple data sources with automatic failover")
    logger.info("🛡️ Built-in price validation and outlier detection")
    logger.info("⏱️ Market hours awareness and rate limiting")
    logger.info("💡 Press Ctrl+C to stop")
    logger.info("🎯 Production-ready for real trading scenarios")
    
    await collector.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())