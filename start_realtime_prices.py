#!/usr/bin/env python3
"""
Main entry point for real-time price collection with:
- WebSocket streaming from Yahoo Finance
- Fallback to REST API polling
- Background daemon mode
- Multi-symbol support (NVDA, AAPL, INTC, MSFT, GOOG, TSLA)
- Market hours awareness
- Automatic reconnection on failures
"""

import argparse
import asyncio
import sys
import signal
import json
import os
from datetime import datetime, time
import logging
from typing import List, Optional
import threading
import time as time_module

# Import our existing modules
from high_frequency_price_collector import HighFrequencyPriceCollector
from price_storage import PriceStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/realtime_prices.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default symbols to track
# OLD: All companies enabled
# DEFAULT_SYMBOLS = ['NVDA', 'AAPL', 'INTC', 'MSFT', 'GOOG', 'TSLA']
# NEW: Only NVIDIA enabled (matches companies.yaml configuration)
DEFAULT_SYMBOLS = ['NVDA']

class RealTimePriceService:
    """Main service for real-time price collection"""
    
    def __init__(self, symbols: List[str], interval: int = 1, use_websocket: bool = False):
        self.symbols = symbols
        self.interval = interval
        self.use_websocket = use_websocket
        self.collector = HighFrequencyPriceCollector(symbols)
        self.storage = PriceStorage()
        self.running = False
        self.daemon_thread = None
        
    def is_market_hours(self) -> bool:
        """Check if current time is during market hours"""
        now = datetime.now()
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = time(9, 30)
        market_close = time(16, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def start(self, daemon: bool = False):
        """Start the price collection service"""
        self.running = True
        logger.info(f"Starting real-time price collection for {self.symbols}")
        
        if daemon:
            self.daemon_thread = threading.Thread(target=self._run_collection, daemon=True)
            self.daemon_thread.start()
            logger.info("Price collection started in daemon mode")
        else:
            self._run_collection()
    
    def _run_collection(self):
        """Main collection loop"""
        while self.running:
            try:
                # # Check if we should collect during market hours only - DISABLED for 24/7 operation
                # if not self.is_market_hours():
                #     logger.debug("Outside market hours, waiting...")
                #     time_module.sleep(60)  # Check every minute
                #     continue
                
                # Collect prices for all symbols
                prices = self.collector.get_current_prices()
                
                if prices:
                    timestamp = datetime.now().isoformat(timespec='seconds')
                    for symbol, price in prices.items():
                        if price and price > 0:
                            # Store in our storage system  
                            price_data = {
                                'symbol': symbol,
                                'price': price,
                                'timestamp': timestamp
                            }
                            self.storage._store_price_sync(price_data)
                            logger.info(f"{symbol}: ${price:.2f}")
                    
                    # Data is saved automatically by storage system
                    logger.debug(f"Prices saved for {len(prices)} symbols")
                else:
                    logger.warning("No prices retrieved")
                
                # Wait for next interval
                time_module.sleep(self.interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time_module.sleep(5)  # Wait before retry
    
    def stop(self):
        """Stop the price collection service"""
        self.running = False
        logger.info("Stopping price collection service")
        if self.daemon_thread:
            self.daemon_thread.join(timeout=5)
    
    def get_status(self) -> dict:
        """Get current service status"""
        stats = self.storage.get_statistics()
        return {
            'running': self.running,
            'symbols': self.symbols,
            'interval': self.interval,
            'market_hours': self.is_market_hours(),
            'storage_stats': stats
        }

def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(description='Real-time price collection service')
    parser.add_argument('--daemon', action='store_true', help='Run as background service')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols to track')
    parser.add_argument('--interval', type=int, default=1, help='Collection frequency in seconds')
    parser.add_argument('--websocket', action='store_true', help='Use WebSocket streaming')
    parser.add_argument('--fallback', action='store_true', help='Use REST API fallback')
    parser.add_argument('--test', action='store_true', help='Test mode with verbose output')
    parser.add_argument('--status', action='store_true', help='Check collection status')
    
    args = parser.parse_args()
    
    # Set logging level for test mode
    if args.test:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Test mode enabled with verbose output")
    
    # Parse symbols or use defaults
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
    
    # Check status
    if args.status:
        storage = PriceStorage()
        stats = storage.get_statistics()
        print("\n=== Price Collection Status ===")
        print(f"Storage type: {stats['storage_type']}")
        print(f"Data directory: {stats['data_dir']}")
        print(f"Total symbols: {stats['total_symbols']}")
        print(f"Symbols: {', '.join(stats['symbols'])}")
        
        if stats['symbol_details']:
            print("\nPer-symbol statistics:")
            for symbol, details in stats['symbol_details'].items():
                print(f"  {symbol}: {details['data_points']} records")
                print(f"    Latest price: ${details['latest_price']:.2f}")
                print(f"    Latest time: {details['latest_time']}")
                print(f"    Price range: ${details['price_range']}")
        else:
            print("No price data available yet")
        return
    
    # Create and start service
    service = RealTimePriceService(
        symbols=symbols,
        interval=args.interval,
        use_websocket=args.websocket
    )
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the service
    try:
        if args.daemon:
            service.start(daemon=True)
            print(f"Price collection service started in background")
            print(f"Tracking symbols: {', '.join(symbols)}")
            print(f"Collection interval: {args.interval} seconds")
            print("Press Ctrl+C to stop")
            
            # Keep main thread alive
            while True:
                time_module.sleep(1)
        else:
            print(f"Starting price collection for {', '.join(symbols)}")
            print(f"Collection interval: {args.interval} seconds")
            print("Press Ctrl+C to stop")
            service.start(daemon=False)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.stop()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    main()