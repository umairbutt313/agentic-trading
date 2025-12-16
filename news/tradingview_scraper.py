#!/usr/bin/env python3
"""
TradingView Scraper - Legacy Compatibility Wrapper
This module provides compatibility with existing scripts by wrapping the 
enhanced Playwright TradingView scraper.
"""

import os
import sys
import asyncio
import logging
from playwright_tradingview_scraper import PlaywrightTradingViewScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingViewScraper:
    """Legacy compatibility wrapper for TradingView scraping"""
    
    def __init__(self):
        """Initialize the TradingView Scraper with Playwright backend"""
        self.playwright_scraper = PlaywrightTradingViewScraper({
            'debug': False,
            'timeout': 60000,
            'concurrency': 2
        })
        logger.info("TradingView scraper initialized with Playwright backend")
    
    async def scrape_all_symbols(self, test_mode: bool = False):
        """
        Scrape all symbols using the enhanced Playwright implementation
        
        Args:
            test_mode: Whether to run in test mode
            
        Returns:
            Dict containing scraped data
        """
        logger.info("Starting TradingView data collection...")
        
        try:
            # Use the working Playwright implementation
            results = await self.playwright_scraper.scrape_all_symbols()
            
            if 'error' in results:
                logger.error(f"Scraping failed: {results['error']}")
                return None
            
            logger.info("TradingView data collection completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to scrape TradingView data: {e}")
            return None
    
    async def scrape_symbol(self, symbol: str, test_mode: bool = False):
        """
        Scrape a single symbol
        
        Args:
            symbol: Stock symbol to scrape
            test_mode: Whether to run in test mode
            
        Returns:
            Dict containing scraped data for the symbol
        """
        logger.info(f"Scraping symbol: {symbol}")
        
        try:
            result = await self.playwright_scraper.scrape_symbol(symbol)
            
            if 'error' in result:
                logger.error(f"Failed to scrape {symbol}: {result['error']}")
                return None
            
            logger.info(f"Successfully scraped {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to scrape {symbol}: {e}")
            return None

# Legacy sync wrappers for compatibility
def run_scraper_all(test_mode: bool = False):
    """Legacy sync wrapper for scraping all symbols"""
    scraper = TradingViewScraper()
    return asyncio.run(scraper.scrape_all_symbols(test_mode=test_mode))

def run_scraper_symbol(symbol: str, test_mode: bool = False):
    """Legacy sync wrapper for scraping single symbol"""
    scraper = TradingViewScraper()
    return asyncio.run(scraper.scrape_symbol(symbol, test_mode=test_mode))

# CLI interface for backwards compatibility
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TradingView Scraper (Playwright Backend)')
    parser.add_argument('--symbol', help='Single symbol to scrape')
    parser.add_argument('--all', action='store_true', help='Scrape all symbols')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    
    args = parser.parse_args()
    
    if args.install_deps:
        print("Installing Playwright dependencies...")
        os.system("cd ../playwright_scrapers && npm install")
        sys.exit(0)
    
    try:
        if args.symbol:
            result = run_scraper_symbol(args.symbol, test_mode=args.test)
            if result:
                print(f"Successfully scraped {args.symbol}")
            else:
                print(f"Failed to scrape {args.symbol}")
                sys.exit(1)
                
        elif args.all:
            result = run_scraper_all(test_mode=args.test)
            if result:
                print("Successfully scraped all symbols")
            else:
                print("Failed to scrape symbols")
                sys.exit(1)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)