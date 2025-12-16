#!/usr/bin/env python3
"""
Enhanced TradingView Scraper - Python Interface for Playwright Implementation

This script provides a Python interface to the enhanced Playwright-based TradingView scraper,
providing enhanced data extraction capabilities:
- Price extraction timing/DOM loading issues
- Daily high/low not extracting
- Volume not finding data
- Post-market data not extracting

Features:
- Python interface to Playwright scraper
- Enhanced data processing and validation
- Integration with existing sentiment analysis pipeline
- Improved error handling and retry mechanisms
- Data persistence and caching
- Performance metrics and monitoring
"""

import os
import sys
import json
import subprocess
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
import aiofiles
import pandas as pd

# Add the project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

class PlaywrightTradingViewScraper:
    """Enhanced TradingView scraper using Playwright backend with Python interface"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Playwright TradingView scraper
        
        Args:
            config: Configuration dictionary with scraper options
        """
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        self.timeout = self.config.get('timeout', 60000)
        self.concurrency = self.config.get('concurrency', 2)
        self.output_dir = self.config.get('output_dir', '../container_output')
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Paths
        self.project_root = project_root
        self.playwright_dir = project_root / 'playwright_scrapers'
        self.scraper_path = self.playwright_dir / 'modules' / 'production_ohlc_extractor.js'
        self.output_path = project_root / 'container_output'
        
        # Ensure output directory exists
        self.output_path.mkdir(exist_ok=True)
        
        # Data storage
        self.last_results = {}
        self.cache_duration = timedelta(minutes=5)  # Cache results for 5 minutes
        
        if self.debug:
            self.logger.info("Playwright TradingView scraper initialized")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('playwright_tradingview_scraper')
        logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    async def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        try:
            # Check if Node.js is available
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.logger.error("Node.js not found. Please install Node.js")
                return False
            
            # Check if Playwright scraper exists
            if not self.scraper_path.exists():
                self.logger.error(f"Playwright scraper not found at {self.scraper_path}")
                return False
            
            # Check if npm packages are installed
            package_json_path = self.playwright_dir / 'package.json'
            if not package_json_path.exists():
                self.logger.error(f"package.json not found at {package_json_path}")
                return False
            
            # Check if node_modules exists
            node_modules_path = self.playwright_dir / 'node_modules'
            if not node_modules_path.exists():
                self.logger.warning("node_modules not found. Installing dependencies...")
                await self._install_dependencies()
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout while checking Node.js version")
            return False
        except Exception as e:
            self.logger.error(f"Error checking dependencies: {e}")
            return False

    async def _install_dependencies(self) -> bool:
        """Install npm dependencies"""
        try:
            self.logger.info("Installing npm dependencies...")
            
            # Change to playwright directory
            original_cwd = os.getcwd()
            os.chdir(self.playwright_dir)
            
            try:
                # Install dependencies
                result = subprocess.run(['npm', 'install'], 
                                      capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    self.logger.error(f"npm install failed: {result.stderr}")
                    return False
                
                # Install Playwright browsers
                result = subprocess.run(['npx', 'playwright', 'install', 'chromium'], 
                                      capture_output=True, text=True, timeout=180)
                
                if result.returncode != 0:
                    self.logger.error(f"Playwright install failed: {result.stderr}")
                    return False
                
                self.logger.info("Dependencies installed successfully")
                return True
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            self.logger.error(f"Error installing dependencies: {e}")
            return False

    async def scrape_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Scrape data for a single symbol
        
        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            
        Returns:
            Dictionary containing scraped data
        """
        if self.debug:
            self.logger.info(f"Scraping symbol: {symbol}")
        
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        if cache_key in self.last_results:
            cache_time = self.last_results[cache_key].get('cache_time')
            if cache_time and datetime.now() - cache_time < self.cache_duration:
                if self.debug:
                    self.logger.info(f"Returning cached result for {symbol}")
                return self.last_results[cache_key]['data']
        
        try:
            # Ensure dependencies are installed
            if not await self.check_dependencies():
                raise Exception("Dependencies not available")
            
            # Prepare command with --json flag for clean output
            cmd = ['node', str(self.scraper_path), symbol, '--json']
            
            # Set environment variables
            env = os.environ.copy()
            env['NODE_ENV'] = 'production'
            
            # Change to playwright directory
            original_cwd = os.getcwd()
            os.chdir(self.playwright_dir)
            
            try:
                # Execute the scraper
                start_time = time.time()
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout / 1000,  # Convert to seconds
                    env=env
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"Scraper failed for {symbol}: {result.stderr}"
                    self.logger.error(error_msg)
                    return {
                        'symbol': symbol,
                        'error': error_msg,
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Parse JSON output
                try:
                    # Extract JSON from output (might have debug info)
                    output_lines = result.stdout.strip().split('\n')
                    json_line = None
                    
                    for line in output_lines:
                        if line.startswith('{') and line.endswith('}'):
                            json_line = line
                            break
                    
                    if not json_line:
                        # Try to find JSON in the output
                        for line in output_lines:
                            if '"symbol"' in line and '"extraction_time"' in line:
                                json_line = line
                                break
                    
                    if json_line:
                        data = json.loads(json_line)
                    else:
                        # Parse full output as JSON
                        data = json.loads(result.stdout)
                    
                    # Add execution metadata
                    data['execution_time'] = execution_time
                    data['scraper_version'] = '2.0.0-playwright'
                    data['python_timestamp'] = datetime.now().isoformat()
                    
                    # Cache the result
                    self.last_results[cache_key] = {
                        'data': data,
                        'cache_time': datetime.now()
                    }
                    
                    if self.debug:
                        self.logger.info(f"Successfully scraped {symbol} in {execution_time:.2f}s")
                    
                    return data
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON output for {symbol}: {e}")
                    self.logger.error(f"Raw output: {result.stdout}")
                    return {
                        'symbol': symbol,
                        'error': f"JSON parse error: {e}",
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }
                
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while scraping {symbol}"
            self.logger.error(error_msg)
            return {
                'symbol': symbol,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Error scraping {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'symbol': symbol,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }

    async def scrape_all_symbols(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Scrape data for all symbols
        
        Args:
            symbols: List of symbols to scrape (uses default if None)
            
        Returns:
            Dictionary containing all scraped data
        """
        if symbols is None:
            symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        
        if self.debug:
            self.logger.info(f"Scraping {len(symbols)} symbols: {symbols}")
        
        try:
            # Ensure dependencies are installed
            if not await self.check_dependencies():
                raise Exception("Dependencies not available")
            
            # Prepare command for all symbols
            cmd = ['node', str(self.scraper_path), '--all']
            
            # Set environment variables
            env = os.environ.copy()
            env['NODE_ENV'] = 'production'
            
            # Change to playwright directory
            original_cwd = os.getcwd()
            os.chdir(self.playwright_dir)
            
            try:
                # Execute the scraper
                start_time = time.time()
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout / 1000 * len(symbols),  # Scale timeout
                    env=env
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode != 0:
                    error_msg = f"Scraper failed: {result.stderr}"
                    self.logger.error(error_msg)
                    return {
                        'error': error_msg,
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Parse and return the results
                if self.debug:
                    self.logger.info(f"All symbols scraped in {execution_time:.2f}s")
                
                # Results are automatically saved by the Node.js scraper
                # Load the latest results
                latest_file = self.output_path / 'playwright-tradingview_latest.json'
                if latest_file.exists():
                    async with aiofiles.open(latest_file, 'r') as f:
                        content = await f.read()
                        return json.loads(content)
                else:
                    return {
                        'error': 'Results file not found',
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }
                
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            error_msg = "Timeout while scraping all symbols"
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Error scraping all symbols: {str(e)}"
            self.logger.error(error_msg)
            return {
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }

    def process_results(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Process scraped results into a pandas DataFrame
        
        Args:
            results: Raw results from scraper
            
        Returns:
            Processed DataFrame
        """
        if 'companies' not in results:
            self.logger.error("No companies data found in results")
            return pd.DataFrame()
        
        processed_data = []
        
        for company in results['companies']:
            row = {
                'symbol': company.get('symbol', 'N/A'),
                'extraction_time': company.get('extraction_time', 'N/A'),
                'errors': len(company.get('errors', []))
            }
            
            # Process basic data
            basic_data = company.get('basic_data', {})
            if basic_data and not basic_data.get('error'):
                row.update({
                    'current_price': basic_data.get('current_price'),
                    'price_change': basic_data.get('price_change'),
                    'daily_high': basic_data.get('daily_high'),
                    'daily_low': basic_data.get('daily_low'),
                    'volume': basic_data.get('volume'),
                    'market_cap': basic_data.get('market_cap'),
                    'post_market_price': basic_data.get('post_market_price'),
                    'post_market_change': basic_data.get('post_market_change')
                })
            
            # Process technical indicators
            technical_data = company.get('technical_indicators', {})
            if technical_data and not technical_data.get('error'):
                row.update({
                    'technical_summary': technical_data.get('technical_summary'),
                    'oscillators': technical_data.get('oscillators'),
                    'moving_averages': technical_data.get('moving_averages')
                })
            
            processed_data.append(row)
        
        df = pd.DataFrame(processed_data)
        
        # Add metadata
        if 'extraction_metadata' in results:
            metadata = results['extraction_metadata']
            df.attrs['extraction_metadata'] = metadata
        
        return df

    async def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest scraped data from file"""
        try:
            latest_file = self.output_path / 'playwright-tradingview_latest.json'
            if latest_file.exists():
                async with aiofiles.open(latest_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            return None
        except Exception as e:
            self.logger.error(f"Error reading latest data: {e}")
            return None

    def get_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from results"""
        if 'extraction_metadata' not in results:
            return {}
        
        metadata = results['extraction_metadata']
        
        return {
            'total_symbols': metadata.get('total_symbols', 0),
            'processing_time_ms': metadata.get('processing_time_ms', 0),
            'processing_time_s': metadata.get('processing_time_ms', 0) / 1000,
            'success_rate': metadata.get('success_rate', 0),
            'scraper_version': metadata.get('scraper_version', 'unknown'),
            'timestamp': metadata.get('timestamp', 'unknown')
        }

    def clear_cache(self):
        """Clear the results cache"""
        self.last_results.clear()
        if self.debug:
            self.logger.info("Cache cleared")


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced TradingView Scraper - Playwright Implementation')
    parser.add_argument('symbol', nargs='?', help='Stock symbol to scrape (optional)')
    parser.add_argument('--all', action='store_true', help='Scrape all symbols')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = PlaywrightTradingViewScraper({
        'debug': args.debug,
        'timeout': 60000,
        'concurrency': 2
    })
    
    try:
        if args.symbol:
            # Scrape single symbol
            result = await scraper.scrape_symbol(args.symbol.upper())
            
            if args.output:
                if args.format == 'json':
                    async with aiofiles.open(args.output, 'w') as f:
                        await f.write(json.dumps(result, indent=2))
                else:
                    df = pd.DataFrame([result])
                    df.to_csv(args.output, index=False)
            else:
                print(json.dumps(result, indent=2))
        
        elif args.all:
            # Scrape all symbols
            results = await scraper.scrape_all_symbols()
            
            if args.output:
                if args.format == 'json':
                    async with aiofiles.open(args.output, 'w') as f:
                        await f.write(json.dumps(results, indent=2))
                else:
                    df = scraper.process_results(results)
                    df.to_csv(args.output, index=False)
            else:
                # Print summary
                if 'extraction_metadata' in results:
                    metadata = results['extraction_metadata']
                    print(f"=== SCRAPING COMPLETED ===")
                    print(f"Total symbols: {metadata.get('total_symbols', 0)}")
                    print(f"Processing time: {metadata.get('processing_time_ms', 0)}ms")
                    print(f"Success rate: {metadata.get('success_rate', 0)}%")
                    print()
                
                # Print individual results
                if 'companies' in results:
                    for company in results['companies']:
                        symbol = company.get('symbol', 'N/A')
                        basic_data = company.get('basic_data', {})
                        technical_data = company.get('technical_indicators', {})
                        
                        price = basic_data.get('current_price', 'N/A')
                        technical = technical_data.get('technical_summary', 'N/A')
                        errors = len(company.get('errors', []))
                        
                        error_str = f" ({errors} errors)" if errors > 0 else ""
                        print(f"{symbol}: ${price} - {technical}{error_str}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 