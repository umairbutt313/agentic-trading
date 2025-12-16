#!/usr/bin/env python3
"""
NVIDIA Stock News Fetcher
Retrieves the latest news about NVIDIA (NVDA) stock from Yahoo Finance
"""

import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import re
import logging
import sys
import os
from typing import List, Dict, Optional

# Configure logging
# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'nvidia_news.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

class NvidiaNewsFetcher:
    def __init__(self):
        self.ticker = "NVDA"
        self.stock = yf.Ticker(self.ticker)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logger.info(f"Initialized NVIDIA News Fetcher for ticker: {self.ticker}")
    
    def get_yahoo_finance_news(self) -> List[Dict]:
        """
        Fetch news from Yahoo Finance using yfinance library
        """
        logger.info("Fetching news from Yahoo Finance API...")
        try:
            start_time = time.time()
            news_data = []
            news = self.stock.news
            
            logger.info(f"Found {len(news)} articles from Yahoo Finance API")
            
            for i, article in enumerate(news, 1):
                logger.debug(f"   Processing article {i}/{len(news)}: {article.get('title', 'No Title')[:50]}...")
                
                news_item = {
                    'title': article.get('title', 'No Title'),
                    'publisher': article.get('publisher', 'Unknown'),
                    'link': article.get('link', ''),
                    'publish_time': datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                    'type': article.get('type', 'NEWS'),
                    'uuid': article.get('uuid', ''),
                    'summary': article.get('summary', 'No summary available')
                }
                news_data.append(news_item)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully fetched {len(news_data)} articles from Yahoo Finance API in {elapsed_time:.2f} seconds")
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return []
    
    def scrape_yahoo_finance_page(self) -> List[Dict]:
        """
        Scrape additional news from Yahoo Finance NVDA page
        """
        logger.info("Scraping additional news from Yahoo Finance web page...")
        try:
            start_time = time.time()
            url = f"https://finance.yahoo.com/quote/{self.ticker}/news"
            logger.debug(f"   Requesting URL: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Successfully retrieved web page (Status: {response.status_code})")
            logger.debug(f"   Response size: {len(response.content)} bytes")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = []
            
            # Find news articles in the page
            news_sections = soup.find_all('div', {'class': re.compile(r'.*news.*', re.I)})
            logger.debug(f"   Found {len(news_sections)} potential news sections")
            
            for section_idx, section in enumerate(news_sections, 1):
                logger.debug(f"   Scanning section {section_idx}/{len(news_sections)}")
                headlines = section.find_all(['h3', 'h4', 'a'], {'class': re.compile(r'.*headline.*|.*title.*', re.I)})
                
                for headline in headlines:
                    if headline.text.strip():
                        link = headline.get('href', '')
                        if link and not link.startswith('http'):
                            link = 'https://finance.yahoo.com' + link
                        
                        news_item = {
                            'title': headline.text.strip(),
                            'link': link,
                            'source': 'Yahoo Finance Scrape',
                            'scraped_time': datetime.now()
                        }
                        news_items.append(news_item)
                        logger.debug(f"     Found article: {headline.text.strip()[:50]}...")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully scraped {len(news_items)} additional articles in {elapsed_time:.2f} seconds")
            return news_items
            
        except requests.RequestException as e:
            logger.error(f"Network error while scraping Yahoo Finance page: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance page: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return []
    
    def get_stock_info(self) -> Dict:
        """
        Get current NVIDIA stock information
        """
        logger.info("Fetching current NVIDIA stock information...")
        try:
            start_time = time.time()
            
            logger.debug("   Retrieving stock info from yfinance...")
            info = self.stock.info
            
            logger.debug("   Retrieving latest price history...")
            history = self.stock.history(period="1d")
            
            current_price = history['Close'].iloc[-1] if not history.empty else None
            
            stock_info = {
                'symbol': info.get('symbol', 'NVDA'),
                'company_name': info.get('longName', 'NVIDIA Corporation'),
                'current_price': current_price,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'last_updated': datetime.now()
            }
            
            elapsed_time = time.time() - start_time
            logger.info(f"Stock information retrieved in {elapsed_time:.2f} seconds")
            logger.info(f"Current Price: ${current_price:.2f}" if current_price else "Current Price: N/A")
            logger.info(f"Market Cap: {info.get('marketCap', 'N/A')}")
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return {}
    
    def save_to_json(self, data: Dict, filename: str = None):
        """
        Save data to JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nvidia_news_{timestamp}.json"
        
        logger.info(f"Saving data to JSON file: {filename}")
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            # Get file size for logging
            import os
            file_size = os.path.getsize(filename)
            logger.info(f"JSON data saved successfully ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            logger.debug("Stack trace:", exc_info=True)
    
    def save_to_csv(self, news_data: List[Dict], filename: str = None):
        """
        Save news data to CSV file
        """
        if not news_data:
            logger.warning("No news data to save to CSV")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nvidia_news_{timestamp}.csv"
        
        logger.info(f"Saving {len(news_data)} articles to CSV file: {filename}")
        try:
            df = pd.DataFrame(news_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            
            # Get file size for logging
            import os
            file_size = os.path.getsize(filename)
            logger.info(f"CSV data saved successfully ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            logger.debug("Stack trace:", exc_info=True)
    
    def print_news_summary(self, news_data: List[Dict], stock_info: Dict):
        """
        Print a formatted summary of the news and stock info
        """
        logger.info("Generating news summary for display...")
        
        print("\n" + "=" * 80)
        print(f"NVIDIA (NVDA) STOCK NEWS SUMMARY")
        print("=" * 80)
        
        if stock_info:
            print(f"\nStock Information:")
            print(f"   Company: {stock_info.get('company_name', 'N/A')}")
            if stock_info.get('current_price'):
                print(f"   Current Price: ${stock_info.get('current_price'):.2f}")
            else:
                print(f"   Current Price: N/A")
            print(f"   Market Cap: {stock_info.get('market_cap', 'N/A')}")
            print(f"   P/E Ratio: {stock_info.get('pe_ratio', 'N/A')}")
            print(f"   Sector: {stock_info.get('sector', 'N/A')}")
        
        print(f"\nLatest News Articles ({len(news_data)} found):")
        print("-" * 80)
        
        displayed_articles = min(10, len(news_data))
        logger.info(f"Displaying top {displayed_articles} articles out of {len(news_data)} total")
        
        for i, article in enumerate(news_data[:10], 1):  # Show top 10 articles
            print(f"\n{i}. {article.get('title', 'No Title')}")
            print(f"   Publisher: {article.get('publisher', article.get('source', 'Unknown'))}")
            
            pub_time = article.get('publish_time') or article.get('scraped_time')
            if pub_time:
                print(f"   Time: {pub_time}")
            
            if article.get('summary'):
                summary = article['summary'][:200] + "..." if len(article['summary']) > 200 else article['summary']
                print(f"   Summary: {summary}")
            
            if article.get('link'):
                print(f"   Link: {article['link']}")
        
        if len(news_data) > 10:
            print(f"\n... and {len(news_data) - 10} more articles (saved to files)")
    
    def fetch_all_news(self, save_json: bool = True, save_csv: bool = True, print_summary: bool = True) -> Dict:
        """
        Main method to fetch all news and stock information
        """
        logger.info("Starting NVIDIA news fetch process...")
        overall_start_time = time.time()
        
        # Get news from different sources
        logger.info("Fetching news from multiple sources...")
        yf_news = self.get_yahoo_finance_news()
        scraped_news = self.scrape_yahoo_finance_page()
        
        # Get stock information
        stock_info = self.get_stock_info()
        
        # Combine all news
        logger.info("Processing and combining news data...")
        all_news = yf_news + scraped_news
        logger.info(f"Combined total: {len(all_news)} articles from all sources")
        
        # Remove duplicates based on title
        logger.info("Removing duplicate articles...")
        seen_titles = set()
        unique_news = []
        duplicates_removed = 0
        
        for article in all_news:
            title = article.get('title', '').strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(article)
            else:
                duplicates_removed += 1
        
        logger.info(f"Removed {duplicates_removed} duplicate articles")
        logger.info(f"Final unique articles: {len(unique_news)}")
        
        # Sort by publish time (newest first)
        logger.info("Sorting articles by publication time...")
        unique_news.sort(key=lambda x: x.get('publish_time') or x.get('scraped_time') or datetime.min, reverse=True)
        
        # Prepare final data structure
        result = {
            'fetch_time': datetime.now(),
            'stock_info': stock_info,
            'news_count': len(unique_news),
            'news_articles': unique_news
        }
        
        # Save data
        if save_json:
            self.save_to_json(result)
        
        if save_csv:
            self.save_to_csv(unique_news)
        
        # Print summary
        if print_summary:
            self.print_news_summary(unique_news, stock_info)
        
        total_elapsed_time = time.time() - overall_start_time
        logger.info(f"News fetch process completed in {total_elapsed_time:.2f} seconds")
        
        return result

def main():
    """
    Main function to run the NVIDIA news fetcher
    """
    logger.info("Starting NVIDIA News Fetcher application...")
    start_time = time.time()
    
    try:
        fetcher = NvidiaNewsFetcher()
        result = fetcher.fetch_all_news()
        
        elapsed_time = time.time() - start_time
        
        print(f"\nSUCCESS SUMMARY:")
        print(f"Successfully fetched {result['news_count']} unique news articles about NVIDIA")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Data saved with timestamp: {result['fetch_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Check the current directory for output files")
        
        logger.info(f"Application completed successfully in {elapsed_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user (Ctrl+C)")
        print("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Critical error in main application: {e}")
        logger.debug("Stack trace:", exc_info=True)
        print(f"Error running NVIDIA news fetcher: {e}")
        print("Check nvidia_news.log for detailed error information")

if __name__ == "__main__":
    main() 