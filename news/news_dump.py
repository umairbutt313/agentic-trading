#!/usr/bin/env python3
"""
News Dump Module
Fetches and stores the latest news for NVIDIA and Apple from NewsAPI.org
No AI sentiment analysis - only raw news collection with timestamps.
"""

import os
import json
import requests
import logging
import sys
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from project root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Configure logging
# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'news_dump.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

class NewsDumper:
    def load_companies_config(self, config_path: str = "companies.yaml") -> Dict:
        """
        Load companies configuration from YAML file
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing company configurations
        """
        try:
            # Try relative path from current directory first
            if os.path.exists(config_path):
                config_file = config_path
            else:
                # Try from parent directory (for news/ subfolder)
                parent_config = os.path.join("..", config_path)
                if os.path.exists(parent_config):
                    config_file = parent_config
                else:
                    logger.error(f"Companies config file not found: {config_path}")
                    return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded companies configuration from: {config_file}")
            return config.get('companies', {})
            
        except Exception as e:
            logger.error(f"Error loading companies config: {e}")
            return {}
    
    def __init__(self, api_key: str = None):
        """
        Initialize NewsDumper with NewsAPI key(s)

        Args:
            api_key: NewsAPI.org API key. If None, will try to get from environment
        """
        # OLD: Single API key only
        # self.api_key = api_key or os.getenv('NEWS_API_KEY')

        # NEW: Support multiple API keys for fallback when rate limited
        # Loads NEWS_API_KEY, NEWS_API_KEY_2, NEWS_API_KEY_3 from .env
        self.api_keys = []
        if api_key:
            self.api_keys.append(api_key)
        else:
            # Load all available API keys from environment
            for i in ['', '_2', '_3']:
                key = os.getenv(f'NEWS_API_KEY{i}')
                if key and key.strip():  # Only add non-empty keys
                    self.api_keys.append(key.strip())

        if not self.api_keys:
            logger.error("No NewsAPI keys found in environment!")
            self.api_keys = [None]  # Fallback to prevent crashes

        self.current_key_index = 0  # Track which key we're using
        self.api_key = self.api_keys[0]  # Current active key
        self.base_url = "https://newsapi.org/v2/everything"

        # Load company configurations from YAML file
        companies_config = self.load_companies_config()

        # Convert YAML config to expected format for NewsAPI
        self.companies = {}
        for company_key, config in companies_config.items():
            self.companies[company_key] = {
                'queries': config.get('queries', []),
                'symbol': config.get('symbol', '')
            }

        logger.info(f"Initialized NewsDumper with {len(self.api_keys)} API key(s)")
        for idx, key in enumerate(self.api_keys, 1):
            logger.info(f"  Key #{idx}: ...{key[-4:] if key else 'None'}")
    
    def fetch_company_news(self, company: str, days: int = 7, page_size: int = 100) -> List[Dict]:
        """
        Fetch news for a specific company from NewsAPI
        
        Args:
            company: Company name ('nvidia' or 'apple')
            days: Number of days to look back for news
            page_size: Number of articles to fetch (max 100)
            
        Returns:
            List of news articles with metadata
        """
        if company not in self.companies:
            logger.error(f"Unknown company: {company}. Available: {list(self.companies.keys())}")
            return []
        
        config = self.companies[company]
        
        # Create search query combining all query terms
        query = ' OR '.join([f'"{q}"' for q in config['queries']])
        
        # Calculate date range
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.api_key,
            'pageSize': page_size
        }
        
        logger.info(f"Fetching {company.upper()} news from {from_date} to {to_date}")
        logger.debug(f"Query: {query}")

        # NEW: Try with first API key
        try:
            response = requests.get(self.base_url, params=params, timeout=30)

            # NEW: If rate limited (429), try backup key
            if response.status_code == 429 and len(self.api_keys) > 1:
                logger.warning(f"⚠️  API key #1 rate limited (429), trying backup key #2")
                params['apiKey'] = self.api_keys[1]  # Use second key
                response = requests.get(self.base_url, params=params, timeout=30)
                if response.status_code == 429:
                    logger.error(f"❌ Both API keys rate limited")
                    return []
                logger.info(f"✅ Backup API key #2 worked!")

            response.raise_for_status()

            data = response.json()

            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error for {company}: {data.get('message', 'Unknown error')}")
                return []

            articles = data.get('articles', [])
            logger.info(f"Successfully fetched {len(articles)} articles for {company.upper()}")

            # Process and enrich articles
            processed_articles = []
            for i, article in enumerate(articles, 1):
                processed_article = {
                    'id': f"{company}_{i}_{int(datetime.now().timestamp())}",
                    'company': company.upper(),
                    'symbol': config['symbol'],
                    'title': article.get('title', 'No Title'),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source_name': article.get('source', {}).get('name', 'Unknown'),
                    'source_id': article.get('source', {}).get('id', ''),
                    'author': article.get('author', ''),
                    'published_at': article.get('publishedAt', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'fetched_at': datetime.now().isoformat(),
                    'fetch_date': datetime.now().strftime("%Y-%m-%d"),
                    'fetch_time': datetime.now().strftime("%H:%M:%S")
                }
                processed_articles.append(processed_article)

            return processed_articles

        except requests.RequestException as e:
            logger.error(f"Network error fetching {company} news: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching {company} news: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return []
    
    def remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """
        Remove duplicate articles based on title and URL
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of unique articles
        """
        seen = set()
        unique_articles = []
        duplicates_removed = 0
        
        for article in articles:
            # Create identifier from title and URL
            identifier = (
                article.get('title', '').strip().lower(),
                article.get('url', '').strip()
            )
            
            if identifier not in seen and identifier[0] and identifier[1]:
                seen.add(identifier)
                unique_articles.append(article)
            else:
                duplicates_removed += 1
        
        logger.info(f"Removed {duplicates_removed} duplicate articles")
        return unique_articles
    
    def save_to_json(self, articles: List[Dict], company: str, include_metadata: bool = True) -> str:
        """
        Save articles to JSON file with timestamp
        
        Args:
            articles: List of article dictionaries
            company: Company name for filename
            include_metadata: Whether to include metadata in output
            
        Returns:
            Path to saved file
        """
        # Create output directory if it doesn't exist
        # Ensure output directory is created relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        dumps_dir = os.path.join(project_root, "container_output", "news")
        os.makedirs(dumps_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(dumps_dir, f"raw-news-images_{timestamp}.json")
        
        # Prepare data structure
        data = {
            'fetch_metadata': {
                'company': company.upper(),
                'fetch_timestamp': datetime.now().isoformat(),
                'fetch_date': datetime.now().strftime("%Y-%m-%d"),
                'fetch_time': datetime.now().strftime("%H:%M:%S"),
                'total_articles': len(articles),
                'api_source': 'NewsAPI.org'
            },
            'articles': articles
        } if include_metadata else articles
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Get file size for logging
            file_size = os.path.getsize(filename)
            logger.info(f"Saved {len(articles)} {company.upper()} articles to {filename} ({file_size} bytes)")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving {company} articles to JSON: {e}")
            return ""
    
    def print_summary(self, articles: List[Dict], company: str):
        """
        Print a formatted summary of fetched articles
        
        Args:
            articles: List of article dictionaries
            company: Company name
        """
        print(f"\n{'='*60}")
        print(f"{company.upper()} NEWS DUMP SUMMARY")
        print(f"{'='*60}")
        print(f"Total Articles: {len(articles)}")
        print(f"Fetch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if articles:
            print(f"\nLatest Articles (showing first 5):")
            print(f"{'-'*60}")
            
            for i, article in enumerate(articles[:5], 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Source: {article['source_name']}")
                print(f"   Published: {article['published_at']}")
                if article['description']:
                    desc = article['description'][:100] + "..." if len(article['description']) > 100 else article['description']
                    print(f"   Description: {desc}")
                print(f"   URL: {article['url']}")
            
            if len(articles) > 5:
                print(f"\n... and {len(articles) - 5} more articles")
    
    def _extract_image_url(self, submission) -> str:
        """
        Extract image URL from Reddit submission if it contains an image
        
        Args:
            submission: praw.models.Submission object
            
        Returns:
            Image URL string or empty string if no image found
        """
        try:
            # Check if submission has image content
            if hasattr(submission, 'post_hint') and submission.post_hint == 'image':
                return submission.url
                
            # Check for common image hosts and file extensions
            if submission.url:
                url = submission.url.lower()
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
                image_hosts = ['i.redd.it', 'i.imgur.com', 'imgur.com', 'reddit.com/media']
                
                # Direct image links
                if any(url.endswith(ext) for ext in image_extensions):
                    return submission.url
                    
                # Known image hosting domains
                if any(host in url for host in image_hosts):
                    return submission.url
                    
                # Imgur links (add .jpg if missing extension)
                if 'imgur.com/' in url and not any(ext in url for ext in image_extensions):
                    if url.count('/') >= 3:  # Valid imgur URL structure
                        return submission.url + '.jpg'
                        
            # Check for preview images in submission data
            if hasattr(submission, 'preview') and submission.preview:
                try:
                    preview_images = submission.preview.get('images', [])
                    if preview_images and 'source' in preview_images[0]:
                        return preview_images[0]['source']['url']
                except Exception:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error extracting image URL from submission {submission.id}: {e}")
            
        return ""
    
    def _download_image(self, image_url: str, company: str, post_id: str, timestamp: str) -> str:
        """
        Download image from URL and save to local storage
        
        Args:
            image_url: URL of the image to download
            company: Company name (NVIDIA, APPLE)
            post_id: Reddit post ID
            timestamp: Timestamp for filename
            
        Returns:
            Local file path of downloaded image or empty string if failed
        """
        if not image_url:
            return ""
            
        try:
            import requests
            import os
            from urllib.parse import urlparse
            
            # Create images directory if it doesn't exist
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            images_dir = os.path.join(project_root, "container_output", "images")
            os.makedirs(images_dir, exist_ok=True)
            
            # Get file extension from URL
            parsed_url = urlparse(image_url)
            path = parsed_url.path.lower()
            
            # Determine file extension
            if path.endswith(('.jpg', '.jpeg')):
                ext = '.jpg'
            elif path.endswith('.png'):
                ext = '.png'
            elif path.endswith('.gif'):
                ext = '.gif'
            elif path.endswith('.webp'):
                ext = '.webp'
            else:
                ext = '.jpg'  # Default to jpg
                
            # Create filename: timestamp_company_postid.ext
            filename = f"{timestamp}_{company}_{post_id}{ext}"
            filepath = os.path.join(images_dir, filename)
            
            # Check if image already exists (avoid duplicates)
            if os.path.exists(filepath):
                logger.debug(f"Image already exists: {filename}")
                return filepath
            
            # Download image with timeout and size limits
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(image_url, headers=headers, timeout=15, stream=True)
            response.raise_for_status()
            
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                logger.debug(f"URL does not return image content: {image_url}")
                return ""
            
            # Check file size (limit to 10MB)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:
                logger.debug(f"Image too large ({content_length} bytes): {image_url}")
                return ""
                
            # Save image with size monitoring
            downloaded_size = 0
            max_size = 10 * 1024 * 1024  # 10MB limit
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded_size += len(chunk)
                    if downloaded_size > max_size:
                        logger.debug(f"Image size exceeded limit during download: {image_url}")
                        os.remove(filepath)  # Clean up partial file
                        return ""
                    f.write(chunk)
                    
            logger.info(f"   📷 Downloaded image: {filename}")
            return filepath
            
        except Exception as e:
            logger.debug(f"Failed to download image {image_url}: {e}")
            return ""
    
    def collect_reddit_data(self) -> List[Dict]:
        """
        Collect Reddit data for both companies
        
        Returns:
            List of Reddit mentions with NewsAPI-compatible format
        """
        # REDDIT FEATURE DISABLED - Uncomment below to re-enable
        return []  # Return empty list when Reddit is disabled
        
        # reddit_mentions = []
        # 
        # # Reddit credentials from environment variables
        # reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        # reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        # reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'stockk/1.0 by uab313')
        # reddit_channels = ['stocks', 'wallstreetbets', 'investing']
        
        # # Load company keywords from YAML configuration
        # companies_config = self.load_companies_config()
        # company_keywords = {}
        # for company_key, config in companies_config.items():
        #     company_name = config.get('name', '').upper()
        #     keywords = config.get('keywords', [])
        #     if company_name and keywords:
        #         company_keywords[company_name] = keywords
        # 
        # try:
        #     import praw
        #     
        #     reddit_client = praw.Reddit(
        #         client_id=reddit_client_id,
        #         client_secret=reddit_client_secret,
        #         user_agent=reddit_user_agent
        #     )
        #     
        #     # Look back 24 hours for Reddit data
        #     since_utc = (datetime.utcnow() - timedelta(hours=24)).timestamp()
        #     
        #     logger.info("Collecting Reddit data from 3 channels...")
        #     
        #     for company, keywords in company_keywords.items():
        #         for channel in reddit_channels:
        #             try:
        #                 logger.info(f"   📡 Scanning r/{channel} for {company}...")
        #                 subreddit = reddit_client.subreddit(channel)
        #                 
        #                 # Get recent posts
        #                 for submission in subreddit.new(limit=50):
        #                     if submission.created_utc < since_utc:
        #                         continue
        #                     
        #                     # Check if post mentions the company
        #                     text = (submission.title + " " + (submission.selftext or "")).lower()
        #                     if any(keyword.lower() in text for keyword in keywords):
        #                         # Detect image content
        #                         image_url = self._extract_image_url(submission)
        #                         has_image = bool(image_url)
        #                         
        #                         # Download image if found
        #                         local_image_path = ""
        #                         if has_image:
        #                             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #                             local_image_path = self._download_image(
        #                                 image_url, company, submission.id, timestamp
        #                             )
        #                         
        #                         reddit_mentions.append({
        #                             'id': f"reddit_{company}_{submission.id}_{int(datetime.now().timestamp())}",
        #                             'company': company.upper(),
        #                             'symbol': 'NVDA' if company == 'NVIDIA' else 'AAPL',
        #                             'title': submission.title,
        #                             'description': submission.selftext or "",
        #                             'content': submission.selftext or "",
        #                             'url': f"https://reddit.com{submission.permalink}",
        #                             'source_name': f"r/{channel}",
        #                             'source_id': channel,
        #                             'author': str(submission.author) if submission.author else "",
        #                             'published_at': datetime.fromtimestamp(submission.created_utc).isoformat(),
        #                             'url_to_image': image_url,
        #                             'fetched_at': datetime.now().isoformat(),
        #                             'fetch_date': datetime.now().strftime("%Y-%m-%d"),
        #                             'fetch_time': datetime.now().strftime("%H:%M:%S"),
        #                             'data_source': 'Reddit',
        #                             'reddit_score': submission.score,
        #                             'reddit_comments': submission.num_comments,
        #                             'reddit_channel': channel,
        #                             'has_image': has_image,
        #                             'reddit_post_id': submission.id,
        #                             'local_image_path': local_image_path
        #                         })
        #                         
        #             except Exception as e:
        #                 logger.error(f"Error scanning r/{channel} for {company}: {e}")
        #     
        #     logger.info(f"Collected {len(reddit_mentions)} Reddit mentions")
        #     return reddit_mentions
        #     
        # except ImportError:
        #     logger.warning("praw library not available - skipping Reddit data collection")
        #     return []
        # except Exception as e:
        #     logger.error(f"Error collecting Reddit data: {e}")
        #     return []
    
    # OLD: def dump_all_news(self, days: int = 7, save_separate: bool = True, print_summaries: bool = True) -> Dict:
    # NEW: Added session_file parameter to support single-file-per-session mode
    def dump_all_news(self, days: int = 7, save_separate: bool = True, print_summaries: bool = True, session_file: str = None) -> Dict:
        """
        Main method to dump news for all configured companies

        Args:
            days: Number of days to look back
            save_separate: Whether to save separate files for each company
            print_summaries: Whether to print summaries
            
        Returns:
            Dictionary with results for each company
        """
        logger.info("Starting news dump process for all companies...")
        start_time = datetime.now()
        
        results = {}
        
        for company in self.companies.keys():
            logger.info(f"Processing {company.upper()}...")
            
            # Fetch articles
            articles = self.fetch_company_news(company, days=days)
            
            if not articles:
                logger.warning(f"No articles found for {company.upper()}")
                results[company] = {
                    'articles': [],
                    'filename': '',
                    'success': False
                }
                continue
            
            # Remove duplicates
            unique_articles = self.remove_duplicates(articles)
            
            # Skip individual file creation - only combined dump needed
            filename = ""
            
            # Print summary
            if print_summaries:
                self.print_summary(unique_articles, company)
            
            results[company] = {
                'articles': unique_articles,
                'filename': filename,
                'success': True
            }
        
        # Collect Reddit data for both companies
        # REDDIT DISABLED - Uncomment line below to re-enable
        # reddit_data = self.collect_reddit_data()
        reddit_data = []  # Reddit disabled - using empty list
        
        # Save combined file (NewsAPI + Reddit)
        if len(results) > 0:
            all_articles = []
            for company_result in results.values():
                if company_result['success']:
                    all_articles.extend(company_result['articles'])

            # Add Reddit data to articles
            all_articles.extend(reddit_data)

            # OLD: Only created file when articles exist (broke session file tracking)
            # if all_articles:

            # NEW: Always create file (even with 0 articles) to maintain session file
            # This ensures session files exist even during NewsAPI rate limits
            if True:  # Always create file for session tracking
                # Create output directory if it doesn't exist
                # Ensure output directory is created relative to project root
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                dumps_dir = os.path.join(project_root, "container_output", "news")
                os.makedirs(dumps_dir, exist_ok=True)

                # OLD: Always created new timestamped file (caused 12 new files/hour in live system)
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # combined_filename = os.path.join(dumps_dir, f"raw-news-images_{timestamp}.json")

                # NEW: Use session file if provided (live_sentiment_system.py sessions)
                # This reuses same file during session, only creates new when restarted
                if session_file:
                    combined_filename = os.path.join(dumps_dir, session_file)
                    logger.info(f"♻️  Updating session file: {session_file}")
                else:
                    # Manual run: create new timestamped file as before
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    combined_filename = os.path.join(dumps_dir, f"raw-news-images_{timestamp}.json")
                    logger.info(f"🆕 Creating new file: raw-news-images_{timestamp}.json")
                
                # Count by source
                newsapi_count = len([a for a in all_articles if a.get('data_source') != 'Reddit'])
                reddit_count = len([a for a in all_articles if a.get('data_source') == 'Reddit'])
                
                combined_data = {
                    'fetch_metadata': {
                        'companies': list(self.companies.keys()),
                        'fetch_timestamp': datetime.now().isoformat(),
                        'total_articles': len(all_articles),
                        'source_breakdown': {
                            'newsapi_articles': newsapi_count,
                            'reddit_mentions': reddit_count
                        },
                        # 'data_sources': 'NewsAPI.org + Reddit (3 channels)'  # Reddit disabled
                        'data_sources': 'NewsAPI.org'  # Reddit disabled - only NewsAPI active
                    },
                    'articles': all_articles
                }
                
                with open(combined_filename, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ Saved combined dump: {combined_filename}")
                logger.info(f"  📰 NewsAPI articles: {newsapi_count}")
                logger.info(f"  🔴 Reddit mentions: {reddit_count}")
                logger.info(f"  📊 Total data points: {len(all_articles)}")
                logger.info(f"  📁 All data consolidated in single file!")
        
        elapsed_time = datetime.now() - start_time
        logger.info(f"News dump process completed in {elapsed_time.total_seconds():.2f} seconds")
        
        return results

def main():
    """
    Main function to run the news dumper
    """
    # NEW: Added argparse to support --session-file parameter from live_sentiment_system.py
    import argparse

    parser = argparse.ArgumentParser(description='News Dump - Fetch news from NewsAPI')
    parser.add_argument('--session-file', type=str, help='Session filename to reuse (for live system)')
    args = parser.parse_args()

    logger.info("Starting News Dump application...")

    try:
        # Initialize dumper
        dumper = NewsDumper()

        # OLD: Always created new file
        # results = dumper.dump_all_news(days=7)

        # NEW: Pass session file if provided (for live system), otherwise creates new file
        results = dumper.dump_all_news(days=7, session_file=args.session_file)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        
        total_articles = 0
        successful_companies = 0
        
        for company, result in results.items():
            if result['success']:
                article_count = len(result['articles'])
                print(f"{company.upper()}: {article_count} articles")
                total_articles += article_count
                successful_companies += 1
            else:
                print(f"{company.upper()}: FAILED")
        
        print(f"\nTotal: {total_articles} articles from {successful_companies} companies")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        logger.warning("Application interrupted by user (Ctrl+C)")
        print("\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Critical error in main application: {e}")
        logger.debug("Stack trace:", exc_info=True)
        print(f"Error running news dumper: {e}")

if __name__ == "__main__":
    main()