#!/usr/bin/env python3
"""
AI Sentiment Analyzer Module
Processes news dump files using Grok API (primary) or OpenAI's ChatGPT API (fallback) for sentiment scoring.
Takes dumped news data and returns numerical sentiment scores (1-10).

# ==============================================================================
# CHANGELOG:
# ==============================================================================
# [2025-12-10] FEATURE: Added Grok API support for real-time sentiment analysis
#              - Grok API is preferred (real-time X/Twitter integration)
#              - Falls back to OpenAI if Grok unavailable
#              - 60-70% cost savings with Grok 4 Fast
#              - Environment variable: GROK_TRADE_API
# ==============================================================================
"""

import os
import json
import logging
import sys
import time
import yaml
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Union
from openai import OpenAI
import praw
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
        logging.FileHandler(os.path.join(logs_dir, 'sentiment_analyzer.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def get_sentiment_file_path(self) -> str:
        """Get the consistent path for the sentiment analysis file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        return os.path.join(project_root, "container_output", "final_score", "news-sentiment-analysis.json")
    
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
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize SentimentAnalyzer with Grok API (preferred) or OpenAI API (fallback)

        Priority order:
        1. Grok API (GROK_TRADE_API) - Real-time X/Twitter sentiment, 60% cheaper
        2. OpenAI API (OPENAI_API_KEY) - Fallback for reliability

        Args:
            openai_api_key: OpenAI API key. If None, will try to get from environment
        """
        # ==============================================================================
        # GROK API INTEGRATION (2025-12-10)
        # Grok provides real-time X/Twitter sentiment analysis
        # OpenAI SDK compatible - just change base_url
        # ==============================================================================

        # Check for Grok API key first (preferred)
        self.grok_api_key = os.getenv('GROK_TRADE_API')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

        # Initialize API client with priority: Grok > OpenAI
        if self.grok_api_key:
            # Use Grok API (preferred - real-time sentiment, cheaper)
            self.client = OpenAI(
                api_key=self.grok_api_key,
                base_url="https://api.x.ai/v1"
            )
            self.model = "grok-3-fast"  # Fast model for sentiment analysis
            self.api_provider = "Grok"
            logger.info(f"🚀 Using GROK API for sentiment analysis (real-time X/Twitter integration)")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   API Key: ...{self.grok_api_key[-4:]}")
        elif self.openai_api_key:
            # Fallback to OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
            self.model = "gpt-3.5-turbo"
            self.api_provider = "OpenAI"
            logger.info(f"📡 Using OpenAI API for sentiment analysis (Grok not configured)")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   API Key: ...{self.openai_api_key[-4:]}")
        else:
            # No API keys available
            self.client = None
            self.model = None
            self.api_provider = None
            logger.error("❌ No API keys found! Set GROK_TRADE_API or OPENAI_API_KEY in .env")

        # Store for backward compatibility
        self.api_key = self.grok_api_key or self.openai_api_key

        # Reddit API credentials from environment variables
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'stockk/1.0 by uab313')
        self.reddit_channels = ['stocks', 'wallstreetbets', 'investing']

        # Load company configurations from YAML file
        self.companies_config = self.load_companies_config()

        # Build company keywords mapping for Reddit search
        self.company_keywords = {}
        for company_key, config in self.companies_config.items():
            company_name = config.get('name', '').upper()
            keywords = config.get('keywords', [])
            if company_name and keywords:
                self.company_keywords[company_name] = keywords

        # Initialize Reddit client
        try:
            self.reddit_client = praw.Reddit(
                client_id=self.reddit_client_id,
                client_secret=self.reddit_client_secret,
                user_agent=self.reddit_user_agent
            )
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit_client = None

        # Sentiment analysis prompt template
        self.prompt_template = """Analyze these news articles and score them from 1 to 10. (10 = best, 1 = worst). Provide only the final score. Do not include explanation or sentiment text.

Article Title: {title}
Article Description: {description}
Published: {published_at}

Score (1-10):"""

        logger.info(f"✅ SentimentAnalyzer initialized with {self.api_provider or 'NO'} API")
    
    def load_news_dump(self, filename: str) -> Dict:
        """
        Load news dump file created by news_dump.py
        
        Args:
            filename: Path to JSON file containing news dump
            
        Returns:
            Dictionary containing articles and metadata
        """
        logger.info(f"Loading news dump from: {filename}")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both structured (with metadata) and direct article list formats
            if isinstance(data, dict) and 'articles' in data:
                articles = data['articles']
                metadata = data.get('fetch_metadata', {})
            elif isinstance(data, list):
                articles = data
                metadata = {}
            else:
                logger.error(f"Unexpected data format in {filename}")
                return {'articles': [], 'metadata': {}}
            
            logger.info(f"Loaded {len(articles)} articles from {filename}")
            return {'articles': articles, 'metadata': metadata}
            
        except FileNotFoundError:
            logger.error(f"News dump file not found: {filename}")
            return {'articles': [], 'metadata': {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            return {'articles': [], 'metadata': {}}
        except Exception as e:
            logger.error(f"Error loading news dump {filename}: {e}")
            logger.debug("Stack trace:", exc_info=True)
            return {'articles': [], 'metadata': {}}
    
    def analyze_company_aggregate_sentiment(self, articles: List[Dict], company: str, max_retries: int = 3) -> Optional[float]:
        """
        Analyze aggregate sentiment for all articles of a company using ChatGPT
        
        Args:
            articles: List of article dictionaries for the company
            company: Company name (NVIDIA, APPLE, etc.)
            max_retries: Maximum number of API call retries
            
        Returns:
            Aggregate sentiment score (1-10) or None if analysis fails
        """
        # Prepare summary of all articles for aggregate analysis
        articles_summary = []
        for i, article in enumerate(articles[:20], 1):  # Limit to first 20 articles to stay within token limits
            title = article.get('title', 'No Title')
            description = article.get('description', 'No Description')
            published_at = article.get('published_at', 'Unknown')
            
            summary = f"{i}. Title: {title}\n   Description: {description[:150]}{'...' if len(description) > 150 else ''}\n   Published: {published_at}\n"
            articles_summary.append(summary)
        
        # Create aggregate analysis prompt
        prompt = f"""Analyze all these news articles about {company} stock and provide ONE overall sentiment score from 1 to 10.
(1 = very negative/bearish, 10 = very positive/bullish)

Consider the overall tone, market impact, and collective sentiment across all articles.
Provide ONLY the final aggregate score as a number. Do not include explanation or individual scores.

News Articles ({len(articles)} total, showing first {min(20, len(articles))}):

{''.join(articles_summary)}

Overall {company} Stock Sentiment Score (1-10):"""
        
        logger.info(f"Analyzing aggregate sentiment for {company} based on {len(articles)} articles...")
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,  # Uses Grok or OpenAI based on config
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a financial sentiment analyzer. Analyze all the {company} news articles and respond only with ONE aggregate sentiment number from 1 to 10, where 1 = very negative/bearish and 10 = very positive/bullish for {company} stock."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=15,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract numeric score
                try:
                    # Try to extract number from response
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        # Ensure score is in valid range
                        if 1 <= score <= 10:
                            logger.info(f"Aggregate sentiment score for {company}: {score}/10")
                            return score
                        else:
                            logger.warning(f"Score {score} out of range for {company}")
                    else:
                        logger.warning(f"No numeric score found in response: {content}")
                
                except ValueError:
                    logger.warning(f"Could not parse score from response: {content}")
                
            except Exception as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting 30 seconds...")
                    time.sleep(30)
                    continue
                
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error analyzing {company} sentiment (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return None
        
        logger.error(f"Failed to analyze {company} sentiment after {max_retries} attempts")
        return None
    
    def collect_reddit_data(self, company: str, hours_back: int = 24) -> List[Dict]:
        """
        Collect Reddit data for a company
        
        Args:
            company: Company name (NVIDIA, APPLE)
            hours_back: How many hours back to search
            
        Returns:
            List of Reddit mentions
        """
        if not self.reddit_client:
            logger.warning("Reddit client not available")
            return []
        
        if company not in self.company_keywords:
            logger.warning(f"No keywords defined for {company}")
            return []
        
        keywords = self.company_keywords[company]
        mentions = []
        since_utc = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp()
        
        logger.info(f"Collecting Reddit data for {company} from last {hours_back} hours...")
        
        for channel in self.reddit_channels:
            try:
                logger.debug(f"   Scanning r/{channel}...")
                subreddit = self.reddit_client.subreddit(channel)
                
                # Get recent posts
                for submission in subreddit.new(limit=50):
                    if submission.created_utc < since_utc:
                        continue
                    
                    # Check if post mentions the company
                    text = (submission.title + " " + (submission.selftext or "")).lower()
                    if any(keyword.lower() in text for keyword in keywords):
                        mentions.append({
                            'source': 'Reddit',
                            'channel': channel,
                            'title': submission.title,
                            'description': submission.selftext or "",
                            'content': submission.selftext or "",
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': submission.created_utc,
                            'published_at': datetime.fromtimestamp(submission.created_utc).isoformat(),
                            'url': submission.url,
                            'source_name': f"r/{channel}"
                        })
                        
            except Exception as e:
                logger.error(f"Error scanning r/{channel}: {e}")
        
        logger.info(f"   Found {len(mentions)} Reddit mentions for {company}")
        return mentions
    
    def analyze_image_sentiment(self, image_path: str, company: str, post_title: str = "", max_retries: int = 3) -> Optional[float]:
        """
        Analyze sentiment from an image using OpenAI GPT-4 Vision API
        
        Args:
            image_path: Local path to the image file
            company: Company name for context
            post_title: Reddit post title for context
            max_retries: Maximum API retries
            
        Returns:
            Sentiment score (1-10) or None if analysis failed
        """
        if not image_path or not os.path.exists(image_path):
            return None
            
        try:
            import base64
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create prompt for financial sentiment analysis
            prompt = f"""
Analyze this image for stock market sentiment about {company}.

Context: This image was posted on Reddit in a stock discussion forum with title: "{post_title}"

Please analyze:
1. Memes or jokes about {company} stock (bullish/bearish?)
2. Charts, graphs, or technical analysis screenshots
3. Company logos, products, or brand imagery
4. Reaction images/GIFs expressing emotions about the stock
5. Text overlays with stock predictions or opinions

Provide ONLY a sentiment score from 1 to 10:
- 1-3: Very bearish/negative sentiment
- 4-6: Neutral sentiment  
- 7-10: Very bullish/positive sentiment

Respond with only the number (1-10):"""

            for attempt in range(max_retries):
                try:
                    # Use Grok vision model if available, otherwise OpenAI GPT-4o
                    vision_model = "grok-2-vision" if self.api_provider == "Grok" else "gpt-4o"
                    response = self.client.chat.completions.create(
                        model=vision_model,  # Grok-2-vision or GPT-4o for image analysis
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=10,
                        temperature=0.3
                    )
                    
                    content = response.choices[0].message.content.strip()
                    
                    # Extract score
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 1 <= score <= 10:
                            logger.info(f"   🖼️  Image sentiment score: {score}/10")
                            return score
                    
                    logger.warning(f"Invalid image sentiment response: {content}")
                    
                except Exception as e:
                    logger.warning(f"Image sentiment analysis attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    
            logger.error(f"Failed to analyze image sentiment after {max_retries} attempts: {image_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing image sentiment for {image_path}: {e}")
            return None
    
    def analyze_combined_data_sentiment(self, combined_data: List[Dict], company: str, max_retries: int = 3) -> Optional[float]:
        """
        Analyze sentiment from combined NewsAPI + Reddit data
        
        Args:
            combined_data: List containing both news articles and Reddit mentions
            company: Company name
            max_retries: Maximum API retries
            
        Returns:
            Combined sentiment score (1-10)
        """
        # Separate data by source for organized prompt
        news_articles = [item for item in combined_data if item.get('data_source', 'NewsAPI') != 'Reddit']
        reddit_mentions = [item for item in combined_data if item.get('data_source') == 'Reddit']
        
        # Prepare comprehensive summary
        summary_parts = []
        
        # News section
        if news_articles:
            summary_parts.append(f"NEWS ARTICLES ({len(news_articles)} total):")
            for i, article in enumerate(news_articles[:10], 1):  # Limit to 10
                title = article.get('title', 'No title')
                desc = article.get('description') or ''
                desc = desc[:150] if desc else ''
                source = article.get('source_name', 'Unknown')
                summary_parts.append(f"{i}. {title}")
                if desc:
                    summary_parts.append(f"   {desc}...")
                summary_parts.append(f"   Source: {source}")
                summary_parts.append("")
        
        # Reddit section with image analysis
        image_sentiments = []
        text_sentiments = []
        
        if reddit_mentions:
            summary_parts.append(f"REDDIT DISCUSSIONS ({len(reddit_mentions)} total):")
            for i, mention in enumerate(reddit_mentions[:10], 1):  # Limit to 10
                title = mention.get('title', 'No title')
                channel = mention.get('reddit_channel', 'unknown')
                score = mention.get('reddit_score', 0)
                comments = mention.get('reddit_comments', 0)
                has_image = mention.get('has_image', False)
                local_image_path = mention.get('local_image_path', '')
                
                summary_parts.append(f"{i}. r/{channel}: {title}")
                summary_parts.append(f"   Upvotes: {score}, Comments: {comments}")
                
                # Analyze image sentiment if available
                if has_image and local_image_path:
                    # Add small delay to avoid rate limiting
                    if image_sentiments:  # Only delay after first image
                        time.sleep(1)
                    
                    image_sentiment = self.analyze_image_sentiment(local_image_path, company, title)
                    if image_sentiment:
                        image_sentiments.append(image_sentiment)
                        summary_parts.append(f"   📷 Image sentiment: {image_sentiment}/10")
                    else:
                        summary_parts.append(f"   📷 Image present (analysis failed)")
                
                summary_parts.append("")
        
        # Add image sentiment summary
        if image_sentiments:
            avg_image_sentiment = sum(image_sentiments) / len(image_sentiments)
            summary_parts.append(f"IMAGE SENTIMENT ANALYSIS:")
            summary_parts.append(f"Total images analyzed: {len(image_sentiments)}")
            summary_parts.append(f"Average image sentiment: {avg_image_sentiment:.1f}/10")
            summary_parts.append("")
        
        # Create comprehensive prompt
        prompt = f"""Analyze ALL the following data about {company} stock and provide ONE overall sentiment score from 1 to 10.
(1 = very negative/bearish, 10 = very positive/bullish)

Consider ALL sources: news articles, Reddit text discussions, and Reddit image sentiment analysis.
Weight each source appropriately:
- News articles: Official/professional sentiment
- Reddit text: Social/community sentiment  
- Reddit images: Visual sentiment (memes, charts, reactions)

Provide ONLY the final aggregate score as a number.

COMBINED DATA ANALYSIS FOR {company}:

{chr(10).join(summary_parts)}

Based on ALL the above sources (news + Reddit text + Reddit images), what is the overall sentiment score for {company} stock?

Overall {company} Combined Sentiment Score (1-10):"""
        
        logger.info(f"Analyzing combined sentiment for {company}...")
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,  # Uses Grok or OpenAI based on config
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a financial sentiment analyzer. Analyze ALL provided data sources (news + Reddit) and respond only with ONE aggregate sentiment number from 1 to 10 for {company} stock."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=15,
                    temperature=0.3
                )
                
                # Add better error checking
                if not response or not hasattr(response, 'choices') or not response.choices:
                    logger.error(f"Invalid API response: {response}")
                    continue
                    
                if not response.choices[0] or not response.choices[0].message:
                    logger.error(f"Invalid choice in response: {response.choices[0]}")
                    continue
                    
                content = response.choices[0].message.content
                if not content:
                    logger.error(f"No content in response message")
                    continue
                    
                content = content.strip()
                
                # Extract score
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', content)
                if numbers:
                    score = float(numbers[0])
                    if 1 <= score <= 10:
                        logger.info(f"Combined sentiment score for {company}: {score}/10")
                        return score
                    else:
                        logger.warning(f"Score {score} out of range for {company}")
                else:
                    logger.warning(f"No numeric score found in response: {content}")
                
            except Exception as e:
                logger.error(f"Error analyzing combined sentiment (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return None
        
        logger.error(f"Failed to analyze combined sentiment after {max_retries} attempts")
        return None
    
    def update_combined_sentiment_with_breakdown(self, company: str, score: float, total_count: int, 
                                               news_count: int, reddit_count: int, image_count: int = 0) -> str:
        """
        Update sentiment file with source breakdown
        """
        filename = self.get_sentiment_file_path()
        
        # Load existing data
        data = self.load_combined_sentiment_file()
        
        # Update with new score and breakdown
        timestamp = datetime.now().isoformat()
        
        data["last_updated"] = timestamp
        data["companies"][company] = {
            "score": score,
            "timestamp": timestamp,
            "articles_analyzed": total_count,
            "source_breakdown": {
                "news_articles": news_count,
                "reddit_mentions": reddit_count,
                "reddit_images": image_count
            },
            "last_analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "last_analysis_time": datetime.now().strftime("%H:%M:%S"),
            "source": "Combined (NewsAPI + Reddit)"
        }
        
        # Save updated data
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(filename)
            logger.info(f"Updated combined sentiment file: {filename} ({file_size} bytes)")
            logger.info(f"{company} combined score: {score}/10 (News: {news_count}, Reddit: {reddit_count}, Images: {image_count})")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving combined sentiment file: {e}")
            return ""
    
    def print_combined_summary(self, company: str, score: float, total_count: int, 
                              news_count: int, reddit_count: int):
        """
        Print summary for combined analysis
        """
        print(f"\n{'='*60}")
        print(f"COMBINED SENTIMENT ANALYSIS - {company}")
        print(f"{'='*60}")
        print(f"Company: {company}")
        print(f"Total Data Points: {total_count}")
        print(f"  📰 NewsAPI Articles: {news_count}")
        print(f"  🔴 Reddit Mentions: {reddit_count}")
        print(f"Combined Sentiment Score: {score:.1f}/10")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Interpretation
        if score >= 8:
            interpretation = "Very Bullish 🚀"
        elif score >= 6:
            interpretation = "Bullish 📈"
        elif score >= 4:
            interpretation = "Neutral ➡️"
        elif score >= 2:
            interpretation = "Bearish 📉"
        else:
            interpretation = "Very Bearish 💥"
        
        print(f"Market Sentiment: {interpretation}")
        print(f"Data Sources: NewsAPI + Reddit (3 channels)")
        print(f"{'='*60}")
    
    def analyze_batch_sentiment(self, articles: List[Dict], delay: float = 1.0) -> List[Dict]:
        """
        Analyze sentiment for a batch of articles
        
        Args:
            articles: List of article dictionaries
            delay: Delay between API calls to respect rate limits
            
        Returns:
            List of articles with added sentiment scores
        """
        logger.info(f"Starting sentiment analysis for {len(articles)} articles...")
        start_time = datetime.now()
        
        analyzed_articles = []
        successful_analyses = 0
        failed_analyses = 0
        
        for i, article in enumerate(articles, 1):
            logger.info(f"Analyzing article {i}/{len(articles)}")
            
            # Analyze sentiment
            sentiment_score = self.analyze_article_sentiment(article)
            
            # Create enhanced article with sentiment data
            enhanced_article = article.copy()
            enhanced_article.update({
                'sentiment_score': sentiment_score,
                'sentiment_analyzed_at': datetime.now().isoformat(),
                'sentiment_analysis_success': sentiment_score is not None
            })
            
            analyzed_articles.append(enhanced_article)
            
            if sentiment_score is not None:
                successful_analyses += 1
            else:
                failed_analyses += 1
            
            # Rate limiting delay
            if i < len(articles):  # Don't delay after last article
                time.sleep(delay)
        
        elapsed_time = datetime.now() - start_time
        
        logger.info(f"Batch analysis completed in {elapsed_time.total_seconds():.2f} seconds")
        logger.info(f"Successful analyses: {successful_analyses}")
        logger.info(f"Failed analyses: {failed_analyses}")
        logger.info(f"Success rate: {(successful_analyses/len(articles)*100):.1f}%")
        
        return analyzed_articles
    
    def calculate_overall_sentiment(self, analyzed_articles: List[Dict]) -> Dict:
        """
        Calculate overall sentiment statistics from analyzed articles
        
        Args:
            analyzed_articles: List of articles with sentiment scores
            
        Returns:
            Dictionary with sentiment statistics
        """
        scores = [article['sentiment_score'] for article in analyzed_articles 
                 if article.get('sentiment_score') is not None]
        
        if not scores:
            return {
                'average_sentiment': None,
                'median_sentiment': None,
                'sentiment_range': None,
                'total_articles': len(analyzed_articles),
                'analyzed_articles': 0
            }
        
        import statistics
        
        stats = {
            'average_sentiment': statistics.mean(scores),
            'median_sentiment': statistics.median(scores),
            'sentiment_range': {'min': min(scores), 'max': max(scores)},
            'total_articles': len(analyzed_articles),
            'analyzed_articles': len(scores),
            'sentiment_distribution': {
                'very_negative': len([s for s in scores if s <= 2]),
                'negative': len([s for s in scores if 2 < s <= 4]),
                'neutral': len([s for s in scores if 4 < s <= 6]),
                'positive': len([s for s in scores if 6 < s <= 8]),
                'very_positive': len([s for s in scores if s > 8])
            }
        }
        
        return stats
    
    def load_combined_sentiment_file(self) -> Dict:
        """
        Load existing combined sentiment analysis file
        
        Returns:
            Dictionary with existing sentiment data or empty structure
        """
        filename = self.get_sentiment_file_path()
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded existing combined sentiment file: {filename}")
            return data
        except FileNotFoundError:
            logger.info(f"Creating new combined sentiment file: {filename}")
            return {
                "last_updated": None,
                "companies": {}
            }
        except Exception as e:
            logger.error(f"Error loading combined sentiment file: {e}")
            return {
                "last_updated": None,
                "companies": {}
            }
    
    def update_combined_sentiment(self, company: str, score: float, article_count: int) -> str:
        """
        Update the combined sentiment analysis file with new company score
        
        Args:
            company: Company name (NVIDIA, APPLE, etc.)
            score: Aggregate sentiment score (1-10)
            article_count: Number of articles analyzed
            
        Returns:
            Path to updated file
        """
        filename = self.get_sentiment_file_path()
        
        # Load existing data
        data = self.load_combined_sentiment_file()
        
        # Update with new score
        timestamp = datetime.now().isoformat()
        
        data["last_updated"] = timestamp
        data["companies"][company] = {
            "score": score,
            "timestamp": timestamp,
            "articles_analyzed": article_count,
            "last_analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "last_analysis_time": datetime.now().strftime("%H:%M:%S")
        }
        
        # Save updated data
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(filename)
            logger.info(f"Updated combined sentiment file: {filename} ({file_size} bytes)")
            logger.info(f"{company} score: {score}/10 (based on {article_count} articles)")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving combined sentiment file: {e}")
            return ""
    
    def save_sentiment_results(self, analyzed_articles: List[Dict], original_metadata: Dict, 
                              company: str = None) -> str:
        """
        Save sentiment analysis results to JSON file
        
        Args:
            analyzed_articles: List of articles with sentiment scores
            original_metadata: Metadata from original news dump
            company: Company name for filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if company:
            filename = f"{company}_sentiment_analysis_{timestamp}.json"
        else:
            filename = f"sentiment_analysis_{timestamp}.json"
        
        # Calculate overall sentiment statistics
        sentiment_stats = self.calculate_overall_sentiment(analyzed_articles)
        
        # Prepare data structure
        data = {
            'analysis_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                'analysis_time': datetime.now().strftime("%H:%M:%S"),
                'analyzer_version': '1.0',
                'api_model': 'gpt-3.5-turbo',
                'total_articles': len(analyzed_articles),
                'successful_analyses': len([a for a in analyzed_articles if a.get('sentiment_score') is not None])
            },
            'original_metadata': original_metadata,
            'sentiment_statistics': sentiment_stats,
            'analyzed_articles': analyzed_articles
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(filename)
            logger.info(f"Saved sentiment analysis results to {filename} ({file_size} bytes)")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving sentiment results: {e}")
            return ""
    
    def print_sentiment_summary(self, analyzed_articles: List[Dict], company: str = None):
        """
        Print a formatted summary of sentiment analysis results
        
        Args:
            analyzed_articles: List of articles with sentiment scores
            company: Company name for header
        """
        stats = self.calculate_overall_sentiment(analyzed_articles)
        
        print(f"\n{'='*60}")
        print(f"SENTIMENT ANALYSIS SUMMARY{' - ' + company.upper() if company else ''}")
        print(f"{'='*60}")
        
        print(f"Total Articles: {stats['total_articles']}")
        print(f"Successfully Analyzed: {stats['analyzed_articles']}")
        
        if stats['average_sentiment'] is not None:
            print(f"Average Sentiment: {stats['average_sentiment']:.2f}/10")
            print(f"Median Sentiment: {stats['median_sentiment']:.2f}/10")
            print(f"Sentiment Range: {stats['sentiment_range']['min']:.1f} - {stats['sentiment_range']['max']:.1f}")
            
            print(f"\nSentiment Distribution:")
            dist = stats['sentiment_distribution']
            print(f"  Very Negative (1-2): {dist['very_negative']}")
            print(f"  Negative (2-4): {dist['negative']}")
            print(f"  Neutral (4-6): {dist['neutral']}")
            print(f"  Positive (6-8): {dist['positive']}")
            print(f"  Very Positive (8-10): {dist['very_positive']}")
        else:
            print("No sentiment scores available")
        
        # Show sample articles with scores
        scored_articles = [a for a in analyzed_articles if a.get('sentiment_score') is not None]
        if scored_articles:
            print(f"\nSample Articles with Scores (showing first 3):")
            print(f"{'-'*60}")
            
            for i, article in enumerate(scored_articles[:3], 1):
                print(f"\n{i}. Score: {article['sentiment_score']:.1f}/10")
                print(f"   Title: {article['title']}")
                print(f"   Source: {article.get('source_name', 'Unknown')}")
    
    def print_aggregate_summary(self, company: str, score: float, article_count: int):
        """
        Print a formatted summary of aggregate sentiment analysis
        
        Args:
            company: Company name
            score: Aggregate sentiment score (1-10)
            article_count: Number of articles analyzed
        """
        print(f"\n{'='*60}")
        print(f"AGGREGATE SENTIMENT ANALYSIS - {company}")
        print(f"{'='*60}")
        print(f"Company: {company}")
        print(f"Articles Analyzed: {article_count}")
        print(f"Aggregate Sentiment Score: {score:.1f}/10")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Interpretation
        if score >= 8:
            interpretation = "Very Bullish"
        elif score >= 6:
            interpretation = "Bullish"
        elif score >= 4:
            interpretation = "Neutral"
        elif score >= 2:
            interpretation = "Bearish"
        else:
            interpretation = "Very Bearish"
        
        print(f"Market Sentiment: {interpretation}")
        print(f"{'='*60}")
    
    def analyze_news_dump_file(self, filename: str, save_results: bool = True, 
                              print_summary: bool = True) -> Dict:
        """
        Main method to analyze aggregate sentiment for a news dump file
        Includes Reddit data for combined analysis
        
        Args:
            filename: Path to news dump JSON file
            save_results: Whether to save results to combined file
            print_summary: Whether to print summary
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting COMBINED sentiment analysis for file: {filename}")
        
        # Load news dump
        dump_data = self.load_news_dump(filename)
        if not dump_data:
            logger.error("Failed to load dump data")
            return {'success': False, 'score': None, 'company': '', 'filename': ''}
            
        news_articles = dump_data.get('articles', [])
        metadata = dump_data.get('metadata', {})
        
        if not news_articles:
            logger.error("No articles found in dump file")
            return {'success': False, 'score': None, 'company': '', 'filename': ''}
        
        # Handle combined data from multiple companies
        # Check if this is a multi-company dump by examining metadata or article companies
        is_multi_company = (
            len(metadata.get('companies', [])) > 1 or
            len(set(article.get('company') for article in news_articles if article.get('company'))) > 1
        )
        
        if is_multi_company:
            # This is a combined dump file - analyze each company separately
            logger.info(f"📊 Found multi-company dump with {len(news_articles)} total data points")
            
            results = {}
            
            # Get all companies from YAML configuration
            for company_name in self.company_keywords.keys():
                # Group data by company
                company_data = [item for item in news_articles if item.get('company') == company_name or company_name.lower() in item.get('title', '').lower()]
                
                logger.info(f"   - {company_name} data points: {len(company_data)}")
                
                if company_data:
                    logger.info(f"Analyzing {company_name} sentiment...")
                    try:
                        company_score = self.analyze_combined_data_sentiment(company_data, company_name)
                        if company_score:
                            company_news = [d for d in company_data if d.get('data_source', 'NewsAPI') != 'Reddit']
                            company_reddit = [d for d in company_data if d.get('data_source') == 'Reddit']
                            company_images = [d for d in company_reddit if d.get('has_image', False)]
                            
                            if save_results:
                                self.update_combined_sentiment_with_breakdown(
                                    company_name, company_score, len(company_data), len(company_news), len(company_reddit), len(company_images)
                                )
                            
                            if print_summary:
                                self.print_combined_summary(company_name, company_score, len(company_data), len(company_news), len(company_reddit))
                            
                            results[company_name] = {
                                'score': company_score,
                                'total_data': len(company_data),
                                'news_articles': len(company_news),
                                'reddit_mentions': len(company_reddit)
                            }
                    except Exception as e:
                        logger.error(f"Error analyzing {company_name} sentiment: {e}")
                        import traceback
                        traceback.print_exc()
            
            return {
                'success': True,
                'companies': results,
                'filename': self.get_sentiment_file_path(),
                'metadata': metadata
            }
        
        else:
            # Single company analysis (original logic)
            # FIX (2025-11-17): Check 'companies' (plural) list first, then fallback to 'company' (singular)
            company = metadata.get('company', '').upper()
            if not company:
                # Try to get from 'companies' list (new format from news_dump.py)
                companies_list = metadata.get('companies', [])
                if companies_list:
                    company = companies_list[0].upper()
                # Fallback: Try to detect from filename
                elif 'nvidia' in filename.lower():
                    company = 'NVIDIA'
                elif 'apple' in filename.lower():
                    company = 'APPLE'
                else:
                    # Last resort: check first article's company field
                    if news_articles and news_articles[0].get('company'):
                        company = news_articles[0].get('company').upper()
                    else:
                        company = 'UNKNOWN'
            
            logger.info(f"📊 Found {len(news_articles)} NewsAPI articles for {company}")
            
            # Collect Reddit data
            reddit_data = self.collect_reddit_data(company)
            logger.info(f"📊 Found {len(reddit_data)} Reddit mentions for {company}")
            
            # Combine all data sources
            all_data = news_articles + reddit_data
            total_data_points = len(all_data)
            
            if not all_data:
                logger.error("No data found from any source")
                return {'success': False, 'score': None, 'company': company, 'filename': ''}
            
            logger.info(f"📊 Total combined data points: {total_data_points}")
            logger.info(f"   - NewsAPI: {len(news_articles)} articles")
            logger.info(f"   - Reddit: {len(reddit_data)} mentions")
            
            # Analyze combined sentiment
            aggregate_score = self.analyze_combined_data_sentiment(all_data, company)
            
            if aggregate_score is None:
                logger.error(f"Failed to analyze combined sentiment for {company}")
                return {'success': False, 'score': None, 'company': company, 'filename': ''}
            
            # Count images in Reddit data
            reddit_images = [d for d in reddit_data if d.get('has_image', False)]
            
            # Update combined sentiment file with breakdown info
            result_filename = ""
            if save_results:
                result_filename = self.update_combined_sentiment_with_breakdown(
                    company, aggregate_score, total_data_points, len(news_articles), len(reddit_data), len(reddit_images)
                )
            
            # Print summary
            if print_summary:
                self.print_combined_summary(company, aggregate_score, total_data_points, len(news_articles), len(reddit_data))
            
            return {
                'success': True,
                'score': aggregate_score,
                'company': company,
                'total_data_points': total_data_points,
                'news_articles': len(news_articles),
                'reddit_mentions': len(reddit_data),
                'filename': result_filename,
                'metadata': metadata
            }

def main():
    """
    Main function to run sentiment analysis
    """
    import sys
    
    logger.info("Starting Sentiment Analyzer application...")
    
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analyzer.py <news_dump_file.json>")
        print("Example: python sentiment_analyzer.py nvidia_news_dump_20241212_143022.json")
        return
    
    filename = sys.argv[1]
    
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Analyze the file
        result = analyzer.analyze_news_dump_file(filename)
        
        if result['success']:
            if 'companies' in result:
                # Combined file with multiple companies
                print(f"\n{'='*60}")
                print("COMBINED ANALYSIS COMPLETE")
                print(f"{'='*60}")
                for company, data in result['companies'].items():
                    print(f"{company}: {data['score']:.1f}/10")
                    print(f"  Total Data: {data['total_data']}")
                    print(f"  📰 News: {data['news_articles']}")
                    print(f"  🔴 Reddit: {data['reddit_mentions']}")
                print(f"Results saved to: {result['filename']}")
                print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # Single company file
                print(f"\n{'='*60}")
                print("COMBINED ANALYSIS COMPLETE")
                print(f"{'='*60}")
                print(f"Company: {result['company']}")
                print(f"Combined Score: {result['score']:.1f}/10")
                print(f"Total Data Points: {result['total_data_points']}")
                print(f"  📰 NewsAPI Articles: {result['news_articles']}")
                print(f"  🔴 Reddit Mentions: {result['reddit_mentions']}")
                if result['filename']:
                    print(f"Results saved to: {result['filename']}")
                print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("Combined analysis failed. Check logs for details.")
            
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user (Ctrl+C)")
        print("\nAnalysis stopped by user")
    except Exception as e:
        logger.error(f"Critical error in sentiment analysis: {e}")
        logger.debug("Stack trace:", exc_info=True)
        print(f"Error running sentiment analyzer: {e}")

if __name__ == "__main__":
    main()