#!/usr/bin/env python3
"""
TradingView Sentiment Analyzer
Processes TradingView data dump files using OpenAI's ChatGPT API for sentiment scoring.
Follows the same pattern as sentiment_analyzer.py for consistency.
Takes dumped TradingView data and returns numerical sentiment scores (1-10).
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
        logging.FileHandler(os.path.join(logs_dir, 'tradingview_sentiment_analyzer.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

class TradingViewSentimentAnalyzer:
    def __init__(self):
        """Initialize the TradingView Sentiment Analyzer with OpenAI client"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with API key ending in: ...{self.api_key[-4:]}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def get_sentiment_file_path(self) -> str:
        """Get the consistent path for the sentiment analysis file"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        return os.path.join(project_root, "container_output", "final_score", "tradingview-sentiment-analysis.json")
    
    def load_companies_config(self, config_path: str = "companies.yaml") -> Dict:
        """Load companies configuration from YAML file"""
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
                    logger.error(f"Cannot find companies config file: {config_path}")
                    return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            return config.get('companies', {})
        except Exception as e:
            logger.error(f"Error loading companies config: {e}")
            return {}
    
    def create_sentiment_prompt(self, trading_data: Dict) -> str:
        """
        Create a detailed prompt for sentiment analysis of TradingView data
        
        Args:
            trading_data: Dictionary containing trading data for a company
            
        Returns:
            Formatted prompt string for OpenAI
        """
        
        company = trading_data.get('company', 'Unknown')
        symbol = trading_data.get('symbol', 'Unknown')
        
        # Extract key data points
        price_data = trading_data.get('price_data', {})
        technical_indicators = trading_data.get('technical_indicators', {})
        sentiment_indicators = trading_data.get('sentiment_indicators', {})
        
        prompt = f"""
Analyze the sentiment of the following TradingView trading data for {company} ({symbol}) and provide a sentiment score from 1-10:

**Price Data:**
- Current Price: {price_data.get('current_price', 'N/A')}
- Price Change: {price_data.get('price_change', 'N/A')}
- Price Change %: {price_data.get('price_change_percent', 'N/A')}
- Volume: {price_data.get('volume', 'N/A')}
- Daily High: {price_data.get('daily_high', 'N/A')}
- Daily Low: {price_data.get('daily_low', 'N/A')}

**Technical Indicators:**
- RSI: {technical_indicators.get('rsi', 'N/A')}
- MACD Signal: {technical_indicators.get('macd_signal', 'N/A')}
- 20-day MA: {technical_indicators.get('ma_20', 'N/A')}
- 50-day MA: {technical_indicators.get('ma_50', 'N/A')}
- 200-day MA: {technical_indicators.get('ma_200', 'N/A')}

**Market Sentiment Indicators:**
- Ideas Count: {sentiment_indicators.get('ideas_count', 'N/A')}
- Social Sentiment: {sentiment_indicators.get('social_sentiment', 'N/A')}
- Analyst Rating: {sentiment_indicators.get('analyst_rating', 'N/A')}

**Analysis Instructions:**
1. Consider price momentum (positive/negative price changes indicate bullish/bearish sentiment)
2. Evaluate technical indicators (RSI levels, MACD signals, moving average relationships)
3. Assess volume patterns (high volume with price increases = strong bullish sentiment)
4. Factor in social sentiment and analyst ratings
5. Consider overall market context and technical patterns

**Sentiment Scoring Scale:**
- 1-2: Very Bearish (strong sell signals, negative momentum)
- 3-4: Bearish (weak fundamentals, downward trends)
- 5-6: Neutral (mixed signals, sideways movement)
- 7-8: Bullish (positive momentum, good fundamentals)
- 9-10: Very Bullish (strong buy signals, exceptional performance)

Please respond with ONLY a single number between 1 and 10 representing the sentiment score. Do not include any explanation or additional text.
"""
        
        return prompt.strip()
    
    def analyze_trading_data_sentiment(self, trading_data: Dict, max_retries: int = 3) -> Optional[float]:
        """
        Analyze sentiment of TradingView trading data using OpenAI
        
        Args:
            trading_data: Dictionary containing trading data
            max_retries: Maximum number of API call retries
            
        Returns:
            Sentiment score (1-10) or None if analysis fails
        """
        
        company = trading_data.get('company', 'Unknown')
        symbol = trading_data.get('symbol', 'Unknown')
        
        # Skip entries with errors
        if 'error' in trading_data:
            logger.warning(f"Skipping {symbol} due to scraping error: {trading_data['error']}")
            return None
        
        prompt = self.create_sentiment_prompt(trading_data)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Analyzing sentiment for {company} ({symbol}) - Attempt {attempt + 1}")
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert financial analyst specializing in technical analysis and market sentiment. You analyze trading data and provide precise sentiment scores."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=10,
                    temperature=0.3
                )
                
                # Extract and validate the sentiment score
                content = response.choices[0].message.content.strip()
                
                # Try to extract a number from the response
                import re
                numbers = re.findall(r'\d+\.?\d*', content)
                if numbers:
                    score = float(numbers[0])
                    if 1 <= score <= 10:
                        logger.info(f"Sentiment score for {symbol}: {score}")
                        return score
                    else:
                        logger.warning(f"Score {score} out of range for {symbol}")
                
                logger.warning(f"Invalid response for {symbol}: {content}")
                
            except Exception as e:
                if "rate limit" in str(e).lower():
                    wait_time = (2 ** attempt) * 60  # Exponential backoff in minutes
                    logger.warning(f"Rate limit hit for {symbol}, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait 5 seconds before retry
                
        logger.error(f"Failed to analyze sentiment for {symbol} after {max_retries} attempts")
        return None
    
    def process_dump_file(self, dump_file_path: str) -> Dict:
        """
        Process a TradingView dump file and analyze sentiment for all companies
        
        Args:
            dump_file_path: Path to the TradingView dump JSON file
            
        Returns:
            Dictionary with sentiment analysis results
        """
        
        logger.info(f"Processing TradingView dump file: {dump_file_path}")
        
        try:
            with open(dump_file_path, 'r', encoding='utf-8') as f:
                dump_data = json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load dump file: {e}")
            raise
        
        # Extract trading data
        trading_data_list = dump_data.get('trading_data', [])
        if not trading_data_list:
            logger.warning("No trading data found in dump file")
            return {}
        
        logger.info(f"Found {len(trading_data_list)} companies in dump file")
        
        # Prepare results structure
        results = {
            "last_updated": datetime.now().isoformat(),
            "companies": {}
        }
        
        # Process each company's trading data
        for trading_data in trading_data_list:
            company = trading_data.get('company', 'Unknown')
            symbol = trading_data.get('symbol', 'Unknown')
            
            logger.info(f"Processing {company} ({symbol})...")
            
            # Analyze sentiment
            sentiment_score = self.analyze_trading_data_sentiment(trading_data)
            
            if sentiment_score is not None:
                results["companies"][company] = {
                    "score": sentiment_score,
                    "timestamp": datetime.now().isoformat(),
                    "data_points_analyzed": self._count_data_points(trading_data),
                    "sentiment_breakdown": self._create_sentiment_breakdown(trading_data, sentiment_score),
                    "analysis_source": "TradingView Data + GPT-4 Analysis",
                    "symbol": symbol,
                    "price_data": self._extract_price_data(trading_data)
                }
                logger.info(f"✓ {company}: {sentiment_score}/10")
            else:
                logger.warning(f"✗ Failed to analyze {company}")
            
            # Add delay between API calls to respect rate limits
            time.sleep(2)
        
        return results
    
    def _count_data_points(self, trading_data: Dict) -> int:
        """Count the number of available data points in trading data"""
        count = 0
        
        # Count price data points
        price_data = trading_data.get('price_data', {})
        count += sum(1 for v in price_data.values() if v != 'N/A' and v is not None)
        
        # Count technical indicators
        technical_indicators = trading_data.get('technical_indicators', {})
        count += sum(1 for v in technical_indicators.values() if v != 'N/A' and v is not None)
        
        # Count sentiment indicators
        sentiment_indicators = trading_data.get('sentiment_indicators', {})
        count += sum(1 for v in sentiment_indicators.values() if v != 'N/A' and v is not None)
        
        return count
    
    def _create_sentiment_breakdown(self, trading_data: Dict, overall_score: float) -> Dict:
        """Create a breakdown of sentiment components"""
        
        # Extract key indicators for breakdown
        price_data = trading_data.get('price_data', {})
        technical_indicators = trading_data.get('technical_indicators', {})
        sentiment_indicators = trading_data.get('sentiment_indicators', {})
        
        # Estimate component scores based on available data
        price_momentum = self._estimate_price_momentum_score(price_data)
        technical_score = self._estimate_technical_score(technical_indicators)
        volume_score = self._estimate_volume_score(price_data)
        social_score = self._estimate_social_score(sentiment_indicators)
        
        return {
            "price_momentum": price_momentum,
            "technical_indicators": technical_score,
            "volume_analysis": volume_score,
            "social_sentiment": social_score
        }
    
    def _estimate_price_momentum_score(self, price_data: Dict) -> float:
        """Estimate price momentum score from price data"""
        price_change = price_data.get('price_change', 'N/A')
        
        if price_change == 'N/A' or not isinstance(price_change, str):
            return 5.0  # Neutral if no data
        
        # Extract percentage or absolute change
        if '%' in price_change:
            try:
                change_value = float(price_change.replace('%', '').replace('+', ''))
                if change_value > 3:
                    return 9.0
                elif change_value > 1:
                    return 7.0
                elif change_value > 0:
                    return 6.0
                elif change_value > -1:
                    return 4.0
                elif change_value > -3:
                    return 3.0
                else:
                    return 1.0
            except:
                return 5.0
        
        # For absolute changes, try to determine direction
        if '+' in price_change:
            return 7.0
        elif '-' in price_change:
            return 3.0
        else:
            return 5.0
    
    def _estimate_technical_score(self, technical_indicators: Dict) -> float:
        """Estimate technical score from indicators"""
        rsi = technical_indicators.get('rsi', 'N/A')
        macd_signal = technical_indicators.get('macd_signal', 'N/A')
        
        score = 5.0  # Start neutral
        
        # RSI analysis
        if rsi != 'N/A':
            try:
                rsi_value = float(rsi)
                if rsi_value > 70:
                    score -= 1  # Overbought
                elif rsi_value < 30:
                    score += 1  # Oversold (potential buy)
                elif 40 <= rsi_value <= 60:
                    score += 0.5  # Healthy range
            except:
                pass
        
        # MACD analysis
        if macd_signal == 'bullish':
            score += 1.5
        elif macd_signal == 'bearish':
            score -= 1.5
        
        return max(1.0, min(10.0, score))
    
    def _estimate_volume_score(self, price_data: Dict) -> float:
        """Estimate volume score from price and volume data"""
        volume = price_data.get('volume', 'N/A')
        
        if volume == 'N/A':
            return 5.0
        
        # Simple volume analysis - higher volume generally positive
        if 'M' in str(volume):  # Millions
            try:
                vol_num = float(str(volume).replace('M', ''))
                if vol_num > 50:
                    return 7.0
                elif vol_num > 20:
                    return 6.0
                else:
                    return 5.0
            except:
                return 5.0
        
        return 5.0
    
    def _estimate_social_score(self, sentiment_indicators: Dict) -> float:
        """Estimate social sentiment score"""
        social_sentiment = sentiment_indicators.get('social_sentiment', 'neutral')
        ideas_count = sentiment_indicators.get('ideas_count', 0)
        
        score = 5.0
        
        if social_sentiment == 'bullish':
            score += 2.0
        elif social_sentiment == 'bearish':
            score -= 2.0
        
        # More ideas generally indicate higher interest
        if isinstance(ideas_count, int) and ideas_count > 100:
            score += 0.5
        
        return max(1.0, min(10.0, score))
    
    def _extract_price_data(self, trading_data: Dict) -> Dict:
        """Extract and clean price data from trading data"""
        price_data = trading_data.get('price_data', {})
        
        # Create clean price data dictionary with only relevant fields
        clean_price_data = {}
        
        # Copy relevant price fields
        price_fields = [
            'current_price', 'price_change', 'price_change_percent',
            'daily_high', 'daily_low', 'open', 'close', 'volume'
        ]
        
        for field in price_fields:
            value = price_data.get(field, 'N/A')
            if value != 'N/A' and value is not None:
                clean_price_data[field] = value
        
        return clean_price_data
    
    def save_sentiment_results(self, results: Dict, output_path: Optional[str] = None) -> str:
        """Save sentiment analysis results to JSON file"""
        
        if output_path is None:
            output_path = self.get_sentiment_file_path()
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Sentiment results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save sentiment results: {e}")
            raise
    
    def find_latest_dump_file(self) -> Optional[str]:
        """Find the most recent TradingView dump file"""
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "container_output", "tradingview")
        
        if not os.path.exists(output_dir):
            logger.error(f"Output directory not found: {output_dir}")
            return None
        
        # Find all TradingView dump files
        dump_files = []
        for filename in os.listdir(output_dir):
            if filename.startswith("raw-tradingview_") and filename.endswith(".json"):
                full_path = os.path.join(output_dir, filename)
                dump_files.append((full_path, os.path.getmtime(full_path)))
        
        if not dump_files:
            logger.error("No TradingView dump files found")
            return None
        
        # Return the most recent file
        latest_file = max(dump_files, key=lambda x: x[1])[0]
        logger.info(f"Found latest dump file: {latest_file}")
        return latest_file

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TradingView Sentiment Analyzer')
    parser.add_argument('dump_file', nargs='?', help='TradingView dump file to analyze (optional - will find latest if not provided)')
    parser.add_argument('--output', '-o', help='Output file path for sentiment results')
    parser.add_argument('--latest', '-l', action='store_true', help='Use the latest dump file')
    
    args = parser.parse_args()
    
    try:
        analyzer = TradingViewSentimentAnalyzer()
        
        # Determine which dump file to process
        if args.dump_file:
            dump_file_path = args.dump_file
        elif args.latest or not args.dump_file:
            dump_file_path = analyzer.find_latest_dump_file()
            if not dump_file_path:
                logger.error("No dump file found. Please run the scraper first.")
                sys.exit(1)
        else:
            logger.error("Please provide a dump file path or use --latest")
            sys.exit(1)
        
        # Process the dump file
        results = analyzer.process_dump_file(dump_file_path)
        
        if not results.get('companies'):
            logger.error("No sentiment analysis results generated")
            sys.exit(1)
        
        # Save results
        output_path = analyzer.save_sentiment_results(results, args.output)
        
        # Print summary
        companies_analyzed = len(results['companies'])
        logger.info(f"Sentiment analysis completed for {companies_analyzed} companies")
        logger.info(f"Results saved to: {output_path}")
        
        # Print scores summary
        print("\nSentiment Scores Summary:")
        print("-" * 40)
        for company, data in results['companies'].items():
            score = data['score']
            symbol = data.get('symbol', 'N/A')
            print(f"{company} ({symbol}): {score}/10")
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()