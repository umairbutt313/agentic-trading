#!/usr/bin/env python3
"""
Weighted Sentiment Score Aggregation System

This script combines news sentiment and image sentiment scores using configurable weights
from the companies.yaml configuration file.

AUTOMATIC DATA GENERATION
=========================
This script now automatically ensures fresh sentiment data before running!

When you run this script, it will:
1. Check if sentiment data files exist and are fresh (< 24 hours old)
2. If data is missing or stale, it will automatically:
   - Run news data collection (news_dump.py)
   - Run news sentiment analysis (sentiment_analyzer.py)
   - Run image sentiment analysis (image_sentiment_analyzer.py)
3. Then proceed with weighted aggregation

Usage:
    python3 weighted_sentiment_aggregator.py                # Interactive mode (prompts for refresh)
    python3 weighted_sentiment_aggregator.py --force-refresh # Force refresh without prompting
    python3 weighted_sentiment_aggregator.py --quiet        # Use existing data if available

The script handles the entire workflow automatically - no manual data generation required!
"""

import json
import yaml
import sys
import subprocess
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)


class WeightedSentimentAggregator:
    def __init__(self):
        self.companies_config = {}
        self.news_scores = {}
        self.image_scores = {}
        self.final_scores = {}
        self.news_dir = "news"
        
        # Configuration from environment variables
        self.data_freshness_threshold = int(os.getenv('DATA_FRESHNESS_THRESHOLD', '24'))  # hours
        self.max_articles = int(os.getenv('MAX_ARTICLES_PER_COMPANY', '100'))
        self.max_reddit_posts = int(os.getenv('MAX_REDDIT_POSTS_PER_COMPANY', '50'))
        self.output_dir = os.getenv('OUTPUT_DIR', 'Output')
        self.log_dir = os.getenv('LOG_DIR', 'logs')
        
        # API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    
    def check_data_freshness(self):
        """Check if sentiment data files exist and are recent."""
        # Check if we're in the news directory or project root
        if os.path.exists("news"):
            # We're in project root
            news_file = os.path.join("container_output", "final_score", "news-sentiment-analysis.json")
            image_file = os.path.join("container_output", "final_score", "image-sentiment-analysis.json")
        else:
            # We're in news directory
            news_file = os.path.join("..", "container_output", "final_score", "news-sentiment-analysis.json")
            image_file = os.path.join("..", "container_output", "final_score", "image-sentiment-analysis.json")
        
        news_exists = os.path.exists(news_file)
        image_exists = os.path.exists(image_file)
        
        # Check if files are less than 24 hours old
        now = datetime.now()
        news_fresh = False
        image_fresh = False
        
        if news_exists:
            news_time = datetime.fromtimestamp(os.path.getmtime(news_file))
            news_fresh = (now - news_time) < timedelta(hours=24)
            
        if image_exists:
            image_time = datetime.fromtimestamp(os.path.getmtime(image_file))
            image_fresh = (now - image_time) < timedelta(hours=24)
        
        return {
            'news_exists': news_exists,
            'image_exists': image_exists,
            'news_fresh': news_fresh,
            'image_fresh': image_fresh,
            'news_file': news_file,
            'image_file': image_file
        }
    
    def run_news_collection(self):
        """Run news data collection script."""
        print("🔄 Running news data collection...")
        try:
            result = subprocess.run(
                [sys.executable, os.path.join("news", "news_dump.py")], 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Go to project root
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                print("✅ News data collection completed successfully")
                return True
            else:
                print(f"❌ News data collection failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ News data collection timed out")
            return False
        except Exception as e:
            print(f"❌ Error running news collection: {e}")
            return False
    
    def find_latest_news_dump(self):
        """Find the most recent news dump file."""
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Try multiple patterns for news dump files
        patterns = [
            os.path.join(project_root, "container_output", "news", "raw-news-images_*.json"),
            os.path.join(project_root, "container_output", "news", "raw-news_*.json"),
            os.path.join(project_root, "container_output", "news", "*news*.json")
        ]
        
        dump_files = []
        for pattern in patterns:
            dump_files.extend(glob.glob(pattern))
        
        if not dump_files:
            return None
        # Sort by modification time, newest first
        latest_file = max(dump_files, key=os.path.getmtime)
        return latest_file
    
    def run_sentiment_analysis(self):
        """Run news sentiment analysis script."""
        print("🔄 Running news sentiment analysis...")
        
        # Find latest news dump file
        latest_dump = self.find_latest_news_dump()
        if not latest_dump:
            print("❌ No news dump file found. Run news collection first.")
            return False
        
        # Get relative path from news directory (we're already in news directory)
        dump_filename = os.path.basename(latest_dump)
        dump_path = os.path.join("container_output", "news", dump_filename)
        
        try:
            result = subprocess.run(
                [sys.executable, os.path.join("news", "sentiment_analyzer.py"), dump_path], 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Go to project root
                capture_output=True, 
                text=True,
                timeout=600  # 10 minute timeout
            )
            if result.returncode == 0:
                print("✅ News sentiment analysis completed successfully")
                return True
            else:
                print(f"❌ News sentiment analysis failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ News sentiment analysis timed out")
            return False
        except Exception as e:
            print(f"❌ Error running sentiment analysis: {e}")
            return False
    
    def run_image_sentiment_analysis(self):
        """Run image sentiment analysis script."""
        print("🔄 Running image sentiment analysis...")
        try:
            result = subprocess.run(
                [sys.executable, os.path.join("news", "image_sentiment_analyzer.py")], 
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Go to project root
                capture_output=True, 
                text=True,
                timeout=600  # 10 minute timeout
            )
            if result.returncode == 0:
                print("✅ Image sentiment analysis completed successfully")
                return True
            else:
                print(f"❌ Image sentiment analysis failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Image sentiment analysis timed out")
            return False
        except Exception as e:
            print(f"❌ Error running image sentiment analysis: {e}")
            return False
    
    def ensure_fresh_data(self, force_refresh=False, quiet=False):
        """Ensure we have fresh sentiment data, generating it if needed."""
        print("🔍 Checking sentiment data freshness...")
        
        data_status = self.check_data_freshness()
        
        # Display current status
        if data_status['news_exists']:
            news_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(data_status['news_file']))
            print(f"📰 News sentiment file: {'✅ Fresh' if data_status['news_fresh'] else '⚠️ Stale'} ({news_age})")
        else:
            print("📰 News sentiment file: ❌ Missing")
            
        if data_status['image_exists']:
            image_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(data_status['image_file']))
            print(f"🖼️ Image sentiment file: {'✅ Fresh' if data_status['image_fresh'] else '⚠️ Stale'} ({image_age})")
        else:
            print("🖼️ Image sentiment file: ❌ Missing")
        
        # In quiet mode, use existing data regardless of freshness
        # OLD: Required both news AND image (but image sentiment is disabled)
        # if quiet:
        #     if data_status['news_exists'] and data_status['image_exists']:
        #         print("✅ Using existing sentiment data (quiet mode)")
        #         return True
        #     else:
        #         print("❌ Required data files missing, cannot proceed in quiet mode")
        #         return False

        # NEW: Only require news sentiment (image sentiment is disabled in system)
        if quiet:
            if data_status['news_exists']:
                print("✅ Using existing news sentiment data (quiet mode)")
                if not data_status['image_exists']:
                    print("ℹ️  Image sentiment disabled - using news-only mode")
                return True
            else:
                print("❌ News sentiment file missing, cannot proceed in quiet mode")
                return False
        
        # Determine what needs to be refreshed
        need_news_collection = force_refresh or not data_status['news_fresh']
        need_image_analysis = force_refresh or not data_status['image_fresh']
        
        if not need_news_collection and not need_image_analysis:
            print("✅ All sentiment data is fresh and ready!")
            return True
        
        # Ask user for confirmation unless forced
        if not force_refresh:
            print("\n🤔 Some sentiment data is missing or stale.")
            response = input("Would you like to generate fresh sentiment data? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("⚠️ Proceeding with existing data (may be stale or incomplete)")
                return True
        
        print("\n🚀 Generating fresh sentiment data...")
        print("=" * 60)
        
        # Step 1: News collection (always needed for fresh sentiment analysis)
        if need_news_collection:
            if not self.run_news_collection():
                print("❌ Failed to collect news data")
                return False
            
            # Step 2: News sentiment analysis
            if not self.run_sentiment_analysis():
                print("❌ Failed to analyze news sentiment")
                return False
        
        # Step 3: Image sentiment analysis
        if need_image_analysis:
            if not self.run_image_sentiment_analysis():
                print("❌ Failed to analyze image sentiment")
                return False
        
        print("\n✅ All sentiment data has been refreshed!")
        return True
        
    def load_companies_config(self, config_path=None):
        """Load companies configuration from YAML file."""
        if config_path is None:
            config_path = "companies.yaml" if os.path.exists("news") else "../companies.yaml"
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.companies_config = config.get('companies', {})
                print(f"✓ Loaded configuration for {len(self.companies_config)} companies")
                return True
        except FileNotFoundError:
            print(f"❌ Error: Configuration file '{config_path}' not found")
            return False
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML file: {e}")
            return False
    
    def load_news_sentiment(self, news_path=None):
        """Load news sentiment scores."""
        if news_path is None:
            news_path = "container_output/final_score/news-sentiment-analysis.json" if os.path.exists("news") else "../container_output/final_score/news-sentiment-analysis.json"
        try:
            with open(news_path, 'r') as file:
                data = json.load(file)
                self.news_scores = data.get('companies', {})
                print(f"✓ Loaded news sentiment for {len(self.news_scores)} companies")
                return True
        except FileNotFoundError:
            print(f"❌ Error: News sentiment file '{news_path}' not found")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing news sentiment JSON: {e}")
            return False
    
    def load_image_sentiment(self, image_path=None):
        """Load image sentiment scores."""
        if image_path is None:
            image_path = "container_output/final_score/image-sentiment-analysis.json" if os.path.exists("news") else "../container_output/final_score/image-sentiment-analysis.json"
        try:
            with open(image_path, 'r') as file:
                data = json.load(file)
                company_results = data.get('company_results', {})
                # Extract average sentiment scores
                self.image_scores = {
                    company: results.get('average_sentiment', 0)
                    for company, results in company_results.items()
                }
                print(f"✓ Loaded image sentiment for {len(self.image_scores)} companies")
                return True
        except FileNotFoundError:
            print(f"⚠️  Warning: Image sentiment file '{image_path}' not found - will use news scores only")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing image sentiment JSON: {e}")
            return False
    
    def calculate_weighted_scores(self):
        """Calculate weighted sentiment scores for all companies."""
        print("\n📊 Calculating weighted sentiment scores...")
        print("=" * 80)
        
        for company_key, config in self.companies_config.items():
            company_name = config.get('name', company_key.upper())
            news_weight = config.get('news_weight', 0.4)
            chart_weight = config.get('chart_weight', 0.6)
            
            # Validate weights
            if abs(news_weight + chart_weight - 1.0) > 0.001:
                print(f"⚠️  Warning: Weights for {company_name} don't sum to 1.0 ({news_weight + chart_weight})")
            
            # Get news score
            news_score = self.news_scores.get(company_name, {}).get('score', None)
            if news_score is None:
                print(f"❌ No news sentiment data for {company_name} - skipping")
                continue
            
            # Get image score (with fallback)
            image_score = self.image_scores.get(company_name, None)
            
            if image_score is not None:
                # Both scores available - calculate weighted average
                final_score = (news_weight * news_score) + (chart_weight * image_score)
                calculation_method = "Weighted Average"
                details = f"({news_weight} × {news_score}) + ({chart_weight} × {image_score})"
            else:
                # Only news score available - use it as fallback
                final_score = news_score
                calculation_method = "News Only (No Image Data)"
                details = f"Using news score only: {news_score}"
                print(f"⚠️  No image sentiment for {company_name} - using news score only")
            
            # Store results
            self.final_scores[company_name] = {
                'final_score': round(final_score, 2),
                'news_score': news_score,
                'image_score': image_score,
                'news_weight': news_weight,
                'chart_weight': chart_weight,
                'calculation_method': calculation_method,
                'calculation_details': details,
                'timestamp': datetime.now().isoformat()
            }
            
            # Display results
            print(f"\n🏢 {company_name}")
            print(f"   News Score: {news_score}")
            print(f"   Image Score: {image_score if image_score else 'N/A'}")
            print(f"   Weights: {news_weight} (news) + {chart_weight} (chart)")
            print(f"   Final Score: {round(final_score, 2)} ({calculation_method})")
            print(f"   Calculation: {details}")
    
    def save_results(self, output_path=None):
        """Save final weighted scores to JSON file."""
        if output_path is None:
            output_path = "container_output/final_score/final-weighted-scores.json" if os.path.exists("news") else "../container_output/final_score/final-weighted-scores.json"
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            output_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_companies': len(self.final_scores),
                    'companies_with_image_data': len([
                        c for c in self.final_scores.values() 
                        if c['image_score'] is not None
                    ]),
                    'calculation_method': 'weighted_average',
                    'default_weights': {
                        'news_weight': 0.4,
                        'chart_weight': 0.6
                    }
                },
                'companies': self.final_scores
            }
            
            with open(output_path, 'w') as file:
                json.dump(output_data, file, indent=2)
            
            print(f"\n💾 Results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def display_summary(self):
        """Display summary statistics."""
        if not self.final_scores:
            print("❌ No results to display")
            return
        
        print("\n📈 SUMMARY")
        print("=" * 80)
        
        scores = [data['final_score'] for data in self.final_scores.values()]
        companies_with_images = [
            name for name, data in self.final_scores.items() 
            if data['image_score'] is not None
        ]
        
        print(f"Total Companies Analyzed: {len(self.final_scores)}")
        print(f"Companies with Image Data: {len(companies_with_images)}")
        print(f"Average Final Score: {sum(scores) / len(scores):.2f}")
        print(f"Highest Score: {max(scores):.2f}")
        print(f"Lowest Score: {min(scores):.2f}")
        
        if companies_with_images:
            print(f"Companies with Image Data: {', '.join(companies_with_images)}")
    
    def run(self, force_refresh=False, quiet=False):
        """Main execution method."""
        print("🎯 Weighted Sentiment Score Aggregation System")
        print("=" * 80)
        
        # Ensure we have fresh sentiment data
        if not self.ensure_fresh_data(force_refresh=force_refresh, quiet=quiet):
            print("❌ Failed to ensure fresh sentiment data")
            return False
        
        print("\n📊 Loading sentiment data...")
        print("=" * 80)
        
        # Load all data
        success = True
        success &= self.load_companies_config()
        success &= self.load_news_sentiment()
        self.load_image_sentiment()  # Image data is optional
        
        if not success:
            print("❌ Failed to load required data files")
            return False
        
        # Calculate scores
        self.calculate_weighted_scores()
        
        # Save and display results
        self.save_results()
        self.display_summary()
        
        print("\n✅ Processing complete!")
        return True


def main():
    """Entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Weighted Sentiment Score Aggregation System')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh of all sentiment data without prompting')
    parser.add_argument('--quiet', action='store_true',
                       help='Skip user prompts and use existing data if available')
    
    args = parser.parse_args()
    
    aggregator = WeightedSentimentAggregator()
    success = aggregator.run(force_refresh=args.force_refresh, quiet=args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()