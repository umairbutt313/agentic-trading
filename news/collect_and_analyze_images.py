#!/usr/bin/env python3
"""
Image Collection and Analysis Orchestration Script
Collects TradingView screenshots and analyzes all images for sentiment.
"""

import os
import sys
import logging
from datetime import datetime

# Add utils directory to path to import headless_charts
sys.path.insert(0, '../utils')
sys.path.insert(0, '.')

from headless_charts import screenshot_tradingview_all_companies
from image_sentiment_analyzer import ImageSentimentAnalyzer

# Configure logging
# Ensure logs directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'image_collection_analysis.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main orchestration function:
    1. Take TradingView screenshots for all companies
    2. Analyze sentiment for all images in ../Output/images folder
    3. Save results to ../container_output/final_score/image-sentiment-analysis.json
    """
    try:
        print("🚀 Starting Image Collection and Sentiment Analysis Pipeline")
        print("="*70)
        
        # Step 1: Collect TradingView screenshots
        print("\n📸 Step 1: Collecting TradingView Screenshots")
        print("-" * 50)
        
        screenshots = screenshot_tradingview_all_companies()
        
        if screenshots:
            print(f"✓ Successfully captured {len(screenshots)} TradingView screenshots")
        else:
            print("⚠️ No TradingView screenshots were captured")
            logger.warning("TradingView screenshot collection failed")
        
        # Step 2: Analyze all images for sentiment
        print("\n🧠 Step 2: Analyzing Image Sentiment")
        print("-" * 50)
        
        analyzer = ImageSentimentAnalyzer()
        results = analyzer.analyze_all_images_in_folder("../Output/images")
        
        if results:
            # Step 3: Save results
            print("\n💾 Step 3: Saving Results")
            print("-" * 50)
            
            analyzer.save_results(results, "news/combine-image-sentimental_analysis.json")
            
            print("\n✅ Image sentiment analysis pipeline completed successfully!")
            print(f"Results saved to: news/combine-image-sentimental_analysis.json")
            
        else:
            print("❌ Image sentiment analysis failed - no results to save")
            logger.error("Image sentiment analysis returned no results")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()