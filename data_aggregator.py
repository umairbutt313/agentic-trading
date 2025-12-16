#!/usr/bin/env python3
"""
Data Aggregator for Multi-Source Sentiment Analysis
Combines news, image, and TradingView sentiment scores with real-time price data
Generates CSV files for chart visualization
"""

import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class SentimentDataAggregator:
  def __init__(self, base_dir: str = "."):
    self.base_dir = base_dir
    self.output_dir = os.path.join(base_dir, "web")
    self.final_score_dir = os.path.join(base_dir, "container_output", "final_score")
    
    # Company symbol mapping
    self.company_symbols = {
      "NVIDIA": "NVDA", 
      "APPLE": "AAPL",
      "INTEL": "INTC",
      "MICROSOFT": "MSFT",
      "GOOGLE": "GOOGL", 
      "AMAZON": "AMZN",
      "TESLA": "TSLA"
    }
    
    # Target companies for file generation
    # OLD: Multiple companies enabled
    # self.target_companies = ["NVIDIA", "APPLE", "INTEL"]
    # NEW: Only NVIDIA enabled (matches companies.yaml configuration)
    self.target_companies = ["NVIDIA"]
    
  def load_news_sentiment(self) -> Dict:
    """Load news sentiment analysis data"""
    try:
      news_file = os.path.join(self.final_score_dir, "news-sentiment-analysis.json")
      with open(news_file, 'r') as f:
        data = json.load(f)
      print(f"✓ Loaded news sentiment data: {len(data.get('companies', {}))} companies")
      return data.get('companies', {})
    except FileNotFoundError:
      print("⚠ news-sentiment-analysis.json not found, using default scores")
      return {}
    except Exception as e:
      print(f"✗ Error loading news sentiment: {e}")
      return {}

  def load_image_sentiment(self) -> Dict:
    """Load image sentiment analysis data"""
    try:
      image_file = os.path.join(self.final_score_dir, "image-sentiment-analysis.json")
      with open(image_file, 'r') as f:
        data = json.load(f)
      
      # Extract company averages from the nested structure
      company_scores = {}
      for company, stats in data.get('company_results', {}).items():
        if company != "Unknown":
          company_scores[company] = stats.get('average_sentiment', 5.0)
      
      print(f"✓ Loaded image sentiment data: {len(company_scores)} companies")
      return company_scores
    except FileNotFoundError:
      print("⚠ image-sentiment-analysis.json not found, using default scores")
      return {}
    except Exception as e:
      print(f"✗ Error loading image sentiment: {e}")
      return {}

  def load_tradingview_sentiment(self) -> Dict:
    """Load TradingView sentiment analysis data (with fallback)"""
    # # TradingView scraping disabled - using neutral scores for all companies
    # # Try multiple possible file locations/names
    # possible_files = [
    #   os.path.join(self.final_score_dir, "tradingview-sentiment-analysis.json"),
    #   os.path.join(self.final_score_dir, "combine-tradingview-sentiment-analysis.json"), 
    #   os.path.join(self.final_score_dir, "final-tradingview.json"),
    #   os.path.join(self.base_dir, "dumps", "final-tradingview.json")
    # ]
    # 
    # for file_path in possible_files:
    #   try:
    #     with open(file_path, 'r') as f:
    #       data = json.load(f)
    #     print(f"✓ Loaded TradingView sentiment from: {os.path.basename(file_path)}")
    #     # Handle nested companies structure
    #     companies_data = data.get('companies', data)
    #     return companies_data
    #   except FileNotFoundError:
    #     continue
    #   except Exception as e:
    #     print(f"✗ Error loading {file_path}: {e}")
    #     continue
    
    print("⚠ TradingView sentiment disabled - using neutral fallback scores (5.0)")
    return {}


  def generate_time_series_data(self, company: str, num_points: int = 100) -> List[Dict]:
    """Generate time series data points using ONLY real prices - enhanced with validation"""
    symbol = self.company_symbols.get(company)
    if not symbol:
      print(f"✗ No symbol mapping for {company}")
      return []
    
    # Get sentiment scores from analysis files with validation
    news_sentiment = self._validate_sentiment_score(
        self.news_data.get(company, {}).get('score', 5.0)
    )
    # # TradingView disabled - commenting out
    # tradingview_sentiment = self._validate_sentiment_score(
    #     self.tradingview_data.get(company, {}).get('score', 5.0)
    # )
    
    data_points = []
    
    try:
      # Import and use real price storage
      from price_storage import PriceStorage
      storage = PriceStorage()
      
      # Get extended real price history (up to 24 hours for better data coverage)
      price_history = storage.get_price_history(symbol, hours_back=24)
      
      if price_history and len(price_history) > 0:
        print(f"✅ Using {len(price_history)} real price points for {symbol}")
        
        # Take the most recent points up to num_points
        recent_prices = price_history[-num_points:] if len(price_history) >= num_points else price_history
        
        # Validate and process each price point
        valid_points = 0
        for price_point in recent_prices:
          if self._validate_price_point(price_point):
            data_points.append({
              'timestamp': price_point['timestamp'],
              # 'score_tradingview': round(tradingview_sentiment, 1),  # Disabled
              'score_sa_news': round(news_sentiment, 1),
              'price': round(float(price_point['price']), 2)
            })
            valid_points += 1
        
        print(f"✅ Generated {valid_points} valid data points for {symbol}")
        
        # Return empty if we don't have enough valid data
        if valid_points < 5:  # Minimum 5 points needed
          print(f"❌ Insufficient valid data points ({valid_points}) for {symbol}")
          return []
          
      else:
        print(f"❌ No real price data found for {symbol} - cannot generate chart data")
        return []
        
    except ImportError:
      print(f"❌ price_storage.py not available for {symbol}")
      return []
      
    except Exception as e:
      print(f"❌ Error accessing real price data for {symbol}: {e}")
      return []
    
    return data_points

  def _validate_sentiment_score(self, score: float) -> float:
    """Validate sentiment score is in valid range (1-10)"""
    try:
      score = float(score)
      return max(1.0, min(10.0, score))
    except (ValueError, TypeError):
      return 5.0  # Neutral fallback
  
  def _validate_price_point(self, point: Dict) -> bool:
    """Validate a price point has required fields and valid data"""
    try:
      # Check required fields exist
      if 'timestamp' not in point or 'price' not in point:
        return False
      
      # Validate price is positive number
      price = float(point['price'])
      if price <= 0:
        return False
      
      # Validate timestamp format (basic check)
      timestamp = point['timestamp']
      if not isinstance(timestamp, str) or len(timestamp) < 10:
        return False
      
      return True
    except (ValueError, TypeError, KeyError):
      return False

  def create_csv_file(self, company: str, data_points: List[Dict]) -> bool:
    """Create CSV file for a company with timestamp-based deduplication"""
    try:
      # Map company name to filename
      filename_map = {
        "NVIDIA": "nvidia_score_price_dump.txt",
        "APPLE": "apple_score_price_dump.txt", 
        "INTEL": "intel_score_price_dump.txt"
      }
      
      filename = filename_map.get(company)
      if not filename:
        print(f"✗ No filename mapping for {company}")
        return False
      
      filepath = os.path.join(self.output_dir, filename)
      
      # Create header
      # header = "timestamp,score_tradingview,score_sa_news,price"  # Old 4-column format
      header = "timestamp,score_sa_news,price"  # New 3-column format without TradingView
      
      # Load existing data if file exists
      existing_data = {}
      if os.path.exists(filepath):
        try:
          with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
              if line.strip():  # Skip empty lines
                parts = line.strip().split(',')
                if len(parts) >= 3:  # Ensure valid format (3-column format: timestamp,score_sa_news,price)
                  timestamp = parts[0]
                  existing_data[timestamp] = line.strip()
          print(f"✓ Loaded {len(existing_data)} existing data points from {filename}")
        except Exception as e:
          print(f"⚠ Error reading existing file {filename}: {e}")
          # If file is corrupted, we'll overwrite it
          existing_data = {}
      
      # Add new data points, updating existing timestamps with latest data
      for point in data_points:
        timestamp = point['timestamp']
        # new_row = f"{timestamp},{point['score_tradingview']},{point['score_sa_news']},{point['price']}"  # Old format
        new_row = f"{timestamp},{point['score_sa_news']},{point['price']}"  # New format without TradingView
        existing_data[timestamp] = new_row
      
      # Sort all data points by timestamp (chronological order)
      sorted_timestamps = sorted(existing_data.keys())
      
      # Write all data (existing + new) in chronological order
      with open(filepath, 'w') as f:
        f.write(header + "\n")
        for timestamp in sorted_timestamps:
          f.write(existing_data[timestamp] + "\n")
      
      new_count = len(data_points)
      total_count = len(existing_data)
      print(f"✓ Updated {filename}: {new_count} new points, {total_count} total points (chronologically sorted)")
      return True
      
    except Exception as e:
      print(f"✗ Error creating {company} CSV file: {e}")
      return False

  def run(self):
    """Main execution function"""
    print("🚀 Starting Multi-Source Sentiment Data Aggregation")
    print("=" * 60)
    
    # Load all sentiment data sources
    print("\n📊 Loading sentiment data sources...")
    self.news_data = self.load_news_sentiment()
    self.image_data = self.load_image_sentiment()
    self.tradingview_data = self.load_tradingview_sentiment()
    
    # Create output directory if needed
    os.makedirs(self.output_dir, exist_ok=True)
    
    print(f"\n📈 Generating data files for companies: {', '.join(self.target_companies)}")
    print("=" * 60)
    
    success_count = 0
    for company in self.target_companies:
      print(f"\n🏢 Processing {company}...")
      
      # Show current sentiment scores
      news_score = self.news_data.get(company, {}).get('score', 5.0)
      image_score = self.image_data.get(company, 5.0)
      tv_score = self.tradingview_data.get(company, {}).get('score', 5.0)
      
      print(f"   News Sentiment: {news_score}/10")
      print(f"   # Image Sentiment: {image_score:.1f}/10 (disabled)")
      # print(f"   TradingView Sentiment: {tv_score}/10")  # Disabled
      
      # Generate time series data
      data_points = self.generate_time_series_data(company)
      
      if data_points:
        # Create CSV file
        if self.create_csv_file(company, data_points):
          success_count += 1
        else:
          print(f"✗ Failed to create file for {company}")
      else:
        print(f"✗ No data points generated for {company}")
    
    print("\n" + "=" * 60)
    print(f"✅ Data aggregation complete: {success_count}/{len(self.target_companies)} files created")
    print(f"📁 Output directory: {self.output_dir}")
    print("\n🎯 Files ready for chart visualization!")

def main():
  """Entry point"""
  aggregator = SentimentDataAggregator()
  aggregator.run()

if __name__ == "__main__":
  main()