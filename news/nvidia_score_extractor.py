#!/usr/bin/env python3
"""
Multi-Company Score & Price Data Extractor
Extracts comprehensive_score and current_price from final-comprehensive-sentiment-analysis.json
for NVIDIA, Apple, and Intel and appends to respective CSV files with timestamp.
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any

class MultiCompanyDataExtractor:
  def __init__(self, base_path: str = None):
    if base_path is None:
      script_dir = os.path.dirname(os.path.abspath(__file__))
      project_root = os.path.dirname(script_dir)
      self.base_path = project_root
    else:
      self.base_path = base_path
    
    self.final_analysis_path = os.path.join(
      self.base_path, "container_output", "final_score", 
      "final-comprehensive-sentiment-analysis.json"
    )
    
    # Define companies and their CSV output paths
    self.companies = {
      "NVIDIA": os.path.join(self.base_path, "web", "nvidia_score_price_dump.txt"),
      "APPLE": os.path.join(self.base_path, "web", "apple_score_price_dump.txt"),
      "INTEL": os.path.join(self.base_path, "web", "intel_score_price_dump.txt")
    }
  
  def load_final_analysis(self) -> Optional[Dict[str, Any]]:
    """Load the final comprehensive sentiment analysis JSON file."""
    try:
      with open(self.final_analysis_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    except FileNotFoundError:
      print(f"❌ Error: {self.final_analysis_path} not found")
      return None
    except json.JSONDecodeError as e:
      print(f"❌ Error reading JSON file: {e}")
      return None
  
  def extract_company_data(self, analysis_data: Dict[str, Any], company_name: str) -> Optional[Dict[str, Any]]:
    """Extract comprehensive score and price data for a specific company."""
    if not analysis_data or "companies" not in analysis_data:
      print("❌ Error: Invalid analysis data structure")
      return None
    
    companies = analysis_data["companies"]
    company_data = companies.get(company_name)
    
    if not company_data:
      print(f"❌ Error: {company_name} data not found in analysis")
      return None
    
    # Extract comprehensive score
    comprehensive_score = company_data.get("final_scores", {}).get("comprehensive_score")
    if comprehensive_score is None:
      print(f"⚠️  Warning: Comprehensive score not found for {company_name}")
      comprehensive_score = 0.0
    
    # Extract price data
    price_data = company_data.get("price_data", {})
    current_price = price_data.get("current_price")
    
    if current_price is None:
      print(f"⚠️  Warning: Current price not found for {company_name}")
      current_price = 0.0
    
    # Extract individual sentiment scores
    scores = company_data.get("final_scores", {})
    tradingview_score = scores.get("tradingview_sentiment", 0.0)
    news_score = scores.get("news_sentiment", 0.0)
    image_score = scores.get("image_sentiment", 0.0)
    
    return {
      "timestamp": datetime.now().isoformat(),
      "score_tradingview": tradingview_score,
      "score_sa_news": news_score,
      "score_sa_image": image_score,
      "price": current_price
    }
  
  def initialize_csv_file(self, csv_path: str, company_name: str):
    """Initialize CSV file with headers if it doesn't exist or is empty."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
      with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("timestamp,score_tradingview,score_sa_news,score_sa_image,price\n")
      print(f"✅ Initialized CSV file for {company_name}: {csv_path}")
    else:
      # Check if file has proper headers
      try:
        df = pd.read_csv(csv_path, nrows=1)
        expected_columns = ['timestamp', 'score_tradingview', 'score_sa_news', 'score_sa_image', 'price']
        if not all(col in df.columns for col in expected_columns):
          print(f"⚠️  Warning: {company_name} CSV file missing expected columns, reinitializing...")
          with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("timestamp,score_tradingview,score_sa_news,score_sa_image,price\n")
      except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"⚠️  Warning: {company_name} CSV file corrupted, reinitializing...")
        with open(csv_path, 'w', encoding='utf-8') as f:
          f.write("timestamp,score_tradingview,score_sa_news,score_sa_image,price\n")
  
  def append_data_to_csv(self, company_data: Dict[str, Any], csv_path: str, company_name: str):
    """Append company data to CSV file using pandas."""
    self.initialize_csv_file(csv_path, company_name)
    
    try:
      # Create new row as DataFrame
      new_row = pd.DataFrame([company_data])
      
      # Read existing data if file exists and has content
      if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
          existing_df = pd.read_csv(csv_path)
          # Combine existing and new data
          updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        except pd.errors.EmptyDataError:
          updated_df = new_row
      else:
        updated_df = new_row
      
      # Save updated data
      updated_df.to_csv(csv_path, index=False)
      
      print(f"✅ {company_name} data appended successfully:")
      print(f"   Timestamp: {company_data['timestamp']}")
      print(f"   TradingView Score: {company_data['score_tradingview']}")
      print(f"   News Score: {company_data['score_sa_news']}")
      print(f"   Image Score: {company_data['score_sa_image']}")
      print(f"   Price: ${company_data['price']}")
      print(f"   Total records: {len(updated_df)}")
      
    except Exception as e:
      print(f"❌ Error appending {company_name} data to CSV: {e}")
      raise
  
  def process_all_companies(self):
    """Process and update data for all companies."""
    print("🚀 Starting multi-company data extraction...")
    
    # Load the final analysis data
    analysis_data = self.load_final_analysis()
    if not analysis_data:
      print("❌ Failed to load analysis data")
      return False
    
    success_count = 0
    total_companies = len(self.companies)
    
    for company_name, csv_path in self.companies.items():
      print(f"\n📊 Processing {company_name}...")
      
      # Extract data for this company
      company_data = self.extract_company_data(analysis_data, company_name)
      
      if company_data:
        # Validate the data
        if self.validate_data(company_data, company_name):
          # Append to CSV
          self.append_data_to_csv(company_data, csv_path, company_name)
          success_count += 1
        else:
          print(f"❌ Validation failed for {company_name}")
      else:
        print(f"❌ Failed to extract data for {company_name}")
    
    print(f"\n✅ Processing complete: {success_count}/{total_companies} companies updated successfully")
    return success_count == total_companies

  def validate_data(self, company_data: Dict[str, Any], company_name: str) -> bool:
    """Validate extracted data before saving."""
    if not company_data:
      return False
    
    # Check required fields
    required_fields = ['timestamp', 'score', 'price']
    for field in required_fields:
      if field not in company_data:
        print(f"❌ Validation error for {company_name}: Missing field '{field}'")
        return False
    
    # Validate score range (should be 0-10)
    score = company_data['score']
    if not isinstance(score, (int, float)) or score < 0 or score > 10:
      print(f"⚠️  Warning for {company_name}: Score {score} is outside expected range (0-10)")
    
    # Validate price (should be positive)
    price = company_data['price']
    if not isinstance(price, (int, float)) or price < 0:
      print(f"⚠️  Warning for {company_name}: Price {price} is invalid")
    
    return True
  
  def extract_and_save(self) -> bool:
    """Main method to extract all company data and save to CSV files."""
    print("🎯 Starting Multi-Company Score & Price Data Extraction...")
    
    return self.process_all_companies()
  
  def display_recent_data(self, num_records: int = 5):
    """Display recent records from all company CSV files."""
    for company_name, csv_path in self.companies.items():
      print(f"\n📊 {company_name} - Recent {num_records} records:")
      print("="*60)
      
      if not os.path.exists(csv_path):
        print(f"❌ {company_name} CSV file not found")
        continue
      
      try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
          print(f"❌ {company_name} CSV file is empty")
          continue
        
        recent_data = df.tail(num_records)
        
        for _, row in recent_data.iterrows():
          timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
          print(f"🕒 {timestamp} | Score: {row['score']:.2f} | Price: ${row['price']:.2f}")
        
        print(f"Total {company_name} records: {len(df)}")
        
      except Exception as e:
        print(f"❌ Error reading {company_name} CSV file: {e}")
      
      print("="*60)

def main():
  """Main execution function."""
  extractor = MultiCompanyDataExtractor()
  
  success = extractor.extract_and_save()
  
  if success:
    # Display recent data for verification
    extractor.display_recent_data()
  
  return success

if __name__ == "__main__":
  import sys
  success = main()
  sys.exit(0 if success else 1)