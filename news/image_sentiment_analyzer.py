#!/usr/bin/env python3
"""
Image-Only Sentiment Analyzer
Processes images from Output/images/ folder and returns sentiment scores (1-10).
Uses OpenAI GPT-4 Vision API for pure image-based sentiment analysis.
"""

import os
import json
import logging
import sys
import time
import base64
import re
from datetime import datetime
from typing import List, Dict, Optional
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
        logging.FileHandler(os.path.join(logs_dir, 'image_sentiment_analyzer.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

class ImageSentimentAnalyzer:
    def __init__(self):
        """Initialize the Image Sentiment Analyzer with OpenAI client"""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client (fallback from Moonshot AI due to authentication issues)
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with API key ending in: ...{self.api_key[-4:]}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def analyze_image_sentiment(self, image_path: str, company: str = "", context: str = "", max_retries: int = 3) -> Optional[float]:
        """
        Analyze sentiment from an image using OpenAI GPT-4 Vision API
        
        Args:
            image_path: Local path to the image file
            company: Company name for context (extracted from filename if not provided)
            context: Additional context (e.g., "TradingView chart" or "Reddit post")
            max_retries: Maximum API retries
            
        Returns:
            Sentiment score (1-10) or None if analysis failed
        """
        if not image_path or not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        # Extract company from filename if not provided
        if not company:
            filename = os.path.basename(image_path)
            # Look for company names in filename
            companies = ['NVIDIA', 'APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON', 'META', 'TESLA']
            for comp in companies:
                if comp in filename.upper():
                    company = comp
                    break
            if not company:
                company = "Unknown"
        
        # Determine context from filename if not provided
        if not context:
            filename = os.path.basename(image_path)
            if 'TRADINGVIEW' in filename.upper():
                context = "TradingView stock chart"
            elif any(x in filename for x in ['reddit', 'Reddit', 'REDDIT']):
                context = "Reddit social media post"
            else:
                context = "Stock-related image"
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create prompt for financial sentiment analysis
            prompt = f"""
Analyze this {context} for stock market sentiment about {company}.

Please analyze:
1. If this is a stock chart: trend direction, volume patterns, support/resistance levels
2. If this is a meme/social post: emotional tone, bullish/bearish indicators  
3. Company logos, products, or brand imagery sentiment
4. Text overlays with predictions, opinions, or reactions
5. Visual elements suggesting optimism or pessimism about the stock

Provide ONLY a sentiment score from 1 to 10:
- 1-3: Very bearish/negative sentiment (sell signals, bad news, fear)
- 4-6: Neutral sentiment (mixed signals, uncertainty)
- 7-10: Very bullish/positive sentiment (buy signals, good news, optimism)

Respond with only the number (1-10):"""

            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",  # GPT-4 with vision capabilities
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
                    numbers = re.findall(r'\d+(?:\.\d+)?', content)
                    if numbers:
                        score = float(numbers[0])
                        if 1 <= score <= 10:
                            logger.info(f"✓ {company} ({context}): {score}/10 - {os.path.basename(image_path)}")
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
    
    def analyze_all_images_in_folder(self, images_dir: str = "../container_output/images") -> Dict:
        """
        Analyze sentiment for all images in the specified folder
        
        Args:
            images_dir: Directory containing images to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(images_dir):
            logger.error(f"Images directory not found: {images_dir}")
            return {}
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
        image_files = []
        
        for filename in os.listdir(images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(images_dir, filename))
        
        if not image_files:
            logger.warning(f"No image files found in {images_dir}")
            return {}
        
        logger.info(f"Found {len(image_files)} image files to analyze")
        
        # Organize results by company and source type
        results = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_images": len(image_files),
                "images_directory": images_dir,
                "model_used": "gpt-4o"
            },
            "company_results": {},
            "detailed_results": []
        }
        
        # Analyze each image
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Analyzing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Add delay between requests to respect rate limits
            if i > 1:
                time.sleep(3)  # 3 second delay between image analyses
            
            sentiment_score = self.analyze_image_sentiment(image_path)
            
            if sentiment_score is not None:
                # Extract details from filename
                filename = os.path.basename(image_path)
                
                # Determine company
                companies = ['NVIDIA', 'APPLE', 'MICROSOFT', 'GOOGLE', 'AMAZON', 'META', 'TESLA']
                company = "Unknown"
                for comp in companies:
                    if comp in filename.upper():
                        company = comp
                        break
                
                # Determine source type
                if 'TRADINGVIEW' in filename.upper():
                    source_type = "TradingView Chart"
                elif any(x in filename for x in ['reddit', 'Reddit', 'REDDIT']):
                    source_type = "Reddit Image"
                else:
                    source_type = "Unknown Source"
                
                # Add to detailed results
                detailed_result = {
                    "filename": filename,
                    "filepath": image_path,
                    "company": company,
                    "source_type": source_type,
                    "sentiment_score": sentiment_score,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                results["detailed_results"].append(detailed_result)
                
                # Aggregate by company
                if company not in results["company_results"]:
                    results["company_results"][company] = {
                        "total_images": 0,
                        "sentiment_scores": [],
                        "average_sentiment": 0.0,
                        "tradingview_images": 0,
                        "reddit_images": 0,
                        "other_images": 0
                    }
                
                company_data = results["company_results"][company]
                company_data["total_images"] += 1
                company_data["sentiment_scores"].append(sentiment_score)
                
                # Count by source type
                if source_type == "TradingView Chart":
                    company_data["tradingview_images"] += 1
                elif source_type == "Reddit Image":
                    company_data["reddit_images"] += 1
                else:
                    company_data["other_images"] += 1
            
            else:
                logger.warning(f"Failed to analyze: {os.path.basename(image_path)}")
        
        # Calculate averages
        for company, data in results["company_results"].items():
            if data["sentiment_scores"]:
                data["average_sentiment"] = sum(data["sentiment_scores"]) / len(data["sentiment_scores"])
        
        # Summary statistics
        total_analyzed = len(results["detailed_results"])
        results["analysis_metadata"]["successfully_analyzed"] = total_analyzed
        results["analysis_metadata"]["failed_analyses"] = len(image_files) - total_analyzed
        
        logger.info(f"Analysis complete: {total_analyzed}/{len(image_files)} images successfully analyzed")
        
        return results
    
    def save_results(self, results: Dict, output_file: str = "../container_output/final_score/image-sentiment-analysis.json"):
        """
        Save analysis results to JSON file
        
        Args:
            results: Analysis results dictionary
            output_file: Output filename
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_file}")
            
            # Print summary
            total_images = results["analysis_metadata"]["total_images"]
            analyzed = results["analysis_metadata"]["successfully_analyzed"]
            companies = len(results["company_results"])
            
            print(f"\n🖼️ IMAGE SENTIMENT ANALYSIS SUMMARY")
            print(f"{'='*50}")
            print(f"Total images processed: {total_images}")
            print(f"Successfully analyzed: {analyzed}")
            print(f"Companies found: {companies}")
            print(f"\nCompany Breakdown:")
            
            for company, data in results["company_results"].items():
                avg_score = data["average_sentiment"]
                total = data["total_images"]
                tv_count = data["tradingview_images"]
                reddit_count = data["reddit_images"]
                
                print(f"  {company}: {avg_score:.1f}/10 avg ({total} images: {tv_count} charts, {reddit_count} social)")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main execution function"""
    try:
        analyzer = ImageSentimentAnalyzer()
        
        # Determine the correct path for images directory
        # Check if we're in the news/ subdirectory or main directory
        images_dir = "../container_output/images"
        if not os.path.exists(images_dir):
            images_dir = "container_output/images"
        if not os.path.exists(images_dir):
            # Try absolute path relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            images_dir = os.path.join(project_root, "container_output", "images")
        
        logger.info(f"Using images directory: {images_dir}")
        
        # Analyze all images in the directory
        results = analyzer.analyze_all_images_in_folder(images_dir)
        
        if results:
            # Save results in the Output/final_score directory
            output_file = "../container_output/final_score/image-sentiment-analysis.json"
            if os.path.basename(os.getcwd()) != "news":
                output_file = "container_output/final_score/image-sentiment-analysis.json"
            
            analyzer.save_results(results, output_file)
        else:
            logger.error("No results to save")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()