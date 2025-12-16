#!/usr/bin/env python3
'''
Trending Stocks Reddit Scraper

This script finds the top N trending stock tickers mentioned in the
selected Reddit communities within a rolling 24‑hour window.

Dependencies:
    pip install praw pandas schedule
    # optional: pip install yfinance
Environment variables required (set in .env file):
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=TrendingStocksScript/0.1
OPENAI_API_KEY=your_openai_api_key
Usage:
    python trending_stocks.py
'''
from datetime import datetime, timezone, timedelta
import praw
import json
import logging
import sys
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from project root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Configuration parameters
WINDOW_HOURS = 24
NEW_SUBREDDITS = ['stocks', 'wallstreetbets', 'investing']
TARGET_COMPANIES = {
    'NVIDIA': ['NVIDIA', 'NVDA', 'nvidia'],
    'APPLE': ['APPLE', 'AAPL', 'apple', 'Apple']
}

# Reddit API credentials from environment variables
NEW_REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
NEW_REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
NEW_REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'stockk/1.0 by uab313')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def create_new_reddit():
    '''Create a praw.Reddit instance with new credentials'''
    return praw.Reddit(
        client_id=NEW_REDDIT_CLIENT_ID,
        client_secret=NEW_REDDIT_CLIENT_SECRET,
        user_agent=NEW_REDDIT_USER_AGENT,
    )

# Reddit sentiment analysis functions
def fetch_company_mentions(reddit, company_name, company_keywords, since_utc):
    """Fetch Reddit posts mentioning specific company from 3 channels"""
    mentions = []
    
    for sub_name in NEW_SUBREDDITS:
        print(f"Scanning r/{sub_name} for {company_name}...")
        subreddit = reddit.subreddit(sub_name)
        
        try:
            # Get recent posts
            for submission in subreddit.new(limit=50):
                if submission.created_utc < since_utc:
                    continue
                    
                # Check if post mentions the company
                text = (submission.title + " " + (submission.selftext or "")).lower()
                if any(keyword.lower() in text for keyword in company_keywords):
                    mentions.append({
                        'subreddit': sub_name,
                        'title': submission.title,
                        'text': submission.selftext or "",
                        'score': submission.score,
                        'created_utc': submission.created_utc,
                    })
        except Exception as e:
            print(f"Error scanning r/{sub_name}: {e}")
    
    return mentions

def analyze_reddit_sentiment(mentions, company_name):
    """Analyze sentiment using OpenAI"""
    if not mentions:
        return None
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Prepare text summary
    text_summary = f"Reddit posts about {company_name}:\n\n"
    for i, mention in enumerate(mentions[:10], 1):
        text_summary += f"{i}. {mention['title']}\n"
        if mention['text']:
            text_summary += f"   Content: {mention['text'][:150]}...\n"
        text_summary += f"   Score: {mention['score']} | r/{mention['subreddit']}\n\n"
    
    prompt = f"""Analyze these Reddit posts about {company_name} and provide ONE sentiment score from 1 to 10.
(1 = very negative/bearish, 10 = very positive/bullish)
Provide ONLY the number.

{text_summary}

Score (1-10):"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Respond only with a sentiment score 1-10 for {company_name} stock."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract score
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        if numbers:
            score = float(numbers[0])
            if 1 <= score <= 10:
                return score
        return None
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None

def update_sentiment_file(company_name, score, mentions_count):
    """Update combine-sentiment_analysis.json"""
    try:
        # Load existing data
        try:
            with open('combine-sentiment_analysis.json', 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"last_updated": None, "companies": {}}
        
        # Update with new data
        timestamp = datetime.now().isoformat()
        data["last_updated"] = timestamp
        data["companies"][company_name] = {
            "score": score,
            "timestamp": timestamp,
            "articles_analyzed": mentions_count,
            "last_analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "last_analysis_time": datetime.now().strftime("%H:%M:%S"),
            "source": "Reddit"
        }
        
        # Save updated data
        with open('combine-sentiment_analysis.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Error updating sentiment file: {e}")
        return False

def run_reddit_sentiment():
    """Main function for Reddit sentiment analysis"""
    print("🚀 Starting Reddit Sentiment Analysis...")
    print(f"📡 Monitoring 3 channels: {', '.join(NEW_SUBREDDITS)}")
    
    # Create Reddit instance with new credentials
    reddit = create_new_reddit()
    
    # Calculate time window (last 24 hours)
    since_utc = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).timestamp()
    
    results = {}
    
    # Analyze each company
    for company_name, keywords in TARGET_COMPANIES.items():
        print(f"\n📊 Analyzing {company_name}...")
        
        # Fetch mentions from 3 Reddit channels
        mentions = fetch_company_mentions(reddit, company_name, keywords, since_utc)
        print(f"   Found {len(mentions)} mentions across 3 channels")
        
        if mentions:
            # Analyze sentiment with OpenAI
            score = analyze_reddit_sentiment(mentions, company_name)
            
            if score:
                print(f"   Reddit Sentiment Score: {score}/10")
                
                # Update combine-sentiment_analysis.json
                if update_sentiment_file(company_name, score, len(mentions)):
                    print(f"   ✅ Updated sentiment file for {company_name}")
                    results[company_name] = {'score': score, 'mentions': len(mentions)}
                else:
                    print(f"   ❌ Failed to update file for {company_name}")
            else:
                print(f"   ❌ Failed to analyze sentiment for {company_name}")
        else:
            print(f"   ⚠️  No mentions found for {company_name}")
    
    # Display final results
    if results:
        print(f"\n🎉 Reddit Analysis Complete!")
        print("="*50)
        for company, data in results.items():
            print(f"{company}: {data['score']}/10 (from {data['mentions']} mentions)")
        print("="*50)
        return True
    
    return False

if __name__ == '__main__':
    # Check if running sentiment analysis
    if len(sys.argv) > 1 and sys.argv[1] == 'sentiment':
        run_reddit_sentiment()
