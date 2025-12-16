import os
import requests
from datetime import datetime, timedelta
import openai
from collections import defaultdict
# Load environment variables from project root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
from dotenv import load_dotenv
load_dotenv(env_path)

# Set API keys from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Configuration
STOCK = "NVIDIA"
NUM_ARTICLES = 50  # Max articles to analyze

def fetch_news(stock, from_date, to_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": stock,
        "from": from_date,
        "to": to_date,
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100
    }
    response = requests.get(url, params=params)
    return response.json().get('articles', [])[:NUM_ARTICLES]

def analyze_sentiment(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"Analyze this stock-related text for market sentiment. Respond with only a number between 1 and 10 (10=very positive, 1=very negative):\n{text}"
            }]
        )
        return float(response.choices[0].message['content'].strip())
    except:
        return 5.0  # Neutral if error

def calculate_weighted_score(articles):
    today = datetime.today().date()
    total_score = 0
    total_weight = 0
    
    for article in articles:
        published_date = datetime.strptime(article['publishedAt'][:10], "%Y-%m-%d").date()
        days_ago = (today - published_date).days
        weight = 1 / (days_ago + 1)  # Recent articles get higher weight
        
        content = f"{article['title']}. {article['description']}"
        sentiment = analyze_sentiment(content)
        
        total_score += sentiment * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight != 0 else 5.0

def get_final_recommendation(score, articles):
    prompt = f"""Analyze these {len(articles)} news articles about {STOCK}. 
    The weighted sentiment score is {score:.2f}/10. Consider recent news trends.
    Write a brief recommendation (1 paragraph) ending with "Final Recommendation Score: X/10" 
    (10=strong buy, 1=strong sell):\n\n"""
    
    for idx, article in enumerate(articles[:5]):  # Top 5 articles
        prompt += f"{idx+1}. {article['title']}\n"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message['content']

def main():
    # Date calculations (last week)
    today = datetime.today().date()
    from_date = (today - timedelta(days=7)).isoformat()
    to_date = today.isoformat()
    
    # Fetch news
    articles = fetch_news(STOCK, from_date, to_date)
    print(f"Found {len(articles)} articles about {STOCK}")
    
    # Calculate weighted score
    weighted_score = calculate_weighted_score(articles)
    final_score = min(10, max(1, round(weighted_score)))
    
    # Generate recommendation
    recommendation = get_final_recommendation(weighted_score, articles)
    
    print("\n=== Market Sentiment Analysis ===")
    print(f"Date Range: {from_date} to {to_date}")
    print(f"Weighted Sentiment Score: {weighted_score:.2f}")
    print(f"Final Recommendation: {final_score}/10")
    print("\n=== AI Analysis ===")
    print(recommendation)

if __name__ == "__main__":
    main()