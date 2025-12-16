"""
News Analysis with Grok/OpenAI API Support

# ==============================================================================
# CHANGELOG:
# ==============================================================================
# [2025-12-10] FEATURE: Added Grok API support for news summarization
#              - Grok API is preferred (real-time X/Twitter integration)
#              - Falls back to OpenAI if Grok unavailable
#              - Environment variable: GROK_TRADE_API
# ==============================================================================
"""

import os
import math
import time
import requests
from datetime import datetime, timedelta
from transformers import pipeline
# Load environment variables from project root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
from dotenv import load_dotenv
load_dotenv(env_path)

###############################
# (A) NEWS FETCHING
###############################
def fetch_stock_news(
    query="NVIDIA",
    days=7,
    api_key=None,
    page_size=20
):
    """
    Example: uses NewsAPI.org to fetch recent articles about `query`.
    For last `days` days, up to `page_size` articles.
    Returns list of dicts with fields: {title, description, url, publishedAt}.
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('NEWS_API_KEY')
        
    # If you don't want NewsAPI, you can adapt to another news source.
    url = "https://newsapi.org/v2/everything"
    
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",  # newest first
        "language": "en",
        "apiKey": api_key,
        "pageSize": page_size
    }
    
    resp = requests.get(url, params=params)
    data = resp.json()
    # Handle potential errors
    if data.get("status") != "ok":
        print("[ERROR] NewsAPI:", data)
        return []
    
    articles = data.get("articles", [])
    
    # Simplify the fields
    results = []
    for a in articles:
        results.append({
            "title": a["title"],
            "description": a.get("description", ""),
            "url": a["url"],
            "publishedAt": a["publishedAt"],  # e.g. "2023-04-01T12:00:00Z"
        })
    return results


###############################
# (B) SENTIMENT ANALYSIS
###############################
def get_huggingface_sentiment(text):
    """
    Example: Use Hugging Face pipeline for sentiment, returning a score in [-1, +1].
    We'll interpret 'negative' as near -1, 'positive' near +1, neutral near 0.
    """
    # We can use a pretrained sentiment model, e.g. "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # or "nlptown/bert-base-multilingual-uncased-sentiment"
    # For simplicity, let's pick a 3-class model from cardiffnlp:

    # The pipeline is usually loaded once outside this function for efficiency
    if not hasattr(get_huggingface_sentiment, "sentiment_pipeline"):
        get_huggingface_sentiment.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    sentiment = get_huggingface_sentiment.sentiment_pipeline(text[:512])[0]  
    # e.g. {'label': 'Positive', 'score': 0.99}
    label = sentiment['label']
    score = sentiment['score']
    
    if label == "Positive":
        return score  # near +1
    elif label == "Negative":
        return -score  # near -1
    else:
        return 0.0      # neutral

###############################
# (C) COMPUTE WEIGHTED SENTIMENT
###############################
def compute_overall_sentiment(articles):
    """
    1) For each article, compute sentiment from [-1, +1].
    2) Apply a time-based weight: more recent = heavier weight.
    3) Combine to produce final rating from [1..10].
    """
    if not articles:
        print("[WARNING] No articles to analyze.")
        return 5.0  # neutral
    
    # Parse article dates, compute "days since publish"
    now_utc = datetime.utcnow()
    # Sort from newest to oldest
    articles_sorted = sorted(
        articles,
        key=lambda x: x['publishedAt'],
        reverse=True
    )
    
    # We'll accumulate weighted sum
    sum_weights = 0.0
    sum_weighted_score = 0.0
    
    for i, art in enumerate(articles_sorted):
        pub_str = art['publishedAt']
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z",""))
        except:
            pub_dt = now_utc
        
        days_old = (now_utc - pub_dt).total_seconds() / 3600 / 24  # fraction of a day
        # Let's define a simple weighting scheme:
        # Weight = e^(-days_old) so that more recent news has higher weight
        w = math.exp(-days_old)
        
        text = art['title'] + ". " + art['description']
        sent_val = get_huggingface_sentiment(text)  # in [-1..+1]
        
        sum_weights += w
        sum_weighted_score += w * sent_val
    
    # Weighted average sentiment in [-1..+1]
    weighted_avg_sentiment = sum_weighted_score / (sum_weights + 1e-9)
    
    # Convert [-1..+1] → [1..10]
    # We'll do: rating = 1 + (weighted_avg_sentiment + 1)*(9/2)
    # So if sentiment=-1 => rating=1, if sentiment=+1 => rating=10
    rating = 1 + (weighted_avg_sentiment + 1) * (9 / 2)
    rating = max(1.0, min(10.0, rating))  # clamp to [1..10]
    
    return rating


###############################
# (D) OPTIONAL: LLM SUMMARY
###############################
def summarize_with_llm(articles, final_rating, openai_api_key=None):
    """
    If you want a large language model to create a textual summary,
    you can call OpenAI's GPT or Grok with a prompt that includes
    the articles + final rating.

    Priority: Grok API > OpenAI API
    """
    from openai import OpenAI

    # ==============================================================================
    # GROK API INTEGRATION (2025-12-10)
    # ==============================================================================
    grok_api_key = os.getenv('GROK_TRADE_API')
    openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')

    # Initialize client with priority: Grok > OpenAI
    if grok_api_key:
        client = OpenAI(
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1"
        )
        model = "grok-3-fast"
        print("🚀 Using Grok API for news summarization")
    elif openai_api_key:
        client = OpenAI(api_key=openai_api_key)
        model = "gpt-3.5-turbo"
        print("📡 Using OpenAI API for news summarization")
    else:
        print("❌ No API keys found! Set GROK_TRADE_API or OPENAI_API_KEY")
        return "Error: No API key configured"

    # Create a short bullet summary of the news for the LLM
    news_bullets = []
    for art in articles[:5]:  # limit
        bullet = f"- {art['title']}, published at {art['publishedAt']}"
        news_bullets.append(bullet)

    prompt = f"""
    I have analyzed the following news for NVIDIA.
    The final aggregated sentiment rating is {final_rating:.2f} out of 10.

    News headlines:
    {chr(10).join(news_bullets)}

    Please provide a short summary of the overall market sentiment and
    whether it suggests a bullish or bearish outlook.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful financial assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0.7
    )
    summary = response.choices[0].message.content.strip()
    return summary


###############################
# MAIN DEMO
###############################
if __name__ == "__main__":
    # 1) Fetch recent news using environment variable for API key
    articles = fetch_stock_news(
        query="NVIDIA",
        days=7,
        page_size=20
    )
    
    # 2) Compute Weighted Sentiment => [1..10]
    final_rating = compute_overall_sentiment(articles)
    print(f"\nFinal Weighted Sentiment (1=Sell, 10=Buy) = {final_rating:.2f}")
    
    # 3) (Optional) Summarize using an LLM (like GPT-3.5)
    #    Uses environment variable for OpenAI API key
    # summary = summarize_with_llm(articles, final_rating)
    # print("\nLLM Summary:\n", summary)
