import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai # Or use 'import openai'

# --- Configuration ---
# Load environment variables from project root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path) # Load environment variables from project root .env file

# --- API Keys (Replace with your actual keys in a .env file) ---
# Example for NewsAPI.org (adjust URL/params for other APIs)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# Example for Google Gemini
LLM_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=LLM_API_KEY)
# --- OR ---
# Example for OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY

# --- Stock & Time ---
STOCK_SYMBOL = "NVDA" # Example: NVIDIA
DAYS_AGO = 7 # Look back period (last week)
CURRENT_TIME_UTC = datetime.utcnow() # Use UTC for consistency
START_DATE = (CURRENT_TIME_UTC - timedelta(days=DAYS_AGO)).strftime('%Y-%m-%d')
END_DATE = CURRENT_TIME_UTC.strftime('%Y-%m-%d')

# --- LLM Model Choice ---
# For Gemini: e.g., 'gemini-1.5-flash', 'gemini-1.0-pro'
LLM_MODEL_NAME = 'gemini-1.5-flash'
# --- OR ---
# For OpenAI: e.g., 'gpt-4o', 'gpt-3.5-turbo'
# LLM_MODEL_NAME = 'gpt-4o'


# --- Function to Fetch News ---
def fetch_stock_news(symbol, start_date, end_date, api_key):
    """Fetches news articles for a given stock symbol and date range."""
    print(f"Fetching news for {symbol} from {start_date} to {end_date}...")

    # --- Adjust parameters based on your chosen News API ---
    params = {
        'q': f'{symbol} OR NVIDIA', # Query term (be specific)
        'from': start_date,
        'to': end_date,
        'language': 'en',
        'sortBy': 'publishedAt', # Get newest first
        'apiKey': api_key,
        'pageSize': 100 # Get max articles (check API limits)
    }
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])
            print(f"Fetched {len(articles)} news articles.")
            # Filter/clean articles if needed
            return articles
        else:
            print(f"Error from News API: {data.get('message')}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Network or API request error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during news fetching: {e}")
        return []

# --- Function to Analyze Sentiment and Score with LLM ---
def get_llm_sentiment_score(articles, stock_symbol, model_name):
    """
    Uses an LLM to analyze news sentiment, consider recency implicitly,
    and provide a buy/sell score (1-10).
    """
    if not articles:
        print("No articles provided for analysis.")
        return None, "No articles to analyze."

    print(f"\nAnalyzing sentiment for {stock_symbol} using {model_name}...")

    # Prepare the news data for the prompt (newest first)
    # Sort again just to be sure, as API might not guarantee perfect sorting
    articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)

    news_prompt_data = ""
    for i, article in enumerate(articles[:30]): # Limit input size for LLM context window
        title = article.get('title', 'N/A')
        published_at_str = article.get('publishedAt', 'N/A')
        # Try parsing the date to show age, helps LLM gauge recency
        try:
            published_dt = datetime.strptime(published_at_str.split('T')[0], '%Y-%m-%d')
            days_old = (CURRENT_TIME_UTC.date() - published_dt.date()).days
            date_info = f"{published_at_str} ({days_old} days ago)"
        except:
            date_info = published_at_str # Fallback if date parsing fails

        news_prompt_data += f"{i+1}. Title: {title}\n   Published: {date_info}\n\n"

    # --- Carefully Craft the Prompt ---
    prompt = f"""
    Analyze the following recent news headlines/articles related to the stock {stock_symbol}.
    Consider the content and the recency (newer articles are generally more relevant).
    Based *only* on the sentiment expressed in these news items regarding {stock_symbol}'s future prospects,
    provide an overall market sentiment score on a scale of 1 to 10.

    Scale:
    1: Extremely Negative Sentiment / Strong Sell Signal inferred from news
    5: Neutral / Mixed Sentiment inferred from news
    10: Extremely Positive Sentiment / Strong Buy Signal inferred from news

    News Items (Newest First):
    ---
    {news_prompt_data}
    ---

    Required Output Format:
    Score: [Your calculated score between 1 and 10]
    Justification: [A brief explanation of your reasoning based on the news sentiment]
    """

    try:
        # --- Using Google Gemini ---
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        llm_output = response.text

        # --- OR ---
        # --- Using OpenAI ---
        # client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # response = client.chat.completions.create(
        #     model=model_name,
        #     messages=[
        #         {"role": "system", "content": "You are a financial news sentiment analyst."},
        #         {"role": "user", "content": prompt}
        #     ]
        # )
        # llm_output = response.choices[0].message.content
        # ---

        print("\n--- LLM Analysis ---")
        print(llm_output)
        print("--------------------")

        # --- Parse the LLM Output (simple parsing, might need refinement) ---
        score = None
        justification = "Could not parse justification from LLM response."

        lines = llm_output.strip().split('\n')
        for line in lines:
            if line.lower().startswith("score:"):
                try:
                    score_str = line.split(":")[1].strip()
                    score = int(float(score_str)) # Allow float then convert
                    if not 1 <= score <= 10:
                         print(f"Warning: LLM score {score} is outside the 1-10 range. Clamping.")
                         score = max(1, min(10, score)) # Clamp to range
                except (IndexError, ValueError) as parse_err:
                    print(f"Error parsing score from line: '{line}'. Error: {parse_err}")
                    score = None # Reset score if parsing fails
            elif line.lower().startswith("justification:"):
                justification = line.split(":", 1)[1].strip()

        if score is None:
             print("Warning: Could not parse a valid score (1-10) from the LLM response.")
             # Fallback: Try to extract any number between 1-10 if direct parsing failed
             import re
             numbers = re.findall(r'\b([1-9]|10)\b', llm_output)
             if numbers:
                 score = int(numbers[0])
                 print(f"Fallback: Extracted score {score} using regex.")
                 justification += " (Score extracted via fallback regex)"
             else:
                 justification = "Failed to parse score and justification." # Overwrite justification if score totally failed


        return score, justification

    except Exception as e:
        print(f"An error occurred during LLM interaction: {e}")
        # Handle specific API errors if needed (e.g., rate limits, auth errors)
        return None, f"LLM API Error: {e}"


# --- Main Execution ---
if __name__ == "__main__":
    if not NEWS_API_KEY or not LLM_API_KEY:
        print("Error: API keys not found. Please set NEWS_API_KEY and GOOGLE_API_KEY (or OPENAI_API_KEY) in your .env file.")
    else:
        # 1. Fetch News
        articles = fetch_stock_news(STOCK_SYMBOL, START_DATE, END_DATE, NEWS_API_KEY)

        # 2. Analyze Sentiment and Get Score
        if articles:
            final_score, justification = get_llm_sentiment_score(articles, STOCK_SYMBOL, LLM_MODEL_NAME)

            # 3. Display Results
            print("\n--- Final Result ---")
            if final_score is not None:
                print(f"Stock: {STOCK_SYMBOL}")
                print(f"Analysis Period: Last {DAYS_AGO} days ({START_DATE} to {END_DATE})")
                print(f"Overall News Sentiment Score (1-10): {final_score}")
                print(f"Justification: {justification}")
            else:
                print(f"Could not determine a final sentiment score for {STOCK_SYMBOL}.")
                print(f"Reason: {justification}") # Display error message
        else:
            print(f"No news articles found for {STOCK_SYMBOL} in the specified period.")

    print("\nDisclaimer: This analysis is based on automated sentiment interpretation of news headlines and should not be considered financial advice. Market sentiment can change rapidly. Always do your own research and consult with a qualified financial advisor before making investment decisions.")