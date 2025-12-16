#!/usr/bin/env python3
"""
⚠️ DISABLED: Chart Image Generation (2025-11-16)

REASON FOR DISABLING:
- Chart API generates PNG images (not usable for trading decisions)
- Trading bot needs CALCULATED INDICATORS (numbers), not images
- Costs money (chart-img.com API)
- We have web dashboard for visual monitoring (better + free)

CODE PRESERVED (NOT DELETED):
- May re-enable in future for enhanced visualization
- All code commented out but kept for reference

CURRENT STATUS:
- Trading system works WITHOUT chart generation
- Use web dashboard (serve_charts.py) for visual monitoring
- Calculated indicators (ADX, RSI, MACD) added to indicator_utils.py instead

Generate technical analysis charts with sentiment overlay
Uses chart-img.com API for static chart generation
"""

# ===== DISABLED CODE BELOW - DO NOT DELETE =====
# Last working version: 2025-11-16
# To re-enable: Remove the triple quotes and uncomment imports

# """
# # DISABLED CHART GENERATION CODE - ALL CODE BELOW IS COMMENTED OUT
# # To re-enable: Remove triple quotes at line 28 and line 286
# 
# import os
# import json
# import pandas as pd
# from datetime import datetime
# import requests
# 
# # API Configuration
# API_KEY = "sRUb8OgTLdc3bGLdQqa1wMFYwTjshdg0d4Vnkf0"
# OUTPUT_DIR = "./container_output/images/technical_charts"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# 
# def generate_tradingview_chart(symbol, indicators=None):
#     """Generate TradingView chart with technical indicators"""
#     
#     # Default indicators if none specified
#     if indicators is None:
#         indicators = "MA:20|MA:50|RSI:14|MACD:12,26,9|BB:20,2|VOL"
#     
#     # TradingView Advanced Chart URL
#     url = f"https://api.chart-img.com/v2/tradingview/advanced-chart"
#     
#     params = {
#         "key": API_KEY,
#         "symbol": symbol,
#         "interval": "D",  # Daily
#         "width": 1200,
#         "height": 600,
#         "theme": "dark",
#         "studies": indicators,
#         "timezone": "America/New_York",
#         "style": "1",  # Candles
#         "locale": "en"
#     }
#     
#     try:
#         response = requests.get(url, params=params, timeout=30)
#         if response.status_code == 200:
#             filename = f"{symbol}_technical_analysis.png"
#             filepath = os.path.join(OUTPUT_DIR, filename)
#             with open(filepath, 'wb') as f:
#                 f.write(response.content)
#             print(f"✅ Saved TradingView chart: {filepath}")
#             return filepath
#         else:
#             print(f"❌ Failed to generate chart for {symbol}: {response.status_code}")
#             return None
#     except Exception as e:
#         print(f"❌ Error generating chart: {e}")
#         return None
# 
# def generate_sentiment_chart(company, data_file):
#     """Generate sentiment analysis chart from score_price_dump files"""
#     
#     try:
#         # Read the data
#         df = pd.read_csv(data_file)
#         
#         # Prepare data for Chart.js
#         chart_data = {
#             "type": "line",
#             "data": {
#                 "labels": df['timestamp'].apply(lambda x: pd.to_datetime(x).strftime('%H:%M')).tolist()[-20:],
#                 "datasets": [
#                     {
#                         "label": "Price",
#                         "data": df['price'].tolist()[-20:],
#                         "borderColor": "rgb(75, 192, 192)",
#                         "yAxisID": "y"
#                     },
#                     {
#                         "label": "TradingView Score",
#                         "data": df['score_tradingview'].tolist()[-20:],
#                         "borderColor": "rgb(255, 99, 132)",
#                         "yAxisID": "y1"
#                     },
#                     {
#                         "label": "News Sentiment",
#                         "data": df['score_sa_news'].tolist()[-20:],
#                         "borderColor": "rgb(54, 162, 235)",
#                         "yAxisID": "y1"
#                     },
#                     {
#                         "label": "Image Sentiment",
#                         "data": df['score_sa_image'].tolist()[-20:],
#                         "borderColor": "rgb(255, 206, 86)",
#                         "yAxisID": "y1"
#                     }
#                 ]
#             },
#             "options": {
#                 "plugins": {
#                     "title": {
#                         "display": True,
#                         "text": f"{company} - Sentiment Analysis & Price"
#                     }
#                 },
#                 "scales": {
#                     "y": {
#                         "type": "linear",
#                         "display": True,
#                         "position": "left",
#                         "title": {
#                             "display": True,
#                             "text": "Price ($)"
#                         }
#                     },
#                     "y1": {
#                         "type": "linear",
#                         "display": True,
#                         "position": "right",
#                         "title": {
#                             "display": True,
#                             "text": "Sentiment Score (0-10)"
#                         },
#                         "min": 0,
#                         "max": 10
#                     }
#                 }
#             }
#         }
#         
#         # Generate chart URL
#         import urllib.parse
#         config_json = json.dumps(chart_data)
#         encoded_config = urllib.parse.quote(config_json)
#         
#         chart_url = f"https://api.chart-img.com/v1/chart?key={API_KEY}&c={encoded_config}&width=1000&height=500&format=png"
#         
#         # Download and save
#         response = requests.get(chart_url, timeout=30)
#         if response.status_code == 200:
#             filename = f"{company.lower()}_sentiment_chart.png"
#             filepath = os.path.join(OUTPUT_DIR, filename)
#             with open(filepath, 'wb') as f:
#                 f.write(response.content)
#             print(f"✅ Saved sentiment chart: {filepath}")
#             return filepath
#         else:
#             print(f"❌ Failed to generate sentiment chart: {response.status_code}")
#             return None
#             
#     except Exception as e:
#         print(f"❌ Error processing {company} data: {e}")
#         return None
# 
# def main():
#     """Generate all technical and sentiment charts"""
#     
#     print("🚀 Generating Technical Analysis Charts with Sentiment Overlay")
#     print("=" * 60)
#     
#     # Company mappings
#     companies = {
#         "NVIDIA": {"symbol": "NVDA", "file": "./web/nvidia_score_price_dump.txt"},
#         "APPLE": {"symbol": "AAPL", "file": "./web/apple_score_price_dump.txt"},
#         "INTEL": {"symbol": "INTC", "file": "./web/intel_score_price_dump.txt"}
#     }
#     
#     # Generate charts for each company
#     for company, info in companies.items():
#         print(f"\n📊 Processing {company} ({info['symbol']})...")
#         
#         # 1. Generate TradingView technical chart
#         print(f"   📈 Generating technical analysis chart...")
#         generate_tradingview_chart(info['symbol'])
#         
#         # 2. Generate sentiment overlay chart
#         if os.path.exists(info['file']):
#             print(f"   🎯 Generating sentiment analysis chart...")
#             generate_sentiment_chart(company, info['file'])
#         else:
#             print(f"   ⚠️  Data file not found: {info['file']}")
#     
#     # 3. Generate comparison chart
#     print("\n📊 Generating comparison chart...")
#     generate_comparison_chart()
#     
#     print(f"\n✅ All charts saved to: {OUTPUT_DIR}")
# 
# def generate_comparison_chart():
#     """Generate multi-company comparison chart"""
#     
#     try:
#         # Load final scores
#         with open("./container_output/final_score/final-weighted-scores.json", "r") as f:
#             data = json.load(f)
#         
#         companies = []
#         news_scores = []
#         tv_scores = []
#         image_scores = []
#         
#         for company, scores in data.get("companies", {}).items():
#             if company in ["NVIDIA", "APPLE", "INTEL"]:
#                 companies.append(company)
#                 news_scores.append(scores.get("news_sentiment", 0))
#                 tv_scores.append(scores.get("tradingview_sentiment", 0))
#                 image_scores.append(scores.get("image_sentiment", 0))
#         
#         # Create comparison chart
#         chart_data = {
#             "type": "bar",
#             "data": {
#                 "labels": companies,
#                 "datasets": [
#                     {
#                         "label": "News Sentiment",
#                         "data": news_scores,
#                         "backgroundColor": "rgba(54, 162, 235, 0.8)"
#                     },
#                     {
#                         "label": "TradingView Score",
#                         "data": tv_scores,
#                         "backgroundColor": "rgba(255, 206, 86, 0.8)"
#                     },
#                     {
#                         "label": "Image Sentiment",
#                         "data": image_scores,
#                         "backgroundColor": "rgba(75, 192, 192, 0.8)"
#                     }
#                 ]
#             },
#             "options": {
#                 "plugins": {
#                     "title": {
#                         "display": True,
#                         "text": "Multi-Company Sentiment Comparison"
#                     }
#                 },
#                 "scales": {
#                     "y": {
#                         "beginAtZero": True,
#                         "max": 10
#                     }
#                 }
#             }
#         }
#         
#         # Generate chart
#         import urllib.parse
#         config_json = json.dumps(chart_data)
#         encoded_config = urllib.parse.quote(config_json)
#         
#         chart_url = f"https://api.chart-img.com/v1/chart?key={API_KEY}&c={encoded_config}&width=800&height=500&format=png"
#         
#         response = requests.get(chart_url, timeout=30)
#         if response.status_code == 200:
#             filepath = os.path.join(OUTPUT_DIR, "sentiment_comparison.png")
#             with open(filepath, 'wb') as f:
#                 f.write(response.content)
#             print(f"   ✅ Saved comparison chart: {filepath}")
#         
#     except Exception as e:
#         print(f"   ❌ Error generating comparison chart: {e}")

# if __name__ == "__main__":
#     main()

# ===== END OF DISABLED CODE =====

# REPLACEMENT: Use calculated indicators instead
# See: trading/indicator_utils.py for ADX, RSI, MACD calculations
# See: serve_charts.py for web-based visual monitoring

if __name__ == "__main__":
    print("⚠️  Chart generation is DISABLED")
    print("📊 Use trading/indicator_utils.py for calculated indicators (ADX, RSI, MACD)")
    print("🌐 Use serve_charts.py for web dashboard visualization")
    print("💡 To re-enable: Uncomment code in this file")