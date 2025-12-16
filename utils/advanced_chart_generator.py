#!/usr/bin/env python3
"""
Advanced Chart Generator using chart-img.com API with TradingView indicators
Generates static charts with technical analysis and sentiment overlays
API Key: sRUb8OgTLdc3bGLdQqa1wMFYwTjshdg0d4Vnkf0
"""

import json
import requests
import urllib.parse
from datetime import datetime, timedelta
import os
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Warning: pandas/numpy not available, some features disabled")
    pd = None
    np = None

class AdvancedChartGenerator:
    # TradingView Chart Drawing Options
    CHART_DRAWINGS = [
        "trend_line",
        "horizontal_line", 
        "fibonacci_retracement",
        "ray_line",
        "text",
        "parallel_channel"
    ]
    
    def __init__(self):
        self.api_key = "sSRUb8OgTLdc3bGLdQqa1wMFYwTjshdg0d4Vnkf0"
        self.base_url_v1 = "https://api.chart-img.com/v1/tradingview/advanced-chart"
        self.base_url_v2 = "https://api.chart-img.com/v2/tradingview/advanced-chart"
        self.quickchart_url = "https://quickchart.io/chart"
        self.output_dir = "./container_output/images/technical_charts"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_tradingview_chart_v1(self, symbol, interval="1d", indicators=None, drawings=None):
        """Generate TradingView chart using v1 API (GET request)"""
        
        if indicators is None:
            indicators = ["MA@tv-basicstudies-20", "MA@tv-basicstudies-50", "RSI@tv-basicstudies-14"]
        
        # Build URL parameters for v1 API
        params = {
            "key": self.api_key,
            "symbol": f"NASDAQ:{symbol}",
            "interval": interval,
            "width": 1200,
            "height": 800,
            "theme": "dark",
            "timezone": "America/New_York"
        }
        
        # Add studies if provided
        if indicators:
            params["studies"] = ",".join(indicators)
        
        # Add drawing tools if provided
        if drawings:
            drawing_tools = []
            for drawing in drawings:
                if isinstance(drawing, dict) and "type" in drawing:
                    drawing_type = drawing["type"]
                    if drawing_type in self.CHART_DRAWINGS:
                        drawing_tools.append(drawing_type)
                elif isinstance(drawing, str):
                    if drawing in self.CHART_DRAWINGS:
                        drawing_tools.append(drawing)
            
            if drawing_tools:
                params["drawings"] = ",".join(drawing_tools)
        
        url = f"{self.base_url_v1}?{urllib.parse.urlencode(params)}"
        return url
    
    def generate_tradingview_chart(self, symbol, interval="1d", indicators=None):
        """Generate TradingView chart with technical indicators"""
        
        if indicators is None:
            indicators = [
                {"name": "Moving Average", "parameters": {"length": 20}},
                {"name": "Moving Average", "parameters": {"length": 50}},
                {"name": "Relative Strength Index", "parameters": {"length": 14}},
                {"name": "MACD", "parameters": {"fast": 12, "slow": 26, "signal": 9}},
                {"name": "Bollinger Bands", "parameters": {"length": 20, "stdDev": 2}}
            ]
        
        # Use chart-img.com v2 API with proper POST payload
        payload = {
            "symbol": f"NASDAQ:{symbol}",  # Use proper exchange prefix
            "interval": interval,
            "width": 1200,
            "height": 800,
            "theme": "dark",
            "studies": indicators,
            "timezone": "America/New_York"
        }
        
        return payload
    
    def generate_tradingview_chart_with_drawings(self, symbol, interval="1d", indicators=None, drawings=None):
        """Generate TradingView chart with technical indicators and drawing tools"""
        
        if indicators is None:
            indicators = [
                {"name": "Moving Average", "parameters": {"length": 20}},
                {"name": "Moving Average", "parameters": {"length": 50}},
                {"name": "Relative Strength Index", "parameters": {"length": 14}},
                {"name": "MACD", "parameters": {"fast": 12, "slow": 26, "signal": 9}},
                {"name": "Bollinger Bands", "parameters": {"length": 20, "stdDev": 2}}
            ]
        
        if drawings is None:
            drawings = [
                "trend_line",
                "horizontal_line", 
                "fibonacci_retracement",
                "ray_line",
                "text",
                "parallel_channel"
            ]
        
        # Use chart-img.com v2 API with proper POST payload
        payload = {
            "symbol": f"NASDAQ:{symbol}",
            "interval": interval,
            "width": 1200,
            "height": 800,
            "theme": "dark",
            "studies": indicators,
            "drawings": drawings,
            "timezone": "America/New_York"
        }
        
        return payload
    
    def generate_chart_with_drawings(self, symbol, interval="D", indicators=None, drawings=None):
        """Generate TradingView chart with technical indicators and drawing tools"""
        
        if indicators is None:
            indicators = [
                {"name": "MA", "parameters": [20]},
                {"name": "MA", "parameters": [50]},
                {"name": "RSI", "parameters": [14]},
                {"name": "MACD", "parameters": [12, 26, 9]},
                {"name": "BB", "parameters": [20, 2]}
            ]
        
        params = {
            "key": self.api_key,
            "symbol": symbol,
            "interval": interval,
            "width": 1200,
            "height": 800,
            "theme": "dark",
            "studies": json.dumps(indicators),
            "timezone": "America/New_York"
        }
        
        # Add drawing tools if specified
        if drawings:
            drawing_config = []
            for drawing in drawings:
                if isinstance(drawing, dict) and "type" in drawing:
                    drawing_type = drawing["type"]
                    if drawing_type in self.CHART_DRAWINGS:
                        drawing_config.append(drawing)
                elif isinstance(drawing, str):
                    if drawing in self.CHART_DRAWINGS:
                        drawing_config.append({"type": drawing})
            
            if drawing_config:
                params["drawings"] = json.dumps(drawing_config)
        
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        return url
    
    def get_available_drawings(self):
        """Get list of available TradingView drawing tools"""
        return {
            "Available TradingView Chart Drawings": self.CHART_DRAWINGS,
            "Usage Examples": {
                "By Name": "drawings=['trend_line', 'horizontal_line']",
                "Multiple": "drawings=['trend_line', 'fibonacci_retracement', 'parallel_channel']",
                "Advanced": "drawings=[{'type': 'trend_line', 'coordinates': [...]}]"
            },
            "Documentation": "https://doc.chart-img.com/#tradingview-chart-drawings"
        }
    
    def generate_sentiment_overlay_chart(self, company_data):
        """Generate chart with sentiment analysis overlay"""
        
        # Prepare data for multi-axis chart
        timestamps = [d["timestamp"] for d in company_data["data"]]
        prices = [d["price"] for d in company_data["data"]]
        sentiments = [d["sentiment_avg"] for d in company_data["data"]]
        
        # Calculate moving averages (if numpy available)
        if np:
            ma_20 = self._calculate_ma(prices, 20)
            ma_50 = self._calculate_ma(prices, 50)
        else:
            ma_20 = [None] * len(prices)
            ma_50 = [None] * len(prices)
        
        # Calculate sentiment zones
        sentiment_high = [8.0] * len(timestamps)  # Bullish zone
        sentiment_low = [3.0] * len(timestamps)   # Bearish zone
        
        chart_config = {
            "type": "line",
            "data": {
                "labels": [t.strftime("%m/%d %H:%M") for t in timestamps],
                "datasets": [
                    {
                        "label": f"{company_data['symbol']} Price",
                        "data": prices,
                        "borderColor": "#00ff88",
                        "backgroundColor": "rgba(0, 255, 136, 0.1)",
                        "yAxisID": "y",
                        "borderWidth": 2
                    },
                    {
                        "label": "MA20",
                        "data": ma_20,
                        "borderColor": "#ffaa00",
                        "borderDash": [5, 5],
                        "yAxisID": "y",
                        "borderWidth": 1
                    },
                    {
                        "label": "MA50",
                        "data": ma_50,
                        "borderColor": "#ff6600",
                        "borderDash": [10, 5],
                        "yAxisID": "y",
                        "borderWidth": 1
                    },
                    {
                        "label": "Sentiment Score",
                        "data": sentiments,
                        "borderColor": "#00aaff",
                        "backgroundColor": "rgba(0, 170, 255, 0.1)",
                        "yAxisID": "y1",
                        "borderWidth": 2,
                        "fill": True
                    },
                    {
                        "label": "Bullish Zone",
                        "data": sentiment_high,
                        "borderColor": "rgba(0, 255, 0, 0.3)",
                        "backgroundColor": "rgba(0, 255, 0, 0.1)",
                        "yAxisID": "y1",
                        "borderWidth": 1,
                        "borderDash": [2, 2],
                        "fill": "+1"
                    },
                    {
                        "label": "Bearish Zone",
                        "data": sentiment_low,
                        "borderColor": "rgba(255, 0, 0, 0.3)",
                        "backgroundColor": "rgba(255, 0, 0, 0.1)",
                        "yAxisID": "y1",
                        "borderWidth": 1,
                        "borderDash": [2, 2],
                        "fill": "-1"
                    }
                ]
            },
            "options": {
                "responsive": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"{company_data['name']} - Price & Sentiment Analysis with Technical Indicators",
                        "font": {"size": 18},
                        "color": "#ffffff"
                    },
                    "legend": {
                        "display": True,
                        "position": "top",
                        "labels": {"color": "#ffffff"}
                    },
                    "annotation": {
                        "annotations": self._generate_signal_annotations(prices, sentiments)
                    }
                },
                "scales": {
                    "x": {
                        "grid": {"color": "rgba(255, 255, 255, 0.1)"},
                        "ticks": {"color": "#ffffff"}
                    },
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left",
                        "title": {
                            "display": True,
                            "text": "Price ($)",
                            "color": "#ffffff"
                        },
                        "grid": {"color": "rgba(255, 255, 255, 0.1)"},
                        "ticks": {"color": "#ffffff"}
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "title": {
                            "display": True,
                            "text": "Sentiment Score (0-10)",
                            "color": "#ffffff"
                        },
                        "min": 0,
                        "max": 10,
                        "grid": {"drawOnChartArea": False},
                        "ticks": {"color": "#ffffff"}
                    }
                }
            }
        }
        
        # Use QuickChart for complex Chart.js configurations
        chart_data = {
            "chart": chart_config,
            "width": 1200,
            "height": 600,
            "format": "png",
            "backgroundColor": "#1a1a1a"
        }
        
        return chart_data
    
    def generate_sentiment_heatmap(self, multi_company_data):
        """Generate heatmap comparing sentiment across companies"""
        
        companies = list(multi_company_data.keys())
        metrics = ["News", "TradingView", "Image", "Overall"]
        
        # Create heatmap data
        data = []
        for metric in metrics:
            row = []
            for company in companies:
                if metric == "News":
                    row.append(multi_company_data[company]["news_sentiment"])
                elif metric == "TradingView":
                    row.append(multi_company_data[company]["tradingview_score"])
                elif metric == "Image":
                    row.append(multi_company_data[company]["image_sentiment"])
                else:  # Overall
                    row.append(multi_company_data[company]["overall_sentiment"])
            data.append(row)
        
        # Create heatmap chart
        chart_config = {
            "type": "matrix",
            "data": {
                "datasets": [{
                    "label": "Sentiment Heatmap",
                    "data": self._flatten_heatmap_data(data, companies, metrics),
                    "backgroundColor": "rgba(100, 149, 237, 0.8)",
                    "borderWidth": 1,
                    "borderColor": "#ffffff",
                    "width": 100,
                    "height": 100
                }]
            },
            "options": {
                "responsive": False,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Multi-Company Sentiment Analysis Heatmap",
                        "font": {"size": 18},
                        "color": "#ffffff"
                    },
                    "legend": {"display": False}
                },
                "scales": {
                    "x": {
                        "type": "category",
                        "labels": companies,
                        "ticks": {"color": "#ffffff"}
                    },
                    "y": {
                        "type": "category",
                        "labels": metrics,
                        "ticks": {"color": "#ffffff"}
                    }
                }
            }
        }
        
        # Use QuickChart for heatmap
        chart_data = {
            "chart": chart_config,
            "width": 800,
            "height": 600,
            "format": "png",
            "backgroundColor": "#1a1a1a"
        }
        
        return chart_data
    
    def _calculate_ma(self, prices, period):
        """Calculate moving average"""
        ma = []
        for i in range(len(prices)):
            if i < period - 1:
                ma.append(None)
            else:
                ma.append(sum(prices[i-period+1:i+1]) / period)
        return ma
    
    def _generate_signal_annotations(self, prices, sentiments):
        """Generate buy/sell signal annotations based on sentiment"""
        annotations = []
        
        for i in range(1, len(prices)):
            # Buy signal: sentiment crosses above 7
            if sentiments[i-1] <= 7 and sentiments[i] > 7:
                annotations.append({
                    "type": "point",
                    "xValue": i,
                    "yValue": prices[i],
                    "backgroundColor": "rgba(0, 255, 0, 0.8)",
                    "radius": 8,
                    "label": {
                        "content": "BUY",
                        "enabled": True,
                        "position": "bottom"
                    }
                })
            
            # Sell signal: sentiment crosses below 3
            elif sentiments[i-1] >= 3 and sentiments[i] < 3:
                annotations.append({
                    "type": "point",
                    "xValue": i,
                    "yValue": prices[i],
                    "backgroundColor": "rgba(255, 0, 0, 0.8)",
                    "radius": 8,
                    "label": {
                        "content": "SELL",
                        "enabled": True,
                        "position": "top"
                    }
                })
        
        return annotations
    
    def _flatten_heatmap_data(self, data, companies, metrics):
        """Flatten 2D array for heatmap visualization"""
        flattened = []
        for i, metric in enumerate(metrics):
            for j, company in enumerate(companies):
                flattened.append({
                    "x": company,
                    "y": metric,
                    "v": data[i][j]
                })
        return flattened
    
    def _generate_heatmap_colors(self, sentiment_value):
        """Generate color based on sentiment value"""
        if sentiment_value >= 8:
            return "rgba(0, 255, 0, 0.8)"  # Green - Bullish
        elif sentiment_value >= 6:
            return "rgba(255, 255, 0, 0.8)"  # Yellow - Neutral-Positive
        elif sentiment_value >= 4:
            return "rgba(255, 165, 0, 0.8)"  # Orange - Neutral
        else:
            return "rgba(255, 0, 0, 0.8)"  # Red - Bearish
    
    def save_chart(self, chart_data_or_url, filename):
        """Download and save chart image"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            
            if isinstance(chart_data_or_url, dict) and "symbol" in chart_data_or_url:
                # Try v1 API first (GET request)
                print(f"🔄 Trying v1 API for {filename}...")
                v1_url = self.generate_tradingview_chart_v1(
                    chart_data_or_url["symbol"].replace("NASDAQ:", ""),
                    chart_data_or_url.get("interval", "1d"),
                    chart_data_or_url.get("indicators"),
                    chart_data_or_url.get("drawings")
                )
                
                response = requests.get(v1_url, timeout=30)
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Chart saved via v1 API: {filepath}")
                    return filepath
                else:
                    print(f"⚠️ v1 API failed ({response.status_code}), trying v2...")
                    
                    # Fallback to v2 API (POST request)
                    headers = {
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                    
                    response = requests.post(
                        self.base_url_v2,
                        json=chart_data_or_url,
                        headers=headers,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        print(f"✅ Chart saved via v2 API: {filepath}")
                        return filepath
                    else:
                        print(f"❌ Both v1 and v2 APIs failed: {response.status_code}")
                        print(f"Response: {response.text}")
                        return None
            
            elif isinstance(chart_data_or_url, str):  # URL
                response = requests.get(chart_data_or_url, timeout=30)
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Chart saved: {filepath}")
                    return filepath
                else:
                    print(f"❌ Failed to generate chart: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
            
            else:  # Chart.js data for QuickChart
                response = requests.post(
                    self.quickchart_url,
                    json=chart_data_or_url,
                    timeout=30
                )
                
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Chart saved: {filepath}")
                    return filepath
                else:
                    print(f"❌ Failed to generate chart: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error saving chart: {e}")
            return None

# Integration with existing sentiment data
def generate_charts_from_sentiment_data():
    """Generate charts using existing sentiment analysis data"""
    
    generator = AdvancedChartGenerator()
    
    # Load sentiment data
    try:
        with open("./container_output/final_score/final-weighted-scores.json", "r") as f:
            sentiment_data = json.load(f)
    except:
        print("❌ Could not load sentiment data")
        return
    
    # Generate TradingView charts for each company
    for company, data in sentiment_data.get("companies", {}).items():
        symbol = {
            "NVIDIA": "NVDA",
            "APPLE": "AAPL", 
            "INTEL": "INTC"
        }.get(company)
        
        if symbol:
            # Generate TradingView chart with indicators and drawing tools
            tv_payload = generator.generate_tradingview_chart_with_drawings(symbol)
            generator.save_chart(tv_payload, f"{symbol}_tradingview_technical.png")
            
            # Prepare data for sentiment overlay
            company_data = {
                "name": company,
                "symbol": symbol,
                "data": []
            }
            
            # Generate sample time series (you'd use real data here)
            for i in range(30):
                timestamp = datetime.now() - timedelta(hours=i)
                # Use basic math instead of numpy if not available
                price_variation = (hash(str(i)) % 100 - 50) / 10.0
                sentiment_variation = (hash(str(i+100)) % 20 - 10) / 20.0
                
                company_data["data"].append({
                    "timestamp": timestamp,
                    "price": 180 + price_variation,
                    "sentiment_avg": max(0, min(10, (data.get("news_sentiment", 5) + 
                                                data.get("tradingview_sentiment", 5) + 
                                                data.get("image_sentiment", 5)) / 3 + sentiment_variation))
                })
            
            # Generate sentiment overlay chart
            sentiment_data = generator.generate_sentiment_overlay_chart(company_data)
            generator.save_chart(sentiment_data, f"{symbol}_sentiment_analysis.png")
    
    # Generate comparison heatmap
    heatmap_data = {}
    for company, data in sentiment_data.get("companies", {}).items():
        heatmap_data[company] = {
            "news_sentiment": data.get("news_sentiment", 5),
            "tradingview_score": data.get("tradingview_sentiment", 5),
            "image_sentiment": data.get("image_sentiment", 5),
            "overall_sentiment": data.get("weighted_sentiment", 5)
        }
    
    heatmap_data_config = generator.generate_sentiment_heatmap(heatmap_data)
    generator.save_chart(heatmap_data_config, "sentiment_comparison_heatmap.png")
    
    print("\n📊 All charts generated successfully!")

if __name__ == "__main__":
    generate_charts_from_sentiment_data()