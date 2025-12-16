#!/usr/bin/env python3
"""
AI Price Prediction Module using Grok API
Implements UCLA research findings on LLM-based price forecasting with chain-of-thought reasoning.

Research Foundation:
- ChatGPT/LLMs can predict next-day stock returns with statistical significance
- GPT-4 achieves ~90% hit rate for initial market reactions
- Chain-of-thought prompting improves financial reasoning
- Best results: news sentiment + technical indicators + price history

# ==============================================================================
# CHANGELOG:
# ==============================================================================
# [2025-12-12] FEATURE: Initial price predictor implementation
#              - Grok API integration matching sentiment_analyzer.py pattern
#              - Chain-of-thought prompting for financial analysis
#              - 30-minute caching to avoid excessive API calls
#              - Prediction horizon: 1-4 hours (swing trading optimized)
#              - Combines: price history, technical indicators, sentiment
# ==============================================================================
"""

import os
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from project root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# Configure logging
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, 'price_predictor.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)


class PricePredictor:
    """
    AI-powered price prediction using Grok API with chain-of-thought reasoning.

    Uses UCLA research methodology:
    - Combines sentiment, technicals, and price history
    - Chain-of-thought prompting for financial reasoning
    - Confidence scoring based on signal alignment
    - Caching to reduce API costs
    """

    def __init__(self, cache_duration_minutes: int = 30):
        """
        Initialize PricePredictor with Grok API (preferred) or OpenAI API (fallback)

        Args:
            cache_duration_minutes: How long to cache predictions (default: 30 minutes)
        """
        # Check for Grok API key first (preferred)
        self.grok_api_key = os.getenv('GROK_TRADE_API')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        # Initialize API client with priority: Grok > OpenAI
        if self.grok_api_key:
            # Use Grok API (preferred - real-time X/Twitter integration)
            self.client = OpenAI(
                api_key=self.grok_api_key,
                base_url="https://api.x.ai/v1"
            )
            self.model = "grok-3-fast"  # Fast model for predictions
            self.api_provider = "Grok"
            logger.info(f"🚀 Using GROK API for price prediction (real-time X/Twitter integration)")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   API Key: ...{self.grok_api_key[-4:]}")
        elif self.openai_api_key:
            # Fallback to OpenAI
            self.client = OpenAI(api_key=self.openai_api_key)
            self.model = "gpt-4o-mini"  # Cost-effective model
            self.api_provider = "OpenAI"
            logger.info(f"📡 Using OpenAI API for price prediction (Grok not configured)")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   API Key: ...{self.openai_api_key[-4:]}")
        else:
            # No API keys available
            self.client = None
            self.model = None
            self.api_provider = None
            logger.error("❌ No API keys found! Set GROK_TRADE_API or OPENAI_API_KEY in .env")

        # Cache configuration
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.prediction_cache = {}  # symbol -> {prediction, timestamp}

        logger.info(f"✅ PricePredictor initialized with {self.api_provider or 'NO'} API")
        logger.info(f"   Cache duration: {cache_duration_minutes} minutes")

    def _is_cache_valid(self, symbol: str) -> bool:
        """
        Check if cached prediction is still valid

        Args:
            symbol: Stock symbol (e.g., 'NVIDIA')

        Returns:
            True if cache is valid, False otherwise
        """
        if symbol not in self.prediction_cache:
            return False

        cache_entry = self.prediction_cache[symbol]
        cache_time = cache_entry.get('timestamp')

        if not cache_time:
            return False

        age = datetime.now() - cache_time
        is_valid = age < self.cache_duration

        if not is_valid:
            logger.debug(f"Cache expired for {symbol} (age: {age.total_seconds():.0f}s)")

        return is_valid

    def _format_price_history(self, price_history: List[float]) -> str:
        """
        Format price history for prompt

        Args:
            price_history: List of recent prices

        Returns:
            Formatted string for prompt
        """
        if not price_history or len(price_history) < 2:
            return "No price history available"

        # Calculate price changes
        prices_str = []
        for i, price in enumerate(price_history[-20:]):  # Last 20 prices
            if i > 0:
                change = price - price_history[i-1]
                change_pct = (change / price_history[i-1] * 100) if price_history[i-1] != 0 else 0
                prices_str.append(f"${price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
            else:
                prices_str.append(f"${price:.2f}")

        return ", ".join(prices_str)

    def _determine_trend(self, price_history: List[float], indicators: Dict) -> str:
        """
        Determine current trend from price history and indicators

        Args:
            price_history: List of recent prices
            indicators: Dictionary with technical indicators

        Returns:
            Trend string: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        # Check price momentum
        if len(price_history) >= 5:
            recent_avg = sum(price_history[-5:]) / 5
            older_avg = sum(price_history[-10:-5]) / 5 if len(price_history) >= 10 else recent_avg
            price_trend = "UP" if recent_avg > older_avg else "DOWN"
        else:
            price_trend = "NEUTRAL"

        # Check ADX for trend strength
        adx = indicators.get('adx', 0)
        di_plus = indicators.get('di_plus', 0)
        di_minus = indicators.get('di_minus', 0)

        # Strong trend if ADX >= 25
        if adx >= 25:
            if di_plus > di_minus:
                return "BULLISH"
            else:
                return "BEARISH"

        # Weak trend - use price momentum
        if price_trend == "UP":
            return "BULLISH"
        elif price_trend == "DOWN":
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _calculate_support_resistance(self, price_history: List[float], current_price: float) -> Tuple[float, float]:
        """
        Calculate simple support and resistance levels

        Args:
            price_history: List of recent prices
            current_price: Current market price

        Returns:
            Tuple of (support, resistance)
        """
        if not price_history or len(price_history) < 10:
            # Default to ±2% if insufficient data
            return (current_price * 0.98, current_price * 1.02)

        # Use recent high/low as resistance/support
        recent_prices = price_history[-20:]
        support = min(recent_prices)
        resistance = max(recent_prices)

        return (support, resistance)

    def get_price_prediction(
        self,
        symbol: str,
        current_price: float,
        price_history: List[float],
        indicators: Dict,
        sentiment: float,
        prediction_hours: int = 2,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Get AI price prediction using chain-of-thought reasoning

        Args:
            symbol: Stock symbol (e.g., 'NVIDIA', 'APPLE')
            current_price: Current market price
            price_history: List of recent prices (last 20 recommended)
            indicators: Dict with technical indicators (rsi, atr, adx, di_plus, di_minus)
            sentiment: Current sentiment score (1-10 scale)
            prediction_hours: Prediction horizon in hours (default: 2)
            max_retries: Maximum API call retries

        Returns:
            Dictionary with prediction data or None if failed:
            {
                'current_price': 176.50,
                'predicted_price': 178.25,
                'prediction_time': '2 hours',
                'direction': 'UP',
                'confidence': 0.72,
                'reasoning': '...',
                'cached': False,
                'timestamp': datetime object
            }
        """
        # Check API availability
        if not self.client:
            logger.error("No API client available for price prediction")
            return None

        # Check cache first
        if self._is_cache_valid(symbol):
            cached_prediction = self.prediction_cache[symbol]['prediction']
            logger.info(f"📦 Using cached prediction for {symbol} (age: {(datetime.now() - self.prediction_cache[symbol]['timestamp']).total_seconds():.0f}s)")
            cached_prediction['cached'] = True
            return cached_prediction

        # Extract indicators
        rsi = indicators.get('rsi', 50)
        atr = indicators.get('atr', 0)
        adx = indicators.get('adx', 0)
        di_plus = indicators.get('di_plus', 0)
        di_minus = indicators.get('di_minus', 0)

        # Determine trend
        trend = self._determine_trend(price_history, indicators)

        # Calculate support/resistance
        support, resistance = self._calculate_support_resistance(price_history, current_price)

        # Format price history
        price_history_str = self._format_price_history(price_history)

        # Interpret sentiment
        if sentiment >= 7.0:
            sentiment_text = "BULLISH"
        elif sentiment <= 4.0:
            sentiment_text = "BEARISH"
        else:
            sentiment_text = "NEUTRAL"

        # Create chain-of-thought prompt
        prompt = f"""You are a professional financial analyst. Predict the {symbol} stock price in {prediction_hours} hours using systematic analysis.

CURRENT DATA:
- Current Price: ${current_price:.2f}
- Recent Price History (last 20 points): {price_history_str}
- Technical Indicators:
  * RSI: {rsi:.1f} ({"Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"})
  * ATR: ${atr:.2f} (volatility measure)
  * ADX: {adx:.1f} ({"Strong Trend" if adx >= 25 else "Weak/No Trend"})
  * DI+: {di_plus:.1f}, DI-: {di_minus:.1f}
- Market Sentiment: {sentiment:.1f}/10 ({sentiment_text})
- Current Trend: {trend}
- Support Level: ${support:.2f}
- Resistance Level: ${resistance:.2f}

ANALYSIS FRAMEWORK (Chain-of-Thought):

1. TREND ANALYSIS:
   - Is the trend BULLISH, BEARISH, or NEUTRAL?
   - Is ADX >= 25 showing strong directional movement?
   - Are DI+ and DI- aligned with price action?

2. SUPPORT/RESISTANCE:
   - Is price near support (buy opportunity) or resistance (sell opportunity)?
   - What's the probability of breaking through current levels?

3. SENTIMENT INTEGRATION:
   - Does sentiment ({sentiment_text}) align with technical trend ({trend})?
   - High sentiment + bullish technicals = strong upside
   - Low sentiment + bearish technicals = strong downside
   - Misalignment = reduced confidence

4. VOLATILITY & RANGE:
   - ATR ${atr:.2f} suggests typical {prediction_hours}hr move of ${atr * prediction_hours / 24:.2f}
   - RSI {rsi:.1f} suggests {"limited upside" if rsi > 70 else "limited downside" if rsi < 30 else "room to move"}

5. PRICE TARGET CALCULATION:
   - Combine all factors to estimate probable price range
   - Consider momentum, sentiment, and technical levels
   - Provide single most likely price target

RESPOND IN THIS EXACT FORMAT:
PREDICTED_PRICE: $XXX.XX
DIRECTION: UP/DOWN/FLAT
CONFIDENCE: XX% (0-100)
REASONING: [One sentence explaining the key driver of this prediction]

Example:
PREDICTED_PRICE: $178.25
DIRECTION: UP
CONFIDENCE: 72%
REASONING: Strong bullish sentiment (8.0/10) aligns with uptrend (ADX 45), RSI not overbought (58), likely to test resistance.
"""

        logger.info(f"🔮 Requesting {prediction_hours}hr price prediction for {symbol} from {self.api_provider}...")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a financial analyst expert in technical analysis and price prediction. Provide structured predictions in the exact format requested."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.3  # Low temperature for consistent predictions
                )

                content = response.choices[0].message.content.strip()

                # Parse response
                prediction_data = self._parse_prediction_response(content, current_price, prediction_hours)

                if prediction_data:
                    # Add metadata
                    prediction_data['symbol'] = symbol
                    prediction_data['timestamp'] = datetime.now()
                    prediction_data['cached'] = False
                    prediction_data['api_provider'] = self.api_provider

                    # Cache the prediction
                    self.prediction_cache[symbol] = {
                        'prediction': prediction_data,
                        'timestamp': datetime.now()
                    }

                    logger.info(f"✅ Prediction for {symbol}: ${prediction_data['predicted_price']:.2f} ({prediction_data['direction']}) - Confidence: {prediction_data['confidence']:.0%}")

                    return prediction_data
                else:
                    logger.warning(f"Failed to parse prediction response (attempt {attempt + 1}/{max_retries})")
                    logger.debug(f"Raw response: {content}")

            except Exception as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting 30 seconds...")
                    time.sleep(30)
                    continue

                logger.error(f"API error during prediction (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return None

        logger.error(f"Failed to get price prediction for {symbol} after {max_retries} attempts")
        return None

    def _parse_prediction_response(self, content: str, current_price: float, prediction_hours: int) -> Optional[Dict]:
        """
        Parse LLM response into structured prediction data

        Args:
            content: Raw LLM response
            current_price: Current market price
            prediction_hours: Prediction horizon

        Returns:
            Parsed prediction dictionary or None
        """
        try:
            import re

            # Extract predicted price
            price_match = re.search(r'PREDICTED_PRICE:\s*\$?(\d+\.?\d*)', content, re.IGNORECASE)
            if not price_match:
                return None
            predicted_price = float(price_match.group(1))

            # Extract direction
            direction_match = re.search(r'DIRECTION:\s*(UP|DOWN|FLAT)', content, re.IGNORECASE)
            if not direction_match:
                # Infer from price
                if predicted_price > current_price:
                    direction = "UP"
                elif predicted_price < current_price:
                    direction = "DOWN"
                else:
                    direction = "FLAT"
            else:
                direction = direction_match.group(1).upper()

            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)%?', content, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1)) / 100.0
            else:
                # Default confidence based on price change magnitude
                price_change_pct = abs(predicted_price - current_price) / current_price
                confidence = max(0.5, min(0.9, 0.7 - price_change_pct * 2))  # Lower confidence for large moves

            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

            # Validate prediction
            if predicted_price <= 0:
                logger.warning(f"Invalid predicted price: ${predicted_price}")
                return None

            # Sanity check: Predicted price shouldn't be more than ±10% from current
            max_change = current_price * 0.10
            if abs(predicted_price - current_price) > max_change:
                logger.warning(f"Prediction too extreme: ${predicted_price:.2f} (current: ${current_price:.2f})")
                # Clamp to max change
                if predicted_price > current_price:
                    predicted_price = current_price + max_change
                else:
                    predicted_price = current_price - max_change
                confidence *= 0.5  # Reduce confidence for clamped predictions

            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': predicted_price - current_price,
                'price_change_pct': (predicted_price - current_price) / current_price,
                'prediction_time': f"{prediction_hours} hours",
                'prediction_hours': prediction_hours,
                'direction': direction,
                'confidence': confidence,
                'reasoning': reasoning
            }

        except Exception as e:
            logger.error(f"Error parsing prediction response: {e}")
            logger.debug(f"Content: {content}")
            return None

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear prediction cache

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            if symbol in self.prediction_cache:
                del self.prediction_cache[symbol]
                logger.info(f"Cleared prediction cache for {symbol}")
        else:
            self.prediction_cache.clear()
            logger.info("Cleared all prediction cache")


def main():
    """
    Test the price predictor with sample data
    """
    print("="*70)
    print("PRICE PREDICTOR TEST")
    print("="*70)

    # Initialize predictor
    predictor = PricePredictor(cache_duration_minutes=30)

    # Test with sample NVIDIA data
    symbol = "NVIDIA"
    current_price = 176.50

    # Sample price history (last 20 prices - simulated uptrend)
    price_history = [
        174.20, 174.35, 174.50, 174.65, 174.80,
        175.00, 175.15, 175.30, 175.50, 175.70,
        175.85, 176.00, 176.10, 176.20, 176.30,
        176.40, 176.45, 176.48, 176.49, 176.50
    ]

    # Sample technical indicators
    indicators = {
        'rsi': 58.5,
        'atr': 2.15,
        'adx': 45.2,
        'di_plus': 28.5,
        'di_minus': 18.3
    }

    # Sample sentiment (bullish)
    sentiment = 8.0

    print(f"\nTest Input:")
    print(f"  Symbol: {symbol}")
    print(f"  Current Price: ${current_price:.2f}")
    print(f"  RSI: {indicators['rsi']:.1f}")
    print(f"  ATR: ${indicators['atr']:.2f}")
    print(f"  ADX: {indicators['adx']:.1f}")
    print(f"  Sentiment: {sentiment:.1f}/10")
    print(f"  Price Trend: {price_history[-5:]} (last 5)")

    # Get prediction
    prediction = predictor.get_price_prediction(
        symbol=symbol,
        current_price=current_price,
        price_history=price_history,
        indicators=indicators,
        sentiment=sentiment,
        prediction_hours=2
    )

    if prediction:
        print("\n" + "="*70)
        print("PREDICTION RESULT")
        print("="*70)
        print(f"  Predicted Price ({prediction['prediction_time']}): ${prediction['predicted_price']:.2f}")
        print(f"  Direction: {prediction['direction']}")
        print(f"  Price Change: ${prediction['price_change']:.2f} ({prediction['price_change_pct']:+.2%})")
        print(f"  Confidence: {prediction['confidence']:.0%}")
        print(f"  Reasoning: {prediction['reasoning']}")
        print(f"  API Provider: {prediction['api_provider']}")
        print(f"  Cached: {prediction['cached']}")
        print("="*70)

        # Test cache
        print("\nTesting cache (should return instantly)...")
        cached_prediction = predictor.get_price_prediction(
            symbol=symbol,
            current_price=current_price,
            price_history=price_history,
            indicators=indicators,
            sentiment=sentiment,
            prediction_hours=2
        )

        if cached_prediction and cached_prediction['cached']:
            print("✅ Cache working correctly!")
        else:
            print("⚠️  Cache may not be working")
    else:
        print("\n❌ Prediction failed - check logs for details")


if __name__ == "__main__":
    main()
