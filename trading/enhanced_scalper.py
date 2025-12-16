#!/usr/bin/env python3
"""
Swing Trading System (1-4 Hour Holds)
Integrates WebSocket data, advanced strategies, and sophisticated position management
Target: 6-10 trades/day with 55%+ win rate and $3.00+ per trade

# ==============================================================================
# CHANGELOG
# ==============================================================================
# [2025-12-12] FEATURE: Price Predictor Integration (price_predictor_integration_001)
#   INTEGRATION: AI-powered price prediction using Grok/GPT-4 with chain-of-thought reasoning
#   METHODOLOGY: UCLA research - LLMs achieve ~90% hit rate for initial market reactions
#   COMPONENTS:
#     - PricePredictor class with 30-minute caching
#     - _get_price_prediction() helper method
#     - Signal confidence adjustment based on prediction alignment
#   IMPACT:
#     - Boost confidence +10% when prediction CONFIRMS signal direction
#     - Reduce confidence -15% when prediction CONFLICTS with signal
#     - Graceful degradation if prediction unavailable (no change)
#   EXPECTED IMPROVEMENT: 5-10% win rate increase through AI-enhanced signal filtering
#   VALIDATION:
#     1. Monitor logs for "🔮 Price prediction CONFIRMS/CONFLICTS" messages
#     2. Verify confidence adjustments appear in signal logs
#     3. Track win rate over next 20 trades for improvement
#     4. Confirm cache working (30-minute duration)
# [2025-12-12] FIX: Over-filtering rate limit (rate_limit_over_filter_001)
#   ISSUE: Static $0.50 threshold too high for NVDA volatility ($0.17-$0.35)
#   IMPACT: 454 rejections out of 456 signals (99.6% rejection rate), only 2 trades in 70+ minutes
#   ROOT CAUSE: Static threshold didn't adapt to market volatility regime
#   FIX: Dynamic ATR-based threshold (1.5× ATR) with $0.20 min, $1.00 max caps
#   EXPECTED IMPACT: Rejection rate drops from 99.6% to < 50%, system executes trades again
#   VALIDATION: Monitor logs for dynamic threshold values, verify trade execution improves
# [2025-12-11] FIX: TypeError - 'int'/'float' object is not subscriptable (subscript_error_001)
#   ISSUE: get_account_info() returned numeric values causing .get() calls to fail
#   ERROR LOG: "Error in scalping cycle: 'int' object is not subscriptable" (15:14:52)
#   ROOT CAUSE: capital_trader.py accounts[0] could be int/float, not validated before return
#   FIX 1: Added type checking in get_account_info() - wrap numeric values in dict
#   FIX 2: Added defensive isinstance() checks before .get() calls in enhanced_scalper.py
#   IMPACT: Prevents crashes during high-frequency scalping, graceful degradation with logging
#   VALIDATION: Monitor logs for "account_info type validation" warnings during next trading session
# [2025-12-10] FIX: Kill switch activation - counter-trend trading (trend_alignment_001)
#   ISSUE: 14 SHORT trades in NVDA uptrend, 62.5% loss rate, system terminated
#   ROOT CAUSE: ADX > 50 filter missed moderate trends (ADX 25-50), allowed counter-trend
#   FIX 1: Check trend direction for ALL ADX >= 25 (not just > 50)
#   FIX 2: Penalize counter-trend trades 80% (was 70%, only ADX > 50)
#   FIX 3: Raise momentum threshold 0.15 → 0.30 to filter noise
#   FIX 4: Increase position size 1.5% → 3.0% for better spread economics
#   IMPACT: Prevents shorting uptrends, reduces signal count ~50%, improves net profit
#   VALIDATION: Monitor next 20 trades for trend alignment, win rate > 50%
# [2025-12-09] FIX: Pre-emptive spread rejection filter (ISSUE_HASH: atr_sf_003)
#              Previous validation was AFTER signal generation, wasting compute
#              Now rejects BEFORE expensive strategy analysis if spread > 50% profit
# 2025-12-09 19:55:00 - FIX: Position-level spread validation (ISSUE_HASH: 8f4a9c2d)
#   ISSUE: 100% loss rate across 8 trades ($-0.27 total loss)
#   ROOT CAUSE: Spread cost ($0.06-0.11) exceeded net profit on fractional shares (0.5-0.6)
#   FIX: Added position-level net profit validation before signal execution
#   IMPACT: Prevents opening trades where spread cost > expected profit
#   VALIDATION: Monitor next 10 trades for net profit > $0.10 after spread
# ==============================================================================
"""

import asyncio
import time
import json
import os
import sys
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import threading
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our components
from trading.capital_trader import CapitalTrader
from trading.websocket_client import WebSocketDataProvider
from trading.scalping_strategies import ScalpingStrategyAggregator, DEFAULT_SCALPING_CONFIG
from trading.position_manager import PositionManager, RiskLimits
from trading.trade_logging.trade_logger import TradeDecisionLogger  # PHASE 4: Trade logging (2025-11-17)
from trading.price_predictor import PricePredictor  # PHASE 5: Price prediction integration (2025-12-12)

@dataclass
class ScalpingConfig:
    # ==============================================================================
    # FIX ATTEMPT [2025-12-10 16:10:00]
    # ==============================================================================
    # ISSUE: Spread cost $0.08 eats 25% of expected profit on 0.08 share positions
    # ISSUE_HASH: position_size_001
    # PREVIOUS ATTEMPTS: None (position size unchanged since system start)
    # LIANG WENFENG REASONING:
    #   1. Market Context: Spread $0.08, position 0.08 shares = $0.006 spread cost
    #   2. Signal Interpretation: Expected $0.03 profit - $0.006 spread = $0.024 net
    #      Margin too thin, any slippage = loss
    #   3. Alternative Evaluation: 3% position size = 0.16 shares = $0.012 spread cost
    #      but gross profit $0.06, net $0.048 (2x better)
    #   4. Risk Management: Larger positions = more risk, but better spread economics
    #   5. Reflection: Spread cost is FIXED per share, so bigger positions needed
    # SOLUTION: Increase position size from 1.5% to 3.0% for better spread economics
    # VALIDATION:
    #   1. Verify position sizes increase from 0.08 to 0.16 shares
    #   2. Verify net profit per trade improves from $0.02 to $0.05+
    #   3. Monitor risk (3% max loss = $30 on $1000 balance, acceptable)
    # ==============================================================================

    # Trading parameters
    symbols: List[str]
    max_concurrent_positions: int = 8
    max_daily_trades: int = 150
    max_daily_loss: float = 200.0
    position_size_pct: float = 0.030  # 3.0% per position (raised from 1.5%)
    
    # ==============================================================================
    # ARCHITECTURE CHANGE [2025-12-12] - SWING TRADING CONVERSION
    # ==============================================================================
    # ISSUE: System converted from intraday scalping to swing trading
    # ISSUE_HASH: swing_trading_conversion_001
    #
    # MATHEMATICAL CHECK:
    #   Win Rate: Target 55%+ - VIABLE for swing trading
    #   Spread Impact: $0.05/$3.00 = 1.67% - EXCELLENT (was 5%)
    #   Timeframe: 1-4 hours - VIABLE (retail-friendly zone)
    #
    # LIANG WENFENG REASONING:
    #   1. Market Context: 40-min scalping had 8.3% win rate (mathematically broken)
    #   2. Signal Interpretation: Trends need 1-4 hours to develop, not 40 minutes
    #   3. Alternative Evaluation: 1-hour viable (55% win rate), 4-hour good (52%)
    #   4. Risk Management: Larger stops ($0.50) but better R:R (1:3 vs 1:1)
    #   5. Reflection: Swing trading eliminates HFT competition, improves economics
    #
    # SOLUTION: Convert to 1-4 hour swing trading with $3.00 targets
    # VALIDATION:
    #   1. Verify positions hold 1-4 hours (not close at 40 min)
    #   2. Verify profit targets ~$3.00 per trade
    #   3. Monitor win rate improves to 55%+ over 20 trades
    #   4. Confirm spread impact drops to <2%
    # ==============================================================================

    # Performance targets
    # UPDATED 2025-12-12: Swing trading (1-4 hour holds)
    target_win_rate: float = 0.55  # 55% win rate required for swing economics
    target_profit_per_trade: float = 3.00  # $3.00 target (was $1.00 scalping)
    target_daily_profit: float = 15.0  # Fewer trades, bigger profits (6-10/day)

    # Risk controls
    max_consecutive_losses: int = 4
    max_hold_time: float = 14400.0  # 4 hours max hold (was 40 minutes)
    latency_threshold: float = 2000.0  # 2 seconds (less critical for swing)
    
    # Strategy weights
    enable_momentum: bool = True
    enable_mean_reversion: bool = True
    enable_orderbook: bool = True
    enable_spread: bool = True
    
    # Advanced features
    use_websocket: bool = True
    use_trailing_stops: bool = True
    dynamic_position_sizing: bool = True
    sentiment_bias_enabled: bool = True


class EnhancedScalper:
    """
    Enhanced scalping system that combines:
    - Real-time WebSocket price feeds
    - Multiple scalping strategies
    - Advanced position management
    - Sentiment-based directional bias
    - Sophisticated risk controls
    """
    
    def __init__(self, config: ScalpingConfig, env_file: str = ".env"):
        self.config = config
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Setup logging
        self._setup_logging()
        
        # Load trading credentials
        self._load_config(env_file)
        
        # Initialize core components
        self.trader = CapitalTrader(
            api_key=self.capital_api_key,
            password=self.capital_password,
            email=self.capital_email,
            demo='--live' not in sys.argv
        )
        
        # Initialize risk limits
        self.risk_limits = RiskLimits(
            max_daily_loss=config.max_daily_loss,
            max_concurrent_positions=config.max_concurrent_positions,
            max_position_size_pct=config.position_size_pct,
            max_symbol_positions=5,  # Increased from 3 to 5 for demo testing (2025-11-17)
            max_consecutive_losses=config.max_consecutive_losses,
            latency_threshold_ms=config.latency_threshold,
            spread_threshold_pct=0.4
        )
        
        # Initialize position manager
        self.position_manager = PositionManager(self.trader, self.risk_limits)
        
        # Initialize strategy aggregator
        self.strategy_config = self._create_strategy_config()
        self.strategy_aggregator = ScalpingStrategyAggregator(self.strategy_config)

        # PHASE 4: Initialize trade decision logger (2025-11-17)
        self.trade_logger = TradeDecisionLogger(base_dir=self.base_dir)
        logging.info("📝 Trade decision logger initialized")

        # Initialize WebSocket data provider (optional)
        self.websocket_provider = None
        if config.use_websocket:
            try:
                self.websocket_provider = WebSocketDataProvider(self.trader, config.symbols)
                logging.info("🌐 WebSocket data provider initialized")
            except Exception as e:
                logging.warning(f"⚠️ WebSocket initialization failed: {e}")
                logging.info("📡 Falling back to REST API for market data")
        
        # Performance tracking
        self.session_start_time = time.time()
        self.total_cycles = 0
        self.signal_count = 0
        self.trades_executed = 0
        self.last_performance_log = time.time()
        
        # Sentiment integration
        self.sentiment_scores = {}
        self.last_sentiment_update = 0
        self.load_sentiment_scores()

        # ==============================================================================
        # FEATURE INTEGRATION [2025-12-12 16:00:00]
        # ==============================================================================
        # FEATURE: Price Predictor Integration (price_predictor_integration_001)
        # PURPOSE: Add AI-powered price prediction to enhance signal confidence
        # METHODOLOGY: UCLA research - LLMs achieve ~90% initial market reaction accuracy
        # INTEGRATION:
        #   - Grok API with chain-of-thought reasoning
        #   - 30-minute cache to reduce API costs
        #   - Prediction horizon: 2 hours (swing trading optimized)
        #   - Combines: price history, technical indicators, sentiment
        # IMPACT:
        #   - Boost confidence +10% when prediction confirms signal direction
        #   - Reduce confidence -15% when prediction conflicts with signal
        #   - No change if prediction unavailable (graceful degradation)
        # ==============================================================================
        # Price prediction integration
        self.price_predictor = PricePredictor(cache_duration_minutes=30)
        logging.info("🔮 Price Predictor initialized")

        # ==============================================================================
        # FIX ATTEMPT [2025-12-12 15:30:00]
        # ==============================================================================
        # ISSUE: Rate limit threshold $0.50 too high for NVDA volatility ($0.17-$0.35)
        # ISSUE_HASH: rate_limit_over_filter_001
        # PREVIOUS ATTEMPTS: Static $0.50 (2025-12-11) caused 99.6% rejection rate
        #
        # MATHEMATICAL CHECK:
        #   Win Rate: 8.3% - NOT VIABLE (caused by over-filtering, not strategy failure)
        #   Spread Impact: 5% - ACCEPTABLE ($0.05 spread vs $1.00 target)
        #   Timeframe: 40-minute - VIABLE for retail
        #   ROOT CAUSE: 454 rejections out of 456 signals = EXECUTION BLOCKED
        #
        # LIANG WENFENG REASONING:
        #   1. Market Context: NVDA volatility $0.17-$0.35, static $0.50 threshold blocks all trades
        #   2. Signal Interpretation: Strategies generate 456 signals but 99.6% rejected by rate limit
        #   3. Alternative Evaluation: Dynamic ATR-based threshold (1.5× ATR) adapts to regime
        #   4. Risk Management: Caps at $0.20 min (prevent zero trading) and $1.00 max (prevent spam)
        #   5. Reflection: Static thresholds fail in variable volatility - must be adaptive
        #
        # SOLUTION: Dynamic rate limit based on current ATR (1.5× multiplier)
        #           - Low volatility (ATR $0.15): threshold = $0.225 → allows trading
        #           - High volatility (ATR $1.00): threshold = $1.00 (capped) → prevents spam
        #           - Recalculates every 60s to adapt to changing market conditions
        #
        # VALIDATION:
        #   1. Verify min_price_difference logs show dynamic values ($0.20-$1.00 range)
        #   2. Verify rejection rate drops from 99.6% to < 50%
        #   3. Monitor next 20 trades for execution improvement
        #   4. Confirm ATR-based threshold prevents over-trading in high volatility
        # ==============================================================================

        # SWING TRADING: Signal rate limiting (ISSUE_HASH: swing_trading_conversion_001)
        # UPDATED 2025-12-12: 1-hour cooldown for swing trading (was 10-min scalping)
        self.last_signal_time = {}  # Track last signal time per symbol
        self.last_signal_price = {}  # Track last signal price per symbol
        self.min_entry_interval = 3600  # 1 hour between signals (swing trading)
        self.min_price_difference = 0.20  # Will be updated dynamically based on ATR
        self.last_rate_limit_update = 0  # Track when we last updated dynamic thresholds

        # Update dynamic rate limits on startup
        self._update_dynamic_rate_limits()

        logging.info(f"🚦 Swing trading rate limits: {self.min_entry_interval}s cooldown (1 hour), ${self.min_price_difference:.2f} min price diff (dynamic ATR-based)")

        # Market session tracking
        self.trading_session_active = False
        self.last_session_check = 0
        
        # Emergency controls
        self.kill_switch_active = False
        self.emergency_reason = ""

        logging.info("🚀 Swing Trading System initialized (1-4 hour holds)")
        logging.info(f"   Symbols: {', '.join(config.symbols)}")
        logging.info(f"   Max positions: {config.max_concurrent_positions}")
        logging.info(f"   Target win rate: {config.target_win_rate*100:.1f}%")
        logging.info(f"   Target profit/trade: ${config.target_profit_per_trade:.2f}")
        logging.info(f"   Max hold time: {config.max_hold_time/3600:.1f} hours")
        logging.info(f"   WebSocket: {'✅ Enabled' if self.websocket_provider else '❌ Disabled'}")
    
    def _setup_logging(self):
        """Setup advanced logging system"""
        logs_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"enhanced_scalper_{timestamp}.log")
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler (detailed)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (simple)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Configure logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.handlers = []  # Clear existing handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logging.info(f"📝 Logging initialized: {log_file}")
    
    def _load_config(self, env_file: str):
        """Load configuration from environment file"""
        env_path = os.path.join(self.base_dir, env_file)
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        value = value.split('#')[0].strip()
                        
                        if key == 'CAPITAL_API_KEY':
                            self.capital_api_key = value
                        elif key == 'CAPITAL_PASSWORD':
                            self.capital_password = value
                        elif key == 'CAPITAL_EMAIL':
                            self.capital_email = value
        else:
            raise FileNotFoundError(f"Configuration file not found: {env_path}")
    
    def _create_strategy_config(self) -> Dict:
        """Create strategy configuration based on settings"""
        config = DEFAULT_SCALPING_CONFIG.copy()
        
        # Adjust weights based on enabled strategies
        weights = {}
        if self.config.enable_momentum:
            weights['momentum'] = 0.35
        if self.config.enable_mean_reversion:
            weights['mean_reversion'] = 0.25
        if self.config.enable_orderbook:
            weights['orderbook'] = 0.25
        if self.config.enable_spread:
            weights['spread'] = 0.15
        
        # Normalize weights
        if weights:
            total_weight = sum(weights.values())
            config['weights'] = {k: v/total_weight for k, v in weights.items()}
        
        # CRITICAL: Take profit must be > spread cost (~0.05% for NVDA)
        # UPDATED 2025-12-11: Adjusted for 40-minute timeframe (was 0.22%/0.20%)
        config['momentum']['momentum_period'] = 25
        config['momentum']['take_profit_pct'] = 0.55  # ~$1.00 on $185 stock (10x spread)
        config['mean_reversion']['take_profit_pct'] = 0.55  # ~$1.00 on $185 stock

        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 21:45:00]
        # ==============================================================================
        # ISSUE: Confidence threshold mismatch (0.6 vs 0.50 per CLAUDE.md)
        # ISSUE_HASH: confidence_config_002
        # SOLUTION: Standardize confidence threshold to 0.50
        # ==============================================================================
        config['min_combined_confidence'] = 0.50  # Was 0.6, standardized per CLAUDE.md
        
        return config
    
    def _update_dynamic_rate_limits(self):
        """
        Update dynamic rate limit thresholds based on current ATR

        ISSUE_HASH: rate_limit_over_filter_001
        Adapts min_price_difference to market volatility using ATR
        """
        try:
            from trading.indicator_utils import IndicatorCalculator
            indicator_calc = IndicatorCalculator()

            # Get ATR for primary symbol (NVDA)
            primary_symbol = self.config.symbols[0] if self.config.symbols else "NVDA"
            atr_info = indicator_calc.get_atr_info(primary_symbol)

            if atr_info and atr_info.get('atr'):
                current_atr = atr_info['atr']

                # Calculate dynamic threshold: 1.5× ATR (covers normal price noise)
                dynamic_threshold = 1.5 * current_atr

                # Apply safety caps
                MIN_THRESHOLD = 0.20  # Prevent zero trading in ultra-low volatility
                MAX_THRESHOLD = 1.00  # Prevent spam in extreme volatility

                self.min_price_difference = max(MIN_THRESHOLD, min(dynamic_threshold, MAX_THRESHOLD))

                logging.info(f"📊 Dynamic rate limit updated: ATR=${current_atr:.3f} → threshold=${self.min_price_difference:.2f}")
                logging.info(f"   Volatility: {atr_info.get('volatility', 'Unknown')}")
            else:
                # Fallback to conservative minimum if ATR unavailable
                self.min_price_difference = 0.20
                logging.warning(f"⚠️ ATR unavailable, using fallback threshold ${self.min_price_difference:.2f}")

        except Exception as e:
            logging.warning(f"⚠️ Could not update dynamic rate limits: {e}, using fallback")
            self.min_price_difference = 0.20

    def load_sentiment_scores(self):
        """Load current sentiment scores for directional bias"""
        try:
            final_score_dir = os.path.join(self.base_dir, "container_output", "final_score")
            weighted_file = os.path.join(final_score_dir, "final-weighted-scores.json")

            logging.info(f"🔄 Loading sentiment from {weighted_file}")

            if os.path.exists(weighted_file):
                with open(weighted_file, 'r') as f:
                    data = json.load(f)

                companies_data = data.get("companies", data)

                # Map company names to symbols
                symbol_mapping = {
                    "NVIDIA": "NVDA",
                    "APPLE": "AAPL",
                    "MICROSOFT": "MSFT",
                    "GOOGLE": "GOOG",
                    "AMAZON": "AMZN",
                    "TESLA": "TSLA",
                    "INTEL": "INTC"
                }

                # Log before update
                logging.info(f"📊 Old sentiment scores: {self.sentiment_scores}")

                # Clear and update scores
                new_scores = {}
                for company, symbol in symbol_mapping.items():
                    if company in companies_data and symbol in self.config.symbols:
                        score = companies_data[company].get("final_score", 5.0)
                        new_scores[symbol] = score

                # Replace old scores with new ones
                self.sentiment_scores.clear()
                self.sentiment_scores.update(new_scores)

                # Log after update
                logging.info(f"✅ New sentiment scores: {self.sentiment_scores}")

                self.last_sentiment_update = time.time()
            else:
                logging.warning(f"⚠️ Sentiment file not found: {weighted_file}")

        except Exception as e:
            logging.warning(f"⚠️ Could not load sentiment: {e}")
            # Default neutral sentiment
            for symbol in self.config.symbols:
                self.sentiment_scores[symbol] = 5.0

    def _get_price_prediction(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Dict]:
        """
        Get AI-powered price prediction for signal enhancement

        INTEGRATION: price_predictor_integration_001
        PURPOSE: Enhance signal confidence using LLM-based price forecasting

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            current_price: Current market price
            market_data: Market data dict with bid/ask/spread

        Returns:
            Prediction dict or None if unavailable:
            {
                'predicted_price': 178.25,
                'direction': 'UP' or 'DOWN',
                'confidence': 0.72,
                'reasoning': '...'
            }
        """
        try:
            # Collect price history from strategy aggregator
            price_history = []
            if hasattr(self.strategy_aggregator, 'momentum') and hasattr(self.strategy_aggregator.momentum, 'price_history'):
                price_history = list(self.strategy_aggregator.momentum.price_history)[-20:]  # Last 20 prices

            # Fallback if no price history available
            if len(price_history) < 5:
                logging.debug(f"🔮 Insufficient price history for prediction ({len(price_history)} points)")
                return None

            # Get technical indicators from indicator_utils
            from trading.indicator_utils import IndicatorCalculator
            indicator_calc = IndicatorCalculator()

            # Collect indicators
            indicators = {}

            # RSI
            # FIX ATTEMPT [2025-12-15] - ISSUE_HASH: price_predictor_rsi_type_001
            # calculate_rsi() returns Optional[float], not dict
            # Changed from rsi_data.get('rsi', 50) to direct float usage
            rsi_value = indicator_calc.calculate_rsi(symbol)
            if rsi_value is not None:
                indicators['rsi'] = rsi_value
            else:
                indicators['rsi'] = 50  # Neutral default

            # ATR
            atr_info = indicator_calc.get_atr_info(symbol)
            if atr_info:
                indicators['atr'] = atr_info.get('atr', 0.5)
            else:
                indicators['atr'] = 0.5  # Conservative default

            # ADX and DI+/DI-
            adx = indicator_calc.calculate_adx(symbol)
            if adx:
                indicators['adx'] = adx
                # Get DI+ and DI- if available
                trend_info = indicator_calc.get_trend_direction(symbol)
                if trend_info:
                    # Estimate DI+/DI- from trend (actual values not returned by get_trend_direction)
                    if trend_info == 'BULLISH':
                        indicators['di_plus'] = 30.0
                        indicators['di_minus'] = 15.0
                    elif trend_info == 'BEARISH':
                        indicators['di_plus'] = 15.0
                        indicators['di_minus'] = 30.0
                    else:
                        indicators['di_plus'] = 20.0
                        indicators['di_minus'] = 20.0
                else:
                    indicators['di_plus'] = 20.0
                    indicators['di_minus'] = 20.0
            else:
                indicators['adx'] = 20  # Weak trend default
                indicators['di_plus'] = 20.0
                indicators['di_minus'] = 20.0

            # Get current sentiment score
            sentiment = self.sentiment_scores.get(symbol, 5.0)

            # Call price predictor
            prediction = self.price_predictor.get_price_prediction(
                symbol=symbol,
                current_price=current_price,
                price_history=price_history,
                indicators=indicators,
                sentiment=sentiment,
                prediction_hours=2  # 2-hour prediction for swing trading
            )

            return prediction

        except Exception as e:
            logging.warning(f"⚠️ Price prediction failed for {symbol}: {e}")
            return None

    async def start_websocket_feed(self):
        """Start WebSocket real-time data feed"""
        if self.websocket_provider:
            try:
                await self.websocket_provider.start()
                logging.info("🌐 WebSocket data feed started")
                return True
            except Exception as e:
                logging.error(f"❌ WebSocket startup failed: {e}")
                self.websocket_provider = None
                return False
        return False

    async def preload_price_history(self):
        """
        Pre-load price history from Capital.com API to avoid cold start
        Fixes: Mean reversion strategy requires 100 ticks before activation
        """
        logging.info("📊 Pre-loading price history for strategies...")

        for symbol in self.config.symbols:
            try:
                # Fetch 100 recent price points
                historical_prices = await self.trader.get_historical_prices_async(
                    symbol=symbol,
                    resolution='MINUTE',
                    max_points=100
                )

                if historical_prices and len(historical_prices) > 0:
                    # Populate price history for all strategies
                    for price in historical_prices:
                        tick_data = {'mid': price, 'symbol': symbol}

                        # Update momentum strategy
                        if hasattr(self.strategy_aggregator, 'momentum'):
                            self.strategy_aggregator.momentum.update_data(tick_data)

                        # Update mean reversion strategy
                        if hasattr(self.strategy_aggregator, 'mean_reversion'):
                            self.strategy_aggregator.mean_reversion.update_data(tick_data)

                    logging.info(f"✅ Pre-loaded {len(historical_prices)} prices for {symbol}")
                    logging.info(f"   Momentum history: {len(self.strategy_aggregator.momentum.price_history)} ticks")
                    logging.info(f"   Mean reversion history: {len(self.strategy_aggregator.mean_reversion.price_history)} ticks")
                else:
                    logging.warning(f"⚠️ No historical prices available for {symbol}, will accumulate live")

            except Exception as e:
                logging.warning(f"⚠️ Failed to pre-load history for {symbol}: {e}")
                logging.info(f"   Strategies will accumulate price history from live data")

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time market data (WebSocket preferred, REST fallback)"""
        
        # Try WebSocket first
        if self.websocket_provider and self.websocket_provider.is_connected():
            price_data = self.websocket_provider.get_current_price(symbol)
            if price_data:
                return {
                    'symbol': symbol,
                    'bid': price_data['bid'],
                    'ask': price_data['ask'],
                    'mid': price_data['mid'],
                    'spread': price_data['spread'],
                    'timestamp': price_data['timestamp']
                }
        
        # Fallback to REST API
        try:
            if not self.trader.cst:
                self.trader.create_session()
            
            market_info = self.trader.get_market_info(symbol)
            
            if market_info and 'snapshot' in market_info:
                snapshot = market_info['snapshot']
                bid = float(snapshot.get('bid', 0))
                offer = float(snapshot.get('offer', 0))
                
                return {
                    'symbol': symbol,
                    'bid': bid,
                    'ask': offer,
                    'mid': (bid + offer) / 2 if bid and offer else 0,
                    'spread': offer - bid if bid and offer else 0,
                    'timestamp': time.time()
                }
        
        except Exception as e:
            logging.error(f"❌ Error getting market data for {symbol}: {e}")
        
        return None
    
    def check_trading_session(self) -> bool:
        """Check if market is open using Capital.com API market status
        Uses the official marketStatus field from API for accurate status
        Falls back to time-based check if API is unavailable
        """
        now = time.time()

        # Check every 30 seconds to avoid excessive API calls
        if now - self.last_session_check < 30:
            return self.trading_session_active

        self.last_session_check = now

        try:
            # Create session if needed (sessions expire after 10 min)
            self.trader.create_session()

            # Get market status from Capital.com API for primary symbol
            primary_symbol = self.config.symbols[0] if self.config.symbols else "NVDA"
            market_info = self.trader.get_market_info(primary_symbol)

            if market_info and 'snapshot' in market_info:
                market_status = market_info['snapshot'].get('marketStatus', 'UNKNOWN')

                # Capital.com returns: "TRADEABLE", "CLOSED", "OFFLINE", etc.
                # Works for both DEMO and LIVE modes
                is_tradeable = market_status in ['TRADEABLE', 'EDITS_ONLY']

                self.trading_session_active = is_tradeable

                if not is_tradeable:
                    logging.debug(f"📊 Market status for {primary_symbol}: {market_status} (Demo: {self.trader.demo})")

                return is_tradeable
            else:
                logging.warning("⚠️ Could not get market status from API, using fallback time check")
                # Fallback to time-based check if API fails
                return self._fallback_time_check()

        except Exception as e:
            logging.debug(f"⚠️ API market check failed ({e}), using fallback time check")
            # Fallback to time-based check on error (silent fallback)
            return self._fallback_time_check()

    def _fallback_time_check(self) -> bool:
        """Fallback time-based market check if API fails"""
        from datetime import datetime, timezone, timedelta

        utc_now = datetime.now(timezone.utc)
        est_now = utc_now - timedelta(hours=5)  # EST = UTC-5

        weekday = est_now.weekday()
        hour = est_now.hour

        # Weekend check
        if weekday >= 5:
            self.trading_session_active = False
            return False

        # NYSE Extended Hours: 4:00 AM - 8:00 PM EST
        if 4 <= hour < 20:
            self.trading_session_active = True
            return True

        self.trading_session_active = False
        return False
    
    async def scalping_cycle(self):
        """Execute one complete scalping cycle"""
        try:
            # Update sentiment periodically
            if time.time() - self.last_sentiment_update > 60:  # Every minute
                self.load_sentiment_scores()

            # Update dynamic rate limits periodically (ISSUE_HASH: rate_limit_over_filter_001)
            if time.time() - self.last_rate_limit_update > 60:  # Every minute
                self._update_dynamic_rate_limits()
                self.last_rate_limit_update = time.time()
            
            # Process each symbol
            for symbol in self.config.symbols:
                
                # Skip if kill switch is active
                if self.kill_switch_active:
                    break
                
                # Get real-time market data
                market_data = await self.get_market_data(symbol)

                if not market_data:
                    logging.debug(f"⚠️ No market data for {symbol}")
                    continue

                # DEBUG: Log market data periodically
                if self.total_cycles % 30 == 0:  # Every 30 cycles (~24s)
                    logging.info(f"🔍 Market data for {symbol}: bid={market_data.get('bid')}, ask={market_data.get('ask')}, mid={market_data.get('mid')}")

                # CRITICAL FIX: Rate limiting check (2025-11-18)
                current_price = market_data.get('mid', 0)
                current_time = time.time()

                # Check time-based cooldown
                if symbol in self.last_signal_time:
                    time_since_last = current_time - self.last_signal_time[symbol]
                    if time_since_last < self.min_entry_interval:
                        logging.debug(f"🚦 Rate limit: {symbol} cooldown ({time_since_last:.1f}s < {self.min_entry_interval}s)")
                        continue

                # Check price movement threshold
                if symbol in self.last_signal_price:
                    price_diff = abs(current_price - self.last_signal_price[symbol])
                    if price_diff < self.min_price_difference:
                        logging.debug(f"🚦 Rate limit: {symbol} insufficient price movement (${price_diff:.2f} < ${self.min_price_difference})")
                        continue

                # Get sentiment bias for this symbol
                sentiment_score = self.sentiment_scores.get(symbol, 5.0)

                # Get combined signal from all strategies
                signal = self.strategy_aggregator.get_combined_signal(market_data, sentiment_score)

                # DEBUG: Log why no signal was generated
                if not signal and self.total_cycles % 60 == 0:  # Every 60 cycles (~48s)
                    logging.info(f"🔍 No signal for {symbol} | Sentiment: {sentiment_score}/10")
                
                if signal:
                    # ==============================================================================
                    # FIX ATTEMPT [2025-12-09 14:45:00]
                    # ==============================================================================
                    # ISSUE: Generating signals that will be rejected anyway, wasting compute
                    # ISSUE_HASH: atr_sf_003
                    # PREVIOUS ATTEMPTS: Position-level validation (8f4a9c2d) but AFTER signal
                    # LIANG WENFENG REASONING:
                    #   1. Market Context: Spread varies $0.06-0.11, some periods have wide spreads
                    #   2. Signal Interpretation: If spread > 50% of profit, guaranteed loss
                    #   3. Alternative Evaluation: Pre-filter saves compute vs post-filter
                    #   4. Risk Management: 50% threshold allows some spread tolerance
                    #   5. Reflection: Catch bad trades early before expensive validations
                    # SOLUTION: Reject signals where spread > 50% of expected profit IMMEDIATELY
                    # VALIDATION: Check logs for "SPREAD FILTER: Rejecting" messages
                    # ==============================================================================

                    # SPREAD REJECTION FILTER: Skip when spread eats too much profit
                    current_spread = market_data.get('spread', 0.10)
                    expected_profit_per_share = abs(signal.take_profit - signal.entry_price)

                    spread_ratio = current_spread / expected_profit_per_share if expected_profit_per_share > 0 else 1.0

                    if spread_ratio > 0.50:  # Spread > 50% of expected profit
                        logging.warning(f"❌ SPREAD FILTER: Rejecting {symbol} - spread ${current_spread:.3f} is {spread_ratio*100:.1f}% of profit ${expected_profit_per_share:.3f}")
                        signal = None
                        continue

                if signal:
                    # ==============================================================================
                    # FIX ATTEMPT [2025-12-09 19:55:00]
                    # ==============================================================================
                    # ISSUE: 100% loss rate - spread cost ($0.06) exceeds net profit on fractional shares
                    # ISSUE_HASH: 8f4a9c2d
                    # PREVIOUS ATTEMPTS: None
                    # LIANG WENFENG REASONING:
                    #   1. Market Context: Low volatility (NVDA $184-185 range), spread $0.11 (0.06%)
                    #   2. Signal Interpretation: Signals validated per-share ($0.41 target) but executed
                    #      with fractional shares (0.5-0.6), making actual profit $0.24, spread cost $0.06
                    #   3. Alternative Evaluation: Old validation compared per-share metrics, not accounting
                    #      for position-level spread cost impact on small positions
                    #   4. Risk Management: R:R appeared 1:2.7 but actual was 1:1.2 after spread, with
                    #      position sizing (1.5% = 0.08 shares) amplifying spread impact
                    #   5. Reflection: Fractional share trading + high spread percentage + low volatility
                    #      = mathematically impossible to profit. Need position-level validation.
                    # SOLUTION: Calculate net profit at position level (quantity × price move - spread cost)
                    #           before opening trade. Reject if net profit ≤ 0 or ≤ minimum threshold.
                    # VALIDATION:
                    #   1. Check logs for "NET PROFIT VALIDATION" showing calculations
                    #   2. Verify rejected signals show negative/insufficient net profit
                    #   3. Verify accepted signals have net profit > $0.10 after spread
                    #   4. Monitor next 10 trades for positive net P&L
                    # ==============================================================================

                    # CRITICAL: Position-level spread validation (accounts for fractional shares)
                    # Calculate actual position size and net profit after spread cost
                    current_spread = market_data.get('spread', 0.10)

                    # Estimate actual quantity (same calculation as position_manager)
                    try:
                        account_info = self.trader.get_account_info()

                        # CRITICAL FIX: Defensive type checking (subscript_error_001)
                        # Handle case where account_info might not be a dict
                        if isinstance(account_info, dict):
                            balance = float(account_info.get('balance', 1000))
                        elif isinstance(account_info, (int, float)):
                            # API returned numeric value directly
                            balance = float(account_info)
                            logging.debug(f"📊 Account info is numeric: {balance}")
                        else:
                            # Unexpected type, use conservative fallback
                            balance = 1000.0
                            logging.warning(f"⚠️ Unexpected account_info type: {type(account_info)}, using fallback")

                        position_value = balance * self.config.position_size_pct
                        estimated_quantity = position_value / signal.entry_price
                    except Exception as e:
                        logging.warning(f"⚠️ Could not get account balance for validation: {e}")
                        estimated_quantity = 0.5  # Conservative fallback

                    # Calculate position-level metrics
                    expected_profit_per_share = abs(signal.take_profit - signal.entry_price)
                    gross_profit_position = expected_profit_per_share * estimated_quantity
                    spread_cost_position = current_spread * estimated_quantity
                    net_profit_position = gross_profit_position - spread_cost_position

                    # Minimum net profit threshold: $0.10 per trade
                    min_net_profit = 0.10

                    logging.info(f"💰 NET PROFIT VALIDATION for {symbol}:")
                    logging.info(f"   Quantity: {estimated_quantity:.3f} shares")
                    logging.info(f"   Profit/share: ${expected_profit_per_share:.3f}")
                    logging.info(f"   Gross profit: ${gross_profit_position:.3f}")
                    logging.info(f"   Spread cost: ${spread_cost_position:.3f}")
                    logging.info(f"   Net profit: ${net_profit_position:.3f}")
                    logging.info(f"   Min required: ${min_net_profit:.2f}")

                    if net_profit_position < min_net_profit:
                        logging.info(f"❌ Signal REJECTED: Net profit ${net_profit_position:.3f} < min ${min_net_profit:.2f} (after spread)")
                        signal = None  # Clear the signal

                # ==============================================================================
                # FIX ATTEMPT [2025-12-11 20:15:00]
                # ==============================================================================
                # ISSUE: Counter-trend trades executing with wrong thresholds (5.0/6.0)
                # ISSUE_HASH: counter_trend_threshold_002
                # PREVIOUS ATTEMPTS:
                #   - counter_trend_validation_001 (2025-12-11 17:45) - Wrong thresholds
                #   - trend_alignment_001 (2025-12-10) - Did not block at execution
                #
                # MATHEMATICAL CHECK:
                #   Win Rate: 8.3% - NOT VIABLE (need > 50%)
                #   ROOT CAUSE: 61.5% counter-trend trades destroying win rate
                #
                # LIANG WENFENG REASONING:
                #   1. Market Context: Sentiment scale 1-10, BUY threshold 7.0, SELL threshold 4.0
                #   2. Signal Interpretation: LONG with sentiment=6.0 is AGAINST bias
                #   3. Alternative: Strict 7.0/4.0 matches signal generation (CHOSEN)
                #   4. Risk Management: Counter-trend trades have negative edge
                #   5. Reflection: Thresholds must match signal generation logic exactly
                #
                # SOLUTION: Match thresholds to signal generation rules:
                #   - LONG: Only allowed if sentiment >= 7.0 (BULLISH)
                #   - SHORT: Only allowed if sentiment <= 4.0 (BEARISH)
                #   - Neutral zone (4.0-7.0): NO directional trades allowed
                # ==============================================================================
                if signal and sentiment_score is not None:
                    if signal.direction == 'LONG' and sentiment_score < 7.0:
                        logging.warning(f"🚫 COUNTER-TREND BLOCKED: LONG rejected - sentiment {sentiment_score:.1f}/10 < 7.0 required")
                        signal = None
                    elif signal.direction == 'SHORT' and sentiment_score > 4.0:
                        logging.warning(f"🚫 COUNTER-TREND BLOCKED: SHORT rejected - sentiment {sentiment_score:.1f}/10 > 4.0 max")
                        signal = None

                # ==============================================================================
                # FEATURE INTEGRATION [2025-12-12 16:00:00]
                # ==============================================================================
                # INTEGRATION: Price Predictor Enhancement (price_predictor_integration_001)
                # PURPOSE: Use AI price prediction to adjust signal confidence
                # METHODOLOGY:
                #   - Get 2-hour price prediction from Grok/GPT-4
                #   - If prediction CONFIRMS signal direction → boost confidence +10%
                #   - If prediction CONFLICTS with signal → reduce confidence -15%
                #   - If prediction unavailable → no change (graceful degradation)
                # IMPACT:
                #   - Filters weak signals that AI predicts will fail
                #   - Boosts strong signals with AI confirmation
                #   - Expected improvement: 5-10% win rate increase
                # ==============================================================================
                if signal:
                    # Get price prediction before trade execution
                    prediction = self._get_price_prediction(symbol, current_price, market_data)

                    if prediction:
                        pred_direction = prediction['direction']  # 'UP' or 'DOWN'
                        signal_direction = 'UP' if signal.direction == 'LONG' else 'DOWN'
                        original_confidence = signal.confidence

                        if pred_direction == signal_direction:
                            # Prediction confirms signal - boost confidence
                            signal.confidence *= 1.10  # +10% boost
                            logging.info(f"🔮 Price prediction CONFIRMS {signal.direction} signal (+10% confidence)")
                            logging.info(f"   Predicted: {pred_direction} to ${prediction['predicted_price']:.2f} ({prediction['confidence']*100:.0f}% confident)")
                            logging.info(f"   Confidence: {original_confidence:.3f} → {signal.confidence:.3f}")
                            logging.info(f"   Reasoning: {prediction.get('reasoning', 'N/A')}")
                        else:
                            # Prediction conflicts with signal - reduce confidence
                            signal.confidence *= 0.85  # -15% reduction
                            logging.warning(f"⚠️ Price prediction CONFLICTS with {signal.direction} signal (-15% confidence)")
                            logging.warning(f"   Predicted: {pred_direction} to ${prediction['predicted_price']:.2f} (vs signal: {signal_direction})")
                            logging.warning(f"   Confidence: {original_confidence:.3f} → {signal.confidence:.3f}")
                            logging.warning(f"   Reasoning: {prediction.get('reasoning', 'N/A')}")
                    else:
                        logging.debug(f"🔮 No price prediction available for {symbol} (proceeding without enhancement)")

                if signal:
                    self.signal_count += 1
                    logging.info(f"📡 Signal: {signal.direction} {symbol} @ ${signal.entry_price:.2f}")
                    logging.info(f"   Strategy: {signal.strategy}")
                    logging.info(f"   Confidence: {signal.confidence:.2f}")
                    logging.info(f"   Spread: ${market_data.get('spread', 0.10):.3f} | Target profit: ${abs(signal.take_profit - signal.entry_price):.2f}")
                    logging.info(f"   Reason: {signal.reason}")

                    # PHASE 4: Log signal analysis (2025-11-17)
                    self.trade_logger.log_trade_decision(
                        symbol=symbol,
                        action="ANALYZE",
                        sentiment=sentiment_score,
                        decision=f"{signal.direction} signal generated",
                        result="SUCCESS",
                        strategy=signal.strategy,
                        confidence=signal.confidence,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        reasoning=signal.reason
                    )

                    # Attempt to open position with error cleanup
                    position = None
                    try:
                        position = await self.position_manager.open_position(signal, market_data)

                        if position:
                            self.trades_executed += 1
                            logging.info(f"✅ Position opened: {position.position_id}")

                            # CRITICAL FIX: Update rate limiting tracking (2025-11-18)
                            self.last_signal_time[symbol] = current_time
                            self.last_signal_price[symbol] = current_price
                            logging.debug(f"🚦 Rate limit updated for {symbol}: time={current_time}, price=${current_price:.2f}")

                            # PHASE 4: Log successful position opening (2025-11-17)
                            self.trade_logger.log_trade_decision(
                                symbol=symbol,
                                action="EXECUTE",
                                sentiment=sentiment_score,
                                decision=f"Open {signal.direction} position",
                                result="SUCCESS",
                                position_id=position.position_id,
                                size=position.quantity,
                                price=signal.entry_price,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                reasoning=signal.reason
                            )

                            # PHASE 3: Start position monitoring (2025-11-17)
                            # Monitor position for stop loss/take profit automatic closure
                            asyncio.create_task(self.position_manager.monitor_position(position))
                            logging.debug(f"🔄 Position monitoring started for {position.position_id}")
                        else:
                            logging.debug(f"❌ Failed to open position for {symbol}")

                            # PHASE 4: Log failed position opening (2025-11-17)
                            self.trade_logger.log_trade_decision(
                                symbol=symbol,
                                action="EXECUTE",
                                sentiment=sentiment_score,
                                decision=f"Attempt to open {signal.direction} position",
                                result="FAILED",
                                reasoning="Position manager rejected (risk limits, API failure, or insufficient funds)"
                            )

                    except Exception as e:
                        logging.error(f"❌ Position opening/monitoring failed: {e}")

                        # Cleanup orphaned position if it was created
                        if position and hasattr(position, 'position_id'):
                            try:
                                logging.warning(f"🧹 Cleaning up orphaned position: {position.position_id}")
                                # Remove from position manager's active positions
                                if hasattr(self.position_manager, 'active_positions'):
                                    self.position_manager.active_positions.pop(position.position_id, None)
                            except Exception as cleanup_error:
                                logging.error(f"❌ Failed to cleanup orphaned position: {cleanup_error}")

                        # Log the failure
                        self.trade_logger.log_trade_decision(
                            symbol=symbol,
                            action="EXECUTE",
                            sentiment=sentiment_score,
                            decision=f"Attempt to open {signal.direction} position",
                            result="FAILED",
                            reasoning=f"Exception during position opening: {str(e)}"
                        )
        
        except Exception as e:
            logging.error(f"❌ Error in scalping cycle: {e}")
    
    async def monitor_performance(self):
        """Monitor and report performance metrics"""
        
        # Update account balance periodically
        self.position_manager.update_account_balance()
        
        # Check risk limits
        performance = self.position_manager.get_performance_summary()
        
        # Check for emergency conditions
        if performance['daily_pnl'] <= -self.config.max_daily_loss:
            await self.activate_kill_switch("Daily loss limit exceeded")

        elif performance['consecutive_losses'] >= self.config.max_consecutive_losses:
            await self.activate_kill_switch("Too many consecutive losses")

        # Log performance every 2 minutes
        if time.time() - self.last_performance_log > 120:
            self.log_performance_update(performance)
            self.last_performance_log = time.time()

        # Check daily trade limit
        if performance['total_trades'] >= self.config.max_daily_trades:
            logging.info(f"📊 Daily trade limit reached ({self.config.max_daily_trades})")
            await self.activate_kill_switch("Daily trade limit reached")
    
    def log_performance_update(self, performance: Dict):
        """Log current performance metrics"""
        session_time = (time.time() - self.session_start_time) / 3600  # hours
        trades_per_hour = performance['total_trades'] / session_time if session_time > 0 else 0
        
        logging.info(f"\n{'='*60}")
        logging.info(f"📊 PERFORMANCE UPDATE - {datetime.now().strftime('%H:%M:%S')}")
        logging.info(f"{'='*60}")
        logging.info(f"💰 Daily P&L: ${performance['daily_pnl']:.2f}")
        logging.info(f"📈 Total Trades: {performance['total_trades']}")
        logging.info(f"🎯 Win Rate: {performance['win_rate']*100:.1f}%")
        logging.info(f"⚡ Trades/Hour: {trades_per_hour:.1f}")
        logging.info(f"🔄 Active Positions: {performance['active_positions']}")
        logging.info(f"📊 Signals Generated: {self.signal_count}")
        logging.info(f"🔥 Win Streak: {performance['current_win_streak']}")
        logging.info(f"❄️ Loss Streak: {performance['current_loss_streak']}")
        
        if performance['trading_halted']:
            logging.warning(f"🛑 Trading Halted: {performance['halt_reason']}")
        
        logging.info(f"{'='*60}")
    
    async def activate_kill_switch(self, reason: str):
        """Activate emergency kill switch and close all positions"""
        self.kill_switch_active = True
        self.emergency_reason = reason
        logging.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")

        # Close all positions immediately
        await self.position_manager.close_all_positions(f"Kill switch: {reason}")

        # Enter emergency mode in position manager
        self.position_manager.enter_emergency_mode()
    
    async def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        logging.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        # Close all positions immediately
        await self.position_manager.close_all_positions("Emergency shutdown")
        
        # Close WebSocket connection
        if self.websocket_provider:
            await self.websocket_provider.close()
        
        logging.critical("🚨 Emergency shutdown complete")
    
    async def graceful_shutdown(self):
        """Graceful shutdown procedure"""
        logging.info("👋 Initiating graceful shutdown...")
        
        # Stop accepting new signals
        self.kill_switch_active = True
        
        # Wait for positions to close naturally (up to 60 seconds)
        timeout = time.time() + 60
        while (len(self.position_manager.positions) > 0 and 
               time.time() < timeout):
            logging.info(f"⏳ Waiting for {len(self.position_manager.positions)} positions to close...")
            await asyncio.sleep(5)
        
        # Force close any remaining positions
        if len(self.position_manager.positions) > 0:
            logging.info("🔄 Force closing remaining positions...")
            await self.position_manager.close_all_positions("Graceful shutdown")
        
        # Close WebSocket
        if self.websocket_provider:
            await self.websocket_provider.close()
        
        # Final performance report
        self.position_manager.print_performance_report()
        
        logging.info("✅ Graceful shutdown complete")
    
    async def run(self):
        """Main swing trading execution loop"""
        print("\n" + "="*80)
        print("🚀 SWING TRADING SYSTEM (1-4 HOUR HOLDS)")
        print("="*80)
        print(f"📊 Symbols: {', '.join(self.config.symbols)}")
        print(f"🎯 Target: {self.config.target_win_rate*100:.0f}% win rate, ${self.config.target_profit_per_trade:.2f}/trade")
        print(f"💰 Daily target: ${self.config.target_daily_profit:.0f}")
        print(f"🛑 Risk limits: ${self.config.max_daily_loss:.0f} max loss, {self.config.max_consecutive_losses} max losses")
        print(f"⚡ Expected: 6-10 swing trades/day")
        print(f"🔄 Max positions: {self.config.max_concurrent_positions}")
        print(f"⏱️ Max hold time: {self.config.max_hold_time/3600:.1f} hours")
        print(f"🌐 Data feed: {'WebSocket' if self.websocket_provider else 'REST API'}")
        print(f"\n⚠️  Press Ctrl+C for graceful shutdown")
        print("="*80 + "\n")
        
        # Start WebSocket if available
        if self.websocket_provider:
            websocket_started = await self.start_websocket_feed()
            if not websocket_started:
                logging.warning("⚠️ WebSocket failed, using REST API only")

        # Pre-load price history to avoid cold start (fixes mean reversion 100-tick requirement)
        await self.preload_price_history()

        cycle_count = 0
        last_status_log = 0  # Initialize to 0 so first market check logs immediately
        last_reconciliation = 0  # Track last position reconciliation time

        try:
            while not self.kill_switch_active:
                cycle_start = time.time()
                
                # Check if market is open
                if not self.check_trading_session():
                    if time.time() - last_status_log > 60:  # Log every 1 minute
                        logging.info("🔒 Market closed, waiting for trading hours...")
                        last_status_log = time.time()
                    await asyncio.sleep(30)  # Check again in 30 seconds
                    continue
                
                # Execute scalping cycle
                await self.scalping_cycle()

                # Reconcile positions with broker every 10 seconds
                if time.time() - last_reconciliation > 10:
                    await self.position_manager.reconcile_positions()
                    last_reconciliation = time.time()

                # Monitor performance and risk
                await self.monitor_performance()
                
                cycle_count += 1
                self.total_cycles = cycle_count
                
                # Ensure minimum cycle time (avoid API spam)
                cycle_elapsed = time.time() - cycle_start
                min_cycle_time = 0.8  # 800ms minimum
                if cycle_elapsed < min_cycle_time:
                    await asyncio.sleep(min_cycle_time - cycle_elapsed)
                
                # Status update every 5 minutes
                if time.time() - last_status_log > 300:
                    performance = self.position_manager.get_performance_summary()
                    session_time = (time.time() - self.session_start_time) / 3600
                    logging.info(f"📊 Status: {cycle_count} cycles, {performance['total_trades']} trades, ${performance['daily_pnl']:.2f} P&L, {session_time:.1f}h runtime")
                    last_status_log = time.time()
        
        except KeyboardInterrupt:
            print(f"\n\n👋 Graceful shutdown requested...")
            await self.graceful_shutdown()
        
        except Exception as e:
            logging.critical(f"💥 CRITICAL ERROR: {e}")
            await self.emergency_shutdown()
        
        finally:
            # Final session summary
            session_time = (time.time() - self.session_start_time) / 3600
            performance = self.position_manager.get_performance_summary()
            
            print(f"\n{'='*80}")
            print(f"📊 FINAL SESSION SUMMARY")
            print(f"{'='*80}")
            print(f"⏱️ Session duration: {session_time:.2f} hours")
            print(f"🔄 Total cycles: {self.total_cycles:,}")
            print(f"📡 Signals generated: {self.signal_count:,}")
            print(f"📈 Trades executed: {self.trades_executed:,}")
            print(f"💰 Final P&L: ${performance['daily_pnl']:.2f}")
            print(f"🎯 Final win rate: {performance['win_rate']*100:.1f}%")
            print(f"⚡ Avg trades/hour: {performance['total_trades']/session_time:.1f}")
            print(f"💵 Avg profit/trade: ${performance['daily_pnl']/max(1, performance['total_trades']):.2f}")
            
            if self.kill_switch_active:
                print(f"🚨 Session ended: {self.emergency_reason}")
            
            print(f"{'='*80}")


# Default configuration
# UPDATED 2025-12-12: Swing trading (1-4 hour holds)
def create_default_config() -> ScalpingConfig:
    """Create default swing trading configuration"""
    return ScalpingConfig(
        symbols=['NVDA'],  # Start with NVIDIA only
        max_concurrent_positions=6,
        max_daily_trades=10,  # ~6-10 swing trades per day (was 36 scalps)
        max_daily_loss=150.0,
        position_size_pct=0.03,  # 3% per position
        target_win_rate=0.55,  # 55% required for swing economics
        target_profit_per_trade=3.00,  # $3.00 target (was $1.00 scalping)
        target_daily_profit=15.0,  # Fewer trades, bigger profits
        max_consecutive_losses=4,
        max_hold_time=14400.0,  # 4 hours (was 40 minutes)
        latency_threshold=2000.0,  # 2 seconds (less critical for swing)
        use_websocket=True,
        use_trailing_stops=True,
        dynamic_position_sizing=True,
        sentiment_bias_enabled=True
    )


# Main execution
if __name__ == "__main__":
    # Safety check for live trading
    if '--live' in sys.argv:
        print("\n" + "="*80)
        print("⚠️  WARNING: LIVE SWING TRADING MODE")
        print("="*80)
        print("🚨 This system executes 6-10 swing trades per day")
        print("💰 Real money will be at risk with every trade")
        print("⏱️ Positions will be held for 1-4 hours each")
        print("🎯 System targets $3.00+ profit per trade")
        print("🛑 Maximum daily loss limit will be enforced")
        print("📊 All trades will be logged and monitored")
        print("="*80)

        response = input("\nAre you absolutely sure you want to trade with REAL MONEY? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("❌ Live trading cancelled. Use without --live for demo mode.")
            sys.exit(0)

        print("✅ Live swing trading mode confirmed. Starting with real money...")
        print("🚨 Use Ctrl+C to stop trading gracefully")
        time.sleep(3)  # Give user time to read
    
    # Create configuration
    config = create_default_config()
    
    # Add more symbols for advanced users
    # OLD: Multi-symbol mode with AAPL, MSFT
    # if '--multi-symbol' in sys.argv:
    #     config.symbols = ['NVDA', 'AAPL', 'MSFT']
    #     config.max_concurrent_positions = 12
    #     print("📊 Multi-symbol mode enabled: NVDA, AAPL, MSFT")
    # NEW: Multi-symbol disabled (only NVDA available in companies.yaml)
    if '--multi-symbol' in sys.argv:
        config.symbols = ['NVDA']  # Only NVIDIA enabled
        config.max_concurrent_positions = 4
        print("📊 NVIDIA-only mode (other companies disabled in companies.yaml)")
    
    # Conservative mode for beginners
    if '--conservative' in sys.argv:
        config.max_daily_loss = 50.0
        config.max_concurrent_positions = 3
        config.position_size_pct = 0.01
        config.max_consecutive_losses = 3
        print("🛡️ Conservative mode enabled: Lower risk limits")
    
    # Aggressive mode for experienced traders
    if '--aggressive' in sys.argv:
        config.max_daily_trades = 200
        config.max_concurrent_positions = 10
        config.position_size_pct = 0.025
        config.target_daily_profit = 50.0
        print("🔥 Aggressive mode enabled: Higher targets and limits")

    # Disable WebSocket (use REST API fallback)
    if '--no-websocket' in sys.argv:
        config.use_websocket = False
        print("🔌 WebSocket disabled: Using REST API fallback")

    # Create and run enhanced scalper
    scalper = EnhancedScalper(config)
    asyncio.run(scalper.run())