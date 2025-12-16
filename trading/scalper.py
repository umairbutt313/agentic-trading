#!/usr/bin/env python3
"""
High-Frequency Scalping Engine
Transforms sentiment-based trading into rapid scalping system with sub-minute holds

# ==============================================================================
# CHANGELOG
# ==============================================================================
# [2025-12-10 17:30:00] FIX: Position hold time monitoring (ISSUE_HASH: position_hold_time_001)
#   ISSUE: Position exceeded max hold time (300s) with no close attempt
#   ROOT CAUSE: Passive monitoring only checked in loop, no proactive threshold
#   FIX: Added synchronous check at 90% of max_hold_time to force early exit
#   LOCATION: monitor_positions() method (lines 501-584)
#   IMPACT: Prevents stuck positions, ensures capital not locked
#   VALIDATION: Positions should close at ~54s (90% of 60s max_hold_time)
#
# [2025-12-10 17:40:00] FIX: Momentum strategy disabled without volume (ISSUE_HASH: momentum_volume_001)
#   ISSUE: Fake/missing volume data causes 50% win rate (should be 65%+)
#   ROOT CAUSE: REST API provides no volume (volume=0), momentum needs confirmation
#   FIX: Check average volume, skip momentum strategy if avg_vol <= 0
#   LOCATION: momentum_strategy() method (lines 227-260)
#   IMPACT: Prevents random signals, improves win rate when WebSocket enabled
#   VALIDATION: Strategy skips when volume=0, activates with WebSocket real volume
#
# [2025-12-10 (Previous Session)] VERIFIED FIXES:
#   - dealReference bug (line 481): Uses correct API field for order tracking
#   - Order ID validation (lines 485-487): Raises error if dealReference missing
# ==============================================================================
"""

import asyncio
import time
import json
import os
import sys
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import threading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.capital_trader import CapitalTrader

@dataclass
class ScalpingSignal:
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy: str
    timestamp: float
    reason: str

@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    quantity: float
    entry_time: float
    stop_loss: float
    take_profit: float
    order_id: Optional[str] = None

class HighFrequencyScalper:
    """
    High-frequency scalping system that:
    - Executes trades every 1-60 seconds
    - Captures micro-profits of $0.10-$0.20 per trade
    - Uses multiple scalping strategies
    - Maintains sentiment bias for direction
    """
    
    def __init__(self, config_file: str = ".env"):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._setup_logging()
        
        # Load trading credentials
        self.load_config(config_file)
        
        # Initialize Capital.com trader
        self.trader = CapitalTrader(
            api_key=self.capital_api_key,
            password=self.capital_password,
            email=self.capital_email,
            demo='--live' not in sys.argv
        )
        
        # Scalping parameters
        self.symbols = ['NVDA']  # Start with NVIDIA only
        self.profit_targets = {
            'NVDA': 0.15  # $0.15 profit target
        }
        self.stop_losses = {
            'NVDA': 0.08  # $0.08 stop loss
        }
        
        # Position management
        self.max_hold_time = 60  # Maximum 60 seconds per trade
        self.max_concurrent_positions = 10
        self.position_size_pct = 0.02  # 2% of balance per trade
        self.cooldown_seconds = 3  # 3 seconds between trades on same symbol
        
        # Strategy weights
        self.strategy_weights = {
            'momentum': 0.35,
            'mean_reversion': 0.25,
            'spread_scalping': 0.20,
            'sentiment_bias': 0.20
        }
        
        # Data storage
        self.price_history = {symbol: deque(maxlen=200) for symbol in self.symbols}
        self.volume_history = {symbol: deque(maxlen=200) for symbol in self.symbols}
        self.spread_history = {symbol: deque(maxlen=100) for symbol in self.symbols}
        self.current_positions = []
        self.completed_trades = []
        self.last_trade_time = {}
        
        # Performance tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        # Initialize sentiment scores
        self.sentiment_scores = {}
        self.load_sentiment_scores()
        
        logging.info("🚀 High-Frequency Scalper initialized")
        logging.info(f"📊 Symbols: {self.symbols}")
        logging.info(f"💰 Max positions: {self.max_concurrent_positions}")
        logging.info(f"⏱️ Max hold time: {self.max_hold_time}s")
    
    def _setup_logging(self):
        """Setup logging system"""
        logs_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"scalper_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ],
            force=True
        )
    
    def load_config(self, config_file: str):
        """Load trading configuration"""
        env_path = os.path.join(self.base_dir, config_file)
        
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
    
    def load_sentiment_scores(self) -> Dict:
        """Load latest sentiment scores for directional bias"""
        try:
            final_score_dir = os.path.join(self.base_dir, "container_output", "final_score")
            weighted_file = os.path.join(final_score_dir, "final-weighted-scores.json")
            
            with open(weighted_file, 'r') as f:
                data = json.load(f)
            
            companies_data = data.get("companies", data)
            
            for company, symbol in [("NVIDIA", "NVDA")]:
                if company in companies_data:
                    self.sentiment_scores[symbol] = companies_data[company]["final_score"]
                else:
                    self.sentiment_scores[symbol] = 5.0  # Neutral
            
            logging.info(f"📊 Sentiment loaded: {self.sentiment_scores}")
            return self.sentiment_scores
            
        except Exception as e:
            logging.warning(f"⚠️ Could not load sentiment: {e}")
            # Default neutral sentiment
            for symbol in self.symbols:
                self.sentiment_scores[symbol] = 5.0
            return self.sentiment_scores
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data"""
        try:
            # Create session if needed
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
    
    def update_price_history(self, symbol: str, market_data: Dict):
        """Update price and volume history"""
        if not market_data:
            return
        
        self.price_history[symbol].append(market_data['mid'])
        self.spread_history[symbol].append(market_data['spread'])

        # ❌ DISABLED: Fake volume data - DO NOT USE IN LIVE TRADING
        # This generated RANDOM volume numbers, not real market data
        # volume = np.random.randint(1000, 5000)  # FAKE DATA

        # Use real volume from market data if available, otherwise use 0
        volume = market_data.get('volume', 0)  # ✅ REAL DATA or 0 if unavailable
        self.volume_history[symbol].append(volume)
    
    def momentum_strategy(self, symbol: str) -> Optional[ScalpingSignal]:
        """Momentum scalping strategy"""
        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 17:40:00]
        # ==============================================================================
        # ISSUE: Fake volume data causes invalid momentum signals (50% win rate vs 65%)
        # ISSUE_HASH: momentum_volume_001
        # PREVIOUS ATTEMPTS: Commented out fake volume generation (line 221)
        # LIANG WENFENG REASONING:
        #   1. Market Context: REST API provides no volume data (volume=0 always)
        #   2. Signal Interpretation: Momentum strategy requires volume confirmation
        #      Without real volume, signals are random (50/50 win rate)
        #   3. Alternative Evaluation: Disable strategy when volume unavailable vs
        #      use price-only momentum (unreliable without volume confirmation)
        #   4. Risk Management: False signals = losses. Better to skip strategy
        #   5. Reflection: WebSocket provides volume, but when disabled, skip momentum
        # SOLUTION: Check if volume data is real (avg > 0), disable if not available
        # VALIDATION:
        #   1. Verify momentum strategy skipped when volume=0
        #   2. Enable WebSocket to restore momentum strategy
        #   3. Monitor win rate improves to 65%+ with real volume
        # ==============================================================================

        if len(self.price_history[symbol]) < 20:
            return None

        prices = np.array(list(self.price_history[symbol]))
        volumes = np.array(list(self.volume_history[symbol]))

        # CRITICAL FIX: Check if volume data is available (not all zeros)
        avg_vol = np.mean(volumes[-20:])
        if avg_vol <= 0:
            logging.debug(f"⏭️ Momentum strategy disabled - no volume data (WebSocket not enabled)")
            return None

        # Calculate momentum indicators
        recent_prices = prices[-10:]
        older_prices = prices[-20:-10]

        recent_avg = np.mean(recent_prices)
        older_avg = np.mean(older_prices)

        # Price momentum
        price_momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        # Volume spike
        recent_vol = np.mean(volumes[-5:])
        volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1
        
        # Current market data
        current_price = prices[-1]
        
        # Signal conditions
        if price_momentum > 0.001 and volume_spike > 1.5:  # 0.1% momentum + volume spike
            return ScalpingSignal(
                symbol=symbol,
                direction='LONG',
                entry_price=current_price,
                stop_loss=current_price - self.stop_losses[symbol],
                take_profit=current_price + self.profit_targets[symbol],
                confidence=min(abs(price_momentum) * 100 + volume_spike / 5, 1.0),
                strategy='momentum',
                timestamp=time.time(),
                reason=f'Momentum surge: {price_momentum:.3f}, Volume: {volume_spike:.1f}x'
            )
        elif price_momentum < -0.001 and volume_spike > 1.5:
            return ScalpingSignal(
                symbol=symbol,
                direction='SHORT',
                entry_price=current_price,
                stop_loss=current_price + self.stop_losses[symbol],
                take_profit=current_price - self.profit_targets[symbol],
                confidence=min(abs(price_momentum) * 100 + volume_spike / 5, 1.0),
                strategy='momentum',
                timestamp=time.time(),
                reason=f'Momentum drop: {price_momentum:.3f}, Volume: {volume_spike:.1f}x'
            )
        
        return None
    
    def mean_reversion_strategy(self, symbol: str) -> Optional[ScalpingSignal]:
        """Mean reversion scalping strategy"""
        if len(self.price_history[symbol]) < 50:
            return None
        
        prices = np.array(list(self.price_history[symbol]))
        
        # Calculate RSI-like indicator
        changes = np.diff(prices[-20:])
        gains = changes[changes > 0]
        losses = -changes[changes < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.01
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.01
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger-like bands
        mean_price = np.mean(prices[-20:])
        std_price = np.std(prices[-20:])
        
        current_price = prices[-1]
        z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
        
        # Mean reversion signals
        if rsi < 20 or z_score < -2:  # Oversold
            return ScalpingSignal(
                symbol=symbol,
                direction='LONG',
                entry_price=current_price,
                stop_loss=current_price - self.stop_losses[symbol],
                take_profit=current_price + self.profit_targets[symbol],
                confidence=min(abs(z_score) / 3 + (30 - rsi) / 30, 1.0),
                strategy='mean_reversion',
                timestamp=time.time(),
                reason=f'Oversold: RSI={rsi:.1f}, Z-score={z_score:.2f}'
            )
        elif rsi > 80 or z_score > 2:  # Overbought
            return ScalpingSignal(
                symbol=symbol,
                direction='SHORT',
                entry_price=current_price,
                stop_loss=current_price + self.stop_losses[symbol],
                take_profit=current_price - self.profit_targets[symbol],
                confidence=min(abs(z_score) / 3 + (rsi - 70) / 30, 1.0),
                strategy='mean_reversion',
                timestamp=time.time(),
                reason=f'Overbought: RSI={rsi:.1f}, Z-score={z_score:.2f}'
            )
        
        return None
    
    def spread_scalping_strategy(self, symbol: str, market_data: Dict) -> Optional[ScalpingSignal]:
        """Spread scalping strategy"""
        if not market_data or len(self.spread_history[symbol]) < 10:
            return None
        
        current_spread = market_data['spread']
        avg_spread = np.mean(list(self.spread_history[symbol]))
        
        # Wide spread opportunity
        if current_spread > avg_spread * 1.3 and current_spread > 0.05:
            # Place orders inside the spread
            mid_price = market_data['mid']
            
            return ScalpingSignal(
                symbol=symbol,
                direction='SPREAD_CAPTURE',
                entry_price=mid_price,
                stop_loss=mid_price - current_spread / 4,
                take_profit=mid_price + current_spread / 4,
                confidence=min(current_spread / avg_spread, 1.0),
                strategy='spread_scalping',
                timestamp=time.time(),
                reason=f'Wide spread: ${current_spread:.3f} vs ${avg_spread:.3f}'
            )
        
        return None
    
    def sentiment_bias_strategy(self, symbol: str) -> Optional[ScalpingSignal]:
        """Use sentiment for directional bias"""
        sentiment = self.sentiment_scores.get(symbol, 5.0)
        
        if len(self.price_history[symbol]) < 10:
            return None
        
        prices = np.array(list(self.price_history[symbol]))
        current_price = prices[-1]
        
        # Strong bullish sentiment
        if sentiment >= 7.0:
            return ScalpingSignal(
                symbol=symbol,
                direction='LONG',
                entry_price=current_price,
                stop_loss=current_price - self.stop_losses[symbol],
                take_profit=current_price + self.profit_targets[symbol] * 1.2,  # Higher target
                confidence=min(sentiment / 10, 1.0),
                strategy='sentiment_bias',
                timestamp=time.time(),
                reason=f'Bullish sentiment: {sentiment}/10'
            )
        # Strong bearish sentiment
        elif sentiment <= 3.0:
            return ScalpingSignal(
                symbol=symbol,
                direction='SHORT',
                entry_price=current_price,
                stop_loss=current_price + self.stop_losses[symbol],
                take_profit=current_price - self.profit_targets[symbol] * 1.2,
                confidence=min((10 - sentiment) / 10, 1.0),
                strategy='sentiment_bias',
                timestamp=time.time(),
                reason=f'Bearish sentiment: {sentiment}/10'
            )
        
        return None
    
    def analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[ScalpingSignal]:
        """Analyze symbol for scalping opportunities"""
        if not market_data:
            return None
        
        # Update data
        self.update_price_history(symbol, market_data)
        
        # Check cooldown
        if symbol in self.last_trade_time:
            if time.time() - self.last_trade_time[symbol] < self.cooldown_seconds:
                return None
        
        # Check if we already have positions
        current_symbol_positions = [p for p in self.current_positions if p.symbol == symbol]
        if len(current_symbol_positions) >= 3:  # Max 3 positions per symbol
            return None
        
        # Get signals from all strategies
        signals = []
        
        momentum_signal = self.momentum_strategy(symbol)
        if momentum_signal:
            signals.append(momentum_signal)
        
        mean_reversion_signal = self.mean_reversion_strategy(symbol)
        if mean_reversion_signal:
            signals.append(mean_reversion_signal)
        
        spread_signal = self.spread_scalping_strategy(symbol, market_data)
        if spread_signal:
            signals.append(spread_signal)
        
        sentiment_signal = self.sentiment_bias_strategy(symbol)
        if sentiment_signal:
            signals.append(sentiment_signal)
        
        # Return strongest signal
        if signals:
            best_signal = max(signals, key=lambda s: s.confidence)
            if best_signal.confidence > 0.6:  # Minimum confidence threshold
                return best_signal
        
        return None
    
    async def execute_signal(self, signal: ScalpingSignal) -> Optional[Position]:
        """Execute scalping signal"""
        try:
            # Calculate position size
            account_info = self.trader.get_account_info()
            balance = float(account_info.get('balance', 1000))
            
            # Get current price for precise sizing
            current_price = signal.entry_price
            position_value = balance * self.position_size_pct
            quantity = max(0.1, round(position_value / current_price, 2))
            
            logging.info(f"🎯 EXECUTING {signal.direction} {signal.symbol}")
            logging.info(f"   Price: ${signal.entry_price:.2f}")
            logging.info(f"   Quantity: {quantity}")
            logging.info(f"   Strategy: {signal.strategy}")
            logging.info(f"   Confidence: {signal.confidence:.2f}")
            logging.info(f"   Reason: {signal.reason}")
            
            # Place order with ATR-based stop loss (PHASE 1.5: 2025-11-12)
            order_result = self.trader.place_order(
                symbol=signal.symbol,
                direction='BUY' if signal.direction == 'LONG' else 'SELL',
                size=quantity,
                stop_loss_price=signal.stop_loss  # Pass ATR stop loss to Capital.com
            )
            
            if order_result:
                position = Position(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    quantity=quantity,
                    entry_time=time.time(),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    order_id=order_result.get('dealReference')  # Capital.com returns dealReference
                )

                # Validate order ID was captured
                if not position.order_id:
                    logging.error(f"❌ No dealReference in order response: {order_result}")
                    raise ValueError(f"Failed to capture dealReference from broker")
                
                self.current_positions.append(position)
                self.last_trade_time[signal.symbol] = time.time()
                self.daily_trades += 1
                
                logging.info(f"✅ Position opened: {signal.direction} {quantity} {signal.symbol}")
                return position
            
        except Exception as e:
            logging.error(f"❌ Failed to execute signal: {e}")
        
        return None
    
    async def monitor_positions(self):
        """Monitor and manage open positions"""
        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 17:30:00]
        # ==============================================================================
        # ISSUE: Position NVDA_1765379861696 exceeded max hold time (300s) with no close
        # ISSUE_HASH: position_hold_time_001
        # PREVIOUS ATTEMPTS: None - hold time check was passive (only checked in loop)
        # LIANG WENFENG REASONING:
        #   1. Market Context: Position held 300+ seconds, exceeding max_hold_time (60s)
        #   2. Signal Interpretation: Monitor loop runs but doesn't force close at 90%
        #   3. Alternative Evaluation: Add synchronous check before market data fetch
        #   4. Risk Management: Stuck positions lock capital, prevent new trades
        #   5. Reflection: Need proactive check at 90% threshold for timely exit
        # SOLUTION: Add synchronous position check for hold times > 90% of max_hold_time
        # VALIDATION:
        #   1. Verify positions close at ~54s (90% of 60s max_hold_time)
        #   2. Monitor logs for "Position approaching max hold time" warnings
        #   3. Confirm no positions exceed max_hold_time in next 50 trades
        # ==============================================================================

        positions_to_close = []

        # CRITICAL FIX: Synchronous check for positions approaching max hold time
        for position in self.current_positions:
            hold_time = time.time() - position.entry_time
            if hold_time > self.max_hold_time * 0.9:  # 90% threshold
                logging.warning(f"⏰ Position approaching max hold time: {hold_time:.1f}s / {self.max_hold_time}s")
                try:
                    market_data = await self.get_market_data(position.symbol)
                    if market_data:
                        current_price = market_data['mid']
                        if position.direction == 'LONG':
                            pnl = (current_price - position.entry_price) * position.quantity
                        else:
                            pnl = (position.entry_price - current_price) * position.quantity
                        positions_to_close.append((position, "MAX_HOLD_TIME_APPROACHING", pnl))
                except Exception as e:
                    logging.error(f"❌ Error checking max hold time: {e}")

        # Regular monitoring for all positions
        for position in self.current_positions:
            try:
                # Get current market data
                market_data = await self.get_market_data(position.symbol)
                if not market_data:
                    continue

                current_price = market_data['mid']
                hold_time = time.time() - position.entry_time

                # Calculate P&L
                if position.direction == 'LONG':
                    pnl = (current_price - position.entry_price) * position.quantity
                    hit_take_profit = current_price > position.take_profit  # FIX: Use > not >= (ISSUE_HASH: take_profit_op_001)
                    hit_stop_loss = current_price <= position.stop_loss
                else:  # SHORT
                    pnl = (position.entry_price - current_price) * position.quantity
                    hit_take_profit = current_price < position.take_profit  # FIX: Use < not <= (ISSUE_HASH: take_profit_op_001)
                    hit_stop_loss = current_price >= position.stop_loss

                # Exit conditions
                should_exit = False
                exit_reason = ""

                if hit_take_profit:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                elif hit_stop_loss:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                elif hold_time > self.max_hold_time:
                    should_exit = True
                    exit_reason = "TIME_EXIT"

                if should_exit:
                    positions_to_close.append((position, exit_reason, pnl))

            except Exception as e:
                logging.error(f"❌ Error monitoring position: {e}")

        # Close positions
        for position, reason, pnl in positions_to_close:
            await self.close_position(position, reason, pnl)
    
    async def close_position(self, position: Position, reason: str, pnl: float):
        """Close a position"""
        try:
            logging.info(f"🔄 CLOSING {position.direction} {position.symbol}")
            logging.info(f"   Reason: {reason}")
            logging.info(f"   P&L: ${pnl:.2f}")
            
            # Remove from current positions
            if position in self.current_positions:
                self.current_positions.remove(position)
            
            # Record trade statistics
            self.daily_pnl += pnl
            
            if pnl > 0:
                self.win_count += 1
                self.total_profit += pnl
            else:
                self.loss_count += 1
                self.total_loss += abs(pnl)
            
            # Log trade
            trade_record = {
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'quantity': position.quantity,
                'pnl': pnl,
                'hold_time': time.time() - position.entry_time,
                'exit_reason': reason,
                'timestamp': time.time()
            }
            
            self.completed_trades.append(trade_record)
            
            # Try to close position via Capital.com (if we have order ID)
            if position.order_id:
                try:
                    close_result = self.trader.close_position(position.order_id)
                    logging.info(f"✅ Position closed via API")
                except Exception as e:
                    logging.warning(f"⚠️ Could not close via API: {e}")
            
        except Exception as e:
            logging.error(f"❌ Error closing position: {e}")
    
    def print_performance_stats(self):
        """Print current performance statistics"""
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        avg_win = self.total_profit / self.win_count if self.win_count > 0 else 0
        avg_loss = self.total_loss / self.loss_count if self.loss_count > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"🏆 SCALPING PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"📊 Total Trades Today: {self.daily_trades}")
        print(f"💰 Daily P&L: ${self.daily_pnl:.2f}")
        print(f"🎯 Win Rate: {win_rate:.1f}% ({self.win_count}W/{self.loss_count}L)")
        print(f"📈 Average Win: ${avg_win:.2f}")
        print(f"📉 Average Loss: ${avg_loss:.2f}")
        print(f"🔄 Open Positions: {len(self.current_positions)}")
        print(f"{'='*60}")
    
    async def scalping_cycle(self):
        """Single scalping cycle across all symbols"""
        try:
            # Reload sentiment periodically
            if time.time() % 30 < 1:  # Every 30 seconds
                self.load_sentiment_scores()
            
            for symbol in self.symbols:
                # Get market data
                market_data = await self.get_market_data(symbol)
                
                if market_data:
                    # Analyze for signals
                    signal = self.analyze_symbol(symbol, market_data)
                    
                    # Execute signal if valid
                    if signal and len(self.current_positions) < self.max_concurrent_positions:
                        await self.execute_signal(signal)
            
            # Monitor existing positions
            await self.monitor_positions()
            
            # Print stats every 50 cycles
            if self.daily_trades % 50 == 0 and self.daily_trades > 0:
                self.print_performance_stats()
                
        except Exception as e:
            logging.error(f"❌ Error in scalping cycle: {e}")
    
    async def run(self):
        """Main scalping loop"""
        print("\n🚀 STARTING HIGH-FREQUENCY SCALPING SYSTEM")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"💰 Profit Target: ${self.profit_targets['NVDA']}")
        print(f"🛑 Stop Loss: ${self.stop_losses['NVDA']}")
        print(f"⏱️ Max Hold: {self.max_hold_time}s")
        print(f"🔄 Max Positions: {self.max_concurrent_positions}")
        print(f"🎯 Target: 50-100 trades/day, 65%+ win rate")
        print(f"\n⚠️  Press Ctrl+C to stop\n")
        
        cycle_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                # Execute scalping cycle
                await self.scalping_cycle()
                
                cycle_count += 1
                
                # Ensure minimum cycle time (don't spam API)
                elapsed = time.time() - start_time
                min_cycle_time = 1.0  # Minimum 1 second per cycle
                if elapsed < min_cycle_time:
                    await asyncio.sleep(min_cycle_time - elapsed)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 SCALPING STOPPED BY USER")
            
            # Close all positions
            if self.current_positions:
                print(f"🔄 Closing {len(self.current_positions)} open positions...")
                for position in self.current_positions.copy():
                    market_data = await self.get_market_data(position.symbol)
                    if market_data:
                        current_price = market_data['mid']
                        if position.direction == 'LONG':
                            pnl = (current_price - position.entry_price) * position.quantity
                        else:
                            pnl = (position.entry_price - current_price) * position.quantity
                        await self.close_position(position, "MANUAL_EXIT", pnl)
            
            # Final performance report
            self.print_performance_stats()
            
            print(f"\n📊 SESSION SUMMARY:")
            print(f"   Total Cycles: {cycle_count}")
            print(f"   Runtime: {(time.time() - start_time) / 60:.1f} minutes")
            print(f"   Trades per Hour: {(self.daily_trades * 60) / ((time.time() - start_time) / 60):.0f}")
            print(f"\n💡 Tips for next session:")
            print(f"   - Adjust profit targets based on performance")
            print(f"   - Monitor sentiment alignment with trades")
            print(f"   - Consider adding more symbols when profitable")


if __name__ == "__main__":
    # Safety check for live trading
    if '--live' in sys.argv:
        print("⚠️  WARNING: LIVE SCALPING MODE WITH REAL MONEY!")
        print(f"   This system can execute 50-100 trades per day")
        print(f"   High-frequency trading with rapid position changes")
        response = input("Are you sure you want to scalp with REAL MONEY? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Live scalping cancelled. Use without --live for demo mode.")
            sys.exit(0)
        print("✅ Live scalping mode confirmed. Starting high-frequency trading...")
    
    # Create and run scalper
    scalper = HighFrequencyScalper()
    asyncio.run(scalper.run())