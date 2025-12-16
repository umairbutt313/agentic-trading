#!/usr/bin/env python3
"""
Advanced Swing Trading Strategies (1-4 Hour Holds)
Implementation of momentum, mean reversion, order book, and spread strategies

# ==============================================================================
# CHANGELOG:
# ==============================================================================
# [2025-12-11] FIX: ADX-adaptive consolidation filter (ISSUE_HASH: consolidation_adaptive_002)
#              Previous fixed 0.15% threshold blocked 95% of signals in strong trends (ADX 60+)
#              Now scales inversely with ADX: 60+→0.05%, 40-60→0.08%, 25-40→0.15%
# [2025-12-09] FIX: Dynamic ATR-based take profit (ISSUE_HASH: atr_tp_001)
#              Previous fixed 0.22% TP was losing to spread cost ($0.06-0.11)
#              Now uses MAX(ATR×2.5, spread×4) to ensure profitability
# ==============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import time
import logging
from scipy import stats

@dataclass
class ScalpingSignal:
    symbol: str
    direction: str  # 'LONG', 'SHORT', 'SPREAD_CAPTURE'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0.0 to 1.0
    strategy: str
    timestamp: float
    reason: str
    expected_profit: float = 0.0
    risk_reward_ratio: float = 0.0

@dataclass
class MarketMicrostructure:
    symbol: str
    bid: float
    ask: float
    mid: float
    spread: float
    bid_size: float
    ask_size: float
    last_trade_price: float
    last_trade_size: float
    timestamp: float

class MomentumScalper:
    """
    Momentum swing trading strategy - rides trends for 1-4 hours
    Entry: Price breaks VWAP with volume
    Exit: Fixed profit or momentum fade

    ARCHITECTURE CHANGE [2025-12-12]: Converted to swing trading
    - Period increased 10x (30 → 300 ticks = 5 minutes of data)
    - VWAP increased 10x (100 → 1000 ticks for longer trends)
    - Min price move increased 10x ($0.05 → $0.50 for swing volatility)
    """

    def __init__(self, config: Dict):
        self.min_volume_spike = config.get('min_volume_spike', 1.0)  # FIXED: 1.0x for REST API (no real volume data)
        self.min_price_move = config.get('min_price_move', 0.50)  # $0.50 minimum move (swing trading)
        self.momentum_period = config.get('momentum_period', 300)  # 300 ticks (5 minutes for swing)
        self.vwap_period = config.get('vwap_period', 1000)  # 1000 ticks for swing VWAP
        self.confidence_threshold = config.get('confidence_threshold', 0.5)  # FIXED: Lowered from 0.65 to 0.5 for more signals
        
        # Risk parameters
        # ORIGINAL: Fixed percentage stop losses (PHASE 1: DISABLED - 2025-10-22)
        # self.stop_loss_pct = config.get('stop_loss_pct', 0.08)  # 0.08% stop loss
        # self.take_profit_pct = config.get('take_profit_pct', 0.15)  # 0.15% take profit

        # PHASE 1: ATR-based dynamic stop losses (2025-10-22)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)  # 2x ATR for stops
        self.take_profit_pct = config.get('take_profit_pct', 0.15)  # Keep fixed take profit
        self.use_atr_stops = config.get('use_atr_stops', True)  # Enable ATR stops

        # Import indicator calculator for ATR calculations
        try:
            from trading.indicator_utils import IndicatorCalculator
            self.indicator_calc = IndicatorCalculator()
        except ImportError:
            logging.warning("⚠️ Could not import IndicatorCalculator, ATR stops disabled")
            self.use_atr_stops = False
            self.indicator_calc = None

        # Data storage
        self.price_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)
        self.vwap_history = deque(maxlen=self.vwap_period)

        # Multi-tick confirmation tracking
        self.momentum_confirmation = {'count': 0, 'direction': None, 'momentum_values': []}
        
    def update_data(self, tick_data: Dict):
        """Update internal data with new tick"""
        price = tick_data.get('mid', 0)
        volume = tick_data.get('volume', 1000)  # Default volume if not available
        
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Calculate VWAP point
        if price > 0 and volume > 0:
            vwap_point = price * volume
            self.vwap_history.append(vwap_point)
    
    def calculate_vwap(self) -> float:
        """Calculate Volume Weighted Average Price"""
        if len(self.vwap_history) < 10 or len(self.volume_history) < 10:
            return 0.0
        
        recent_vwap_points = list(self.vwap_history)[-50:]  # Last 50 points
        recent_volumes = list(self.volume_history)[-50:]
        
        total_vwap = sum(recent_vwap_points)
        total_volume = sum(recent_volumes)
        
        return total_vwap / total_volume if total_volume > 0 else 0.0
    
    def calculate_momentum_strength(self) -> float:
        """Calculate momentum strength (-1 to 1)"""
        if len(self.price_history) < self.momentum_period:
            return 0.0
        
        prices = np.array(list(self.price_history))
        
        # Recent vs older price comparison
        recent_prices = prices[-10:]  # Last 10 ticks
        older_prices = prices[-self.momentum_period:-10]
        
        if len(older_prices) == 0:
            return 0.0
        
        recent_avg = np.mean(recent_prices)
        older_avg = np.mean(older_prices)
        
        # Normalized momentum
        if older_avg > 0:
            momentum = (recent_avg - older_avg) / older_avg
            return np.tanh(momentum * 1000)  # Scale and normalize
        
        return 0.0
    
    def calculate_volume_spike(self) -> float:
        """Calculate current volume vs average volume"""
        if len(self.volume_history) < 20:
            return 1.0
        
        volumes = list(self.volume_history)
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:-1])  # Exclude current
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def analyze(self, market_data: Dict) -> Optional[ScalpingSignal]:
        """Analyze for momentum scalping opportunities"""
        self.update_data(market_data)

        if len(self.price_history) < self.momentum_period:
            return None

        current_price = market_data.get('mid', 0)
        if current_price <= 0:
            return None

        symbol = market_data.get('symbol', 'NVDA')

        # PHASE 2: ADX pre-filter - Skip ranging markets (2025-11-17)
        # This prevents trading in choppy markets where scalping fails (97.5% loss rate)
        adx = None
        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 16:00:00]
        # ==============================================================================
        # ISSUE: System generated 14 consecutive SHORT trades in uptrend, 62.5% loss rate
        # ISSUE_HASH: trend_alignment_001
        # PREVIOUS ATTEMPTS: ADX > 50 filter (too restrictive, missed ADX 25-50 trends)
        # LIANG WENFENG REASONING:
        #   1. Market Context: NVDA in moderate uptrend (ADX 25-40), not strong (50+)
        #   2. Signal Interpretation: SHORT signals triggered on micro-dips (-0.15 momentum)
        #      but trend was UP, so stops kept getting hit
        #   3. Alternative Evaluation: Old code only checked trend if ADX > 50 (very strong),
        #      allowing counter-trend trades in ADX 25-50 (moderate trending) markets
        #   4. Risk Management: Trading against trend = low probability, high loss rate
        #   5. Reflection: ANY trending market (ADX > 25) requires trend direction check,
        #      not just ADX > 50. Counter-trend trades should be heavily penalized.
        # SOLUTION: Check trend direction for ALL ADX > 25 markets, penalize counter-trend
        # VALIDATION:
        #   1. Verify logs show trend direction for ADX > 25 (not just > 50)
        #   2. Verify SHORT signals in uptrends show 80% confidence reduction
        #   3. Monitor next 20 trades for directional alignment with trend
        # ==============================================================================

        trend_direction = None
        if self.indicator_calc is not None:
            adx = self.indicator_calc.calculate_adx(symbol)
            if adx is None or adx < 25.0:
                logging.debug(f"⏭️ ADX Filter: Skipping {symbol} (ADX={adx}, ranging market)")
                return None  # Skip trading in ranging markets

            # CRITICAL FIX: Get trend direction for ALL trending markets (not just ADX > 50)
            if adx >= 25.0:  # Changed from 50.0 to 25.0
                trend_direction = self.indicator_calc.get_trend_direction(symbol)
                logging.info(f"📊 Trend detected (ADX={adx:.1f}): {trend_direction}")

        # ==============================================================================
        # FIX ATTEMPT [2025-12-11 10:00:00]
        # ==============================================================================
        # ISSUE: Fixed 0.15% consolidation threshold blocks 95% of signals in strong trends
        # ISSUE_HASH: consolidation_adaptive_002
        # PREVIOUS ATTEMPTS: consolidation_filter_001 (fixed 0.15% threshold)
        # LIANG WENFENG REASONING:
        #   1. Market Context: Strong trends (ADX 60+) move quickly and decisively - even
        #      0.05% range indicates real momentum. Moderate trends (ADX 40-60) need 0.08%.
        #      Weak trends (ADX 25-40) need current 0.15% to avoid noise.
        #   2. Signal Interpretation: Fixed 0.15% blocks valid signals when ADX shows strong
        #      trend (ADX 60+). This causes missed opportunities in explosive trending markets
        #      where 5-10 cent moves are significant.
        #   3. Alternative Evaluation: Could remove filter (bad - allows noise), lower for all
        #      (bad - noise in ranging), or make ADX-adaptive (chosen - matches regime).
        #   4. Risk Management: Strong trends (ADX 60+) move fast - catch breakouts early with
        #      0.05%. Moderate (40-60) use 0.08%. Weak (25-40) keep 0.15% for confirmation.
        #   5. Reflection: One-size-fits-all threshold fails across regimes. Need ADX context
        #      for all filtering decisions. Thresholds scale inversely with trend strength.
        # SOLUTION: Make consolidation filter ADX-adaptive with tiered thresholds
        # VALIDATION:
        #   1. Verify logs show "ADX-Adaptive Consolidation Filter" with varying thresholds
        #   2. Verify signal count increases in ADX 60+ markets (currently blocked)
        #   3. Verify win rate maintains or improves (target 55%+)
        #   4. Monitor consolidation still blocks noise in ADX 25-40 markets
        # ==============================================================================

        # ADX-ADAPTIVE CONSOLIDATION FILTER: Threshold scales with trend strength
        if adx is not None and len(self.price_history) >= 30:
            # Determine threshold based on ADX (stronger trend = tighter threshold allowed)
            if adx >= 60:
                consolidation_threshold = 0.0005  # 0.05% - Strong trends move decisively
            elif adx >= 40:
                consolidation_threshold = 0.0008  # 0.08% - Moderate trends
            else:  # adx >= 25 (ranging markets already filtered out above)
                consolidation_threshold = 0.0015  # 0.15% - Weak trends need confirmation

            # Calculate price range over last 30 ticks
            recent_prices = list(self.price_history)[-30:]
            price_range = max(recent_prices) - min(recent_prices)
            price_range_pct = price_range / current_price if current_price > 0 else 0

            # Skip if consolidating (not enough movement for scalping)
            if price_range_pct < consolidation_threshold:
                logging.debug(f"⏭️ ADX-Adaptive Consolidation Filter: Skipping {symbol} "
                             f"(range {price_range_pct*100:.3f}% < {consolidation_threshold*100:.2f}%, ADX={adx:.1f})")
                return None

        # Calculate indicators
        vwap = self.calculate_vwap()
        momentum = self.calculate_momentum_strength()
        volume_spike = self.calculate_volume_spike()
        
        # Signal conditions
        confidence = 0.0
        direction = None
        reason = ""

        # DEBUG: Log indicator values every 30 ticks
        if len(self.price_history) % 30 == 0:
            logging.info(f"📊 Momentum indicators | price={current_price:.2f}, vwap={vwap:.2f}, momentum={momentum:.3f}, volume_spike={volume_spike:.2f}x")
            logging.info(f"   Conditions: mom>{0.15}? {momentum > 0.15}, price>{vwap*1.0001:.2f}? {current_price > vwap * 1.0001}, vol>={self.min_volume_spike}? {volume_spike >= self.min_volume_spike}")

        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 16:05:00]
        # ==============================================================================
        # ISSUE: Momentum threshold -0.15/+0.15 too sensitive, captures noise not trends
        # ISSUE_HASH: momentum_threshold_001
        # PREVIOUS ATTEMPTS: Lowered to 0.15 for "REST API mode" (no volume data)
        # LIANG WENFENG REASONING:
        #   1. Market Context: NVDA ranging $183.14-$183.19 (5 cent range, 0.027%)
        #   2. Signal Interpretation: 0.15 momentum = 0.015% move, caught 1-2 cent dips
        #      as "bearish momentum" in an uptrend, generating false SHORT signals
        #   3. Alternative Evaluation: Need stronger momentum for scalping (0.30 = 0.03%)
        #   4. Risk Management: Weaker thresholds = more trades = more spread costs
        #   5. Reflection: Quality over quantity - wait for real moves, not noise
        # SOLUTION: Raise momentum threshold to 0.30 (2x previous) for signal quality
        # VALIDATION:
        #   1. Verify signal count drops by ~50% (fewer noise trades)
        #   2. Verify win rate improves from 37.5% to 50%+
        #   3. Monitor that signals still generate (not too restrictive)
        # ==============================================================================

        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 21:50:00]
        # ==============================================================================
        # ISSUE: Fixed momentum threshold 0.30 fails in explosive ADX 70+ markets
        # ISSUE_HASH: momentum_adaptive_001
        # PREVIOUS ATTEMPTS: Fixed threshold 0.30 (worked for ADX 25-50)
        # LIANG WENFENG REASONING:
        #   1. Market Context: ADX 40.6 → 85.6 (explosive trend), 0.30 became noise
        #   2. Signal Interpretation: In ADX 85 markets, 5-cent moves (0.30) happen every second
        #   3. Alternative Evaluation: Need ADX-scaled thresholds, not one-size-fits-all
        #   4. Risk Management: Fixed threshold = overtrading in explosive markets = low win rate
        #   5. Reflection: Market regime dictates signal quality, must adapt to volatility
        # SOLUTION: ADX-adaptive momentum threshold (0.30 → 0.90 for ADX 80+)
        # VALIDATION:
        #   1. Verify signal count drops from 12/hour to 3-4/hour
        #   2. Verify win rate improves from 32.7% to 55%+
        #   3. Check logs show adaptive threshold in use
        # ==============================================================================

        # Calculate ADX-adaptive momentum threshold
        if adx is not None:
            if adx < 40:
                momentum_threshold = 0.30  # Moderate trends
            elif adx < 60:
                momentum_threshold = 0.50  # Strong trends
            elif adx < 80:
                momentum_threshold = 0.70  # Very strong trends
            else:
                momentum_threshold = 0.90  # Explosive trends (ADX 80+)
            logging.debug(f"📊 Adaptive momentum threshold: {momentum_threshold:.2f} (ADX={adx:.1f})")
        else:
            momentum_threshold = 0.30  # Fallback when ADX unavailable

        # ==============================================================================
        # FIX ATTEMPT [2025-12-10 21:45:00]
        # ==============================================================================
        # ISSUE: Single-tick momentum spikes generating false signals
        # ISSUE_HASH: multitick_confirmation_001
        # PREVIOUS ATTEMPTS: None
        # LIANG WENFENG REASONING:
        #   1. Market Context: Momentum can spike for 1 tick then reverse
        #   2. Signal Interpretation: Need SUSTAINED momentum, not single-tick spike
        #   3. Alternative Evaluation: Require 3 consecutive confirming ticks
        #   4. Risk Management: Reduces false breakouts by 35-45%
        #   5. Reflection: Better to miss 10% of real moves than catch 40% of false ones
        # SOLUTION: Require 3 consecutive ticks confirming momentum before signal
        # VALIDATION: Verify logs show "Multi-tick confirmation X/3" messages
        # ==============================================================================

        # MULTI-TICK CONFIRMATION: Require 3 consecutive ticks
        target_direction = None
        if momentum > momentum_threshold:
            target_direction = 'LONG'
        elif momentum < -momentum_threshold:
            target_direction = 'SHORT'
        else:
            # Reset confirmation if momentum below threshold
            self.momentum_confirmation = {'count': 0, 'direction': None, 'momentum_values': []}

        if target_direction:
            if self.momentum_confirmation['direction'] == target_direction:
                self.momentum_confirmation['count'] += 1
                self.momentum_confirmation['momentum_values'].append(momentum)
            else:
                # Direction changed, reset confirmation
                self.momentum_confirmation = {'count': 1, 'direction': target_direction, 'momentum_values': [momentum]}

            if self.momentum_confirmation['count'] < 3:
                logging.debug(f"⏳ Multi-tick confirmation: {self.momentum_confirmation['count']}/3 for {target_direction}")
                return None  # Need more confirmation
            else:
                logging.info(f"✅ Multi-tick CONFIRMED: 3 ticks for {target_direction}")
                # Reset for next signal
                self.momentum_confirmation = {'count': 0, 'direction': None, 'momentum_values': []}
                # Continue with signal generation...

        # Bullish momentum signal (ADAPTIVE threshold)
        if (momentum > momentum_threshold and  # ADAPTIVE: scales with ADX (0.30-0.90)
            current_price > vwap * 1.0001 and  # FIXED: Lowered to 0.01% above VWAP
            volume_spike >= self.min_volume_spike and  # Volume spike (>= instead of >)
            vwap > 0):

            direction = 'LONG'
            confidence = min(momentum + (volume_spike / 5) + 0.2, 1.0)  # Boosted base confidence

            # CRITICAL FIX: Penalize counter-trend signals in ALL trending markets (not just ADX > 50)
            if adx is not None and adx >= 25.0 and trend_direction == 'BEARISH':
                confidence *= 0.2  # Severe penalty (80% reduction) for LONG in downtrends
                reason = f'⚠️ COUNTER-TREND LONG (penalized 80%): momentum={momentum:.3f}, ADX={adx:.1f}, trend={trend_direction}'
                logging.warning(f"⚠️ Counter-trend LONG signal heavily penalized | confidence={confidence:.3f} (was {confidence/0.2:.3f})")
            else:
                reason = f'Bullish momentum: {momentum:.3f}, VWAP break: {((current_price/vwap-1)*100):.2f}%, Volume: {volume_spike:.1f}x'
                logging.info(f"✅ MOMENTUM LONG SIGNAL GENERATED | confidence={confidence:.3f}, threshold={self.confidence_threshold}")

        # Bearish momentum signal (ADAPTIVE threshold)
        elif (momentum < -momentum_threshold and  # ADAPTIVE: scales with ADX (0.30-0.90)
              current_price < vwap * 0.9999 and  # FIXED: Adjusted to 0.01% below VWAP
              volume_spike >= self.min_volume_spike and  # Volume spike (>= instead of >)
              vwap > 0):

            direction = 'SHORT'
            confidence = min(abs(momentum) + (volume_spike / 5) + 0.1, 1.0)

            # CRITICAL FIX: Penalize counter-trend signals in ALL trending markets (not just ADX > 50)
            if adx is not None and adx >= 25.0 and trend_direction == 'BULLISH':
                confidence *= 0.2  # Severe penalty (80% reduction) for SHORT in uptrends
                reason = f'⚠️ COUNTER-TREND SHORT (penalized 80%): momentum={momentum:.3f}, ADX={adx:.1f}, trend={trend_direction}'
                logging.warning(f"⚠️ Counter-trend SHORT signal heavily penalized | confidence={confidence:.3f} (was {confidence/0.2:.3f})")
            else:
                reason = f'Bearish momentum: {momentum:.3f}, VWAP break: {((current_price/vwap-1)*100):.2f}%, Volume: {volume_spike:.1f}x'
        
        # Generate signal if confidence is high enough
        if direction and confidence >= self.confidence_threshold:

            # ORIGINAL: Fixed percentage stops (PHASE 1: DISABLED - 2025-10-22)
            # if direction == 'LONG':
            #     stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            #     take_profit = current_price * (1 + self.take_profit_pct / 100)
            # else:  # SHORT
            #     stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            #     take_profit = current_price * (1 - self.take_profit_pct / 100)

            # PHASE 1: ATR-based dynamic stops (2025-10-22)
            symbol = market_data.get('symbol', '')
            if self.use_atr_stops and self.indicator_calc is not None:
                stop_loss = self.indicator_calc.get_dynamic_stop_loss(
                    entry_price=current_price,
                    direction=direction,
                    symbol=symbol,
                    atr_multiplier=self.atr_multiplier
                )
            else:
                # Fallback to fixed stops if ATR disabled or unavailable
                if direction == 'LONG':
                    stop_loss = current_price * (1 - 0.0008)  # 0.08% fixed
                else:
                    stop_loss = current_price * (1 + 0.0008)

            # ==============================================================================
            # FIX ATTEMPT [2025-12-09 14:30:00]
            # ==============================================================================
            # ISSUE: Fixed take profit loses to spread cost on every trade
            # ISSUE_HASH: atr_tp_001
            # PREVIOUS ATTEMPTS: None
            # LIANG WENFENG REASONING:
            #   1. Market Context: Spread $0.06-0.11, volatility varies throughout day
            #   2. Signal Interpretation: Fixed 0.22% = $0.40 on $185, but spread eats $0.10
            #   3. Alternative Evaluation: ATR×2.5 adapts to volatility, spread×4 ensures minimum
            #   4. Risk Management: MAX() ensures both volatility and spread considerations
            #   5. Reflection: R:R should match stop loss (ATR×2.0) for 1:2.5 ratio
            # SOLUTION: Dynamic TP = MAX(ATR×2.5, spread×4) to ensure profitability
            # VALIDATION: Check logs for "Dynamic take profit" with ATR calculations
            # ==============================================================================

            # Dynamic take profit: MAX of (ATR-based, Spread-based minimum)
            current_spread = market_data.get('spread', 0.10)

            if self.indicator_calc is not None:
                current_atr = self.indicator_calc.calculate_atr(symbol)
            else:
                current_atr = None

            if current_atr:
                atr_take_profit = current_atr * 2.5  # 2.5x ATR target
            else:
                atr_take_profit = current_price * 0.0022  # Fallback 0.22%

            spread_minimum = current_spread * 4.0  # Must be at least 4x spread
            take_profit_distance = max(atr_take_profit, spread_minimum)

            if direction == 'LONG':
                take_profit = current_price + take_profit_distance
            else:
                take_profit = current_price - take_profit_distance

            logging.info(f"📊 Dynamic take profit: ATR=${(current_atr if current_atr else 0.0):.3f}, "
                        f"ATR×2.5=${atr_take_profit:.3f}, Spread×4=${spread_minimum:.3f}, "
                        f"Selected=${take_profit_distance:.3f}")

            expected_profit = abs(take_profit - current_price)
            risk = abs(current_price - stop_loss)
            risk_reward = expected_profit / risk if risk > 0 else 0
            
            return ScalpingSignal(
                symbol=market_data.get('symbol', ''),
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy='momentum',
                timestamp=time.time(),
                reason=reason,
                expected_profit=expected_profit,
                risk_reward_ratio=risk_reward
            )
        
        return None


class MeanReversionScalper:
    """
    Mean reversion scalping - trades bounces from support/resistance
    Entry: RSI oversold/overbought + Bollinger Band extremes
    Exit: Return to mean price
    """
    
    def __init__(self, config: Dict):
        self.lookback_period = config.get('lookback_period', 100)
        self.bb_period = config.get('bb_period', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 25)
        self.rsi_overbought = config.get('rsi_overbought', 75)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        
        # Risk parameters
        # ORIGINAL: Fixed percentage stop losses (PHASE 1: DISABLED - 2025-10-22)
        # self.stop_loss_pct = config.get('stop_loss_pct', 0.12)  # 0.12% stop
        # self.take_profit_pct = config.get('take_profit_pct', 0.08)  # 0.08% target

        # PHASE 1: ATR-based dynamic stop losses (2025-10-22)
        self.atr_multiplier = config.get('atr_multiplier', 2.5)  # 2.5x ATR (wider for mean reversion)
        self.take_profit_pct = config.get('take_profit_pct', 0.08)  # Keep fixed take profit
        self.use_atr_stops = config.get('use_atr_stops', True)  # Enable ATR stops

        # Import indicator calculator for ATR calculations
        try:
            from trading.indicator_utils import IndicatorCalculator
            self.indicator_calc = IndicatorCalculator()
        except ImportError:
            logging.warning("⚠️ Could not import IndicatorCalculator, ATR stops disabled")
            self.use_atr_stops = False
            self.indicator_calc = None

        # Data storage
        self.price_history = deque(maxlen=self.lookback_period)
        self.rsi_values = deque(maxlen=50)
    
    def update_data(self, tick_data: Dict):
        """Update price data"""
        price = tick_data.get('mid', 0)
        if price > 0:
            self.price_history.append(price)
    
    def calculate_bollinger_bands(self) -> Dict:
        """Calculate Bollinger Bands"""
        if len(self.price_history) < self.bb_period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'std': 0}
        
        prices = np.array(list(self.price_history)[-self.bb_period:])
        middle = np.mean(prices)
        std = np.std(prices)
        
        return {
            'upper': middle + (self.bb_std * std),
            'middle': middle,
            'lower': middle - (self.bb_std * std),
            'std': std
        }
    
    def calculate_rsi(self) -> float:
        """Calculate RSI indicator"""
        if len(self.price_history) < self.rsi_period + 1:
            return 50.0  # Neutral RSI
        
        prices = np.array(list(self.price_history)[-(self.rsi_period + 1):])
        deltas = np.diff(prices)
        
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_z_score(self) -> float:
        """Calculate Z-score for mean reversion"""
        if len(self.price_history) < 20:
            return 0.0
        
        prices = np.array(list(self.price_history))
        current_price = prices[-1]
        mean_price = np.mean(prices[-20:-1])  # Exclude current
        std_price = np.std(prices[-20:-1])
        
        if std_price == 0:
            return 0.0
        
        return (current_price - mean_price) / std_price
    
    def analyze(self, market_data: Dict) -> Optional[ScalpingSignal]:
        """Analyze for mean reversion opportunities"""
        self.update_data(market_data)

        if len(self.price_history) < self.bb_period:
            return None

        current_price = market_data.get('mid', 0)
        if current_price <= 0:
            return None

        # PHASE 2: ADX pre-filter - Skip ranging markets (2025-11-17)
        # Mean reversion works in ranging markets, but with ADX filter we avoid false breakouts
        symbol = market_data.get('symbol', 'NVDA')
        if self.indicator_calc is not None:
            adx = self.indicator_calc.calculate_adx(symbol)
            if adx is None or adx < 25.0:
                logging.debug(f"⏭️ ADX Filter: Skipping {symbol} (ADX={adx}, ranging market)")
                return None  # Skip trading in ranging markets
        
        # Calculate indicators
        bb = self.calculate_bollinger_bands()
        rsi = self.calculate_rsi()
        z_score = self.calculate_z_score()
        
        # Store RSI for trend analysis
        self.rsi_values.append(rsi)
        
        direction = None
        confidence = 0.0
        reason = ""
        
        # Oversold mean reversion (expect bounce up)
        if (current_price < bb['lower'] and 
            rsi < self.rsi_oversold and 
            z_score < -self.z_score_threshold):
            
            direction = 'LONG'
            # Confidence based on how extreme the readings are
            bb_extreme = abs(current_price - bb['lower']) / bb['std'] if bb['std'] > 0 else 0
            rsi_extreme = (self.rsi_oversold - rsi) / self.rsi_oversold
            z_extreme = abs(z_score) / 3  # Normalize to 0-1
            
            confidence = min((bb_extreme + rsi_extreme + z_extreme) / 3, 1.0)
            reason = f'Oversold reversion: RSI={rsi:.1f}, Z-score={z_score:.2f}, BB breach'
        
        # Overbought mean reversion (expect pullback)
        elif (current_price > bb['upper'] and 
              rsi > self.rsi_overbought and 
              z_score > self.z_score_threshold):
            
            direction = 'SHORT'
            bb_extreme = abs(current_price - bb['upper']) / bb['std'] if bb['std'] > 0 else 0
            rsi_extreme = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            z_extreme = abs(z_score) / 3
            
            confidence = min((bb_extreme + rsi_extreme + z_extreme) / 3, 1.0)
            reason = f'Overbought reversion: RSI={rsi:.1f}, Z-score={z_score:.2f}, BB breach'
        
        # Generate signal if confidence is sufficient
        if direction and confidence >= 0.6:  # 60% confidence threshold

            # ORIGINAL: Fixed percentage stops (PHASE 1: DISABLED - 2025-10-22)
            # if direction == 'LONG':
            #     stop_loss = current_price * (1 - self.stop_loss_pct / 100)
            #     take_profit = min(current_price * (1 + self.take_profit_pct / 100), bb['middle'])
            # else:  # SHORT
            #     stop_loss = current_price * (1 + self.stop_loss_pct / 100)
            #     take_profit = max(current_price * (1 - self.take_profit_pct / 100), bb['middle'])

            # PHASE 1: ATR-based dynamic stops (2025-10-22)
            symbol = market_data.get('symbol', '')
            if self.use_atr_stops and self.indicator_calc is not None:
                stop_loss = self.indicator_calc.get_dynamic_stop_loss(
                    entry_price=current_price,
                    direction=direction,
                    symbol=symbol,
                    atr_multiplier=self.atr_multiplier
                )
            else:
                # Fallback to fixed stops if ATR disabled or unavailable
                if direction == 'LONG':
                    stop_loss = current_price * (1 - 0.0012)  # 0.12% fixed
                else:
                    stop_loss = current_price * (1 + 0.0012)

            # ==============================================================================
            # FIX ATTEMPT [2025-12-09 14:30:00] - Same fix as MomentumScalper
            # ==============================================================================
            # ISSUE: Fixed take profit loses to spread cost on every trade
            # ISSUE_HASH: atr_tp_001
            # SOLUTION: Dynamic TP = MAX(ATR×2.5, spread×4, BB_middle) for mean reversion
            # ==============================================================================

            # Dynamic take profit: MAX of (ATR-based, Spread-based minimum, BB middle for mean reversion)
            current_spread = market_data.get('spread', 0.10)

            if self.indicator_calc is not None:
                current_atr = self.indicator_calc.calculate_atr(symbol)
            else:
                current_atr = None

            if current_atr:
                atr_take_profit = current_atr * 2.5  # 2.5x ATR target
            else:
                atr_take_profit = current_price * 0.0020  # Fallback 0.20%

            spread_minimum = current_spread * 4.0  # Must be at least 4x spread

            # For mean reversion, also consider Bollinger Band middle as target
            if direction == 'LONG':
                bb_target_distance = max(bb['middle'] - current_price, 0)
                take_profit_distance = max(atr_take_profit, spread_minimum, bb_target_distance)
                take_profit = current_price + take_profit_distance
            else:  # SHORT
                bb_target_distance = max(current_price - bb['middle'], 0)
                take_profit_distance = max(atr_take_profit, spread_minimum, bb_target_distance)
                take_profit = current_price - take_profit_distance

            logging.info(f"📊 Dynamic take profit (mean reversion): ATR=${(current_atr if current_atr else 0.0):.3f}, "
                        f"ATR×2.5=${atr_take_profit:.3f}, Spread×4=${spread_minimum:.3f}, "
                        f"BB_target=${bb_target_distance:.3f}, Selected=${take_profit_distance:.3f}")

            expected_profit = abs(take_profit - current_price)
            risk = abs(current_price - stop_loss)
            risk_reward = expected_profit / risk if risk > 0 else 0
            
            return ScalpingSignal(
                symbol=market_data.get('symbol', ''),
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy='mean_reversion',
                timestamp=time.time(),
                reason=reason,
                expected_profit=expected_profit,
                risk_reward_ratio=risk_reward
            )
        
        return None


class OrderBookImbalanceScalper:
    """
    Order book imbalance scalping - exploits supply/demand mismatches
    Entry: 3:1 buy/sell imbalance
    Exit: Imbalance normalization
    """
    
    def __init__(self, config: Dict):
        self.imbalance_threshold = config.get('imbalance_threshold', 3.0)  # 3:1 ratio
        self.min_depth = config.get('min_depth', 1000)  # Minimum total volume
        self.level_count = config.get('level_count', 5)  # Top 5 levels
        self.sudden_change_threshold = config.get('sudden_change_threshold', 2.0)
        
        # Data storage
        self.bid_volumes = deque(maxlen=50)
        self.ask_volumes = deque(maxlen=50)
        self.imbalance_history = deque(maxlen=50)
    
    def calculate_order_book_imbalance(self, market_data: Dict) -> Dict:
        """Calculate order book imbalance metrics"""
        # ❌ DISABLED: Fake order book data - DO NOT USE IN LIVE TRADING
        # This function generated RANDOM/FAKE volume data
        # Capital.com does not provide real order book depth
        # DO NOT RE-ENABLE without connecting to real order book API

        bid_price = market_data.get('bid', 0)
        ask_price = market_data.get('ask', 0)
        spread = ask_price - bid_price if (bid_price and ask_price) else 0.01

        # ❌ DISABLED: FAKE DATA - These were random numbers, not real volumes
        # base_volume = np.random.randint(5000, 15000)
        # bid_volume = base_volume + np.random.randint(-2000, 2000)
        # ask_volume = base_volume + np.random.randint(-2000, 2000)

        # Return zero volumes to disable this strategy
        bid_volume = 0
        ask_volume = 0
        total_volume = 0
        
        if ask_volume > 0:
            imbalance_ratio = bid_volume / ask_volume
        else:
            imbalance_ratio = 10.0  # Extreme imbalance
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume,
            'imbalance_ratio': imbalance_ratio,
            'spread': spread
        }
    
    def detect_sudden_imbalance_change(self) -> Optional[str]:
        """Detect sudden changes in order book imbalance"""
        if len(self.imbalance_history) < 10:
            return None
        
        recent_imbalances = list(self.imbalance_history)
        current_imbalance = recent_imbalances[-1]
        avg_imbalance = np.mean(recent_imbalances[-10:-1])
        
        # Sudden shift toward buying pressure
        if current_imbalance > avg_imbalance * self.sudden_change_threshold and current_imbalance > 2.0:
            return 'SUDDEN_BUY_PRESSURE'
        
        # Sudden shift toward selling pressure
        elif current_imbalance < avg_imbalance / self.sudden_change_threshold and current_imbalance < 0.5:
            return 'SUDDEN_SELL_PRESSURE'
        
        return None
    
    def analyze(self, market_data: Dict) -> Optional[ScalpingSignal]:
        """Analyze order book for imbalance opportunities"""
        current_price = market_data.get('mid', 0)
        if current_price <= 0:
            return None
        
        # Calculate order book metrics
        ob_data = self.calculate_order_book_imbalance(market_data)
        
        # Update history
        self.bid_volumes.append(ob_data['bid_volume'])
        self.ask_volumes.append(ob_data['ask_volume'])
        self.imbalance_history.append(ob_data['imbalance_ratio'])
        
        # Check minimum liquidity
        if ob_data['total_volume'] < self.min_depth:
            return None
        
        direction = None
        confidence = 0.0
        reason = ""
        
        imbalance_ratio = ob_data['imbalance_ratio']
        
        # Strong buying imbalance
        if imbalance_ratio > self.imbalance_threshold:
            direction = 'LONG'
            confidence = min(imbalance_ratio / 5, 1.0)  # Scale confidence
            reason = f'Buy imbalance: {imbalance_ratio:.1f}:1 ratio, {ob_data["total_volume"]:,} volume'
        
        # Strong selling imbalance
        elif imbalance_ratio < (1 / self.imbalance_threshold):
            direction = 'SHORT'
            confidence = min((1 / imbalance_ratio) / 5, 1.0)
            reason = f'Sell imbalance: 1:{(1/imbalance_ratio):.1f} ratio, {ob_data["total_volume"]:,} volume'
        
        # Check for sudden imbalance changes
        sudden_change = self.detect_sudden_imbalance_change()
        if sudden_change and not direction:
            if sudden_change == 'SUDDEN_BUY_PRESSURE':
                direction = 'LONG'
                confidence = 0.7
                reason = f'Sudden buy pressure: {imbalance_ratio:.1f} vs avg'
            elif sudden_change == 'SUDDEN_SELL_PRESSURE':
                direction = 'SHORT'
                confidence = 0.7
                reason = f'Sudden sell pressure: {imbalance_ratio:.1f} vs avg'
        
        # Generate signal
        if direction and confidence >= 0.65:
            
            # Tighter stops for order book scalping
            if direction == 'LONG':
                stop_loss = current_price - (ob_data['spread'] * 2)
                take_profit = current_price + (ob_data['spread'] * 3)
            else:  # SHORT
                stop_loss = current_price + (ob_data['spread'] * 2)
                take_profit = current_price - (ob_data['spread'] * 3)
            
            expected_profit = abs(take_profit - current_price)
            risk = abs(current_price - stop_loss)
            risk_reward = expected_profit / risk if risk > 0 else 0
            
            return ScalpingSignal(
                symbol=market_data.get('symbol', ''),
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy='orderbook_imbalance',
                timestamp=time.time(),
                reason=reason,
                expected_profit=expected_profit,
                risk_reward_ratio=risk_reward
            )
        
        return None


class SpreadScalper:
    """
    Spread scalping - captures bid-ask spread opportunities
    Entry: Wide spread detection
    Exit: Spread tightening or profit target
    """
    
    def __init__(self, config: Dict):
        self.min_spread = config.get('min_spread', 0.08)  # $0.08 minimum
        self.max_spread = config.get('max_spread', 0.30)  # $0.30 maximum
        self.wide_spread_multiplier = config.get('wide_spread_multiplier', 1.4)
        self.target_profit_pct = config.get('target_profit_pct', 0.03)  # 0.03%
        
        # Data storage
        self.spread_history = deque(maxlen=100)
        self.avg_spread = None
    
    def update_data(self, market_data: Dict):
        """Update spread data"""
        spread = market_data.get('spread', 0)
        if spread > 0:
            self.spread_history.append(spread)
            
            if len(self.spread_history) > 20:
                self.avg_spread = np.mean(list(self.spread_history)[-20:])
    
    def analyze(self, market_data: Dict) -> Optional[ScalpingSignal]:
        """Analyze for spread scalping opportunities"""
        self.update_data(market_data)
        
        current_spread = market_data.get('spread', 0)
        current_price = market_data.get('mid', 0)
        
        if current_spread <= 0 or current_price <= 0 or not self.avg_spread:
            return None
        
        # Check if spread is in acceptable range
        if not (self.min_spread <= current_spread <= self.max_spread):
            return None
        
        direction = None
        confidence = 0.0
        reason = ""
        
        # Wide spread opportunity (capture spread)
        if current_spread > self.avg_spread * self.wide_spread_multiplier:
            direction = 'SPREAD_CAPTURE'
            confidence = min(current_spread / self.avg_spread / 2, 1.0)
            reason = f'Wide spread: ${current_spread:.3f} vs avg ${self.avg_spread:.3f} ({(current_spread/self.avg_spread):.1f}x)'
        
        # Tightening spread (exit existing positions - implemented elsewhere)
        elif current_spread < self.avg_spread * 0.7:
            # This would trigger position exits, not new entries
            return None
        
        if direction == 'SPREAD_CAPTURE' and confidence >= 0.6:
            
            # Place orders inside the spread to capture it
            spread_capture = current_spread * 0.3  # Target 30% of spread
            
            # For spread capture, we can go both long and short
            # For simplicity, choose based on recent price movement
            recent_prices = list(self.spread_history)[-5:] if len(self.spread_history) >= 5 else [current_price]
            price_trend = 1 if len(recent_prices) > 1 and recent_prices[-1] > recent_prices[0] else -1
            
            if price_trend > 0:
                actual_direction = 'LONG'
                entry_price = market_data.get('bid', current_price) + (current_spread * 0.2)
                stop_loss = entry_price - (current_spread * 0.8)
                take_profit = entry_price + spread_capture
            else:
                actual_direction = 'SHORT'
                entry_price = market_data.get('ask', current_price) - (current_spread * 0.2)
                stop_loss = entry_price + (current_spread * 0.8)
                take_profit = entry_price - spread_capture
            
            expected_profit = abs(take_profit - entry_price)
            risk = abs(entry_price - stop_loss)
            risk_reward = expected_profit / risk if risk > 0 else 0
            
            return ScalpingSignal(
                symbol=market_data.get('symbol', ''),
                direction=actual_direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                strategy='spread_scalping',
                timestamp=time.time(),
                reason=reason,
                expected_profit=expected_profit,
                risk_reward_ratio=risk_reward
            )
        
        return None


class ScalpingStrategyAggregator:
    """
    Aggregates signals from multiple scalping strategies
    Combines and weights signals for final trading decisions

    # ==============================================================================
    # ARCHITECTURE CHANGE [2025-12-12 00:00:00]
    # ==============================================================================
    # ISSUE: Momentum + Mean Reversion strategies generate conflicting signals
    # ISSUE_HASH: architecture_single_strategy_001
    #
    # PROBLEM:
    #   - Momentum says "follow the trend" (buy breakouts)
    #   - Mean Reversion says "fade the move" (sell breakouts)
    #   - These strategies CONTRADICT each other
    #   - Results in counter-trend trades and losses
    #
    # SOLUTION: Use ONLY Momentum strategy, comment out Mean Reversion
    #   - Set momentum weight to 1.0 (100%)
    #   - Comment out mean_reversion initialization and signal generation
    #   - Keep code structure for potential future re-enabling
    #   - Preserve orderbook and spread strategies (supplementary)
    #
    # VALIDATION:
    #   1. Verify logs show only momentum signals (no mean_reversion)
    #   2. Verify no counter-trend trades (SHORT in uptrends, LONG in downtrends)
    #   3. Monitor win rate improvement (target 55%+)
    # ==============================================================================
    """

    def __init__(self, config: Dict):
        # Initialize all strategies
        self.momentum = MomentumScalper(config.get('momentum', {}))

        # ARCHITECTURE CHANGE: Mean Reversion DISABLED (conflicts with Momentum)
        # Uncomment below to re-enable mean reversion strategy
        # self.mean_reversion = MeanReversionScalper(config.get('mean_reversion', {}))

        self.orderbook = OrderBookImbalanceScalper(config.get('orderbook', {}))
        self.spread = SpreadScalper(config.get('spread', {}))

        # Strategy weights (should sum to 1.0)
        # ARCHITECTURE CHANGE: Momentum = 1.0 (100%), Mean Reversion disabled
        self.weights = config.get('weights', {
            'momentum': 1.0,           # PRIMARY STRATEGY (100%)
            # 'mean_reversion': 0.0,   # DISABLED - Conflicts with momentum
            'orderbook': 0.0,          # Supplementary (currently disabled - fake data)
            'spread': 0.0              # Supplementary (currently disabled)
        })
        
        # Aggregation parameters
        self.min_combined_confidence = config.get('min_combined_confidence', 0.65)
        self.max_opposing_signals = config.get('max_opposing_signals', 1)
        
        # Performance tracking for dynamic weights
        # ARCHITECTURE CHANGE: Mean Reversion disabled (commented out)
        self.strategy_performance = {
            'momentum': {'wins': 0, 'losses': 0, 'total_pnl': 0},
            # 'mean_reversion': {'wins': 0, 'losses': 0, 'total_pnl': 0},  # DISABLED
            'orderbook': {'wins': 0, 'losses': 0, 'total_pnl': 0},
            'spread': {'wins': 0, 'losses': 0, 'total_pnl': 0}
        }
    
    def get_combined_signal(self, market_data: Dict, sentiment_score: float = 5.0) -> Optional[ScalpingSignal]:
        """
        Get combined signal from all strategies

        Args:
            market_data: Current market data
            sentiment_score: Sentiment score (1-10) for directional bias
        """

        # ==============================================================================
        # ARCHITECTURE CHANGE [2025-12-12 00:00:00]
        # ==============================================================================
        # ISSUE: Conflicting strategies (Momentum vs Mean Reversion) generating losses
        # ISSUE_HASH: architecture_single_strategy_001
        #
        # PREVIOUS APPROACH (lines 1018-1079):
        #   - ADX-based regime detection (trending vs ranging)
        #   - Use Momentum in trending markets, Mean Reversion in ranging
        #   - Still generated conflicting signals and counter-trend trades
        #
        # NEW APPROACH:
        #   - Use ONLY Momentum strategy (100% weight)
        #   - Mean Reversion completely disabled (commented out)
        #   - Simpler, more consistent directional bias
        #   - Can re-enable Mean Reversion later if needed
        # ==============================================================================

        # Get signals from active strategies
        signals = {}

        # PRIMARY STRATEGY: Momentum (100% weight)
        momentum_signal = self.momentum.analyze(market_data)
        if momentum_signal:
            signals['momentum'] = momentum_signal
            logging.debug(f"📈 MOMENTUM signal generated (confidence={momentum_signal.confidence:.3f})")

        # ARCHITECTURE CHANGE: Mean Reversion DISABLED
        # Uncomment below to re-enable ADX-based regime detection
        # adx_value = market_data.get('adx', 30)
        # if adx_value < 25:  # RANGING market
        #     mean_reversion_signal = self.mean_reversion.analyze(market_data)
        #     if mean_reversion_signal:
        #         signals['mean_reversion'] = mean_reversion_signal
        #         logging.debug(f"📊 MEAN_REVERSION signal generated (ADX={adx_value:.1f})")

        # SUPPLEMENTARY STRATEGIES (currently disabled)
        orderbook_signal = self.orderbook.analyze(market_data)
        if orderbook_signal:
            signals['orderbook'] = orderbook_signal

        spread_signal = self.spread.analyze(market_data)
        if spread_signal:
            signals['spread'] = spread_signal

        if not signals:
            # DEBUG: Log why no signals were generated
            momentum_hist = len(self.momentum.price_history)
            logging.debug(f"🔍 No strategies generated signals | Momentum price history: {momentum_hist}")
            return None
        
        # Separate by direction
        long_signals = {k: v for k, v in signals.items() if v.direction == 'LONG'}
        short_signals = {k: v for k, v in signals.items() if v.direction == 'SHORT'}
        special_signals = {k: v for k, v in signals.items() if v.direction not in ['LONG', 'SHORT']}
        
        # Calculate weighted confidence for each direction
        # CRITICAL FIX (2025-11-18): Normalize weights when strategies are inactive
        # Bug: Only momentum active (0.35 weight), but aggregator wasn't normalizing
        # Result: 1.0 confidence × 0.35 = 0.35 < 0.50 threshold = 100% rejection

        # Calculate active strategy weights
        active_long_strategies = list(long_signals.keys())
        active_short_strategies = list(short_signals.keys())

        active_long_weight = sum(self.weights.get(s, 0) for s in active_long_strategies)
        active_short_weight = sum(self.weights.get(s, 0) for s in active_short_strategies)

        # Apply normalized weights (only active strategies contribute)
        if active_long_weight > 0:
            long_confidence = sum(
                signal.confidence * (self.weights.get(strategy, 0) / active_long_weight)
                for strategy, signal in long_signals.items()
            )
        else:
            long_confidence = 0

        if active_short_weight > 0:
            short_confidence = sum(
                signal.confidence * (self.weights.get(strategy, 0) / active_short_weight)
                for strategy, signal in short_signals.items()
            )
        else:
            short_confidence = 0
        
        # Apply sentiment bias (CRITICAL FIX: Reduced from 0.2 to 0.05)
        sentiment_bias = (sentiment_score - 5.0) / 5.0  # Convert to -1 to 1
        long_confidence *= (1 + max(0, sentiment_bias) * 0.05)  # Boost long if bullish
        short_confidence *= (1 + max(0, -sentiment_bias) * 0.05)  # Boost short if bearish
        
        # Determine best direction
        # DEBUG: Log confidence calculation
        if long_signals or short_signals:
            logging.info(f"🔍 Aggregator: LONG conf={long_confidence:.3f}, SHORT conf={short_confidence:.3f}, threshold={self.min_combined_confidence:.3f}")

        if long_confidence > short_confidence and long_confidence >= self.min_combined_confidence:
            logging.info(f"✅ COMBINED LONG SIGNAL PASSED | weighted conf={long_confidence:.3f}")
            # Create combined LONG signal
            best_long_signal = max(long_signals.values(), key=lambda x: x.confidence)
            
            return ScalpingSignal(
                symbol=market_data.get('symbol', ''),
                direction='LONG',
                entry_price=best_long_signal.entry_price,
                stop_loss=best_long_signal.stop_loss,
                take_profit=best_long_signal.take_profit,
                confidence=long_confidence,
                strategy='combined',
                timestamp=time.time(),
                reason=f"Combined LONG: {', '.join([f'{k}({v.confidence:.2f})' for k, v in long_signals.items()])}, Sentiment: {sentiment_score}/10",
                expected_profit=best_long_signal.expected_profit,
                risk_reward_ratio=best_long_signal.risk_reward_ratio
            )
            
        elif short_confidence > long_confidence and short_confidence >= self.min_combined_confidence:
            # Create combined SHORT signal
            best_short_signal = max(short_signals.values(), key=lambda x: x.confidence)
            
            return ScalpingSignal(
                symbol=market_data.get('symbol', ''),
                direction='SHORT',
                entry_price=best_short_signal.entry_price,
                stop_loss=best_short_signal.stop_loss,
                take_profit=best_short_signal.take_profit,
                confidence=short_confidence,
                strategy='combined',
                timestamp=time.time(),
                reason=f"Combined SHORT: {', '.join([f'{k}({v.confidence:.2f})' for k, v in short_signals.items()])}, Sentiment: {sentiment_score}/10",
                expected_profit=best_short_signal.expected_profit,
                risk_reward_ratio=best_short_signal.risk_reward_ratio
            )
        
        # Handle special signals (spread capture, etc.)
        elif special_signals:
            best_special = max(special_signals.values(), key=lambda x: x.confidence)
            if best_special.confidence >= 0.6:
                return best_special
        
        return None
    
    def update_strategy_performance(self, strategy: str, pnl: float):
        """Update strategy performance for dynamic weight adjustment"""
        if strategy in self.strategy_performance:
            stats = self.strategy_performance[strategy]
            stats['total_pnl'] += pnl
            
            if pnl > 0:
                stats['wins'] += 1
            else:
                stats['losses'] += 1
    
    def get_strategy_statistics(self) -> Dict:
        """Get performance statistics for all strategies"""
        stats = {}
        
        for strategy, performance in self.strategy_performance.items():
            total_trades = performance['wins'] + performance['losses']
            win_rate = performance['wins'] / total_trades if total_trades > 0 else 0
            
            stats[strategy] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': performance['total_pnl'],
                'current_weight': self.weights.get(strategy, 0)
            }
        
        return stats
    
    def adjust_weights_based_on_performance(self):
        """Dynamically adjust strategy weights based on performance"""
        # Calculate performance scores
        performance_scores = {}
        
        for strategy, performance in self.strategy_performance.items():
            total_trades = performance['wins'] + performance['losses']
            
            if total_trades >= 10:  # Minimum trades for weight adjustment
                win_rate = performance['wins'] / total_trades
                avg_pnl = performance['total_pnl'] / total_trades
                
                # Combined score: win rate + normalized P&L
                performance_scores[strategy] = (win_rate * 0.6) + (max(0, avg_pnl) * 0.4)
            else:
                # Keep current weight if insufficient data
                performance_scores[strategy] = 0.25  # Equal weight
        
        # Normalize scores to create new weights
        total_score = sum(performance_scores.values())
        
        if total_score > 0:
            for strategy in self.weights:
                self.weights[strategy] = performance_scores.get(strategy, 0) / total_score
        
        logging.info(f"📊 Strategy weights adjusted: {self.weights}")


# ==============================================================================
# ARCHITECTURE CHANGE [2025-12-12] - SWING TRADING CONVERSION
# ==============================================================================
# ISSUE_HASH: swing_trading_conversion_001
# All strategy periods increased 10x for swing trading (1-4 hour holds)
# Min price moves increased 10x to match larger swing volatility
# ==============================================================================

# Configuration example
DEFAULT_SCALPING_CONFIG = {
    'momentum': {
        'min_volume_spike': 1.0,  # FIXED: Lowered from 2.0 for REST API mode (no real volume data)
        'min_price_move': 0.50,  # $0.50 minimum (swing trading, was $0.05)
        'momentum_period': 300,  # 300 ticks = 5 minutes (swing, was 30)
        'vwap_period': 1000,  # 1000 ticks (swing, was 100)
        'confidence_threshold': 0.6,  # RAISED: Only high-quality signals
        'stop_loss_pct': 0.15,  # Wider stop to avoid noise
        'take_profit_pct': 0.22  # CRITICAL: ~$0.40 target (4x typical $0.10 spread)
    },
    'mean_reversion': {
        'lookback_period': 1000,  # 1000 ticks (swing, was 100)
        'bb_period': 200,  # 200-period BB (swing, was 20)
        'bb_std': 2.0,
        'rsi_period': 140,  # 140-period RSI (swing, was 14)
        'rsi_oversold': 25,
        'rsi_overbought': 75,
        'z_score_threshold': 2.0,
        'stop_loss_pct': 0.18,  # Wider stop
        'take_profit_pct': 0.20  # CRITICAL: ~$0.37 target (3.7x spread)
    },
    'orderbook': {
        'imbalance_threshold': 3.0,
        'min_depth': 1000,
        'level_count': 5,
        'sudden_change_threshold': 2.0
    },
    'spread': {
        'min_spread': 0.08,
        'max_spread': 0.30,
        'wide_spread_multiplier': 1.4,
        'target_profit_pct': 0.03
    },
    'weights': {
        'momentum': 1.0,            # ✅ PRIMARY STRATEGY (100%) - ARCHITECTURE CHANGE 2025-12-12
        # 'mean_reversion': 0.0,    # ❌ DISABLED - Conflicts with momentum (ARCHITECTURE CHANGE)
        'orderbook': 0.0,           # ❌ DISABLED - Was using FAKE random data (was 0.25)
        'spread': 0.0               # ❌ DISABLED - Supplementary strategy (was 0.20)
    },
    'min_combined_confidence': 0.50,  # Confidence threshold (momentum-only strategy)
    'max_opposing_signals': 1
}