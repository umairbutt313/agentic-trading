#!/usr/bin/env python3
"""
Technical Indicator Utilities using pandas-ta
All indicators use REAL Capital.com price data

PHASE 1: ATR (Average True Range) for dynamic stop losses - Created: 2025-10-22
PHASE 2: ADX (Trend Filter) + RSI (Momentum) - Created: 2025-11-17
PHASE 3: MACD (Trend Confirmation) - Created: 2025-11-17
"""

import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from price_storage import PriceStorage

logger = logging.getLogger('IndicatorCalculator')

class IndicatorCalculator:
    """
    Centralized technical indicator calculations using pandas-ta
    All indicators calculate from REAL Capital.com price data

    PHASE 1: ATR dynamic stop losses
    PHASE 2: ADX trend filter + RSI momentum
    PHASE 3: MACD trend confirmation
    """

    def __init__(self):
        self.storage = PriceStorage()
        logger.info("📊 IndicatorCalculator initialized - Phases 1-3: ATR, ADX, RSI, MACD")

    def _get_dataframe(self, symbol: str, hours: int = 2) -> Optional[pd.DataFrame]:
        """
        Get price history as DataFrame for pandas-ta calculations
        Returns DataFrame with OHLC columns from REAL Capital.com data

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            hours: Hours of historical data to retrieve

        Returns:
            DataFrame with high, low, close columns or None if insufficient data
        """
        try:
            prices = self.storage.get_price_history(symbol, hours_back=hours)

            if not prices or len(prices) < 14:
                logger.warning(f"⚠️ Insufficient price data for {symbol}: {len(prices) if prices else 0} points")
                return None

            df = pd.DataFrame(prices)

            # Ensure we have required OHLC columns
            # Capital.com provides: timestamp, price (mid), bid, ask
            # We need: high, low, close for technical indicators

            if 'close' not in df.columns:
                if 'price' in df.columns:
                    df['close'] = df['price']
                else:
                    logger.error(f"❌ No price data in columns: {df.columns.tolist()}")
                    return None

            # If high/low not provided, estimate from bid/ask spread
            if 'high' not in df.columns:
                if 'ask' in df.columns:
                    df['high'] = df['ask']
                else:
                    df['high'] = df['close']

            if 'low' not in df.columns:
                if 'bid' in df.columns:
                    df['low'] = df['bid']
                else:
                    df['low'] = df['close']

            logger.debug(f"✅ Got {len(df)} price points for {symbol}")
            return df

        except Exception as e:
            logger.error(f"⚠️ Error getting price data for {symbol}: {e}")
            return None

    # ========================================
    # PHASE 1: ATR (Average True Range)
    # ========================================

    def calculate_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range from REAL Capital.com data

        ATR measures market volatility - essential for dynamic stop losses.
        Higher ATR = more volatile = wider stops needed.
        Lower ATR = less volatile = tighter stops possible.

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            period: ATR period (default 14)

        Returns:
            Current ATR value or None if insufficient data

        Example:
            If NVDA ATR = $1.50, this means typical price movement is $1.50/period
        """
        df = self._get_dataframe(symbol, hours=2)

        if df is None or len(df) < period:
            logger.warning(f"⚠️ Cannot calculate ATR for {symbol}: insufficient data")
            return None

        try:
            # Calculate ATR using pandas-ta with REAL data
            atr_series = ta.atr(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=period
            )

            # Get the most recent ATR value
            current_atr = float(atr_series.iloc[-1])

            # Validate ATR is reasonable (not NaN, not zero)
            if pd.isna(current_atr) or current_atr <= 0:
                logger.warning(f"⚠️ Invalid ATR calculated for {symbol}: {current_atr}")
                return None

            # CRITICAL: Enforce minimum ATR (0.05% of price)
            current_price = float(df['close'].iloc[-1])
            min_atr = current_price * 0.0005  # 0.05% minimum

            if current_atr < min_atr:
                logger.warning(f"⚠️ ATR {current_atr:.4f} below minimum {min_atr:.4f}, using minimum")
                current_atr = min_atr

            logger.debug(f"📊 ATR for {symbol}: ${current_atr:.3f}")
            return current_atr

        except Exception as e:
            logger.error(f"⚠️ Error calculating ATR for {symbol}: {e}")
            return None

    def get_dynamic_stop_loss(
        self,
        entry_price: float,
        direction: str,
        symbol: str,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate dynamic stop loss based on ATR (Average True Range)

        This replaces fixed percentage stops with volatility-based stops:
        - High volatility = wider stops (avoid premature stop-outs)
        - Low volatility = tighter stops (better risk control)

        Args:
            entry_price: Position entry price
            direction: 'LONG' or 'SHORT'
            symbol: Stock symbol
            atr_multiplier: How many ATRs away (default 2.0)
                           2.0 = conservative (wider stops)
                           1.5 = moderate
                           1.0 = aggressive (tighter stops)

        Returns:
            Stop loss price based on current ATR

        Example:
            Entry: $150, ATR: $1.50, Multiplier: 2.0, Direction: LONG
            Stop Loss: $150 - (2.0 × $1.50) = $147.00
        """
        current_atr = self.calculate_atr(symbol)

        # Fallback to fixed percentage if ATR unavailable
        if current_atr is None:
            logger.warning(f"⚠️ ATR unavailable for {symbol}, using fixed 0.08% stop")
            if direction == 'LONG':
                return entry_price * (1 - 0.0008)  # 0.08% fixed stop
            else:
                return entry_price * (1 + 0.0008)

        # Calculate ATR-based stop loss
        atr_distance = atr_multiplier * current_atr

        if direction == 'LONG':
            stop_loss = entry_price - atr_distance
        else:  # SHORT
            stop_loss = entry_price + atr_distance

        # Calculate percentage for logging
        stop_distance_pct = abs(stop_loss - entry_price) / entry_price * 100

        # Log ATR-based stop calculation
        logger.info(f"📊 ATR Stop Loss Calculation:")
        logger.info(f"   Symbol: {symbol} | Direction: {direction}")
        logger.info(f"   Entry: ${entry_price:.2f}")
        logger.info(f"   ATR: ${current_atr:.3f} × {atr_multiplier} = ${atr_distance:.3f}")
        logger.info(f"   Stop Loss: ${stop_loss:.2f} ({stop_distance_pct:.2f}% from entry)")

        return stop_loss

    def get_atr_info(self, symbol: str) -> Dict:
        """
        Get detailed ATR information for a symbol
        Useful for monitoring and debugging

        Args:
            symbol: Stock symbol

        Returns:
            Dict with ATR details and recommendations
        """
        current_atr = self.calculate_atr(symbol)

        if current_atr is None:
            return {
                'atr': None,
                'status': 'unavailable',
                'recommendation': 'Use fixed stops'
            }

        # Get recent price for context
        df = self._get_dataframe(symbol, hours=1)
        current_price = float(df['close'].iloc[-1]) if df is not None and len(df) > 0 else 0

        # Calculate ATR as percentage of price
        atr_pct = (current_atr / current_price * 100) if current_price > 0 else 0

        # Determine volatility level
        if atr_pct > 2.0:
            volatility = "Very High"
            recommendation = "Use 2.5x ATR multiplier (wider stops)"
        elif atr_pct > 1.0:
            volatility = "High"
            recommendation = "Use 2.0x ATR multiplier (standard)"
        elif atr_pct > 0.5:
            volatility = "Moderate"
            recommendation = "Use 1.5x ATR multiplier"
        else:
            volatility = "Low"
            recommendation = "Use 1.0x ATR multiplier (tight stops)"

        return {
            'atr': current_atr,
            'atr_pct': atr_pct,
            'current_price': current_price,
            'volatility': volatility,
            'status': 'available',
            'recommendation': recommendation
        }

    # ========================================
    # PHASE 2: ADX (Average Directional Index)
    # ========================================

    def calculate_adx(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate ADX (Average Directional Index) - Trend Strength Filter

        ADX measures trend strength (NOT direction):
        - ADX > 25 = Strong trend (trade)
        - ADX 20-25 = Weak trend (caution)
        - ADX < 20 = No trend/ranging (skip)

        Critical for scalping: Avoid ranging markets where indicators fail.

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            period: ADX period (default 14)

        Returns:
            ADX value 0-100 (higher = stronger trend) or None if insufficient data

        Example:
            ADX = 32 → Strong trend, good for scalping
            ADX = 18 → Ranging market, skip trading
        """
        df = self._get_dataframe(symbol, hours=4)  # Need more data for ADX

        if df is None or len(df) < period * 2:
            logger.warning(f"⚠️ Insufficient data for ADX: {len(df) if df is not None else 0} points (need {period * 2})")
            return None

        try:
            # Calculate ADX using pandas-ta
            adx_df = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=period
            )

            # ADX column name format: ADX_{period}
            adx_col = f'ADX_{period}'
            if adx_col not in adx_df.columns:
                logger.error(f"⚠️ ADX column {adx_col} not found. Available: {adx_df.columns.tolist()}")
                return None

            current_adx = float(adx_df[adx_col].iloc[-1])

            # Validate ADX is reasonable
            if pd.isna(current_adx) or current_adx < 0 or current_adx > 100:
                logger.warning(f"⚠️ Invalid ADX for {symbol}: {current_adx}")
                return None

            logger.debug(f"📊 ADX for {symbol}: {current_adx:.1f}")
            return current_adx

        except Exception as e:
            logger.error(f"⚠️ Error calculating ADX for {symbol}: {e}")
            return None

    def is_trending_market(self, symbol: str, threshold: float = 25.0) -> bool:
        """
        Check if market is trending (ADX above threshold)

        Use this as a FILTER before trading:
        - Only trade when is_trending_market() returns True
        - Skip trades when False (ranging market)

        Args:
            symbol: Stock symbol
            threshold: ADX threshold
                      20 = Aggressive (more trades, lower quality)
                      25 = Standard (balanced) ✅ RECOMMENDED
                      30 = Conservative (fewer trades, higher quality)

        Returns:
            True if trending (ADX >= threshold), False otherwise

        Example:
            if not is_trending_market('NVDA', 25):
                logger.info("⏭️ SKIPPING - Ranging market")
                return None  # Skip this trade
        """
        adx = self.calculate_adx(symbol)

        if adx is None:
            logger.warning(f"⚠️ ADX unavailable for {symbol}, assuming not trending (conservative)")
            return False  # Conservative: if can't calculate, don't trade

        is_trending = adx >= threshold

        logger.info(f"📊 ADX Filter: {symbol} ADX={adx:.1f} | Threshold={threshold} | "
                   f"Trending={'YES ✅' if is_trending else 'NO ❌'}")

        return is_trending

    def get_trend_direction(self, symbol: str, period: int = 14) -> Optional[str]:
        """
        Get trend direction using DI+ and DI- from ADX calculation

        CRITICAL FIX: Prevents trading against strong trends
        - If DI+ > DI-: Uptrend (favor LONG)
        - If DI- > DI+: Downtrend (favor SHORT)

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            period: ADX period (default 14)

        Returns:
            'BULLISH' if DI+ > DI-, 'BEARISH' if DI- > DI+, None if insufficient data
        """
        df = self._get_dataframe(symbol, hours=4)

        if df is None or len(df) < period * 2:
            logger.warning(f"⚠️ Insufficient data for trend direction: {len(df) if df is not None else 0} points")
            return None

        try:
            # Calculate ADX (includes DI+ and DI- as DMP and DMN)
            adx_df = ta.adx(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                length=period
            )

            # Column names: DMP_{period} (DI+) and DMN_{period} (DI-)
            dmp_col = f'DMP_{period}'
            dmn_col = f'DMN_{period}'

            if dmp_col not in adx_df.columns or dmn_col not in adx_df.columns:
                logger.error(f"⚠️ DI columns not found. Available: {adx_df.columns.tolist()}")
                return None

            di_plus = float(adx_df[dmp_col].iloc[-1])
            di_minus = float(adx_df[dmn_col].iloc[-1])

            # Validate values
            if pd.isna(di_plus) or pd.isna(di_minus):
                logger.warning(f"⚠️ Invalid DI values for {symbol}: DI+={di_plus}, DI-={di_minus}")
                return None

            trend = 'BULLISH' if di_plus > di_minus else 'BEARISH'
            logger.debug(f"📊 Trend direction for {symbol}: {trend} (DI+={di_plus:.1f}, DI-={di_minus:.1f})")

            return trend

        except Exception as e:
            logger.error(f"⚠️ Error calculating trend direction for {symbol}: {e}")
            return None

    # ========================================
    # PHASE 2: RSI (Relative Strength Index)
    # ========================================

    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate RSI (Relative Strength Index) - Momentum Indicator

        RSI measures momentum on 0-100 scale:
        - RSI > 70 = Overbought (potential reversal)
        - RSI 50-70 = Bullish momentum
        - RSI 30-50 = Bearish momentum
        - RSI < 30 = Oversold (potential reversal)

        Best use for scalping: Confirmation, NOT entry signal

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            period: RSI period (default 14)

        Returns:
            RSI value 0-100 or None if insufficient data

        Example:
            Signal = LONG, RSI = 55 → Good confirmation (bullish momentum)
            Signal = LONG, RSI = 25 → Weak confirmation (oversold)
        """
        df = self._get_dataframe(symbol, hours=2)

        if df is None or len(df) < period + 10:
            logger.warning(f"⚠️ Insufficient data for RSI: {len(df) if df is not None else 0} points")
            return None

        try:
            # Calculate RSI using pandas-ta
            rsi_series = ta.rsi(df['close'], length=period)
            current_rsi = float(rsi_series.iloc[-1])

            # Validate RSI
            if pd.isna(current_rsi) or current_rsi < 0 or current_rsi > 100:
                logger.warning(f"⚠️ Invalid RSI for {symbol}: {current_rsi}")
                return None

            logger.debug(f"📊 RSI for {symbol}: {current_rsi:.1f}")
            return current_rsi

        except Exception as e:
            logger.error(f"⚠️ Error calculating RSI for {symbol}: {e}")
            return None

    # ========================================
    # PHASE 3: MACD (Moving Average Convergence Divergence)
    # ========================================

    def calculate_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        MACD shows trend momentum and direction:
        - MACD line > Signal line = Bullish
        - MACD line < Signal line = Bearish
        - Histogram > 0 = Bullish momentum increasing
        - Histogram < 0 = Bearish momentum increasing

        For scalping: Use fast settings (3/10/5) instead of standard (12/26/9)

        Args:
            symbol: Stock symbol (e.g., 'NVDA')
            fast: Fast EMA period (default 12, use 3 for scalping)
            slow: Slow EMA period (default 26, use 10 for scalping)
            signal: Signal line period (default 9, use 5 for scalping)

        Returns:
            Dict with 'macd', 'signal', 'histogram' or None if insufficient data

        Example:
            macd = calculate_macd('NVDA', fast=3, slow=10, signal=5)
            if macd['histogram'] > 0:
                # Bullish MACD confirmation
        """
        df = self._get_dataframe(symbol, hours=4)

        if df is None or len(df) < slow + signal + 10:
            logger.warning(f"⚠️ Insufficient data for MACD: {len(df) if df is not None else 0} points")
            return None

        try:
            # Calculate MACD using pandas-ta
            macd_df = ta.macd(df['close'], fast=fast, slow=slow, signal=signal)

            # MACD columns format: MACD_{fast}_{slow}_{signal}, MACDh_{fast}_{slow}_{signal}, MACDs_{fast}_{slow}_{signal}
            macd_col = f'MACD_{fast}_{slow}_{signal}'
            signal_col = f'MACDs_{fast}_{slow}_{signal}'
            hist_col = f'MACDh_{fast}_{slow}_{signal}'

            if macd_col not in macd_df.columns:
                logger.error(f"⚠️ MACD columns not found. Available: {macd_df.columns.tolist()}")
                return None

            result = {
                'macd': float(macd_df[macd_col].iloc[-1]),
                'signal': float(macd_df[signal_col].iloc[-1]),
                'histogram': float(macd_df[hist_col].iloc[-1])
            }

            # Validate values
            if any(pd.isna(v) for v in result.values()):
                logger.warning(f"⚠️ Invalid MACD values for {symbol}: {result}")
                return None

            logger.debug(f"📊 MACD for {symbol}: MACD={result['macd']:.4f}, "
                        f"Signal={result['signal']:.4f}, Histogram={result['histogram']:.4f}")
            return result

        except Exception as e:
            logger.error(f"⚠️ Error calculating MACD for {symbol}: {e}")
            return None

    # ========================================
    # COMBINED INDICATOR METHODS
    # ========================================

    def get_momentum_confirmation(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive momentum confirmation using RSI + MACD

        Combines RSI and MACD into a single confidence score:
        - bullish_score: 0.0 to 1.0 (higher = stronger bullish)
        - bearish_score: 0.0 to 1.0 (higher = stronger bearish)

        Use this to validate signals from your momentum/mean reversion scalpers.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with RSI, MACD values, and bullish/bearish scores

        Example:
            confirm = get_momentum_confirmation('NVDA')
            if signal == 'LONG' and confirm['bullish_score'] < 0.5:
                logger.info("⏭️ SKIPPING - Weak momentum confirmation")
                return None
        """
        rsi = self.calculate_rsi(symbol)
        macd = self.calculate_macd(symbol, fast=12, slow=26, signal=9)  # Standard for now

        if rsi is None or macd is None:
            logger.warning(f"⚠️ Cannot calculate momentum confirmation for {symbol} (RSI or MACD unavailable)")
            return None

        try:
            # Calculate bullish/bearish scores (0.0 to 1.0)
            bullish_score = 0.0
            bearish_score = 0.0

            # RSI component (0.5 weight)
            # RSI > 50 = bullish, RSI < 50 = bearish
            if rsi > 50:
                bullish_score += ((rsi - 50) / 50) * 0.5  # 0.0 to 0.5
            else:
                bearish_score += ((50 - rsi) / 50) * 0.5  # 0.0 to 0.5

            # MACD component (0.5 weight)
            # Histogram > 0 = bullish, Histogram < 0 = bearish
            if macd['histogram'] > 0:
                bullish_score += 0.5
            else:
                bearish_score += 0.5

            # Determine dominant direction
            dominant = 'BULLISH' if bullish_score > bearish_score else 'BEARISH'
            confidence = max(bullish_score, bearish_score)

            result = {
                'rsi': rsi,
                'macd': macd['macd'],
                'macd_signal': macd['signal'],
                'macd_histogram': macd['histogram'],
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'dominant': dominant,
                'confidence': confidence
            }

            logger.debug(f"📊 Momentum Confirmation for {symbol}: "
                        f"{dominant} (confidence={confidence:.2f}), "
                        f"RSI={rsi:.1f}, MACD_hist={macd['histogram']:.4f}")

            return result

        except Exception as e:
            logger.error(f"⚠️ Error calculating momentum confirmation for {symbol}: {e}")
            return None


# Convenience function for quick ATR calculation
def calculate_atr_stop(symbol: str, entry_price: float, direction: str, multiplier: float = 2.0) -> float:
    """
    Quick function to calculate ATR-based stop loss

    Usage:
        stop = calculate_atr_stop('NVDA', 150.00, 'LONG', 2.0)
    """
    calc = IndicatorCalculator()
    return calc.get_dynamic_stop_loss(entry_price, direction, symbol, multiplier)


if __name__ == "__main__":
    # Test ALL indicators
    print("🧪 Testing Indicator Calculator - Phases 1-3\n")
    print("=" * 60)

    calc = IndicatorCalculator()
    symbol = 'NVDA'

    # ========================================
    # PHASE 1: ATR Tests
    # ========================================
    print("\n📊 PHASE 1: ATR (Average True Range)")
    print("-" * 60)

    atr = calc.calculate_atr(symbol)
    if atr:
        print(f"✅ ATR for {symbol}: ${atr:.3f}")

        # Test stop loss calculation
        entry_price = 150.00
        print(f"\nDynamic Stop Loss Examples (Entry: ${entry_price:.2f}):")

        for direction in ['LONG', 'SHORT']:
            print(f"\n  {direction} Position:")
            for multiplier in [1.5, 2.0, 2.5]:
                stop = calc.get_dynamic_stop_loss(entry_price, direction, symbol, multiplier)
                distance = abs(stop - entry_price)
                pct = (distance / entry_price) * 100
                print(f"    {multiplier}x ATR: ${stop:.2f} (${distance:.2f} / {pct:.2f}%)")

        # Get detailed info
        print("\n  ATR Details:")
        info = calc.get_atr_info(symbol)
        for key, value in info.items():
            print(f"    {key}: {value}")
    else:
        print(f"❌ Could not calculate ATR for {symbol}")

    # ========================================
    # PHASE 2: ADX Tests
    # ========================================
    print("\n\n📊 PHASE 2: ADX (Trend Strength Filter)")
    print("-" * 60)

    adx = calc.calculate_adx(symbol)
    if adx:
        print(f"✅ ADX for {symbol}: {adx:.1f}")

        # Interpret ADX
        if adx > 30:
            trend_strength = "Very Strong Trend"
            recommendation = "EXCELLENT for scalping ✅"
        elif adx > 25:
            trend_strength = "Strong Trend"
            recommendation = "GOOD for scalping ✅"
        elif adx > 20:
            trend_strength = "Weak Trend"
            recommendation = "CAUTION - marginal ⚠️"
        else:
            trend_strength = "No Trend / Ranging"
            recommendation = "SKIP trading ❌"

        print(f"  Strength: {trend_strength}")
        print(f"  Recommendation: {recommendation}")

        # Test is_trending_market with different thresholds
        print("\n  Trend Filter Tests:")
        for threshold in [20, 25, 30]:
            is_trending = calc.is_trending_market(symbol, threshold)
            status = "✅ TRADE" if is_trending else "❌ SKIP"
            print(f"    Threshold {threshold}: {status}")
    else:
        print(f"❌ Could not calculate ADX for {symbol}")

    # ========================================
    # PHASE 2: RSI Tests
    # ========================================
    print("\n\n📊 PHASE 2: RSI (Momentum Indicator)")
    print("-" * 60)

    rsi = calc.calculate_rsi(symbol)
    if rsi:
        print(f"✅ RSI for {symbol}: {rsi:.1f}")

        # Interpret RSI
        if rsi > 70:
            rsi_zone = "Overbought"
            signal = "Potential reversal DOWN ⚠️"
        elif rsi > 50:
            rsi_zone = "Bullish Momentum"
            signal = "Good for LONG signals ✅"
        elif rsi > 30:
            rsi_zone = "Bearish Momentum"
            signal = "Good for SHORT signals ✅"
        else:
            rsi_zone = "Oversold"
            signal = "Potential reversal UP ⚠️"

        print(f"  Zone: {rsi_zone}")
        print(f"  Signal: {signal}")
    else:
        print(f"❌ Could not calculate RSI for {symbol}")

    # ========================================
    # PHASE 3: MACD Tests
    # ========================================
    print("\n\n📊 PHASE 3: MACD (Trend Confirmation)")
    print("-" * 60)

    # Test standard MACD
    macd_std = calc.calculate_macd(symbol)
    if macd_std:
        print(f"✅ MACD Standard (12/26/9) for {symbol}:")
        print(f"  MACD Line: {macd_std['macd']:.4f}")
        print(f"  Signal Line: {macd_std['signal']:.4f}")
        print(f"  Histogram: {macd_std['histogram']:.4f}")

        # Interpret MACD
        if macd_std['histogram'] > 0:
            macd_signal = "BULLISH - Histogram > 0 ✅"
        else:
            macd_signal = "BEARISH - Histogram < 0 ⚠️"
        print(f"  Signal: {macd_signal}")
    else:
        print(f"❌ Could not calculate MACD for {symbol}")

    # Test fast MACD for scalping
    macd_fast = calc.calculate_macd(symbol, fast=3, slow=10, signal=5)
    if macd_fast:
        print(f"\n✅ MACD Fast (3/10/5) for scalping:")
        print(f"  MACD Line: {macd_fast['macd']:.4f}")
        print(f"  Signal Line: {macd_fast['signal']:.4f}")
        print(f"  Histogram: {macd_fast['histogram']:.4f}")

        if macd_fast['histogram'] > 0:
            macd_signal = "BULLISH - Histogram > 0 ✅"
        else:
            macd_signal = "BEARISH - Histogram < 0 ⚠️"
        print(f"  Signal: {macd_signal}")

    # ========================================
    # COMBINED: Momentum Confirmation
    # ========================================
    print("\n\n📊 COMBINED: Momentum Confirmation (RSI + MACD)")
    print("-" * 60)

    momentum = calc.get_momentum_confirmation(symbol)
    if momentum:
        print(f"✅ Momentum Confirmation for {symbol}:")
        print(f"  RSI: {momentum['rsi']:.1f}")
        print(f"  MACD Histogram: {momentum['macd_histogram']:.4f}")
        print(f"  Bullish Score: {momentum['bullish_score']:.2f} (0.0-1.0)")
        print(f"  Bearish Score: {momentum['bearish_score']:.2f} (0.0-1.0)")
        print(f"  Dominant: {momentum['dominant']}")
        print(f"  Confidence: {momentum['confidence']:.2f}")

        # Trading recommendation
        print("\n  Trading Recommendation:")
        if momentum['dominant'] == 'BULLISH' and momentum['confidence'] > 0.6:
            print("    ✅ STRONG LONG signal confirmation")
        elif momentum['dominant'] == 'BEARISH' and momentum['confidence'] > 0.6:
            print("    ✅ STRONG SHORT signal confirmation")
        elif momentum['confidence'] > 0.5:
            print(f"    ⚠️ MODERATE {momentum['dominant']} confirmation")
        else:
            print("    ❌ WEAK confirmation - consider skipping")
    else:
        print(f"❌ Could not calculate momentum confirmation for {symbol}")

    # ========================================
    # Summary
    # ========================================
    print("\n\n📋 IMPLEMENTATION SUMMARY")
    print("=" * 60)
    print("✅ Phase 1: ATR (Dynamic Stop Losses)")
    print("✅ Phase 2: ADX (Trend Filter) + RSI (Momentum)")
    print("✅ Phase 3: MACD (Trend Confirmation)")
    print("✅ Combined: get_momentum_confirmation()")
    print("\n💡 Ready for integration with enhanced_scalper.py")
    print("=" * 60)
