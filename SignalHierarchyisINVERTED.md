# Signal Hierarchy Inversion Fix Plan

## WHY WE ARE MAKING THESE CHANGES

### The Problem We Discovered

On **2025-12-16**, after multiple losing trades, we conducted deep research using the Research Trading Agent with web searches across 30+ academic papers and professional trading resources (Stanford CS229, SSRN, ScienceDirect, arXiv).

**Key Finding**: Our trading system's signal hierarchy is **INVERTED** from how professional trading systems work.

### What Was Wrong

| Component | Our System (WRONG) | Professional Systems (CORRECT) |
|-----------|-------------------|-------------------------------|
| **Sentiment** | PRIMARY signal (blocks trades) | SECONDARY (adjusts position size) |
| **Technical Indicators** | Secondary (generates candidates) | PRIMARY (generates signals) |
| **LLM Prediction** | Adjusts confidence ±10-15% | Logging only (59% accuracy) |

### Evidence from Research

1. **LLM Price Prediction Accuracy**: Research shows LLMs achieve only **59% accuracy** on price prediction - barely better than a coin flip. Our system was using Grok predictions to adjust trade confidence.

2. **Sentiment Role**: Studies show sentiment should "supplement rather than replace" technical analysis. Best results when sentiment **confirms** technical signals, not when it **blocks** them.

3. **Multi-Indicator Consensus**: Professional systems require **3+ indicators to agree** before generating signals. Our system used only momentum strategy (100% weight).

4. **ADX Regime Filtering**: Trading in ranging markets (ADX < 25) generates false signals. Our system had no regime filter.

### Specific Problems in Our Code

```python
# enhanced_scalper.py Lines 945-951 - THIS IS WRONG
if signal.direction == 'LONG' and sentiment_score < 7.0:
    signal = None  # BLOCKS technical signal based on sentiment
elif signal.direction == 'SHORT' and sentiment_score > 4.0:
    signal = None  # BLOCKS technical signal based on sentiment
```

**Why This Is Wrong**:
- Technical indicators analyze PRICE DATA (objective)
- Sentiment analyzes NEWS DATA (subjective, can lag)
- Blocking objective signals with subjective data = poor decisions
- Result: Missing profitable trades when technicals are right but sentiment is neutral

### What We Learned

1. **Technical indicators should GENERATE signals** - they analyze actual price movement
2. **Sentiment should ADJUST position size** - not block trades entirely
3. **LLM predictions are unreliable** - use for logging/context only
4. **ADX regime filter is essential** - avoid trading in ranging markets

### Expected Improvements After Fix

| Metric | Current | Expected | Source |
|--------|---------|----------|--------|
| Win Rate | ~40% | +10-15% | Multi-indicator confirmation |
| False Signals | High | -40-60% | Consensus requirement |
| Drawdown | High | -20-30% | Regime filtering |

### Trade Examples That Prompted This Research

- Multiple SHORT trades in BULLISH sentiment = losses
- Trades blocked when technicals were correct but sentiment was neutral
- LLM always predicting "UP" at 75% confidence (not variable)
- Position close logic was fixed but still losing trades

---

## Executive Summary

**Problem**: Current trading system uses **Sentiment as PRIMARY signal** and Technical Indicators as secondary. Research shows this is **inverted** from professional trading systems.

**Solution**: Restructure signal hierarchy to:
1. Technical Indicators = PRIMARY (signal generation)
2. Sentiment = SECONDARY (position sizing/bias)
3. LLM Prediction = LOGGING ONLY (59% accuracy is unreliable)

---

## Current Architecture (WRONG)

```
CURRENT FLOW (enhanced_scalper.py):

1. Technical Signal Generation (scalping_strategies.py:1044-1191)
   |-- Momentum Strategy generates LONG/SHORT signal

2. SENTIMENT FILTER (PRIMARY) <-- PROBLEM HERE
   |-- Lines 945-951: Block if sentiment doesn't match
   |-- LONG blocked if sentiment < 7.0
   |-- SHORT blocked if sentiment > 4.0

3. LLM Prediction Enhancement
   |-- Lines 968-992: Adjust confidence +/-10-15%
```

**Key Problem Files**:
- `trading/enhanced_scalper.py` - Lines 945-951 (sentiment as gate)
- `trading/scalping_strategies.py` - Lines 1138-1141 (sentiment bias only 5%)

---

## Target Architecture (CORRECT)

```
NEW FLOW:

LEVEL 0: REGIME FILTER (Binary Gate)
|-- ADX > 25 = Allow trades
|-- ADX < 25 = NO TRADE (ranging market)
|-- High ATR = Reduce position size

LEVEL 1: TECHNICAL SIGNALS (PRIMARY - 50% weight)
|-- MACD crossover
|-- MA crossover
|-- Price breakout/momentum
|-- Require 2 of 3 to agree

LEVEL 2: TECHNICAL CONFIRMATION (30% weight)
|-- Volume surge (>10% above average)
|-- RSI in valid range (30-70)
|-- ALL must pass

LEVEL 3: SENTIMENT BIAS (SECONDARY - 20% weight)
|-- Adjusts position size (0.5x to 1.5x)
|-- Does NOT block signals
|-- Provides early warning only

LEVEL 4: LLM CONTEXT (0% weight)
|-- Logging/explanation ONLY
|-- Remove from signal generation
```

---

## Implementation Steps

### Phase 1: Add Regime Filter (ADX Gate)

**File**: `trading/enhanced_scalper.py`

**Location**: Before signal generation (around line 820)

**Add New Method**:
```python
def _check_regime_filter(self, symbol: str) -> tuple[bool, str]:
    """
    LEVEL 0: Regime filter - determines if trading conditions are favorable.
    Returns (should_trade, reason)
    """
    try:
        indicator_calc = IndicatorCalculator()
        adx = indicator_calc.calculate_adx(symbol)

        if adx is None:
            return True, "ADX unavailable - allowing trade"

        if adx < 25:
            return False, f"REGIME FILTER: ADX {adx:.1f} < 25 (ranging market)"

        # Optional: Check volatility for position sizing
        atr_info = indicator_calc.get_atr_info(symbol)
        if atr_info and atr_info.get('volatility') == 'very_high':
            # Don't block, but flag for reduced size
            logging.info(f"HIGH VOLATILITY: Consider reduced position size")

        return True, f"REGIME OK: ADX {adx:.1f} (trending)"

    except Exception as e:
        logging.warning(f"Regime filter error: {e}")
        return True, "Regime filter error - allowing trade"
```

**Integration Point**: Line ~820, before `get_combined_signal()`
```python
# NEW: Check regime filter first
should_trade, regime_reason = self._check_regime_filter(symbol)
if not should_trade:
    logging.info(f"LEVEL 0 REJECT: {regime_reason}")
    continue
```

---

### Phase 2: Add Multi-Indicator Consensus

**File**: `trading/enhanced_scalper.py`

**Add New Method**:
```python
def _check_technical_consensus(self, symbol: str, current_price: float) -> tuple[bool, str, str]:
    """
    LEVEL 1: Technical consensus - requires 2 of 3 primary indicators to agree.
    Returns (has_consensus, direction, reason)
    """
    try:
        indicator_calc = IndicatorCalculator()
        signals = []

        # Signal 1: MACD
        macd_data = indicator_calc.calculate_macd(symbol)
        if macd_data:
            macd_signal = 'LONG' if macd_data['histogram'] > 0 else 'SHORT'
            signals.append(('MACD', macd_signal))

        # Signal 2: Trend Direction (DI+/DI-)
        trend = indicator_calc.get_trend_direction(symbol)
        if trend:
            trend_signal = 'LONG' if trend == 'BULLISH' else 'SHORT'
            signals.append(('TREND', trend_signal))

        # Signal 3: Momentum (from existing strategy)
        # Use the momentum signal if available from scalping_strategies

        if len(signals) < 2:
            return False, None, "Insufficient indicators for consensus"

        # Count votes
        long_votes = sum(1 for _, s in signals if s == 'LONG')
        short_votes = sum(1 for _, s in signals if s == 'SHORT')

        if long_votes >= 2:
            return True, 'LONG', f"CONSENSUS: {long_votes} indicators agree LONG"
        elif short_votes >= 2:
            return True, 'SHORT', f"CONSENSUS: {short_votes} indicators agree SHORT"
        else:
            return False, None, f"NO CONSENSUS: LONG={long_votes}, SHORT={short_votes}"

    except Exception as e:
        logging.warning(f"Consensus check error: {e}")
        return False, None, str(e)
```

---

### Phase 3: Add Confirmation Layer

**File**: `trading/enhanced_scalper.py`

**Add New Method**:
```python
def _check_confirmations(self, symbol: str, direction: str) -> tuple[bool, str]:
    """
    LEVEL 2: Confirmation filters - ALL must pass.
    Returns (confirmed, reason)
    """
    try:
        indicator_calc = IndicatorCalculator()
        confirmations = []

        # Confirmation 1: ADX strength (already checked in regime, but verify > 25)
        adx = indicator_calc.calculate_adx(symbol)
        adx_ok = adx is not None and adx > 25
        confirmations.append(('ADX>25', adx_ok))

        # Confirmation 2: RSI not at extremes
        rsi = indicator_calc.calculate_rsi(symbol)
        if rsi is not None:
            if direction == 'LONG':
                rsi_ok = rsi < 70  # Not overbought for longs
            else:
                rsi_ok = rsi > 30  # Not oversold for shorts
        else:
            rsi_ok = True  # Allow if unavailable
        confirmations.append(('RSI_VALID', rsi_ok))

        # Confirmation 3: Volume (if available)
        # Note: Capital.com has limited volume data
        volume_ok = True  # Default to true, enhance later
        confirmations.append(('VOLUME', volume_ok))

        # All must pass
        all_confirmed = all(ok for _, ok in confirmations)
        failed = [name for name, ok in confirmations if not ok]

        if all_confirmed:
            return True, "All confirmations passed"
        else:
            return False, f"CONFIRMATION FAILED: {', '.join(failed)}"

    except Exception as e:
        logging.warning(f"Confirmation check error: {e}")
        return True, "Confirmation error - allowing trade"
```

---

### Phase 4: Convert Sentiment to Position Size Modifier (CRITICAL)

**File**: `trading/enhanced_scalper.py`

**REMOVE** (Lines 945-951):
```python
# DELETE THIS BLOCK - Sentiment should NOT block signals
if signal.direction == 'LONG' and sentiment_score < 7.0:
    logging.warning(f"COUNTER-TREND BLOCKED: LONG rejected...")
    signal = None
elif signal.direction == 'SHORT' and sentiment_score > 4.0:
    logging.warning(f"COUNTER-TREND BLOCKED: SHORT rejected...")
    signal = None
```

**REPLACE WITH** Position Size Modifier:
```python
def _apply_sentiment_bias(self, base_position_pct: float, sentiment_score: float,
                          direction: str) -> float:
    """
    LEVEL 3: Sentiment adjusts position SIZE, not signal direction.
    Returns adjusted position percentage.
    """
    # Sentiment alignment check
    sentiment_bullish = sentiment_score >= 7.0
    sentiment_bearish = sentiment_score <= 4.0
    sentiment_neutral = 4.0 < sentiment_score < 7.0

    if direction == 'LONG':
        if sentiment_bullish:
            multiplier = 1.5  # Strong alignment - increase size
            logging.info(f"SENTIMENT BOOST: LONG + Bullish ({sentiment_score}/10) = 1.5x size")
        elif sentiment_neutral:
            multiplier = 1.0  # Neutral - normal size
            logging.info(f"SENTIMENT NEUTRAL: LONG + Neutral ({sentiment_score}/10) = 1.0x size")
        else:  # sentiment_bearish
            multiplier = 0.5  # Conflicting - reduce size (but don't block!)
            logging.warning(f"SENTIMENT CONFLICT: LONG + Bearish ({sentiment_score}/10) = 0.5x size")

    elif direction == 'SHORT':
        if sentiment_bearish:
            multiplier = 1.5  # Strong alignment
            logging.info(f"SENTIMENT BOOST: SHORT + Bearish ({sentiment_score}/10) = 1.5x size")
        elif sentiment_neutral:
            multiplier = 1.0  # Neutral
            logging.info(f"SENTIMENT NEUTRAL: SHORT + Neutral ({sentiment_score}/10) = 1.0x size")
        else:  # sentiment_bullish
            multiplier = 0.5  # Conflicting - reduce size
            logging.warning(f"SENTIMENT CONFLICT: SHORT + Bullish ({sentiment_score}/10) = 0.5x size")

    adjusted_pct = base_position_pct * multiplier

    # Enforce min/max bounds
    return max(0.05, min(adjusted_pct, 0.20))  # 5-20% range
```

---

### Phase 5: Remove LLM from Signal Generation

**File**: `trading/enhanced_scalper.py`

**MODIFY** (Lines 968-992):

**FROM**:
```python
if prediction:
    if pred_direction == signal_direction:
        signal.confidence *= 1.10  # +10% boost
    else:
        signal.confidence *= 0.85  # -15% reduction
```

**TO**:
```python
if prediction:
    # LEVEL 4: LLM for LOGGING ONLY - do NOT adjust confidence
    # Research shows 59% accuracy - unreliable for trading decisions
    logging.info(f"LLM Context (logging only): {prediction['direction']} "
                 f"to ${prediction['predicted_price']:.2f} "
                 f"({prediction['confidence']*100:.0f}% confident)")
    logging.info(f"   Reasoning: {prediction.get('reasoning', 'N/A')}")

    # Track for analysis but don't use in decision
    # signal.confidence remains unchanged
```

---

### Phase 6: Integrate New Flow

**File**: `trading/enhanced_scalper.py`

**New Signal Processing Flow** (replace existing logic around lines 820-1000):

```python
# === NEW SIGNAL HIERARCHY ===

# LEVEL 0: Regime Filter
should_trade, regime_reason = self._check_regime_filter(symbol)
if not should_trade:
    logging.info(f"LEVEL 0 REJECT: {regime_reason}")
    continue

# LEVEL 1: Technical Consensus (PRIMARY)
has_consensus, consensus_direction, consensus_reason = self._check_technical_consensus(
    symbol, current_price
)
if not has_consensus:
    logging.debug(f"LEVEL 1 NO SIGNAL: {consensus_reason}")
    continue

logging.info(f"LEVEL 1 SIGNAL: {consensus_direction} ({consensus_reason})")

# LEVEL 2: Confirmations
confirmed, confirm_reason = self._check_confirmations(symbol, consensus_direction)
if not confirmed:
    logging.info(f"LEVEL 2 REJECT: {confirm_reason}")
    continue

logging.info(f"LEVEL 2 CONFIRMED: {confirm_reason}")

# LEVEL 3: Sentiment Bias (position sizing)
sentiment_score = self.sentiment_scores.get(symbol, 5.0)
base_position_pct = 0.10  # 10% base
adjusted_position_pct = self._apply_sentiment_bias(
    base_position_pct, sentiment_score, consensus_direction
)

logging.info(f"LEVEL 3 POSITION SIZE: {adjusted_position_pct*100:.1f}% "
             f"(sentiment: {sentiment_score}/10)")

# LEVEL 4: LLM Context (logging only)
prediction = self._get_price_prediction(symbol, current_price, market_data)
# Logged but not used in decision

# Generate final signal using existing ScalpingSignal structure
signal = ScalpingSignal(
    symbol=symbol,
    direction=consensus_direction,
    entry_price=current_price,
    confidence=0.70,  # Base confidence from technical consensus
    # ... other fields
)

# Execute with adjusted position size
self._execute_signal(signal, adjusted_position_pct)
```

---

## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `trading/enhanced_scalper.py` | 945-951 | DELETE sentiment blocking logic |
| `trading/enhanced_scalper.py` | 968-992 | MODIFY LLM to logging only |
| `trading/enhanced_scalper.py` | ~820 | ADD regime filter check |
| `trading/enhanced_scalper.py` | NEW | ADD `_check_regime_filter()` method |
| `trading/enhanced_scalper.py` | NEW | ADD `_check_technical_consensus()` method |
| `trading/enhanced_scalper.py` | NEW | ADD `_check_confirmations()` method |
| `trading/enhanced_scalper.py` | NEW | ADD `_apply_sentiment_bias()` method |
| `trading/scalping_strategies.py` | 1138-1141 | KEEP existing 5% bias (now secondary) |

---

## Testing Plan

1. **Unit Test**: Each new method independently
2. **Integration Test**: Full signal flow with mock data
3. **Paper Trade**: Run on demo account for 1 week
4. **Compare**: Track metrics before/after

---

## Rollback Plan

If issues occur:
1. Revert `enhanced_scalper.py` to previous version
2. Git: `git checkout HEAD~1 -- trading/enhanced_scalper.py`

---

## Research Sources

- Stanford CS229: Algorithmic Trading using Sentiment Analysis
- SSRN: Enhanced Financial Sentiment Analysis (2024)
- ScienceDirect: Sentiment trading with LLMs
- arXiv: Neural Network-Based Algorithmic Trading Systems
- ExtractAlpha: Top 7 Trading Signals Every Quant Should Track
- Multiple Medium articles on multi-indicator trading systems

---

## Summary

**Key Changes**:
1. **ADD** ADX regime filter (Level 0)
2. **ADD** Multi-indicator consensus (Level 1)
3. **ADD** Confirmation layer (Level 2)
4. **CONVERT** Sentiment from blocker to position sizer (Level 3)
5. **REMOVE** LLM from signal generation (Level 4 = logging only)

**Architecture Change**:
```
BEFORE: Technical -> Sentiment (BLOCKS) -> LLM (adjusts)
AFTER:  Regime -> Technical (PRIMARY) -> Confirmation -> Sentiment (SIZE) -> LLM (LOG)
```

---

**Plan Created**: 2025-12-16
**Status**: NOT EXECUTED - Saved for future implementation
**Author**: Claude Code with Research Trading Agent
