#!/usr/bin/env python3
"""
PRICE PREDICTOR - QUICK INTEGRATION SNIPPET
Copy this code into enhanced_scalper.py to add price prediction

RECOMMENDED: Pattern 1 - Signal Confidence Boost
"""

# ==============================================================================
# STEP 1: Add import at top of enhanced_scalper.py
# ==============================================================================

from trading.price_predictor import PricePredictor


# ==============================================================================
# STEP 2: Initialize in __init__ method
# ==============================================================================

class EnhancedScalper:
    def __init__(self, config: ScalpingConfig):
        # ... existing initialization ...

        # Add price predictor (30-min cache)
        self.price_predictor = PricePredictor(cache_duration_minutes=30)
        logger.info("✅ Price predictor initialized")


# ==============================================================================
# STEP 3: Add prediction to signal evaluation
# ==============================================================================

async def _evaluate_signals(self, symbol: str, data: Dict) -> Optional[Dict]:
    """
    Evaluate signals and generate trade decision
    ADD THIS CODE AFTER signal generation but BEFORE confidence check
    """

    # ... existing signal generation code ...
    # combined_signal = {...}  # Your existing signal

    # ========================================================================
    # PRICE PREDICTION INTEGRATION (Pattern 1: Confidence Boost)
    # ========================================================================

    # Get price history from data (or build from websocket)
    price_history = data.get('price_history', [])

    # If no history, skip prediction
    if len(price_history) >= 10:
        try:
            # Get price prediction
            prediction = self.price_predictor.get_price_prediction(
                symbol=symbol,
                current_price=data['current_price'],
                price_history=price_history[-20:],  # Last 20 prices
                indicators={
                    'rsi': data['indicators']['rsi'],
                    'atr': data['indicators']['atr'],
                    'adx': data['indicators']['adx'],
                    'di_plus': data['indicators'].get('di_plus', 0),
                    'di_minus': data['indicators'].get('di_minus', 0)
                },
                sentiment=data.get('sentiment', 5.0),
                prediction_hours=2  # 2-hour prediction
            )

            if prediction:
                # Log prediction
                logger.info(f"🔮 Price Prediction:")
                logger.info(f"   Current: ${prediction['current_price']:.2f}")
                logger.info(f"   Predicted (2hr): ${prediction['predicted_price']:.2f} ({prediction['direction']})")
                logger.info(f"   Confidence: {prediction['confidence']:.0%}")
                logger.info(f"   Reasoning: {prediction['reasoning']}")

                # Check alignment with signal
                signal_direction = combined_signal['signal'].upper()
                pred_direction = prediction['direction']
                pred_confidence = prediction['confidence']

                if signal_direction == pred_direction:
                    # ALIGNMENT: Boost confidence
                    alignment_bonus = pred_confidence * 0.2  # Up to +20%
                    old_confidence = combined_signal['confidence']
                    combined_signal['confidence'] += alignment_bonus

                    logger.info(f"✅ PREDICTION ALIGNED ({pred_direction})")
                    logger.info(f"   Original Confidence: {old_confidence:.1%}")
                    logger.info(f"   Alignment Bonus: +{alignment_bonus:.1%}")
                    logger.info(f"   Boosted Confidence: {combined_signal['confidence']:.1%}")

                    # Add prediction to signal metadata
                    combined_signal['prediction'] = {
                        'predicted_price': prediction['predicted_price'],
                        'direction': prediction['direction'],
                        'confidence': prediction['confidence'],
                        'aligned': True
                    }
                else:
                    # MISALIGNMENT: Reduce confidence
                    penalty = 0.10  # -10%
                    old_confidence = combined_signal['confidence']
                    combined_signal['confidence'] -= penalty

                    logger.warning(f"⚠️  PREDICTION MISALIGNED")
                    logger.warning(f"   Signal: {signal_direction}, Prediction: {pred_direction}")
                    logger.warning(f"   Original Confidence: {old_confidence:.1%}")
                    logger.warning(f"   Misalignment Penalty: -{penalty:.1%}")
                    logger.warning(f"   Reduced Confidence: {combined_signal['confidence']:.1%}")

                    # Add prediction to signal metadata
                    combined_signal['prediction'] = {
                        'predicted_price': prediction['predicted_price'],
                        'direction': prediction['direction'],
                        'confidence': prediction['confidence'],
                        'aligned': False
                    }
            else:
                logger.debug("Price prediction unavailable (API error or cache miss)")

        except Exception as e:
            logger.error(f"Error getting price prediction: {e}")
            # Continue without prediction (graceful degradation)

    # ========================================================================
    # END PRICE PREDICTION INTEGRATION
    # ========================================================================

    # Continue with existing signal evaluation...
    if combined_signal['confidence'] < self.config.min_combined_confidence:
        logger.info(f"Signal confidence too low: {combined_signal['confidence']:.1%}")
        return None

    return combined_signal


# ==============================================================================
# STEP 4: Add price history tracking to data collection
# ==============================================================================

async def _collect_market_data(self, symbol: str) -> Dict:
    """
    Collect market data from WebSocket or REST
    ADD price history tracking
    """

    # ... existing data collection ...

    # Initialize price history storage (add to class __init__ if not exists)
    if not hasattr(self, 'price_history'):
        self.price_history = {}

    # Add current price to history
    if symbol not in self.price_history:
        self.price_history[symbol] = []

    self.price_history[symbol].append(data['current_price'])

    # Keep last 50 prices (enough for 20-point history with buffer)
    if len(self.price_history[symbol]) > 50:
        self.price_history[symbol] = self.price_history[symbol][-50:]

    # Add to data dict
    data['price_history'] = self.price_history[symbol]

    return data


# ==============================================================================
# OPTIONAL: Use prediction for dynamic take profit (Pattern 3)
# ==============================================================================

async def _calculate_position_params(self, signal: Dict, data: Dict):
    """
    Calculate position parameters with optional prediction-based TP
    """
    current_price = signal['price']
    atr = signal['atr']

    # Default take profit (ATR-based)
    take_profit = current_price + (atr * 2.0)

    # If prediction available and aligned, use predicted price
    if 'prediction' in signal and signal['prediction']['aligned']:
        predicted_price = signal['prediction']['predicted_price']
        predicted_move = abs(predicted_price - current_price)

        # Use 80% of predicted move as conservative TP
        if signal['signal'].upper() == 'LONG':
            prediction_tp = current_price + (predicted_move * 0.8)
        else:
            prediction_tp = current_price - (predicted_move * 0.8)

        # Use whichever is more conservative (closer to entry)
        if signal['signal'].upper() == 'LONG':
            take_profit = min(take_profit, prediction_tp)
        else:
            take_profit = max(take_profit, prediction_tp)

        logger.info(f"🎯 Take profit adjusted using prediction: ${take_profit:.2f}")

    return {
        'entry_price': current_price,
        'take_profit': take_profit,
        'stop_loss': signal['stop_loss']
    }


# ==============================================================================
# CONFIGURATION CHECKLIST
# ==============================================================================

"""
BEFORE RUNNING:

1. Add API key to .env:
   GROK_TRADE_API=xai-xxx...
   (or OPENAI_API_KEY=sk-xxx... as fallback)

2. Test standalone:
   python3 trading/price_predictor.py

3. Test integration:
   python3 test_price_prediction_integration.py --offline

4. Monitor logs:
   tail -f logs/price_predictor.log

5. Track metrics after 50+ trades:
   - Alignment win rate (should be > 55%)
   - Misalignment win rate (should be < 45%)
   - Cache hit rate (should be > 80%)
   - Prediction accuracy (directional)

EXPECTED BEHAVIOR:

- First call: 3-5 second API latency
- Cached calls: <1ms response time
- Aligned signals: +10-20% confidence boost
- Misaligned signals: -10% confidence penalty
- API failures: Graceful degradation, continues without prediction

TROUBLESHOOTING:

- No predictions appearing in logs:
  → Check .env has GROK_TRADE_API or OPENAI_API_KEY
  → Check logs/price_predictor.log for errors

- Too many API calls:
  → Increase cache_duration_minutes in __init__

- Low prediction accuracy:
  → Track over 50+ predictions before adjusting
  → May need to adjust alignment bonus/penalty weights
"""
