"""
Strategy Optimizer for Continuous Improvement
Analyzes trading performance and suggests parameter adjustments.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import statistics


class ParameterType(Enum):
  """Types of tunable parameters."""
  THRESHOLD = "THRESHOLD"
  MULTIPLIER = "MULTIPLIER"
  INTERVAL = "INTERVAL"
  DISTANCE = "DISTANCE"
  WEIGHT = "WEIGHT"


@dataclass
class StrategyParameter:
  """Represents a tunable strategy parameter."""
  name: str
  param_type: ParameterType
  current_value: float
  min_value: float
  max_value: float
  step_size: float
  last_updated: Optional[datetime] = None
  performance_impact: float = 0.0  # Correlation with win rate
  description: str = ""

  def suggest_new_value(self, direction: str = "increase") -> float:
    """
    Suggest a new value based on direction.

    Args:
      direction: 'increase', 'decrease', or 'reset'

    Returns:
      New suggested value
    """
    if direction == "reset":
      return (self.min_value + self.max_value) / 2

    if direction == "increase":
      new_value = self.current_value + self.step_size
      return min(new_value, self.max_value)

    if direction == "decrease":
      new_value = self.current_value - self.step_size
      return max(new_value, self.min_value)

    return self.current_value

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""
    return {
      "name": self.name,
      "type": self.param_type.value,
      "current_value": self.current_value,
      "min_value": self.min_value,
      "max_value": self.max_value,
      "step_size": self.step_size,
      "last_updated": self.last_updated.isoformat() if self.last_updated else None,
      "performance_impact": self.performance_impact,
      "description": self.description,
    }


@dataclass
class ImprovementSuggestion:
  """Suggested parameter adjustment."""
  parameter: StrategyParameter
  suggested_value: float
  direction: str  # increase, decrease, reset
  reasoning: str
  expected_impact: str
  confidence: float  # 0.0 to 1.0
  requires_human_review: bool = True
  created_at: datetime = field(default_factory=datetime.now)

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""
    return {
      "parameter_name": self.parameter.name,
      "current_value": self.parameter.current_value,
      "suggested_value": self.suggested_value,
      "direction": self.direction,
      "reasoning": self.reasoning,
      "expected_impact": self.expected_impact,
      "confidence": self.confidence,
      "requires_human_review": self.requires_human_review,
      "created_at": self.created_at.isoformat(),
    }


@dataclass
class TradeRecord:
  """Record of a single trade for analysis."""
  timestamp: datetime
  symbol: str
  direction: str
  entry_price: float
  exit_price: float
  pnl: float
  strategy: str
  confidence: float
  sentiment_score: float
  hold_time_seconds: float
  indicators: Dict[str, float] = field(default_factory=dict)


class StrategyOptimizer:
  """
  Analyzes trading performance and suggests parameter improvements.

  Features:
  - Parameter performance correlation analysis
  - Trend detection in strategy performance
  - Automated improvement suggestions
  - Human review workflow
  """

  # Default tunable parameters
  DEFAULT_PARAMETERS = [
    StrategyParameter(
      name="min_combined_confidence",
      param_type=ParameterType.THRESHOLD,
      current_value=0.50,
      min_value=0.30,
      max_value=0.80,
      step_size=0.05,
      description="Minimum confidence to enter a trade"
    ),
    StrategyParameter(
      name="atr_multiplier",
      param_type=ParameterType.MULTIPLIER,
      current_value=1.5,
      min_value=1.0,
      max_value=3.0,
      step_size=0.1,
      description="ATR multiplier for stop loss distance"
    ),
    StrategyParameter(
      name="min_entry_interval",
      param_type=ParameterType.INTERVAL,
      current_value=15,
      min_value=5,
      max_value=60,
      step_size=5,
      description="Minimum seconds between entries"
    ),
    StrategyParameter(
      name="trailing_stop_distance",
      param_type=ParameterType.DISTANCE,
      current_value=0.15,
      min_value=0.10,
      max_value=0.60,
      step_size=0.05,
      description="Trailing stop distance in dollars"
    ),
    StrategyParameter(
      name="sentiment_bias_weight",
      param_type=ParameterType.WEIGHT,
      current_value=0.05,
      min_value=0.0,
      max_value=0.30,
      step_size=0.05,
      description="Weight of sentiment in confidence calculation"
    ),
  ]

  def __init__(self, parameters: Optional[List[StrategyParameter]] = None):
    """
    Initialize the optimizer.

    Args:
      parameters: List of tunable parameters (uses defaults if not provided)
    """
    self.parameters = {p.name: p for p in (parameters or self.DEFAULT_PARAMETERS)}
    self.trade_history: List[TradeRecord] = []
    self.suggestions: List[ImprovementSuggestion] = []

  def add_trade(self, trade: TradeRecord):
    """Add a trade to the history for analysis."""
    self.trade_history.append(trade)

  def add_trades(self, trades: List[Dict[str, Any]]):
    """Add multiple trades from dictionary format."""
    for t in trades:
      record = TradeRecord(
        timestamp=datetime.fromisoformat(t['timestamp']) if isinstance(t['timestamp'], str) else t['timestamp'],
        symbol=t.get('symbol', 'UNKNOWN'),
        direction=t.get('direction', 'UNKNOWN'),
        entry_price=t.get('entry_price', 0),
        exit_price=t.get('exit_price', 0),
        pnl=t.get('pnl', 0),
        strategy=t.get('strategy', 'unknown'),
        confidence=t.get('confidence', 0),
        sentiment_score=t.get('sentiment_score', 5),
        hold_time_seconds=t.get('hold_time_seconds', 0),
        indicators=t.get('indicators', {}),
      )
      self.trade_history.append(record)

  def analyze_performance(self, window_size: int = 50) -> Dict[str, Any]:
    """
    Analyze recent trading performance.

    Args:
      window_size: Number of recent trades to analyze

    Returns:
      Performance metrics
    """
    if not self.trade_history:
      return {"error": "No trade history"}

    recent = self.trade_history[-window_size:]

    # Calculate metrics
    wins = [t for t in recent if t.pnl > 0]
    losses = [t for t in recent if t.pnl <= 0]

    total_pnl = sum(t.pnl for t in recent)
    win_rate = len(wins) / len(recent) if recent else 0

    avg_win = statistics.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = statistics.mean([t.pnl for t in losses]) if losses else 0

    # Calculate profit factor
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Analyze by confidence level
    high_conf = [t for t in recent if t.confidence >= 0.6]
    low_conf = [t for t in recent if t.confidence < 0.6]

    high_conf_win_rate = len([t for t in high_conf if t.pnl > 0]) / len(high_conf) if high_conf else 0
    low_conf_win_rate = len([t for t in low_conf if t.pnl > 0]) / len(low_conf) if low_conf else 0

    # Analyze by hold time
    avg_hold_time = statistics.mean([t.hold_time_seconds for t in recent]) if recent else 0

    # Analyze trend
    if len(recent) >= 20:
      first_half = recent[:len(recent)//2]
      second_half = recent[len(recent)//2:]
      first_win_rate = len([t for t in first_half if t.pnl > 0]) / len(first_half)
      second_win_rate = len([t for t in second_half if t.pnl > 0]) / len(second_half)
      trend = "improving" if second_win_rate > first_win_rate else "declining"
    else:
      trend = "insufficient_data"

    return {
      "total_trades": len(recent),
      "total_pnl": total_pnl,
      "win_rate": win_rate,
      "profit_factor": profit_factor,
      "avg_win": avg_win,
      "avg_loss": avg_loss,
      "avg_hold_time_seconds": avg_hold_time,
      "high_confidence_win_rate": high_conf_win_rate,
      "low_confidence_win_rate": low_conf_win_rate,
      "trend": trend,
      "analysis_timestamp": datetime.now().isoformat(),
    }

  def generate_suggestions(self, performance: Optional[Dict[str, Any]] = None) -> List[ImprovementSuggestion]:
    """
    Generate improvement suggestions based on performance.

    Args:
      performance: Performance metrics (analyzes if not provided)

    Returns:
      List of improvement suggestions
    """
    if performance is None:
      performance = self.analyze_performance()

    if "error" in performance:
      return []

    suggestions = []

    win_rate = performance.get("win_rate", 0)
    profit_factor = performance.get("profit_factor", 1)
    trend = performance.get("trend", "unknown")
    high_conf_wr = performance.get("high_confidence_win_rate", 0)
    low_conf_wr = performance.get("low_confidence_win_rate", 0)

    # Suggestion 1: Confidence threshold
    if low_conf_wr < 0.35 and high_conf_wr > 0.50:
      # Low confidence trades performing poorly
      param = self.parameters.get("min_combined_confidence")
      if param:
        suggestion = ImprovementSuggestion(
          parameter=param,
          suggested_value=param.suggest_new_value("increase"),
          direction="increase",
          reasoning=f"Low confidence trades ({low_conf_wr:.1%} win rate) underperforming "
                   f"high confidence ({high_conf_wr:.1%}). Increase threshold to filter.",
          expected_impact="Fewer trades but higher win rate",
          confidence=0.7,
        )
        suggestions.append(suggestion)

    elif high_conf_wr > 0.60 and win_rate > 0.55:
      # Strong performance, could consider lowering threshold
      param = self.parameters.get("min_combined_confidence")
      if param and param.current_value > 0.50:
        suggestion = ImprovementSuggestion(
          parameter=param,
          suggested_value=param.suggest_new_value("decrease"),
          direction="decrease",
          reasoning=f"High win rate ({win_rate:.1%}) with good confidence correlation. "
                   f"Consider lowering threshold for more opportunities.",
          expected_impact="More trades with acceptable win rate",
          confidence=0.5,
        )
        suggestions.append(suggestion)

    # Suggestion 2: Stop loss (ATR multiplier)
    avg_loss = abs(performance.get("avg_loss", 0))
    avg_win = performance.get("avg_win", 0)

    if avg_loss > avg_win * 1.5 and win_rate < 0.55:
      # Losses too large relative to wins
      param = self.parameters.get("atr_multiplier")
      if param:
        suggestion = ImprovementSuggestion(
          parameter=param,
          suggested_value=param.suggest_new_value("decrease"),
          direction="decrease",
          reasoning=f"Average loss (${avg_loss:.2f}) exceeds average win (${avg_win:.2f}) "
                   f"by too much. Tighten stops.",
          expected_impact="Smaller losses but potentially more stop-outs",
          confidence=0.6,
        )
        suggestions.append(suggestion)

    elif win_rate > 0.60 and avg_loss < avg_win * 0.5:
      # Stops might be too tight
      param = self.parameters.get("atr_multiplier")
      if param and param.current_value < 2.0:
        suggestion = ImprovementSuggestion(
          parameter=param,
          suggested_value=param.suggest_new_value("increase"),
          direction="increase",
          reasoning=f"High win rate but small average loss suggests stops may be "
                   f"triggering prematurely. Consider wider stops.",
          expected_impact="Fewer premature stop-outs, potentially larger wins",
          confidence=0.5,
        )
        suggestions.append(suggestion)

    # Suggestion 3: Entry interval
    if trend == "declining" and win_rate < 0.45:
      param = self.parameters.get("min_entry_interval")
      if param:
        suggestion = ImprovementSuggestion(
          parameter=param,
          suggested_value=param.suggest_new_value("increase"),
          direction="increase",
          reasoning=f"Declining performance ({trend}) with low win rate ({win_rate:.1%}). "
                   f"Slow down entry frequency to be more selective.",
          expected_impact="Fewer but more carefully selected trades",
          confidence=0.65,
        )
        suggestions.append(suggestion)

    # Suggestion 4: Trailing stop
    avg_hold = performance.get("avg_hold_time_seconds", 0)
    if avg_hold < 60 and win_rate < 0.50:
      # Short hold times with poor win rate
      param = self.parameters.get("trailing_stop_distance")
      if param:
        suggestion = ImprovementSuggestion(
          parameter=param,
          suggested_value=param.suggest_new_value("increase"),
          direction="increase",
          reasoning=f"Average hold time ({avg_hold:.0f}s) very short with poor win rate. "
                   f"Trailing stop may be too tight.",
          expected_impact="Longer hold times, fewer premature exits",
          confidence=0.6,
        )
        suggestions.append(suggestion)

    # Store suggestions
    self.suggestions.extend(suggestions)

    return suggestions

  def apply_suggestion(self, suggestion: ImprovementSuggestion, approved: bool = False) -> Dict[str, Any]:
    """
    Apply an approved suggestion.

    Args:
      suggestion: The suggestion to apply
      approved: Whether human has approved

    Returns:
      Result of application
    """
    if not approved:
      return {"status": "pending_approval", "suggestion": suggestion.to_dict()}

    param_name = suggestion.parameter.name
    if param_name not in self.parameters:
      return {"status": "error", "message": f"Unknown parameter: {param_name}"}

    # Update parameter
    old_value = self.parameters[param_name].current_value
    self.parameters[param_name].current_value = suggestion.suggested_value
    self.parameters[param_name].last_updated = datetime.now()

    return {
      "status": "applied",
      "parameter": param_name,
      "old_value": old_value,
      "new_value": suggestion.suggested_value,
      "timestamp": datetime.now().isoformat(),
    }

  def get_current_parameters(self) -> Dict[str, Any]:
    """Get all current parameter values."""
    return {
      name: {
        "value": param.current_value,
        "min": param.min_value,
        "max": param.max_value,
        "last_updated": param.last_updated.isoformat() if param.last_updated else None,
      }
      for name, param in self.parameters.items()
    }

  def generate_config_update(self) -> str:
    """
    Generate config update for the trading system.

    Returns:
      Python code to update trading config
    """
    lines = [
      "# Strategy Parameter Update",
      f"# Generated: {datetime.now().isoformat()}",
      "",
      "STRATEGY_CONFIG = {",
    ]

    for name, param in self.parameters.items():
      lines.append(f'    "{name}": {param.current_value},  # {param.description}')

    lines.append("}")

    return '\n'.join(lines)
