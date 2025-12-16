"""
Trigger Rules Engine for Watcher Agent
Determines when events should be emitted to the Supervisor Agent.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import hashlib

from .event_schemas import (
  BaseEvent, EventType, EventSeverity,
  TradeEvent, ErrorEvent, PerformanceDropEvent
)


@dataclass
class TriggerConfig:
  """Configuration for trigger thresholds."""
  # Performance thresholds
  max_daily_loss_threshold: float = 50.0  # USD
  min_win_rate_threshold: float = 0.40  # 40%
  win_rate_window: int = 10  # trades
  max_drawdown_threshold: float = 0.05  # 5%

  # Latency thresholds
  latency_threshold_ms: float = 500.0

  # Sentiment thresholds
  sentiment_change_threshold: float = 2.0  # Score change to trigger

  # Cooldown settings (prevent event flooding)
  error_cooldown_seconds: int = 60
  performance_cooldown_seconds: int = 300
  sentiment_cooldown_seconds: int = 30

  # Deduplication
  dedup_window_seconds: int = 60
  max_events_per_minute: int = 30


@dataclass
class TradeStats:
  """Running statistics for trade performance."""
  recent_trades: deque = field(default_factory=lambda: deque(maxlen=100))
  daily_pnl: float = 0.0
  daily_trades_count: int = 0
  wins: int = 0
  losses: int = 0
  starting_balance: Optional[float] = None
  current_balance: Optional[float] = None
  peak_balance: Optional[float] = None
  last_trade_time: Optional[datetime] = None
  consecutive_losses: int = 0
  day_start: Optional[datetime] = None

  def reset_daily(self):
    """Reset daily statistics."""
    self.daily_pnl = 0.0
    self.daily_trades_count = 0
    self.wins = 0
    self.losses = 0
    self.consecutive_losses = 0
    self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


class TriggerRules:
  """
  Determines when events should be emitted to the Supervisor Agent.

  Features:
  - Performance threshold monitoring
  - Event deduplication
  - Rate limiting
  - Cooldown management
  """

  def __init__(self, config: Optional[TriggerConfig] = None):
    """
    Initialize trigger rules engine.

    Args:
      config: Trigger configuration (uses defaults if not provided)
    """
    self.config = config or TriggerConfig()
    self.stats = TradeStats()
    self.stats.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Event tracking for deduplication and rate limiting
    self._recent_event_hashes: deque = deque(maxlen=1000)
    self._event_timestamps: deque = deque(maxlen=100)
    self._last_event_by_type: Dict[EventType, datetime] = {}

  def should_emit(self, event: BaseEvent) -> bool:
    """
    Determine if an event should be sent to the Supervisor.

    Args:
      event: The event to evaluate

    Returns:
      True if event should be emitted, False otherwise
    """
    # Check for new day and reset stats
    self._check_day_rollover()

    # Check rate limiting first
    if self._is_rate_limited():
      return False

    # Check deduplication
    if self._is_duplicate(event):
      return False

    # Check cooldown for event type
    if self._is_on_cooldown(event):
      return False

    # Always emit CRITICAL severity
    if event.severity == EventSeverity.CRITICAL:
      self._record_event(event)
      return True

    # Always emit ERROR type
    if event.event_type == EventType.ERROR:
      self._record_event(event)
      return True

    # Always emit POSITION_DESYNC (critical bug indicator)
    if event.event_type == EventType.POSITION_DESYNC:
      self._record_event(event)
      return True

    # Always emit CIRCUIT_BREAKER
    if event.event_type == EventType.CIRCUIT_BREAKER:
      self._record_event(event)
      return True

    # Handle TRADE events
    if event.event_type == EventType.TRADE:
      if self._should_emit_trade(event):
        self._record_event(event)
        return True
      return False

    # Check performance thresholds for any event
    if self._check_performance_thresholds():
      self._record_event(event)
      return True

    # Emit SENTIMENT changes above threshold
    if event.event_type == EventType.SENTIMENT:
      if hasattr(event, 'change_magnitude'):
        if event.change_magnitude >= self.config.sentiment_change_threshold:
          self._record_event(event)
          return True

    # Emit high latency events
    if event.event_type == EventType.EXECUTION_LAG:
      if hasattr(event, 'latency_ms'):
        if event.latency_ms >= self.config.latency_threshold_ms:
          self._record_event(event)
          return True

    return False

  def _should_emit_trade(self, event: TradeEvent) -> bool:
    """
    Determine if a trade event should be emitted.

    All trades are recorded for analysis, but only significant ones
    are emitted immediately.
    """
    # Update trade statistics
    self._update_trade_stats(event)

    # Always emit losing trades (for analysis)
    if event.pnl is not None and event.pnl < 0:
      return True

    # Always emit if win rate has dropped below threshold
    if self._get_recent_win_rate() < self.config.min_win_rate_threshold:
      return True

    # Emit if daily loss threshold breached
    if self.stats.daily_pnl < -self.config.max_daily_loss_threshold:
      return True

    # Emit if drawdown threshold breached
    if self._get_current_drawdown() > self.config.max_drawdown_threshold:
      return True

    # Emit if multiple consecutive losses
    if self.stats.consecutive_losses >= 3:
      return True

    # Emit all trades for complete analysis
    # (Can be disabled for high-frequency systems)
    return True

  def _update_trade_stats(self, event: TradeEvent):
    """Update running trade statistics."""
    # Add to recent trades
    self.stats.recent_trades.append({
      'timestamp': event.timestamp,
      'pnl': event.pnl,
      'direction': event.direction,
      'confidence': event.confidence,
      'strategy': event.strategy,
    })

    # Update daily P&L
    if event.pnl is not None:
      self.stats.daily_pnl += event.pnl
      self.stats.daily_trades_count += 1

      if event.pnl > 0:
        self.stats.wins += 1
        self.stats.consecutive_losses = 0
      else:
        self.stats.losses += 1
        self.stats.consecutive_losses += 1

    self.stats.last_trade_time = event.timestamp

  def _get_recent_win_rate(self) -> float:
    """Calculate win rate over the configured window."""
    if len(self.stats.recent_trades) < self.config.win_rate_window:
      return 1.0  # Not enough data, assume OK

    recent = list(self.stats.recent_trades)[-self.config.win_rate_window:]
    wins = sum(1 for t in recent if t.get('pnl') and t['pnl'] > 0)
    return wins / len(recent)

  def _get_current_drawdown(self) -> float:
    """Calculate current drawdown from peak."""
    if not self.stats.starting_balance or not self.stats.current_balance:
      return 0.0

    peak = self.stats.peak_balance or self.stats.starting_balance
    if peak <= 0:
      return 0.0

    return (peak - self.stats.current_balance) / peak

  def _check_performance_thresholds(self) -> bool:
    """Check if any performance threshold is breached."""
    # Check daily loss
    if self.stats.daily_pnl < -self.config.max_daily_loss_threshold:
      return True

    # Check win rate
    if len(self.stats.recent_trades) >= self.config.win_rate_window:
      if self._get_recent_win_rate() < self.config.min_win_rate_threshold:
        return True

    # Check drawdown
    if self._get_current_drawdown() > self.config.max_drawdown_threshold:
      return True

    return False

  def _check_day_rollover(self):
    """Check if a new trading day has started and reset stats."""
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if self.stats.day_start and self.stats.day_start < today_start:
      self.stats.reset_daily()

  def _is_rate_limited(self) -> bool:
    """Check if event emission is rate limited."""
    now = datetime.now()
    cutoff = now - timedelta(seconds=60)

    # Count events in the last minute
    recent_count = sum(1 for ts in self._event_timestamps if ts > cutoff)
    return recent_count >= self.config.max_events_per_minute

  def _is_duplicate(self, event: BaseEvent) -> bool:
    """Check if this event is a duplicate of a recent event."""
    event_hash = self._compute_event_hash(event)
    now = datetime.now()

    # Check if hash exists in recent events
    for stored_hash, stored_time in self._recent_event_hashes:
      if stored_hash == event_hash:
        # Check if within dedup window
        if (now - stored_time).total_seconds() < self.config.dedup_window_seconds:
          return True

    return False

  def _is_on_cooldown(self, event: BaseEvent) -> bool:
    """Check if event type is on cooldown."""
    if event.event_type not in self._last_event_by_type:
      return False

    last_time = self._last_event_by_type[event.event_type]
    now = datetime.now()

    # Get cooldown for event type
    cooldown_seconds = self._get_cooldown_seconds(event.event_type)
    return (now - last_time).total_seconds() < cooldown_seconds

  def _get_cooldown_seconds(self, event_type: EventType) -> int:
    """Get cooldown period for an event type."""
    cooldowns = {
      EventType.ERROR: self.config.error_cooldown_seconds,
      EventType.PERFORMANCE_DROP: self.config.performance_cooldown_seconds,
      EventType.SENTIMENT: self.config.sentiment_cooldown_seconds,
      EventType.INDICATOR_ANOMALY: 120,
      EventType.EXECUTION_LAG: 60,
    }
    return cooldowns.get(event_type, 30)

  def _compute_event_hash(self, event: BaseEvent) -> str:
    """Compute a hash for deduplication."""
    # Include event type, symbol, and key identifying features
    hash_input = f"{event.event_type.value}:{event.symbol}"

    if hasattr(event, 'error_type'):
      hash_input += f":{event.error_type}"
    if hasattr(event, 'error_message'):
      hash_input += f":{event.error_message[:50]}"
    if hasattr(event, 'trigger_reason'):
      hash_input += f":{event.trigger_reason}"

    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

  def _record_event(self, event: BaseEvent):
    """Record that an event was emitted."""
    now = datetime.now()

    # Record timestamp
    self._event_timestamps.append(now)

    # Record hash for deduplication
    event_hash = self._compute_event_hash(event)
    self._recent_event_hashes.append((event_hash, now))

    # Record last event time for cooldown
    self._last_event_by_type[event.event_type] = now

  def update_balance(self, current_balance: float, starting_balance: Optional[float] = None):
    """
    Update balance information for drawdown calculations.

    Args:
      current_balance: Current account balance
      starting_balance: Starting balance (if first update)
    """
    if starting_balance is not None:
      self.stats.starting_balance = starting_balance

    self.stats.current_balance = current_balance

    # Update peak balance
    if self.stats.peak_balance is None or current_balance > self.stats.peak_balance:
      self.stats.peak_balance = current_balance

  def create_performance_drop_event(self) -> Optional[PerformanceDropEvent]:
    """
    Create a performance drop event if thresholds are breached.

    Returns:
      PerformanceDropEvent if thresholds breached, None otherwise
    """
    if not self._check_performance_thresholds():
      return None

    # Determine which threshold was breached
    metric_name = "unknown"
    current_value = 0.0
    threshold_value = 0.0

    if self.stats.daily_pnl < -self.config.max_daily_loss_threshold:
      metric_name = "daily_pnl"
      current_value = self.stats.daily_pnl
      threshold_value = -self.config.max_daily_loss_threshold
    elif self._get_recent_win_rate() < self.config.min_win_rate_threshold:
      metric_name = "win_rate"
      current_value = self._get_recent_win_rate()
      threshold_value = self.config.min_win_rate_threshold
    elif self._get_current_drawdown() > self.config.max_drawdown_threshold:
      metric_name = "drawdown"
      current_value = self._get_current_drawdown()
      threshold_value = self.config.max_drawdown_threshold

    # Get recent trades summary
    recent = list(self.stats.recent_trades)[-5:]

    return PerformanceDropEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.PERFORMANCE_DROP,
      severity=EventSeverity.CRITICAL,
      timestamp=datetime.now(),
      symbol="PORTFOLIO",
      source_file="trigger_rules.py",
      source_line=0,
      metric_name=metric_name,
      current_value=current_value,
      threshold_value=threshold_value,
      window_trades=len(recent),
      recent_trades=recent
    )

  def get_stats_summary(self) -> Dict[str, Any]:
    """Get a summary of current trading statistics."""
    return {
      'daily_pnl': self.stats.daily_pnl,
      'daily_trades': self.stats.daily_trades_count,
      'wins': self.stats.wins,
      'losses': self.stats.losses,
      'win_rate': self._get_recent_win_rate(),
      'consecutive_losses': self.stats.consecutive_losses,
      'drawdown': self._get_current_drawdown(),
      'current_balance': self.stats.current_balance,
      'peak_balance': self.stats.peak_balance,
      'events_last_minute': len([
        ts for ts in self._event_timestamps
        if (datetime.now() - ts).total_seconds() < 60
      ]),
    }
