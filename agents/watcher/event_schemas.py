"""
Event Schemas for Autonomous Trading Developer System
All events are designed to serialize to <1KB JSON for efficient webhook transmission.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import json
import uuid


class EventType(Enum):
  """Types of events the Watcher Agent can emit."""
  TRADE = "TRADE"
  ERROR = "ERROR"
  SENTIMENT = "SENTIMENT"
  PERFORMANCE_DROP = "PERFORMANCE_DROP"
  INDICATOR_ANOMALY = "INDICATOR_ANOMALY"
  EXECUTION_LAG = "EXECUTION_LAG"
  POSITION_DESYNC = "POSITION_DESYNC"
  CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
  MANUAL_TRIGGER = "MANUAL_TRIGGER"


class EventSeverity(Enum):
  """Severity levels for events."""
  INFO = "INFO"
  WARNING = "WARNING"
  CRITICAL = "CRITICAL"


@dataclass
class BaseEvent:
  """
  Base event structure - all events must serialize to <1KB JSON.

  Attributes:
    event_id: Unique identifier for this event
    event_type: Type of event (from EventType enum)
    severity: Severity level (from EventSeverity enum)
    timestamp: When the event occurred
    symbol: Trading symbol (e.g., "NVDA")
    source_file: File that generated the event
    source_line: Line number in source file
  """
  event_id: str
  event_type: EventType
  severity: EventSeverity
  timestamp: datetime
  symbol: str
  source_file: str
  source_line: int

  @classmethod
  def generate_id(cls) -> str:
    """Generate a unique event ID."""
    return str(uuid.uuid4())[:8]

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
    data = asdict(self)
    data['event_type'] = self.event_type.value
    data['severity'] = self.severity.value
    data['timestamp'] = self.timestamp.isoformat()
    return data

  def to_json(self) -> str:
    """Serialize to JSON string."""
    return json.dumps(self.to_dict(), default=str)

  def validate_size(self) -> bool:
    """Ensure event serializes to <1KB."""
    return len(self.to_json().encode('utf-8')) < 1024


@dataclass
class TradeEvent(BaseEvent):
  """
  Emitted for every trade execution.
  Contains essential trade details for analysis.
  """
  direction: str  # LONG or SHORT
  entry_price: float
  exit_price: Optional[float]
  quantity: float
  pnl: Optional[float]
  strategy: str
  confidence: float
  sentiment_score: float
  hold_time_seconds: Optional[float]
  exit_reason: Optional[str]
  indicators: Dict[str, float] = field(default_factory=dict)
  # Example: {"adx": 32.5, "rsi": 55.2, "atr": 1.45}

  def __post_init__(self):
    """Validate and truncate indicators if needed to stay under 1KB."""
    if self.indicators and len(self.indicators) > 5:
      # Keep only top 5 most important indicators
      important = ['adx', 'rsi', 'atr', 'macd', 'volume']
      self.indicators = {
        k: v for k, v in self.indicators.items()
        if k.lower() in important
      }


@dataclass
class ErrorEvent(BaseEvent):
  """
  Emitted for ERROR or CRITICAL log entries.
  Stack traces are truncated to fit within 1KB limit.
  """
  error_type: str
  error_message: str  # Max 200 chars
  stack_trace_summary: str  # First 200 chars only
  function_name: str
  recovery_attempted: bool

  def __post_init__(self):
    """Truncate fields to ensure <1KB serialization."""
    if len(self.error_message) > 200:
      self.error_message = self.error_message[:197] + "..."
    if len(self.stack_trace_summary) > 200:
      self.stack_trace_summary = self.stack_trace_summary[:197] + "..."


@dataclass
class SentimentEvent(BaseEvent):
  """Emitted when sentiment changes significantly."""
  old_score: float
  new_score: float
  change_magnitude: float
  sentiment_label: str  # bullish, bearish, neutral


@dataclass
class PerformanceDropEvent(BaseEvent):
  """
  Emitted when performance thresholds are breached.
  Includes summary of recent trades for context.
  """
  metric_name: str  # win_rate, daily_pnl, drawdown
  current_value: float
  threshold_value: float
  window_trades: int
  recent_trades: List[Dict[str, Any]] = field(default_factory=list)
  # Last 5 trades summary only

  def __post_init__(self):
    """Limit recent trades to 5 to stay under 1KB."""
    if len(self.recent_trades) > 5:
      self.recent_trades = self.recent_trades[-5:]
    # Minimize each trade dict
    self.recent_trades = [
      {
        'pnl': t.get('pnl'),
        'dir': t.get('direction', t.get('dir'))[:1] if t.get('direction', t.get('dir')) else 'U',
        'ts': t.get('timestamp', t.get('ts', ''))[:10]
      }
      for t in self.recent_trades
    ]


@dataclass
class IndicatorAnomalyEvent(BaseEvent):
  """Emitted when indicator readings are unusual."""
  indicator_name: str
  current_value: float
  expected_min: float
  expected_max: float
  deviation_pct: float


@dataclass
class ExecutionLagEvent(BaseEvent):
  """Emitted when API latency is high."""
  operation: str  # open_position, close_position, get_market_data
  latency_ms: float
  threshold_ms: float
  retry_count: int


@dataclass
class PositionDesyncEvent(BaseEvent):
  """
  Emitted when local/broker position counts differ.
  CRITICAL: This indicates the phantom position bug.
  """
  local_position_count: int
  broker_position_count: int
  phantom_position_ids: List[str] = field(default_factory=list)
  orphaned_position_ids: List[str] = field(default_factory=list)

  def __post_init__(self):
    """Truncate position ID lists to stay under 1KB."""
    if len(self.phantom_position_ids) > 10:
      self.phantom_position_ids = self.phantom_position_ids[:10]
    if len(self.orphaned_position_ids) > 10:
      self.orphaned_position_ids = self.orphaned_position_ids[:10]


@dataclass
class CircuitBreakerEvent(BaseEvent):
  """Emitted when circuit breaker is triggered."""
  trigger_reason: str  # max_loss, consecutive_losses, volatility
  cooldown_seconds: int
  trades_before_trigger: int
  pnl_at_trigger: float


@dataclass
class ManualTriggerEvent(BaseEvent):
  """Emitted when a human manually triggers analysis."""
  description: str
  context: Dict[str, Any] = field(default_factory=dict)

  def __post_init__(self):
    """Truncate description and context."""
    if len(self.description) > 200:
      self.description = self.description[:197] + "..."
    # Limit context keys
    if len(self.context) > 5:
      self.context = dict(list(self.context.items())[:5])


# Factory function for creating events
def create_event(event_type: EventType, **kwargs) -> BaseEvent:
  """
  Factory function to create the appropriate event type.

  Args:
    event_type: The type of event to create
    **kwargs: Event-specific parameters

  Returns:
    An instance of the appropriate event class
  """
  event_classes = {
    EventType.TRADE: TradeEvent,
    EventType.ERROR: ErrorEvent,
    EventType.SENTIMENT: SentimentEvent,
    EventType.PERFORMANCE_DROP: PerformanceDropEvent,
    EventType.INDICATOR_ANOMALY: IndicatorAnomalyEvent,
    EventType.EXECUTION_LAG: ExecutionLagEvent,
    EventType.POSITION_DESYNC: PositionDesyncEvent,
    EventType.CIRCUIT_BREAKER: CircuitBreakerEvent,
    EventType.MANUAL_TRIGGER: ManualTriggerEvent,
  }

  event_class = event_classes.get(event_type)
  if not event_class:
    raise ValueError(f"Unknown event type: {event_type}")

  # Add common fields if not present
  if 'event_id' not in kwargs:
    kwargs['event_id'] = BaseEvent.generate_id()
  if 'timestamp' not in kwargs:
    kwargs['timestamp'] = datetime.now()
  if 'event_type' not in kwargs:
    kwargs['event_type'] = event_type

  return event_class(**kwargs)
