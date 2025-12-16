"""
Log Parser for Trading System
Parses various log formats and converts them to structured events.
"""

import re
from datetime import datetime
from typing import Optional, Dict, List, Any, Generator, Tuple
from dataclasses import dataclass

from .event_schemas import (
  EventType, EventSeverity, BaseEvent, TradeEvent, ErrorEvent,
  SentimentEvent, PerformanceDropEvent, IndicatorAnomalyEvent,
  ExecutionLagEvent, PositionDesyncEvent, CircuitBreakerEvent,
  create_event
)


@dataclass
class ParsedLogEntry:
  """Intermediate representation of a parsed log line."""
  timestamp: datetime
  level: str  # INFO, WARNING, ERROR, CRITICAL
  source_file: str
  source_line: int
  message: str
  raw_line: str


class LogParser:
  """
  Parses trading system logs into structured events.

  Supports multiple log formats:
  - Trade decision logs (TradeDecisionLogger format)
  - Standard Python logging format
  - Enhanced scalper logs
  """

  # Regular expression patterns for different log formats
  PATTERNS = {
    # Trade decision format: [2024-01-01 12:00:00] | NVDA | EXECUTE | 7.5 BULLISH | LONG | SUCCESS
    'trade_decision': re.compile(
      r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] \| '
      r'(\w+) \| '
      r'(\w+) \| '
      r'([\d.]+|N/A) ?(\w+)? \| '
      r'(.+?) \| '
      r'(\w+)'
    ),

    # Standard Python logging: 2024-01-01 12:00:00,000 - ERROR - module:123 - Message
    'standard_log': re.compile(
      r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),?(\d+)? - '
      r'(DEBUG|INFO|WARNING|ERROR|CRITICAL) - '
      r'(?:(\w+):(\d+) - )?'
      r'(.+)'
    ),

    # Position opened: NVDA_1234567890
    'position_opened': re.compile(r'Position opened: (\w+_\d+)'),

    # Position closed: P&L $123.45 or P&L -$45.67
    'position_closed': re.compile(r'Position closed: P&L \$?(-?[\d.]+)'),

    # Phantom positions detected
    'phantom_positions': re.compile(r'PHANTOM POSITIONS DETECTED: (\d+) positions?'),

    # Circuit breaker triggered
    'circuit_breaker': re.compile(r'CIRCUIT[ _]?BREAKER (TRIGGERED|ACTIVATED)'),

    # API latency log
    'api_latency': re.compile(r'API (call|request) took (\d+)ms'),

    # Sentiment change
    'sentiment_change': re.compile(r'Sentiment changed?: (\d+\.?\d*) -> (\d+\.?\d*)'),

    # Performance metrics
    'win_rate': re.compile(r'Win rate: ([\d.]+)%'),
    'daily_pnl': re.compile(r'Daily P&L: \$?(-?[\d.]+)'),
    'drawdown': re.compile(r'Drawdown: ([\d.]+)%'),

    # Indicator values
    'indicator_adx': re.compile(r'ADX[=: ]+(\d+\.?\d*)'),
    'indicator_atr': re.compile(r'ATR[=: ]+(\d+\.?\d*)'),
    'indicator_rsi': re.compile(r'RSI[=: ]+(\d+\.?\d*)'),

    # Error patterns
    'exception': re.compile(r'(Exception|Error|Failed|Traceback)'),

    # Trade details (from multi-line logs)
    'entry_price': re.compile(r'Entry: \$?([\d.]+)'),
    'exit_price': re.compile(r'Exit: \$?([\d.]+)'),
    'pnl_line': re.compile(r'P&L: [€$]?(-?[\d.]+)'),
    'hold_duration': re.compile(r'Hold Duration: (\d+:\d{2}:\d{2}|\d+s|\d+m)'),
    'confidence': re.compile(r'Confidence: (\d+\.?\d*)'),
    'strategy': re.compile(r'Strategy: (\w+)'),
  }

  def __init__(self, default_symbol: str = "NVDA"):
    """
    Initialize the log parser.

    Args:
      default_symbol: Default trading symbol when not specified in log
    """
    self.default_symbol = default_symbol
    self._context = {}  # For multi-line log context
    self._pending_trade = None  # For accumulating trade details

  def parse_line(self, line: str, source_file: str = "unknown") -> Optional[ParsedLogEntry]:
    """
    Parse a single log line into a ParsedLogEntry.

    Args:
      line: Raw log line text
      source_file: Path to the source log file

    Returns:
      ParsedLogEntry or None if line is not parseable
    """
    line = line.strip()
    if not line:
      return None

    # Try trade decision format first
    match = self.PATTERNS['trade_decision'].match(line)
    if match:
      timestamp_str, symbol, action, sentiment, label, decision, result = match.groups()
      return ParsedLogEntry(
        timestamp=datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"),
        level="INFO",
        source_file=source_file,
        source_line=0,
        message=line,
        raw_line=line
      )

    # Try standard logging format
    match = self.PATTERNS['standard_log'].match(line)
    if match:
      groups = match.groups()
      timestamp_str = groups[0]
      ms = groups[1] or "0"
      level = groups[2]
      module = groups[3] or "unknown"
      line_num = int(groups[4]) if groups[4] else 0
      message = groups[5]

      return ParsedLogEntry(
        timestamp=datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"),
        level=level,
        source_file=f"{source_file}:{module}",
        source_line=line_num,
        message=message,
        raw_line=line
      )

    # For unstructured lines, try to extract timestamp
    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if timestamp_match:
      return ParsedLogEntry(
        timestamp=datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S"),
        level="INFO",
        source_file=source_file,
        source_line=0,
        message=line,
        raw_line=line
      )

    return None

  def parse_to_events(self, line: str, source_file: str = "unknown") -> List[BaseEvent]:
    """
    Parse a log line and generate appropriate events.

    Args:
      line: Raw log line text
      source_file: Path to the source log file

    Returns:
      List of events (may be empty or contain multiple events)
    """
    events = []
    entry = self.parse_line(line, source_file)

    if not entry:
      return events

    # Check for error/critical events first
    if entry.level in ("ERROR", "CRITICAL"):
      events.append(self._create_error_event(entry))

    # Check for trade decision
    if self._is_trade_decision(entry.message):
      trade_event = self._parse_trade_decision(entry)
      if trade_event:
        events.append(trade_event)

    # Check for position desync
    phantom_match = self.PATTERNS['phantom_positions'].search(entry.message)
    if phantom_match:
      events.append(self._create_position_desync_event(entry, phantom_match))

    # Check for circuit breaker
    if self.PATTERNS['circuit_breaker'].search(entry.message):
      events.append(self._create_circuit_breaker_event(entry))

    # Check for API latency
    latency_match = self.PATTERNS['api_latency'].search(entry.message)
    if latency_match and int(latency_match.group(2)) > 500:
      events.append(self._create_latency_event(entry, latency_match))

    # Check for sentiment changes
    sentiment_match = self.PATTERNS['sentiment_change'].search(entry.message)
    if sentiment_match:
      events.append(self._create_sentiment_event(entry, sentiment_match))

    return events

  def _is_trade_decision(self, message: str) -> bool:
    """Check if message contains trade decision keywords."""
    keywords = ['EXECUTE', 'LONG', 'SHORT', 'CLOSE', 'Position opened', 'Position closed']
    return any(kw in message for kw in keywords)

  def _parse_trade_decision(self, entry: ParsedLogEntry) -> Optional[TradeEvent]:
    """Parse a trade decision from log entry."""
    match = self.PATTERNS['trade_decision'].match(entry.message)
    if not match:
      # Try position opened/closed patterns
      return self._parse_position_event(entry)

    timestamp_str, symbol, action, sentiment_str, label, decision, result = match.groups()

    # Determine direction
    direction = "UNKNOWN"
    if "LONG" in decision.upper():
      direction = "LONG"
    elif "SHORT" in decision.upper():
      direction = "SHORT"

    # Parse sentiment
    try:
      sentiment = float(sentiment_str) if sentiment_str != "N/A" else 0.0
    except (ValueError, TypeError):
      sentiment = 0.0

    # Extract indicators from context
    indicators = self._extract_indicators(entry.message)

    # Determine severity
    severity = EventSeverity.INFO
    if result == "FAILED":
      severity = EventSeverity.WARNING

    return TradeEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.TRADE,
      severity=severity,
      timestamp=entry.timestamp,
      symbol=symbol,
      source_file=entry.source_file,
      source_line=entry.source_line,
      direction=direction,
      entry_price=self._extract_price(entry.message, 'entry'),
      exit_price=self._extract_price(entry.message, 'exit'),
      quantity=0.0,  # Will be updated from context
      pnl=self._extract_pnl(entry.message),
      strategy=self._extract_strategy(entry.message),
      confidence=self._extract_confidence(entry.message),
      sentiment_score=sentiment,
      hold_time_seconds=self._extract_hold_time(entry.message),
      exit_reason=decision if action == "CLOSE" else None,
      indicators=indicators
    )

  def _parse_position_event(self, entry: ParsedLogEntry) -> Optional[TradeEvent]:
    """Parse position opened/closed events."""
    # Check for position opened
    opened_match = self.PATTERNS['position_opened'].search(entry.message)
    if opened_match:
      position_id = opened_match.group(1)
      symbol = position_id.split('_')[0] if '_' in position_id else self.default_symbol

      return TradeEvent(
        event_id=BaseEvent.generate_id(),
        event_type=EventType.TRADE,
        severity=EventSeverity.INFO,
        timestamp=entry.timestamp,
        symbol=symbol,
        source_file=entry.source_file,
        source_line=entry.source_line,
        direction="LONG",  # Default, will be updated
        entry_price=0.0,
        exit_price=None,
        quantity=0.0,
        pnl=None,
        strategy="unknown",
        confidence=0.0,
        sentiment_score=0.0,
        hold_time_seconds=None,
        exit_reason=None,
        indicators={}
      )

    # Check for position closed
    closed_match = self.PATTERNS['position_closed'].search(entry.message)
    if closed_match:
      pnl = float(closed_match.group(1))
      severity = EventSeverity.INFO if pnl >= 0 else EventSeverity.WARNING

      return TradeEvent(
        event_id=BaseEvent.generate_id(),
        event_type=EventType.TRADE,
        severity=severity,
        timestamp=entry.timestamp,
        symbol=self.default_symbol,
        source_file=entry.source_file,
        source_line=entry.source_line,
        direction="UNKNOWN",
        entry_price=0.0,
        exit_price=0.0,
        quantity=0.0,
        pnl=pnl,
        strategy="unknown",
        confidence=0.0,
        sentiment_score=0.0,
        hold_time_seconds=None,
        exit_reason="CLOSED",
        indicators={}
      )

    return None

  def _create_error_event(self, entry: ParsedLogEntry) -> ErrorEvent:
    """Create an error event from a log entry."""
    # Extract error details
    error_type = "UnknownError"
    if "Exception" in entry.message:
      error_type = entry.message.split("Exception")[0].strip().split()[-1] + "Exception"
    elif "Error" in entry.message:
      error_type = entry.message.split("Error")[0].strip().split()[-1] + "Error"

    # Extract function name from source file
    func_match = re.search(r'(\w+)\(\)', entry.message)
    function_name = func_match.group(1) if func_match else "unknown"

    return ErrorEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.ERROR,
      severity=EventSeverity.CRITICAL if entry.level == "CRITICAL" else EventSeverity.WARNING,
      timestamp=entry.timestamp,
      symbol=self.default_symbol,
      source_file=entry.source_file,
      source_line=entry.source_line,
      error_type=error_type,
      error_message=entry.message[:200],
      stack_trace_summary="",
      function_name=function_name,
      recovery_attempted=False
    )

  def _create_position_desync_event(
    self, entry: ParsedLogEntry, match: re.Match
  ) -> PositionDesyncEvent:
    """Create a position desync event."""
    phantom_count = int(match.group(1))

    return PositionDesyncEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.POSITION_DESYNC,
      severity=EventSeverity.CRITICAL,
      timestamp=entry.timestamp,
      symbol=self.default_symbol,
      source_file=entry.source_file,
      source_line=entry.source_line,
      local_position_count=0,
      broker_position_count=phantom_count,
      phantom_position_ids=[],
      orphaned_position_ids=[]
    )

  def _create_circuit_breaker_event(self, entry: ParsedLogEntry) -> CircuitBreakerEvent:
    """Create a circuit breaker event."""
    # Extract reason from message
    reason = "unknown"
    if "drawdown" in entry.message.lower():
      reason = "max_drawdown"
    elif "loss" in entry.message.lower():
      reason = "max_loss"
    elif "consecutive" in entry.message.lower():
      reason = "consecutive_losses"

    return CircuitBreakerEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.CIRCUIT_BREAKER,
      severity=EventSeverity.CRITICAL,
      timestamp=entry.timestamp,
      symbol=self.default_symbol,
      source_file=entry.source_file,
      source_line=entry.source_line,
      trigger_reason=reason,
      cooldown_seconds=86400,  # 24 hours default
      trades_before_trigger=0,
      pnl_at_trigger=0.0
    )

  def _create_latency_event(
    self, entry: ParsedLogEntry, match: re.Match
  ) -> ExecutionLagEvent:
    """Create an execution lag event."""
    latency_ms = float(match.group(2))
    operation = match.group(1)

    return ExecutionLagEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.EXECUTION_LAG,
      severity=EventSeverity.WARNING if latency_ms < 1000 else EventSeverity.CRITICAL,
      timestamp=entry.timestamp,
      symbol=self.default_symbol,
      source_file=entry.source_file,
      source_line=entry.source_line,
      operation=operation,
      latency_ms=latency_ms,
      threshold_ms=500.0,
      retry_count=0
    )

  def _create_sentiment_event(
    self, entry: ParsedLogEntry, match: re.Match
  ) -> SentimentEvent:
    """Create a sentiment change event."""
    old_score = float(match.group(1))
    new_score = float(match.group(2))
    change = abs(new_score - old_score)

    # Determine label
    if new_score >= 7:
      label = "bullish"
    elif new_score <= 4:
      label = "bearish"
    else:
      label = "neutral"

    return SentimentEvent(
      event_id=BaseEvent.generate_id(),
      event_type=EventType.SENTIMENT,
      severity=EventSeverity.INFO if change < 2 else EventSeverity.WARNING,
      timestamp=entry.timestamp,
      symbol=self.default_symbol,
      source_file=entry.source_file,
      source_line=entry.source_line,
      old_score=old_score,
      new_score=new_score,
      change_magnitude=change,
      sentiment_label=label
    )

  def _extract_indicators(self, message: str) -> Dict[str, float]:
    """Extract indicator values from message."""
    indicators = {}

    for name, pattern in [
      ('adx', self.PATTERNS['indicator_adx']),
      ('atr', self.PATTERNS['indicator_atr']),
      ('rsi', self.PATTERNS['indicator_rsi']),
    ]:
      match = pattern.search(message)
      if match:
        indicators[name] = float(match.group(1))

    return indicators

  def _extract_price(self, message: str, price_type: str) -> float:
    """Extract entry or exit price from message."""
    if price_type == 'entry':
      match = self.PATTERNS['entry_price'].search(message)
    else:
      match = self.PATTERNS['exit_price'].search(message)

    return float(match.group(1)) if match else 0.0

  def _extract_pnl(self, message: str) -> Optional[float]:
    """Extract P&L from message."""
    match = self.PATTERNS['pnl_line'].search(message)
    return float(match.group(1)) if match else None

  def _extract_strategy(self, message: str) -> str:
    """Extract strategy name from message."""
    match = self.PATTERNS['strategy'].search(message)
    return match.group(1) if match else "unknown"

  def _extract_confidence(self, message: str) -> float:
    """Extract confidence from message."""
    match = self.PATTERNS['confidence'].search(message)
    return float(match.group(1)) if match else 0.0

  def _extract_hold_time(self, message: str) -> Optional[float]:
    """Extract hold duration in seconds from message."""
    match = self.PATTERNS['hold_duration'].search(message)
    if not match:
      return None

    duration_str = match.group(1)

    # Parse different formats
    if ':' in duration_str:
      # Format: HH:MM:SS or MM:SS
      parts = duration_str.split(':')
      if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
      elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif duration_str.endswith('s'):
      return float(duration_str[:-1])
    elif duration_str.endswith('m'):
      return float(duration_str[:-1]) * 60

    return None

  def parse_file(self, file_path: str) -> Generator[BaseEvent, None, None]:
    """
    Parse an entire log file and yield events.

    Args:
      file_path: Path to the log file

    Yields:
      Events parsed from each line
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
      for line in f:
        events = self.parse_to_events(line, file_path)
        for event in events:
          yield event
