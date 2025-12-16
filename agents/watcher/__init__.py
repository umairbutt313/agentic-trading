"""
Watcher Agent Package
Monitors trading logs and emits structured events to Supervisor Agent.
"""

from .event_schemas import (
  EventType,
  EventSeverity,
  BaseEvent,
  TradeEvent,
  ErrorEvent,
  SentimentEvent,
  PerformanceDropEvent,
  IndicatorAnomalyEvent,
  ExecutionLagEvent,
  PositionDesyncEvent,
  CircuitBreakerEvent,
  ManualTriggerEvent,
  create_event,
)

from .log_parser import LogParser, ParsedLogEntry
from .trigger_rules import TriggerRules, TriggerConfig, TradeStats

__all__ = [
  # Event types
  'EventType',
  'EventSeverity',
  'BaseEvent',
  'TradeEvent',
  'ErrorEvent',
  'SentimentEvent',
  'PerformanceDropEvent',
  'IndicatorAnomalyEvent',
  'ExecutionLagEvent',
  'PositionDesyncEvent',
  'CircuitBreakerEvent',
  'ManualTriggerEvent',
  'create_event',
  # Parser
  'LogParser',
  'ParsedLogEntry',
  # Trigger rules
  'TriggerRules',
  'TriggerConfig',
  'TradeStats',
]
