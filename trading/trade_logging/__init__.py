"""
Trade Decision Logging Module
Provides comprehensive logging for all trading decisions and outcomes

NOTE: Renamed from 'logging' to 'trade_logging' to avoid shadowing
      Python's standard library 'logging' module.
"""

from .trade_logger import TradeDecisionLogger, get_trade_logger

__all__ = ['TradeDecisionLogger', 'get_trade_logger']
