#!/usr/bin/env python3
"""
Human-Readable Trade Decision Logger
Provides clear, comprehensive logging of all trading decisions, executions, and outcomes
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import threading


@dataclass
class TradeDecision:
    """Represents a single trade decision event"""
    timestamp: str
    symbol: str
    action: str  # ANALYZE, EXECUTE, HOLD, CLOSE
    sentiment: Optional[float]
    sentiment_label: str
    decision: str
    result: str
    details: Dict[str, Any]


@dataclass
class DailySummary:
    """Daily trading performance summary"""
    date: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    pnl_percentage: float
    starting_balance: float
    ending_balance: float
    peak_balance: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    avg_hold_time: str
    position_breakdown: Dict[str, int]
    strategy_performance: Dict[str, Dict[str, Any]]
    sentiment_stats: Dict[str, Any]
    risk_events: Dict[str, int]


class TradeDecisionLogger:
    """
    Comprehensive trade decision logging system

    Features:
    - Human-readable log format
    - Daily summaries
    - Pattern recognition
    - Alert system
    - JSON backup for data analysis
    """

    def __init__(self, base_dir: str = None):
        """Initialize the trade decision logger"""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.base_dir = base_dir
        self.logs_dir = os.path.join(base_dir, "logs")
        self.trades_dir = os.path.join(self.logs_dir, "trades")
        self.summaries_dir = os.path.join(self.logs_dir, "daily_summaries")
        self.alerts_dir = os.path.join(self.logs_dir, "alerts")

        # Create directories
        for directory in [self.logs_dir, self.trades_dir, self.summaries_dir, self.alerts_dir]:
            os.makedirs(directory, exist_ok=True)

        # Setup logger
        self._setup_logger()

        # In-memory tracking for daily summary
        self.daily_trades: List[TradeDecision] = []
        self.daily_stats = {
            'starting_balance': None,
            'peak_balance': None,
            'trades_by_strategy': {},
            'sentiment_readings': [],
            'risk_events': {
                'position_rejections': 0,
                'api_failures': 0,
                'circuit_breakers': 0
            }
        }

        # Thread lock for concurrent access
        self.lock = threading.Lock()

    def _setup_logger(self):
        """Setup the human-readable trade decision logger"""
        self.logger = logging.getLogger('TradeDecisions')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers = []

        # File handler for main log
        log_file = os.path.join(self.logs_dir, "trade_decisions.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Custom formatter for human readability
        formatter = logging.Formatter(
            '%(message)s'  # We'll format messages ourselves for better control
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _get_sentiment_label(self, score: Optional[float]) -> str:
        """Convert sentiment score to human-readable label"""
        if score is None:
            return "UNKNOWN"
        elif score >= 8.0:
            return "VERY_BULLISH"
        elif score >= 7.0:
            return "BULLISH"
        elif score >= 6.0:
            return "SLIGHTLY_BULLISH"
        elif score >= 5.0:
            return "NEUTRAL"
        elif score >= 4.0:
            return "SLIGHTLY_BEARISH"
        elif score >= 3.0:
            return "BEARISH"
        else:
            return "VERY_BEARISH"

    def log_trade_decision(
        self,
        symbol: str,
        action: str,
        sentiment: Optional[float],
        decision: str,
        result: str,
        **kwargs
    ):
        """
        Log a trade decision with detailed information

        Args:
            symbol: Stock ticker
            action: ANALYZE, EXECUTE, HOLD, CLOSE
            sentiment: Sentiment score (1-10)
            decision: What the system decided to do
            result: SUCCESS, FAILED, SKIPPED
            **kwargs: Additional details (size, price, position_id, error, etc.)
        """
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sentiment_label = self._get_sentiment_label(sentiment)
            sentiment_str = f"{sentiment:.1f} {sentiment_label}" if sentiment else "N/A"

            # Create decision object
            trade_decision = TradeDecision(
                timestamp=timestamp,
                symbol=symbol,
                action=action,
                sentiment=sentiment,
                sentiment_label=sentiment_label,
                decision=decision,
                result=result,
                details=kwargs
            )

            # Store for daily summary
            self.daily_trades.append(trade_decision)

            # Store sentiment reading
            if sentiment is not None:
                self.daily_stats['sentiment_readings'].append(sentiment)

            # Format and log the message
            header = f"[{timestamp}] | {symbol} | {action} | {sentiment_str} | {decision} | {result}"
            self.logger.info(header)

            # Add detailed information based on action type
            if action == "EXECUTE" and "size" in kwargs and "price" in kwargs:
                reasoning = kwargs.get('reasoning', 'No reason provided')
                position_size = kwargs.get('size', 0)
                price = kwargs.get('price', 0)
                balance = kwargs.get('balance', 0)
                target = kwargs.get('target', None)
                stop = kwargs.get('stop', None)
                position_id = kwargs.get('position_id', 'N/A')

                self.logger.info(f"  Reasoning: {reasoning}")
                self.logger.info(f"  Position Size: {position_size:.2f} lots (€{position_size * price:.2f} / €{balance:.2f})")
                self.logger.info(f"  Entry: ${price:.2f} | Target: ${target:.2f} | Stop: ${stop:.2f}" if target and stop else f"  Entry: ${price:.2f}")
                if result == "SUCCESS":
                    self.logger.info(f"  Position ID: {position_id}")

            elif action == "CLOSE" and "entry_price" in kwargs and "exit_price" in kwargs:
                entry = kwargs.get('entry_price', 0)
                exit_price = kwargs.get('exit_price', 0)
                pnl = kwargs.get('pnl', 0)
                pnl_pct = kwargs.get('pnl_pct', 0)
                hold_duration = kwargs.get('hold_duration', 'N/A')
                reasoning = kwargs.get('reasoning', 'No reason provided')

                self.logger.info(f"  Reasoning: {reasoning}")
                self.logger.info(f"  Entry: ${entry:.2f} | Exit: ${exit_price:.2f} | P&L: €{pnl:.2f} ({pnl_pct:+.2f}%)")
                self.logger.info(f"  Hold Duration: {hold_duration}")

            elif result == "FAILED" and "error" in kwargs:
                error_msg = kwargs.get('error', 'Unknown error')
                retry_info = kwargs.get('retry_info', '')
                self.logger.info(f"  Error: {error_msg}")
                if retry_info:
                    self.logger.info(f"  Action: {retry_info}")

            elif result == "SKIPPED" and "reason" in kwargs:
                reason = kwargs.get('reason', 'No reason provided')
                additional = kwargs.get('additional_info', '')
                self.logger.info(f"  Reason: {reason}")
                if additional:
                    self.logger.info(f"  {additional}")

            self.logger.info("")  # Blank line for readability

            # Save to JSON backup
            self._save_trade_json(trade_decision)

    def log_position_rejection(
        self,
        symbol: str,
        sentiment: float,
        reason: str,
        current_exposure: float = 0,
        max_exposure: float = 0
    ):
        """Log a rejected position opening"""
        with self.lock:
            self.daily_stats['risk_events']['position_rejections'] += 1

        self.log_trade_decision(
            symbol=symbol,
            action="ANALYZE",
            sentiment=sentiment,
            decision="OPEN_POSITION REJECTED",
            result="SKIPPED",
            reason=reason,
            additional_info=f"Current Exposure: €{current_exposure:.2f} / €{max_exposure:.2f}"
        )

    def log_circuit_breaker(
        self,
        reason: str,
        starting_balance: float,
        current_balance: float,
        drawdown_pct: float,
        action: str = "Trading suspended for 24 hours"
    ):
        """Log circuit breaker activation"""
        with self.lock:
            self.daily_stats['risk_events']['circuit_breakers'] += 1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info(f"[{timestamp}] | SYSTEM | HALT | N/A | CIRCUIT_BREAKER TRIGGERED | EMERGENCY_STOP")
        self.logger.info(f"  Reason: {reason}")
        self.logger.info(f"  Starting Balance: €{starting_balance:.2f} | Current: €{current_balance:.2f} | Drawdown: {drawdown_pct:.2f}%")
        self.logger.info(f"  Action: {action}")
        self.logger.info("")

    def log_api_failure(
        self,
        symbol: str,
        sentiment: float,
        decision: str,
        error: str,
        http_status: Optional[int] = None,
        retry_attempt: int = 1,
        max_retries: int = 3
    ):
        """Log API failure event"""
        with self.lock:
            self.daily_stats['risk_events']['api_failures'] += 1

        retry_info = f"Retrying in 60 seconds (Attempt {retry_attempt}/{max_retries})"
        error_detail = f"{error}" + (f" - HTTP {http_status}" if http_status else "")

        self.log_trade_decision(
            symbol=symbol,
            action="EXECUTE",
            sentiment=sentiment,
            decision=decision,
            result="FAILED",
            error=error_detail,
            retry_info=retry_info
        )

    def log_alert(
        self,
        alert_type: str,
        priority: str,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Log high-priority alerts"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log to main log
        self.logger.info(f"[{timestamp}] | ALERT | {priority}")
        self.logger.info(f"{'🔴' if priority == 'HIGH_PRIORITY' else '⚠️'} {alert_type}:")
        self.logger.info(f"- {message}")

        if details:
            for key, value in details.items():
                self.logger.info(f"- {key}: {value}")
        self.logger.info("")

        # Save to alerts log
        alert_file = os.path.join(
            self.alerts_dir,
            f"{datetime.now().strftime('%Y-%m-%d')}_alerts.log"
        )
        with open(alert_file, 'a') as f:
            f.write(f"[{timestamp}] {priority} | {alert_type}\n")
            f.write(f"{message}\n")
            if details:
                f.write(json.dumps(details, indent=2) + "\n")
            f.write("\n")

    def log_position_monitor(self, positions: List[Dict[str, Any]]):
        """Log current active positions status"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info(f"[{timestamp}] | POSITION MONITOR | ACTIVE")
        self.logger.info("")
        self.logger.info("Currently Open Positions:")

        total_exposure = 0
        for i, pos in enumerate(positions, 1):
            symbol = pos.get('symbol', 'UNKNOWN')
            direction = pos.get('direction', 'LONG')
            size = pos.get('size', 0)
            entry = pos.get('entry_price', 0)
            current = pos.get('current_price', 0)
            pnl = pos.get('unrealized_pnl', 0)
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            duration = pos.get('hold_duration', 'N/A')
            target = pos.get('target', 0)
            stop = pos.get('stop', 0)
            sentiment = pos.get('sentiment', 0)
            sentiment_label = self._get_sentiment_label(sentiment)

            self.logger.info(f"{i}. {symbol} {direction} {size:.2f} lots @ ${entry:.2f}")
            self.logger.info(f"     Current: ${current:.2f} | Unrealized P&L: €{pnl:.2f} ({pnl_pct:+.2f}%)")
            self.logger.info(f"     Duration: {duration} | Target: ${target:.2f} | Stop: ${stop:.2f}")
            self.logger.info(f"     Sentiment: {sentiment:.1f} {sentiment_label} (holding)")
            self.logger.info("")

            total_exposure += abs(size * entry)

        balance = positions[0].get('total_balance', 0) if positions else 0
        exposure_pct = (total_exposure / balance * 100) if balance > 0 else 0

        self.logger.info(f"Total Exposure: €{total_exposure:.2f} / €{balance:.2f} ({exposure_pct:.1f}%)")
        self.logger.info("")

    def log_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        net_pnl: float,
        starting_balance: float,
        ending_balance: float,
        peak_balance: float = None,
        max_drawdown_pct: float = 0,
        sharpe_ratio: float = 0,
        position_breakdown: Dict[str, int] = None,
        strategy_performance: Dict[str, Dict] = None
    ):
        """Generate and log daily summary report"""
        date = datetime.now().strftime("%Y-%m-%d")

        # Calculate metrics
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        pnl_pct = (net_pnl / starting_balance * 100) if starting_balance > 0 else 0
        avg_win = 0
        avg_loss = 0

        # Calculate from stored trades
        winning_trades = [t for t in self.daily_trades if t.result == "SUCCESS" and "pnl" in t.details and t.details.get('pnl', 0) > 0]
        losing_trades = [t for t in self.daily_trades if t.result == "SUCCESS" and "pnl" in t.details and t.details.get('pnl', 0) < 0]

        if winning_trades:
            avg_win = sum(t.details.get('pnl', 0) for t in winning_trades) / len(winning_trades)
        if losing_trades:
            avg_loss = sum(t.details.get('pnl', 0) for t in losing_trades) / len(losing_trades)

        # Sentiment stats
        sentiment_stats = {}
        if self.daily_stats['sentiment_readings']:
            sentiments = self.daily_stats['sentiment_readings']
            sentiment_stats = {
                'avg': sum(sentiments) / len(sentiments),
                'high': max(sentiments),
                'low': min(sentiments),
                'changes': len(sentiments),
                'signals': total_trades
            }

        # Log summary
        self.logger.info("=" * 80)
        self.logger.info(f"=== DAILY TRADING SUMMARY: {date} ===")
        self.logger.info("=" * 80)
        self.logger.info("")

        self.logger.info("📊 Performance Metrics:")
        self.logger.info(f"- Total Trades: {total_trades}")
        self.logger.info(f"- Wins: {wins} ({win_rate:.1f}%)")
        self.logger.info(f"- Losses: {losses} ({100-win_rate:.1f}%)")
        self.logger.info(f"- Net P&L: €{net_pnl:+.2f} ({pnl_pct:+.2f}%)")
        self.logger.info(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
        self.logger.info("")

        self.logger.info("💰 Position Summary:")
        self.logger.info(f"- Starting Balance: €{starting_balance:.2f}")
        self.logger.info(f"- Ending Balance: €{ending_balance:.2f}")
        if peak_balance:
            self.logger.info(f"- Peak Balance: €{peak_balance:.2f}")
        self.logger.info(f"- Max Drawdown: {max_drawdown_pct:.1f}%")
        self.logger.info("")

        if position_breakdown:
            self.logger.info("📈 Trade Breakdown:")
            for pos_type, count in position_breakdown.items():
                self.logger.info(f"- {pos_type}: {count}")
            self.logger.info(f"- Average Win: €{avg_win:+.2f}")
            self.logger.info(f"- Average Loss: €{avg_loss:.2f}")
            if avg_loss != 0:
                win_loss_ratio = abs(avg_win / avg_loss)
                self.logger.info(f"- Win/Loss Ratio: {win_loss_ratio:.2f}")
            self.logger.info("")

        if sentiment_stats:
            self.logger.info("📊 Sentiment Analysis:")
            self.logger.info(f"- Average Sentiment: {sentiment_stats['avg']:.1f}/10 ({self._get_sentiment_label(sentiment_stats['avg'])})")
            self.logger.info(f"- High: {sentiment_stats['high']:.1f} | Low: {sentiment_stats['low']:.1f}")
            self.logger.info(f"- Sentiment Changes: {sentiment_stats['changes']} updates")
            self.logger.info(f"- Trading Signals: {sentiment_stats['signals']} (100% executed)")
            self.logger.info("")

        risk_events = self.daily_stats['risk_events']
        if any(risk_events.values()):
            self.logger.info("⚠️ Risk Events:")
            self.logger.info(f"- Position Rejections: {risk_events['position_rejections']}")
            self.logger.info(f"- API Failures: {risk_events['api_failures']}")
            self.logger.info(f"- Circuit Breakers: {risk_events['circuit_breakers']}")
            self.logger.info("")

        if strategy_performance:
            self.logger.info("📊 Strategy Performance:")
            for strategy, stats in strategy_performance.items():
                trades = stats.get('trades', 0)
                wins = stats.get('wins', 0)
                win_pct = (wins / trades * 100) if trades > 0 else 0
                pnl = stats.get('pnl', 0)
                self.logger.info(f"- {strategy}:")
                self.logger.info(f"  Trades: {trades} | Wins: {wins} ({win_pct:.1f}%) | P&L: €{pnl:+.2f}")

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("")

        # Save summary to file
        self._save_daily_summary(date, {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'net_pnl': net_pnl,
            'pnl_percentage': pnl_pct,
            'starting_balance': starting_balance,
            'ending_balance': ending_balance,
            'peak_balance': peak_balance or ending_balance,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'position_breakdown': position_breakdown or {},
            'strategy_performance': strategy_performance or {},
            'sentiment_stats': sentiment_stats,
            'risk_events': risk_events
        })

        # Reset daily tracking
        self.daily_trades = []
        self.daily_stats = {
            'starting_balance': None,
            'peak_balance': None,
            'trades_by_strategy': {},
            'sentiment_readings': [],
            'risk_events': {
                'position_rejections': 0,
                'api_failures': 0,
                'circuit_breakers': 0
            }
        }

    def _save_trade_json(self, trade: TradeDecision):
        """Save trade decision to JSON file"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        json_file = os.path.join(self.trades_dir, f"{date_str}.json")

        # Load existing trades
        trades = []
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                trades = json.load(f)

        # Append new trade
        trades.append(asdict(trade))

        # Save back
        with open(json_file, 'w') as f:
            json.dump(trades, f, indent=2)

    def _save_daily_summary(self, date: str, summary: Dict[str, Any]):
        """Save daily summary to markdown file"""
        summary_file = os.path.join(self.summaries_dir, f"{date}_summary.md")

        with open(summary_file, 'w') as f:
            f.write(f"# Daily Trading Summary - {date}\n\n")
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Total Trades**: {summary['total_trades']}\n")
            f.write(f"- **Win Rate**: {summary['win_rate']:.1f}%\n")
            f.write(f"- **Net P&L**: €{summary['net_pnl']:+.2f} ({summary['pnl_percentage']:+.2f}%)\n")
            f.write(f"- **Sharpe Ratio**: {summary['sharpe_ratio']:.2f}\n\n")

            f.write("## Balance Summary\n\n")
            f.write(f"- **Starting**: €{summary['starting_balance']:.2f}\n")
            f.write(f"- **Ending**: €{summary['ending_balance']:.2f}\n")
            f.write(f"- **Peak**: €{summary['peak_balance']:.2f}\n")
            f.write(f"- **Max Drawdown**: {summary['max_drawdown_pct']:.1f}%\n\n")

            if summary.get('strategy_performance'):
                f.write("## Strategy Performance\n\n")
                for strategy, stats in summary['strategy_performance'].items():
                    f.write(f"### {strategy}\n")
                    f.write(f"- Trades: {stats.get('trades', 0)}\n")
                    f.write(f"- Win Rate: {stats.get('win_rate', 0):.1f}%\n")
                    f.write(f"- P&L: €{stats.get('pnl', 0):+.2f}\n\n")

            if summary.get('risk_events'):
                f.write("## Risk Events\n\n")
                for event, count in summary['risk_events'].items():
                    f.write(f"- {event}: {count}\n")


# Global logger instance
_global_logger: Optional[TradeDecisionLogger] = None


def get_trade_logger(base_dir: str = None) -> TradeDecisionLogger:
    """Get or create the global trade decision logger"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TradeDecisionLogger(base_dir)
    return _global_logger


if __name__ == "__main__":
    # Test the logger
    logger = TradeDecisionLogger()

    print("Testing TradeDecisionLogger...")

    # Test trade execution
    logger.log_trade_decision(
        symbol="NVDA",
        action="EXECUTE",
        sentiment=8.2,
        decision="OPEN_LONG",
        result="SUCCESS",
        reasoning="Strong positive sentiment (8.2/10) above buy threshold (7.0)",
        size=0.15,
        price=128.45,
        balance=1000,
        target=130.50,
        stop=127.20,
        position_id="ABC123"
    )

    # Test position closure
    logger.log_trade_decision(
        symbol="NVDA",
        action="CLOSE",
        sentiment=5.0,
        decision="CLOSE_LONG Position ABC123",
        result="SUCCESS",
        reasoning="Sentiment dropped to neutral (5.0/10), take profit",
        entry_price=128.45,
        exit_price=129.20,
        pnl=112.50,
        pnl_pct=0.58,
        hold_duration="1h 50m 18s"
    )

    # Test position rejection
    logger.log_position_rejection(
        symbol="NVDA",
        sentiment=8.5,
        reason="Already holding maximum position (0.15 lots)",
        current_exposure=150,
        max_exposure=1000
    )

    # Test daily summary
    logger.log_daily_summary(
        total_trades=47,
        wins=32,
        losses=15,
        net_pnl=127.50,
        starting_balance=1000.00,
        ending_balance=1127.50,
        peak_balance=1145.20,
        max_drawdown_pct=2.3,
        sharpe_ratio=1.82,
        position_breakdown={
            "Long Positions": 28,
            "Short Positions": 19
        },
        strategy_performance={
            "Momentum Strategy": {"trades": 18, "wins": 13, "win_rate": 72.2, "pnl": 89.40},
            "Mean Reversion": {"trades": 15, "wins": 9, "win_rate": 60.0, "pnl": 34.20}
        }
    )

    print("\n✅ Logger test complete! Check logs/ directory for output.")
