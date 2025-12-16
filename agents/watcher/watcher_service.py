#!/usr/bin/env python3
"""
Watcher Agent - Local VPS Service
Monitors trading logs and emits structured events via webhook to Supervisor Agent.

Usage:
  python watcher_service.py [--config config.yaml] [--port 5000]
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests
from flask import Flask, jsonify, request
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watcher.event_schemas import (
  BaseEvent, EventType, EventSeverity, ManualTriggerEvent
)
from watcher.log_parser import LogParser
from watcher.trigger_rules import TriggerRules, TriggerConfig


# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WatcherAgent')


# Default configuration
DEFAULT_CONFIG = {
  "logs_dir": "/root/arslan-chart/agentic-trading-dec2025/stocks/logs",
  "webhook_url": os.getenv("SUPERVISOR_WEBHOOK_URL", "http://localhost:5001/event"),
  "poll_interval": 1.0,
  "batch_size": 10,
  "batch_timeout": 5.0,
  "retry_attempts": 3,
  "retry_delay": 2.0,

  # Trigger thresholds
  "max_daily_loss_threshold": 50.0,
  "min_win_rate_threshold": 0.40,
  "win_rate_window": 10,
  "max_drawdown_threshold": 0.05,
  "latency_threshold_ms": 500,

  # API settings
  "api_port": 5000,
  "api_host": "0.0.0.0",
}


# Global event queue for batching
event_queue: queue.Queue = queue.Queue()


class WebhookEmitter:
  """Sends events to Supervisor Agent via webhook with retry logic."""

  def __init__(
    self,
    webhook_url: str,
    retry_attempts: int = 3,
    retry_delay: float = 2.0
  ):
    """
    Initialize webhook emitter.

    Args:
      webhook_url: URL to send events to
      retry_attempts: Number of retry attempts on failure
      retry_delay: Delay between retries in seconds
    """
    self.webhook_url = webhook_url
    self.retry_attempts = retry_attempts
    self.retry_delay = retry_delay
    self._failed_events: List[Dict] = []
    self._lock = threading.Lock()

  def emit(self, event: BaseEvent) -> bool:
    """
    Send event to webhook with retry logic.

    Args:
      event: Event to send

    Returns:
      True if sent successfully, False otherwise
    """
    # Serialize event to JSON (must be <1KB)
    if hasattr(event, 'to_dict'):
      event_data = event.to_dict()
    else:
      event_data = event.__dict__ if hasattr(event, '__dict__') else event

    event_json = json.dumps(event_data, default=str)

    # Validate size constraint (<1KB)
    if len(event_json.encode('utf-8')) > 1024:
      logger.warning(f"Event too large ({len(event_json)} bytes), truncating")
      event_data = self._truncate_event(event_data)
      event_json = json.dumps(event_data, default=str)

    for attempt in range(self.retry_attempts):
      try:
        response = requests.post(
          self.webhook_url,
          json=event_data,
          headers={"Content-Type": "application/json"},
          timeout=10
        )

        if response.status_code == 200:
          logger.info(f"Event {event_data.get('event_id', 'unknown')} sent successfully")
          return True
        else:
          logger.warning(
            f"Webhook returned {response.status_code}: {response.text[:100]}"
          )

      except requests.exceptions.RequestException as e:
        logger.error(f"Webhook attempt {attempt + 1} failed: {e}")

      if attempt < self.retry_attempts - 1:
        time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

    # Store failed event for later retry
    with self._lock:
      self._failed_events.append(event_data)
      if len(self._failed_events) > 100:
        self._failed_events = self._failed_events[-100:]

    return False

  def emit_batch(self, events: List[BaseEvent]) -> int:
    """
    Send multiple events in batch.

    Args:
      events: List of events to send

    Returns:
      Number of successfully sent events
    """
    success_count = 0
    for event in events:
      if self.emit(event):
        success_count += 1
    return success_count

  def retry_failed_events(self) -> int:
    """
    Retry sending failed events.

    Returns:
      Number of successfully retried events
    """
    with self._lock:
      to_retry = self._failed_events.copy()
      self._failed_events = []

    success_count = 0
    for event_data in to_retry:
      try:
        response = requests.post(
          self.webhook_url,
          json=event_data,
          headers={"Content-Type": "application/json"},
          timeout=10
        )
        if response.status_code == 200:
          success_count += 1
        else:
          with self._lock:
            self._failed_events.append(event_data)
      except requests.exceptions.RequestException:
        with self._lock:
          self._failed_events.append(event_data)

    return success_count

  def _truncate_event(self, event_data: Dict) -> Dict:
    """Truncate event data to fit within 1KB limit."""
    truncated = event_data.copy()

    # Truncate large string fields
    for key in ['stack_trace_summary', 'error_message', 'description']:
      if key in truncated and isinstance(truncated[key], str):
        if len(truncated[key]) > 100:
          truncated[key] = truncated[key][:97] + "..."

    # Truncate lists
    for key in ['phantom_position_ids', 'orphaned_position_ids', 'recent_trades']:
      if key in truncated and isinstance(truncated[key], list):
        if len(truncated[key]) > 5:
          truncated[key] = truncated[key][:5]

    return truncated

  @property
  def failed_count(self) -> int:
    """Get count of failed events awaiting retry."""
    return len(self._failed_events)


class TradingLogHandler(FileSystemEventHandler):
  """Watchdog handler for trading log files."""

  def __init__(
    self,
    trigger_rules: TriggerRules,
    emitter: WebhookEmitter,
    parser: LogParser
  ):
    """
    Initialize log handler.

    Args:
      trigger_rules: Rules for determining which events to emit
      emitter: Webhook emitter for sending events
      parser: Log parser for converting log lines to events
    """
    self.trigger_rules = trigger_rules
    self.emitter = emitter
    self.parser = parser
    self.file_positions: Dict[str, int] = {}
    self._lock = threading.Lock()

  # Files to exclude from monitoring (disabled features)
  EXCLUDED_FILES = [
    'image_sentiment_analyzer.log',  # Image sentiment feature disabled
    'reddit_scraper.log',            # Reddit feature disabled
  ]

  def on_modified(self, event):
    """Handle file modification events."""
    if event.is_directory:
      return

    # Only process log files
    if not (event.src_path.endswith('.log') or event.src_path.endswith('.json')):
      return

    # Skip excluded files (disabled features)
    for excluded in self.EXCLUDED_FILES:
      if excluded in event.src_path:
        return

    # Skip very frequent updates (debounce)
    self._process_new_lines(event.src_path)

  def _process_new_lines(self, file_path: str):
    """Process new lines from a modified log file."""
    with self._lock:
      last_pos = self.file_positions.get(file_path, 0)

      try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
          f.seek(last_pos)
          new_content = f.read()
          self.file_positions[file_path] = f.tell()

        if not new_content:
          return

        # Parse lines into events
        for line in new_content.strip().split('\n'):
          if not line.strip():
            continue

          events = self.parser.parse_to_events(line, file_path)

          for evt in events:
            if self.trigger_rules.should_emit(evt):
              # Add to queue for batch sending
              event_queue.put(evt)

      except FileNotFoundError:
        logger.error(f"Log file not found: {file_path}")
      except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

  def get_monitored_files(self) -> List[str]:
    """Get list of currently monitored files."""
    return list(self.file_positions.keys())


class EventBatcher(threading.Thread):
  """Background thread for batching and sending events."""

  def __init__(
    self,
    emitter: WebhookEmitter,
    batch_size: int = 10,
    batch_timeout: float = 5.0
  ):
    """
    Initialize event batcher.

    Args:
      emitter: Webhook emitter for sending events
      batch_size: Maximum batch size before sending
      batch_timeout: Maximum time to wait before sending incomplete batch
    """
    super().__init__(daemon=True)
    self.emitter = emitter
    self.batch_size = batch_size
    self.batch_timeout = batch_timeout
    self._running = True
    self._events_sent = 0

  def run(self):
    """Main loop for batch processing."""
    batch: List[BaseEvent] = []
    last_send_time = time.time()

    while self._running:
      try:
        # Try to get event with timeout
        event = event_queue.get(timeout=0.5)
        batch.append(event)

        # Send if batch is full
        if len(batch) >= self.batch_size:
          self._send_batch(batch)
          batch = []
          last_send_time = time.time()

      except queue.Empty:
        # Check if we should send incomplete batch due to timeout
        if batch and (time.time() - last_send_time) >= self.batch_timeout:
          self._send_batch(batch)
          batch = []
          last_send_time = time.time()

    # Send remaining events on shutdown
    if batch:
      self._send_batch(batch)

  def _send_batch(self, batch: List[BaseEvent]):
    """Send a batch of events."""
    success = self.emitter.emit_batch(batch)
    self._events_sent += success
    logger.info(f"Sent batch of {success}/{len(batch)} events")

  def stop(self):
    """Stop the batcher thread."""
    self._running = False

  @property
  def events_sent(self) -> int:
    """Get total number of events sent."""
    return self._events_sent


# Flask application for health checks and manual triggers
app = Flask(__name__)

# Global references for Flask routes
trigger_rules: Optional[TriggerRules] = None
emitter: Optional[WebhookEmitter] = None
log_handler: Optional[TradingLogHandler] = None
observer: Optional[Observer] = None
batcher: Optional[EventBatcher] = None
start_time: datetime = datetime.now()


@app.route('/health', methods=['GET'])
def health_check():
  """Health check endpoint."""
  return jsonify({
    "status": "healthy",
    "timestamp": datetime.now().isoformat(),
    "uptime_seconds": (datetime.now() - start_time).total_seconds(),
    "queue_size": event_queue.qsize(),
    "failed_events": emitter.failed_count if emitter else 0,
  })


@app.route('/stats', methods=['GET'])
def get_stats():
  """Get current monitoring statistics."""
  stats = trigger_rules.get_stats_summary() if trigger_rules else {}

  return jsonify({
    "trading_stats": stats,
    "monitored_files": log_handler.get_monitored_files() if log_handler else [],
    "events_sent": batcher.events_sent if batcher else 0,
    "queue_size": event_queue.qsize(),
    "failed_events_pending": emitter.failed_count if emitter else 0,
  })


@app.route('/trigger', methods=['POST'])
def manual_trigger():
  """Manually trigger analysis for a specific issue."""
  data = request.json or {}

  manual_event = ManualTriggerEvent(
    event_id=BaseEvent.generate_id(),
    event_type=EventType.MANUAL_TRIGGER,
    severity=EventSeverity.WARNING,
    timestamp=datetime.now(),
    symbol=data.get("symbol", "NVDA"),
    source_file="manual_trigger",
    source_line=0,
    description=data.get("description", "Manual analysis trigger"),
    context=data.get("context", {})
  )

  if emitter:
    success = emitter.emit(manual_event)
    status = "sent" if success else "queued"
  else:
    event_queue.put(manual_event)
    status = "queued"

  return jsonify({
    "status": status,
    "event_id": manual_event.event_id,
    "timestamp": manual_event.timestamp.isoformat()
  })


@app.route('/retry-failed', methods=['POST'])
def retry_failed():
  """Retry sending failed events."""
  if emitter:
    count = emitter.retry_failed_events()
    return jsonify({
      "status": "success",
      "retried_count": count,
      "remaining_failed": emitter.failed_count
    })

  return jsonify({"status": "error", "message": "Emitter not initialized"})


@app.route('/balance', methods=['POST'])
def update_balance():
  """Update account balance for drawdown calculations."""
  data = request.json or {}

  if trigger_rules:
    trigger_rules.update_balance(
      current_balance=data.get("current", 0),
      starting_balance=data.get("starting")
    )
    return jsonify({"status": "updated"})

  return jsonify({"status": "error", "message": "Trigger rules not initialized"})


def load_config(config_path: Optional[str] = None) -> Dict:
  """Load configuration from file or use defaults."""
  config = DEFAULT_CONFIG.copy()

  if config_path and os.path.exists(config_path):
    import yaml
    with open(config_path, 'r') as f:
      file_config = yaml.safe_load(f)
      if file_config:
        config.update(file_config)

  # Override with environment variables
  env_mappings = {
    'SUPERVISOR_WEBHOOK_URL': 'webhook_url',
    'LOGS_DIR': 'logs_dir',
    'WATCHER_PORT': 'api_port',
  }

  for env_key, config_key in env_mappings.items():
    if os.getenv(env_key):
      value = os.getenv(env_key)
      # Convert to int for port
      if config_key == 'api_port':
        value = int(value)
      config[config_key] = value

  return config


def start_watcher(config: Dict):
  """Start the log watcher service."""
  global trigger_rules, emitter, log_handler, observer, batcher, start_time

  start_time = datetime.now()

  # Initialize trigger rules
  trigger_config = TriggerConfig(
    max_daily_loss_threshold=config["max_daily_loss_threshold"],
    min_win_rate_threshold=config["min_win_rate_threshold"],
    win_rate_window=config["win_rate_window"],
    max_drawdown_threshold=config["max_drawdown_threshold"],
    latency_threshold_ms=config["latency_threshold_ms"],
  )
  trigger_rules = TriggerRules(trigger_config)

  # Initialize emitter
  emitter = WebhookEmitter(
    webhook_url=config["webhook_url"],
    retry_attempts=config["retry_attempts"],
    retry_delay=config["retry_delay"]
  )

  # Initialize parser
  parser = LogParser()

  # Initialize log handler
  log_handler = TradingLogHandler(trigger_rules, emitter, parser)

  # Initialize observer
  observer = Observer()
  logs_dir = config["logs_dir"]

  # Create logs directory if it doesn't exist
  os.makedirs(logs_dir, exist_ok=True)

  observer.schedule(log_handler, logs_dir, recursive=True)
  observer.start()

  # Start event batcher
  batcher = EventBatcher(
    emitter,
    batch_size=config["batch_size"],
    batch_timeout=config["batch_timeout"]
  )
  batcher.start()

  logger.info(f"Watcher started - monitoring {logs_dir}")
  logger.info(f"Webhook URL: {config['webhook_url']}")


def stop_watcher():
  """Stop the watcher service gracefully."""
  global observer, batcher

  if batcher:
    batcher.stop()
    batcher.join(timeout=5)

  if observer:
    observer.stop()
    observer.join(timeout=5)

  logger.info("Watcher stopped")


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(description="Trading Log Watcher Agent")
  parser.add_argument('--config', type=str, help="Path to config file (YAML)")
  parser.add_argument('--port', type=int, help="API port (default: 5000)")
  parser.add_argument('--host', type=str, default="0.0.0.0", help="API host")
  args = parser.parse_args()

  # Load configuration
  config = load_config(args.config)

  if args.port:
    config['api_port'] = args.port
  if args.host:
    config['api_host'] = args.host

  # Start watcher in background thread
  watcher_thread = threading.Thread(target=start_watcher, args=(config,), daemon=True)
  watcher_thread.start()

  # Give watcher time to initialize
  time.sleep(1)

  try:
    # Run Flask app
    logger.info(f"Starting API server on {config['api_host']}:{config['api_port']}")
    app.run(
      host=config['api_host'],
      port=config['api_port'],
      debug=False,
      use_reloader=False
    )
  except KeyboardInterrupt:
    logger.info("Shutting down...")
  finally:
    stop_watcher()


if __name__ == "__main__":
  main()
