#!/usr/bin/env python3
"""
Supervisor Agent API Server
Receives events from Watcher Agent via webhook and processes them.
"""

import os
import sys
import asyncio
import json
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from flask import Flask, jsonify, request

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisor.supervisor_agent import SupervisorAgent, TriageResult

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SupervisorAPI')


# Flask application
app = Flask(__name__)

# Global state
event_queue: queue.Queue = queue.Queue()
processing_results: List[Dict[str, Any]] = []
supervisor: Optional[SupervisorAgent] = None
start_time: datetime = datetime.now()

# Configuration
CONFIG = {
  "api_host": os.getenv("SUPERVISOR_HOST", "0.0.0.0"),
  "api_port": int(os.getenv("SUPERVISOR_PORT", "5001")),
  "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
  "max_queue_size": 100,
  "max_results_history": 1000,
}


class EventProcessor(threading.Thread):
  """Background thread for processing events asynchronously."""

  def __init__(self, supervisor_agent: SupervisorAgent):
    super().__init__(daemon=True)
    self.supervisor = supervisor_agent
    self._running = True
    self._loop = None

  def run(self):
    """Main processing loop."""
    self._loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._loop)

    while self._running:
      try:
        # Get event from queue with timeout
        event = event_queue.get(timeout=1.0)

        # Process the event
        result = self._loop.run_until_complete(
          self._process_event(event)
        )

        # Store result
        processing_results.append(result)

        # Trim results history
        while len(processing_results) > CONFIG["max_results_history"]:
          processing_results.pop(0)

      except queue.Empty:
        continue
      except Exception as e:
        logger.error(f"Error processing event: {e}")

    self._loop.close()

  async def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single event."""
    logger.info(f"Processing event: {event.get('event_id', 'unknown')}")

    try:
      # Triage the event
      triage_result = await self.supervisor.triage_event(event)

      # Dispatch to expert if needed
      dispatch_result = await self.supervisor.dispatch_to_expert(triage_result)

      return {
        "event_id": event.get('event_id'),
        "status": "processed",
        "triage": asdict(triage_result),
        "dispatch": dispatch_result,
        "processed_at": datetime.now().isoformat()
      }

    except Exception as e:
      logger.error(f"Error in event processing: {e}")
      return {
        "event_id": event.get('event_id'),
        "status": "error",
        "error": str(e),
        "processed_at": datetime.now().isoformat()
      }

  def stop(self):
    """Stop the processor."""
    self._running = False


# Flask Routes

@app.route('/health', methods=['GET'])
def health_check():
  """Health check endpoint."""
  return jsonify({
    "status": "healthy",
    "timestamp": datetime.now().isoformat(),
    "uptime_seconds": (datetime.now() - start_time).total_seconds(),
    "queue_size": event_queue.qsize(),
    "results_count": len(processing_results),
  })


@app.route('/event', methods=['POST'])
def receive_event():
  """
  Receive event from Watcher Agent.

  Expected JSON body:
  {
    "event_id": "...",
    "event_type": "TRADE|ERROR|POSITION_DESYNC|...",
    "severity": "INFO|WARNING|CRITICAL",
    "timestamp": "...",
    "symbol": "NVDA",
    ...event-specific fields...
  }
  """
  try:
    event = request.json

    if not event:
      return jsonify({"status": "error", "message": "No JSON body"}), 400

    # Validate required fields
    required_fields = ['event_id', 'event_type']
    missing = [f for f in required_fields if f not in event]
    if missing:
      return jsonify({
        "status": "error",
        "message": f"Missing required fields: {missing}"
      }), 400

    # Check queue capacity
    if event_queue.qsize() >= CONFIG["max_queue_size"]:
      return jsonify({
        "status": "error",
        "message": "Queue full, try again later"
      }), 503

    # Add to processing queue
    event_queue.put(event)

    logger.info(f"Event queued: {event['event_id']} ({event['event_type']})")

    return jsonify({
      "status": "queued",
      "event_id": event['event_id'],
      "queue_position": event_queue.qsize(),
      "timestamp": datetime.now().isoformat()
    })

  except Exception as e:
    logger.error(f"Error receiving event: {e}")
    return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/event/<event_id>', methods=['GET'])
def get_event_result(event_id: str):
  """Get the processing result for a specific event."""
  for result in reversed(processing_results):
    if result.get('event_id') == event_id:
      return jsonify(result)

  # Check if still in queue
  return jsonify({
    "status": "not_found",
    "message": f"Event {event_id} not found in results"
  }), 404


@app.route('/results', methods=['GET'])
def get_all_results():
  """Get recent processing results."""
  limit = request.args.get('limit', 50, type=int)
  offset = request.args.get('offset', 0, type=int)

  results = processing_results[-(limit + offset):]
  if offset > 0:
    results = results[:-offset]

  return jsonify({
    "count": len(results),
    "total": len(processing_results),
    "results": results
  })


@app.route('/results/by-type/<event_type>', methods=['GET'])
def get_results_by_type(event_type: str):
  """Get processing results filtered by event type."""
  matching = [
    r for r in processing_results
    if r.get('triage', {}).get('category') == event_type or
       (r.get('triage') and 'event_type' in str(r.get('triage', {})) and
        event_type.upper() in str(r.get('triage', {})).upper())
  ]

  return jsonify({
    "count": len(matching),
    "event_type": event_type,
    "results": matching[-50:]  # Last 50
  })


@app.route('/results/critical', methods=['GET'])
def get_critical_results():
  """Get all CRITICAL severity results."""
  critical = [
    r for r in processing_results
    if r.get('triage', {}).get('severity') == 'CRITICAL'
  ]

  return jsonify({
    "count": len(critical),
    "results": critical
  })


@app.route('/stats', methods=['GET'])
def get_stats():
  """Get processing statistics."""
  # Count by status
  status_counts = {}
  category_counts = {}
  severity_counts = {}

  for r in processing_results:
    status = r.get('status', 'unknown')
    status_counts[status] = status_counts.get(status, 0) + 1

    triage = r.get('triage', {})
    category = triage.get('category', 'unknown')
    category_counts[category] = category_counts.get(category, 0) + 1

    severity = triage.get('severity', 'unknown')
    severity_counts[severity] = severity_counts.get(severity, 0) + 1

  return jsonify({
    "uptime_seconds": (datetime.now() - start_time).total_seconds(),
    "total_processed": len(processing_results),
    "queue_size": event_queue.qsize(),
    "by_status": status_counts,
    "by_category": category_counts,
    "by_severity": severity_counts,
  })


@app.route('/triage/history', methods=['GET'])
def get_triage_history():
  """Get triage history from the Supervisor Agent."""
  if supervisor:
    history = [asdict(t) for t in supervisor._triage_history[-100:]]
    return jsonify({
      "count": len(history),
      "history": history
    })

  return jsonify({"status": "error", "message": "Supervisor not initialized"}), 500


@app.route('/dispatch', methods=['POST'])
def manual_dispatch():
  """Manually dispatch an event to the Expert Agent."""
  data = request.json or {}

  # Create a manual event
  event = {
    "event_id": f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    "event_type": data.get('event_type', 'MANUAL_TRIGGER'),
    "severity": data.get('severity', 'WARNING'),
    "timestamp": datetime.now().isoformat(),
    "symbol": data.get('symbol', 'NVDA'),
    "source_file": data.get('source_file', 'manual'),
    "source_line": 0,
    "description": data.get('description', 'Manual dispatch request'),
    **data.get('context', {})
  }

  event_queue.put(event)

  return jsonify({
    "status": "queued",
    "event_id": event['event_id']
  })


def initialize_supervisor():
  """Initialize the Supervisor Agent."""
  global supervisor

  supervisor = SupervisorAgent(
    model=CONFIG["model"],
  )

  logger.info(f"Supervisor Agent initialized with model: {CONFIG['model']}")


def start_server():
  """Start the API server."""
  global processor, start_time

  start_time = datetime.now()

  # Initialize supervisor
  initialize_supervisor()

  # Start event processor
  processor = EventProcessor(supervisor)
  processor.start()

  logger.info(f"Starting API server on {CONFIG['api_host']}:{CONFIG['api_port']}")

  # Run Flask app
  app.run(
    host=CONFIG['api_host'],
    port=CONFIG['api_port'],
    debug=False,
    use_reloader=False
  )


def stop_server():
  """Stop the API server gracefully."""
  if processor:
    processor.stop()
    processor.join(timeout=5)

  # Close supervisor
  if supervisor:
    asyncio.run(supervisor.close())

  logger.info("Server stopped")


if __name__ == "__main__":
  try:
    start_server()
  except KeyboardInterrupt:
    logger.info("Shutting down...")
  finally:
    stop_server()
