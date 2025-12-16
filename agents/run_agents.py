#!/usr/bin/env python3
"""
Autonomous Trading Developer System - Main Entry Point
Launches all agents and coordinates the autonomous development loop.

Usage:
  # Start all agents
  python run_agents.py

  # Start specific agent
  python run_agents.py --agent watcher
  python run_agents.py --agent supervisor
  python run_agents.py --agent expert

  # Test mode (no actual file modifications)
  python run_agents.py --test-mode
"""

import os
import sys
import argparse
import asyncio
import logging
import signal
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging FIRST before any other imports (prevents other basicConfig calls)
LOG_FILE = '/root/arslan-chart/agentic-trading-dec2025/stocks/logs/agents.log'
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[
    logging.StreamHandler(),
    logging.FileHandler(LOG_FILE, mode='a')
  ],
  force=True  # Override any existing configuration
)
logger = logging.getLogger('AgentSystem')
logger.info(f"Logging initialized to {LOG_FILE}")

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watcher.watcher_service import start_watcher, stop_watcher, app as watcher_app
from supervisor.api_server import start_server as start_supervisor
from expert.expert_agent import ExpertAgent


class AgentSystem:
  """
  Coordinates all agents in the autonomous trading developer system.

  Components:
  - Watcher Agent: Monitors logs, emits events
  - Supervisor Agent: Triages events, routes to expert
  - Expert Agent: Analyzes issues, generates patches
  """

  def __init__(self, config_path: Optional[str] = None, test_mode: bool = False):
    """
    Initialize the agent system.

    Args:
      config_path: Path to config.yaml
      test_mode: If True, don't apply patches
    """
    self.config_path = config_path or os.path.join(
      os.path.dirname(__file__), 'config.yaml'
    )
    self.test_mode = test_mode

    self.watcher_thread: Optional[threading.Thread] = None
    self.supervisor_thread: Optional[threading.Thread] = None
    self.expert_agent: Optional[ExpertAgent] = None

    self._running = False
    self._shutdown_event = threading.Event()

  def start_all(self):
    """Start all agents."""
    logger.info("Starting Autonomous Trading Developer System...")

    self._running = True

    # Start Watcher Agent
    logger.info("Starting Watcher Agent...")
    self.watcher_thread = threading.Thread(
      target=self._run_watcher,
      daemon=True
    )
    self.watcher_thread.start()

    # Give watcher time to start
    time.sleep(2)

    # Start Supervisor Agent
    logger.info("Starting Supervisor Agent...")
    self.supervisor_thread = threading.Thread(
      target=self._run_supervisor,
      daemon=True
    )
    self.supervisor_thread.start()

    # Give supervisor time to start
    time.sleep(2)

    # Initialize Expert Agent
    logger.info("Initializing Expert Agent...")
    self.expert_agent = ExpertAgent()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Autonomous Trading Developer System RUNNING")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Endpoints:")
    logger.info("  Watcher:    http://localhost:5000/health")
    logger.info("  Supervisor: http://localhost:5001/health")
    logger.info("")
    logger.info("Manual trigger: POST http://localhost:5001/dispatch")
    logger.info("")
    logger.info(f"Test Mode: {'ENABLED' if self.test_mode else 'DISABLED'}")
    logger.info("")
    logger.info("Press Ctrl+C to stop...")

  def stop_all(self):
    """Stop all agents gracefully."""
    logger.info("Stopping Autonomous Trading Developer System...")

    self._running = False
    self._shutdown_event.set()

    # Stop watcher
    try:
      stop_watcher()
    except Exception as e:
      logger.error(f"Error stopping watcher: {e}")

    # Close expert agent
    if self.expert_agent:
      asyncio.run(self.expert_agent.close())

    logger.info("All agents stopped.")

  def _run_watcher(self):
    """Run the watcher agent."""
    from watcher.watcher_service import load_config, start_watcher, app

    config = load_config(self.config_path)

    # Start watcher components
    watcher_thread = threading.Thread(
      target=lambda: start_watcher(config),
      daemon=True
    )
    watcher_thread.start()

    # Run Flask app
    try:
      app.run(
        host=config['api_host'],
        port=config['api_port'],
        debug=False,
        use_reloader=False
      )
    except Exception as e:
      if self._running:
        logger.error(f"Watcher error: {e}")

  def _run_supervisor(self):
    """Run the supervisor agent."""
    try:
      start_supervisor()
    except Exception as e:
      if self._running:
        logger.error(f"Supervisor error: {e}")

  def wait_for_shutdown(self):
    """Wait for shutdown signal."""
    try:
      while not self._shutdown_event.is_set():
        self._shutdown_event.wait(timeout=1.0)
    except KeyboardInterrupt:
      pass

  async def process_manual_issue(self, issue_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manually process an issue through the system.

    Args:
      issue_data: Issue data to process

    Returns:
      Analysis result
    """
    if not self.expert_agent:
      return {"error": "Expert agent not initialized"}

    try:
      # Create a mock triage result
      triage_result = {
        "event_id": issue_data.get('event_id', f"manual_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
        "issue_hash": issue_data.get('issue_hash', 'manual_hash'),
        "severity": issue_data.get('severity', 'WARNING'),
        "category": issue_data.get('category', 'MANUAL_TRIGGER'),
        "requires_expert": True,
        "previous_attempts": issue_data.get('previous_attempts', []),
        "context_summary": issue_data.get('description', 'Manual analysis request'),
        "recommended_files": issue_data.get('files', []),
      }

      result = await self.expert_agent.analyze_issue(triage_result)

      return {
        "issue_id": result.issue_id,
        "status": result.status,
        "summary": result.summary,
        "patch_id": result.patch.patch_id if result.patch else None,
        "test_mode": self.test_mode,
      }

    except Exception as e:
      logger.error(f"Error processing manual issue: {e}")
      return {"error": str(e)}


def run_watcher_only():
  """Run only the Watcher Agent."""
  from watcher.watcher_service import main as watcher_main
  watcher_main()


def run_supervisor_only():
  """Run only the Supervisor Agent."""
  from supervisor.api_server import start_server
  start_server()


async def run_expert_only():
  """Run Expert Agent in interactive mode."""
  expert = ExpertAgent()

  print("\nExpert Agent Interactive Mode")
  print("=" * 40)
  print("Enter issue descriptions to analyze.")
  print("Type 'quit' to exit.\n")

  try:
    while True:
      issue_desc = input("Issue: ").strip()
      if issue_desc.lower() == 'quit':
        break

      if not issue_desc:
        continue

      # Create mock triage
      triage = {
        "event_id": f"interactive_{datetime.now().strftime('%H%M%S')}",
        "issue_hash": "interactive_hash",
        "severity": "WARNING",
        "category": "MANUAL_TRIGGER",
        "requires_expert": True,
        "previous_attempts": [],
        "context_summary": issue_desc,
        "recommended_files": [],
      }

      print("\nAnalyzing...")
      result = await expert.analyze_issue(triage)

      print("\n" + "=" * 40)
      print(f"Status: {result.status}")
      print(f"\nSummary:\n{result.summary}")
      print("=" * 40 + "\n")

  finally:
    await expert.close()


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Autonomous Trading Developer System"
  )
  parser.add_argument(
    '--agent',
    choices=['watcher', 'supervisor', 'expert', 'all'],
    default='all',
    help="Which agent(s) to start"
  )
  parser.add_argument(
    '--test-mode',
    action='store_true',
    help="Run in test mode (no file modifications)"
  )
  parser.add_argument(
    '--config',
    type=str,
    help="Path to config.yaml"
  )

  args = parser.parse_args()

  # Setup signal handlers
  def signal_handler(signum, frame):
    logger.info("Received shutdown signal")
    if 'system' in dir():
      system.stop_all()
    sys.exit(0)

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  # Create logs directory
  os.makedirs('/root/arslan-chart/agentic-trading-dec2025/stocks/logs', exist_ok=True)

  if args.agent == 'watcher':
    run_watcher_only()

  elif args.agent == 'supervisor':
    run_supervisor_only()

  elif args.agent == 'expert':
    asyncio.run(run_expert_only())

  else:
    # Run all agents
    system = AgentSystem(
      config_path=args.config,
      test_mode=args.test_mode
    )

    try:
      system.start_all()
      system.wait_for_shutdown()
    finally:
      system.stop_all()


if __name__ == "__main__":
  main()
