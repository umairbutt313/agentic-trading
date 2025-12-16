#!/usr/bin/env python3
"""
Supervisor Agent - Claude Agent SDK Implementation
Receives events from Watcher, triages them, and dispatches to Expert Agent.

Uses the official query() pattern from claude-agent-sdk for AI-powered triage.
"""

# CHANGELOG:
# [2025-12-15] SIMPLIFY: Removed complex pending patch tracking, added simple error list
# [2025-12-08] FIX: Handle None pnl in TRADE event hash computation (ISSUE_HASH: 8f3a2b1c)

# ==============================================================================
# FIX ATTEMPT [2025-12-08 20:45:00]
# ==============================================================================
# ISSUE: TypeError in _compute_hash when processing TRADE events
#        '<' not supported between instances of 'NoneType' and 'int'
# ISSUE_HASH: 8f3a2b1c
#
# ROOT CAUSE:
#   Line 217: pnl = event.get('pnl', 0)
#   Line 219: error_type = 'TradeIssue' if pnl < 0 else 'TradeSuccess'
#
#   dict.get('key', default) returns the default only when key is MISSING.
#   If key exists with value None, it returns None, not the default.
#   Then pnl < 0 raises TypeError because None < int is invalid.
#
# LIANG WENFENG REASONING:
#   1. Market Context: N/A (agent infrastructure bug, not trading)
#   2. Signal Interpretation: TRADE events from Watcher have pnl=None for
#      open positions (no exit yet), causing crash during triage
#   3. Alternative Evaluation: Could use (pnl or 0), but explicit None
#      check is clearer and handles edge cases like pnl=0 correctly
#   4. Risk Management: Self-referential bug - Supervisor can't triage
#      its own errors, causing silent failure of TRADE event processing
#   5. Reflection: Agent files were not in tracked_files, preventing
#      self-healing. Added to config.yaml for future auto-detection.
#
# SOLUTION: Explicit None check before comparison
#   pnl = event.get('pnl')
#   if pnl is None:
#       pnl = 0
#
# VALIDATION:
#   1. Send TRADE event with pnl=None -> should not crash
#   2. Send TRADE event with pnl=-0.50 -> should classify as TradeIssue
#   3. Send TRADE event with pnl=0.50 -> should classify as TradeSuccess
# ==============================================================================

import os
import sys
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supervisor.memory_tools import (
  lookup_fix_history_sync,
  compute_issue_hash_sync,
  ALLOWED_BASE_PATH,
)

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SupervisorAgent')


# System prompt for the Supervisor Agent
SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent in an Autonomous Trading Developer System.

Your role is to:
1. Receive events from the Watcher Agent (errors, trades, performance issues)
2. Triage events by severity and type
3. Look up previous FIX ATTEMPT blocks in the codebase
4. Compute issue hashes for deduplication
5. Route events to the Expert Agent with context

CRITICAL RULES:
- Always check for previous FIX ATTEMPTS before creating new tickets
- Never suggest solutions that have already been tried and failed
- Preserve all historical FIX ATTEMPT comments in code
- Use issue hashes to prevent duplicate work

When you receive an event, follow this workflow:
1. Parse the event to understand the issue
2. Compute an issue hash using compute_issue_hash tool
3. Search for previous fix attempts using lookup_fix_history tool
4. If duplicates exist, reference them in your analysis
5. Prepare a context summary for the Expert Agent

You have access to these tools:
- lookup_fix_history: Search code for FIX ATTEMPT blocks
- compute_issue_hash: Generate unique hash for deduplication
- read_trading_file: Read source files in 250-line chunks

Always be systematic and thorough in your analysis.
"""


@dataclass
class TriageResult:
  """Result of event triage."""
  event_id: str
  issue_hash: str
  severity: str
  category: str
  requires_expert: bool
  previous_attempts: List[Dict[str, Any]]
  context_summary: str
  recommended_files: List[str]
  timestamp: str


@dataclass
class PendingError:
  """Simple: An error that needs to be fixed."""
  issue_hash: str
  description: str
  severity: str
  source_file: str
  created_at: str


class SupervisorAgent:
  """
  Supervisor Agent - SIMPLE VERSION.

  - Receives errors from Watcher
  - Tracks pending errors (what still needs fixing)
  - Routes to Expert Agent
  - Tells you what's pending
  """

  def __init__(
    self,
    model: str = "claude-sonnet-4-20250514",
    cwd: str = ALLOWED_BASE_PATH,
    tracked_files: Optional[List[str]] = None
  ):
    """Initialize the Supervisor Agent."""
    self.model = model
    self.cwd = cwd
    self.tracked_files = tracked_files or [
      f"{ALLOWED_BASE_PATH}/trading/position_manager.py",
      f"{ALLOWED_BASE_PATH}/trading/scalping_strategies.py",
      f"{ALLOWED_BASE_PATH}/trading/enhanced_scalper.py",
      f"{ALLOWED_BASE_PATH}/trading/indicator_utils.py",
      f"{ALLOWED_BASE_PATH}/trading/capital_trader.py",
    ]
    self._triage_history: List[TriageResult] = []
    self._sdk_available = self._check_sdk_available()

    # Simple: Track errors that need fixing
    self._pending_errors: Dict[str, PendingError] = {}

  # ============================================================================
  # SIMPLE ERROR TRACKING
  # ============================================================================

  def add_pending_error(self, issue_hash: str, description: str, severity: str, source_file: str):
    """Add an error to pending list."""
    self._pending_errors[issue_hash] = PendingError(
      issue_hash=issue_hash,
      description=description,
      severity=severity,
      source_file=source_file,
      created_at=datetime.now().isoformat()
    )
    logger.info(f"📋 Added pending error: {description[:50]}...")

  def mark_error_fixed(self, issue_hash: str):
    """Mark an error as fixed."""
    if issue_hash in self._pending_errors:
      err = self._pending_errors.pop(issue_hash)
      logger.info(f"✅ Fixed: {err.description[:50]}...")

  def get_pending_errors(self) -> List[PendingError]:
    """Get all errors that need fixing."""
    return list(self._pending_errors.values())

  def show_pending_errors(self):
    """Print all pending errors to console."""
    errors = self.get_pending_errors()

    if not errors:
      logger.info("✅ No pending errors!")
      return

    logger.info("=" * 60)
    logger.info(f"📋 PENDING ERRORS TO FIX: {len(errors)}")
    logger.info("=" * 60)

    for i, err in enumerate(errors, 1):
      logger.info(f"{i}. [{err.severity}] {err.description[:60]}")
      logger.info(f"   File: {err.source_file}")
      logger.info(f"   Hash: {err.issue_hash}")

    logger.info("=" * 60)

  def _check_sdk_available(self) -> bool:
    """Check if Claude Agent SDK is available."""
    try:
      from claude_agent_sdk import query, ClaudeAgentOptions
      logger.info("Claude Agent SDK available - using real AI for triage")
      return True
    except ImportError:
      logger.warning("Claude Agent SDK not installed - using simple triage")
      return False

  async def close(self):
    """Close any resources (no persistent client in new SDK pattern)."""
    pass  # No persistent client to close with query() pattern

  async def triage_event(self, event: Dict[str, Any]) -> TriageResult:
    """
    Triage an incoming event from the Watcher.

    Args:
      event: Event data from Watcher Agent

    Returns:
      TriageResult with analysis and recommendations
    """
    event_type = event.get('event_type', 'UNKNOWN')
    event_id = event.get('event_id', 'unknown')
    severity = event.get('severity', 'INFO')

    logger.info(f"Triaging event: {event_id} (type: {event_type}, severity: {severity})")

    # Step 1: Compute issue hash
    issue_hash = self._compute_hash(event)

    # Step 2: Check for duplicate issues
    duplicates = self._check_duplicates(issue_hash)
    duplicate_count = len(duplicates)
    if duplicates:
      logger.info(f"Found {duplicate_count} existing issues with same hash")

    # ==============================================================================
    # FIX ATTEMPT [2025-12-11 20:20:00]
    # ==============================================================================
    # ISSUE: Same error dispatched 79+ times to Expert Agent with no resolution
    # ISSUE_HASH: recurring_fundamental_001
    # PREVIOUS ATTEMPTS: None - architectural gap since system start
    #
    # LIANG WENFENG REASONING:
    #   1. Market Context: Same issue hash appearing 79+ times indicates fundamental problem
    #   2. Signal Interpretation: Expert Agent cannot fix architectural issues via code patches
    #   3. Alternative: Stop dispatching after 50 occurrences, flag for human review (CHOSEN)
    #   4. Risk Management: Continuing to dispatch wastes tokens and creates noise
    #   5. Reflection: Some problems require human architectural decisions, not patches
    #
    # SOLUTION: Block Expert dispatch when duplicate_count > 50
    # ==============================================================================
    if duplicate_count > 50:
      logger.critical(f"🚨 RECURRING FUNDAMENTAL ISSUE: Hash {issue_hash} has {duplicate_count} occurrences")
      logger.critical(f"   STOPPING Expert dispatch - this requires HUMAN ARCHITECTURAL REVIEW")
      logger.critical(f"   Event type: {event_type}, Severity: {severity}")
      # Set requires_expert to False below by passing duplicate_count

    # Step 3: Look up previous fix attempts
    previous_attempts = self._lookup_previous_attempts(event)

    # Step 4: Determine category and if expert is needed
    category = self._categorize_event(event)
    requires_expert = self._requires_expert_analysis(event, previous_attempts, duplicate_count)

    # Step 5: Generate context summary using Claude
    context_summary = await self._generate_context_summary(event, previous_attempts)

    # Step 6: Identify relevant files
    recommended_files = self._identify_relevant_files(event)

    result = TriageResult(
      event_id=event_id,
      issue_hash=issue_hash,
      severity=severity,
      category=category,
      requires_expert=requires_expert,
      previous_attempts=previous_attempts,
      context_summary=context_summary,
      recommended_files=recommended_files,
      timestamp=datetime.now().isoformat()
    )

    self._triage_history.append(result)
    return result

  def _compute_hash(self, event: Dict[str, Any]) -> str:
    """Compute issue hash for deduplication."""
    event_type = event.get('event_type', 'UNKNOWN')
    symbol = event.get('symbol', 'UNKNOWN')

    # Build description from event
    if event_type == 'ERROR':
      description = event.get('error_message', '')
      error_type = event.get('error_type', 'UnknownError')
    elif event_type == 'TRADE':
      pnl = event.get('pnl')
      if pnl is None:
        pnl = 0
      description = f"Trade {event.get('direction', 'UNKNOWN')} PnL: {pnl}"
      error_type = 'TradeIssue' if pnl < 0 else 'TradeSuccess'
    elif event_type == 'POSITION_DESYNC':
      description = f"Position desync: local={event.get('local_position_count')} broker={event.get('broker_position_count')}"
      error_type = 'PositionDesync'
    else:
      description = str(event)[:100]
      error_type = event_type

    affected_file = event.get('source_file', 'unknown.py')

    return compute_issue_hash_sync(description, affected_file, error_type)

  def _check_duplicates(self, issue_hash: str) -> List[TriageResult]:
    """Check if we've already triaged this issue."""
    return [t for t in self._triage_history if t.issue_hash == issue_hash]

  def _lookup_previous_attempts(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Look up previous FIX ATTEMPT blocks related to this event."""
    # Build search pattern from event
    event_type = event.get('event_type', '')
    error_type = event.get('error_type', '')
    error_message = event.get('error_message', '')

    search_patterns = []
    if error_type:
      search_patterns.append(error_type)
    if error_message:
      # Extract key terms
      words = error_message.split()[:5]
      search_patterns.extend(words)

    # Special cases
    if event_type == 'POSITION_DESYNC':
      search_patterns.extend(['dealId', 'dealReference', 'phantom', 'position'])

    results = []
    for pattern in search_patterns:
      if len(pattern) < 3:
        continue

      result = lookup_fix_history_sync(pattern, self.tracked_files)
      text = result.get('content', [{}])[0].get('text', '')

      if 'Found' in text and 'previous fix attempts' in text:
        # Parse the results
        results.append({
          'pattern': pattern,
          'summary': text[:500]
        })

    return results

  def _categorize_event(self, event: Dict[str, Any]) -> str:
    """Categorize the event for routing."""
    event_type = event.get('event_type', '')

    categories = {
      'ERROR': 'CODE_BUG',
      'POSITION_DESYNC': 'CRITICAL_BUG',
      'CIRCUIT_BREAKER': 'RISK_MANAGEMENT',
      'PERFORMANCE_DROP': 'STRATEGY_ISSUE',
      'TRADE': 'TRADE_ANALYSIS',
      'SENTIMENT': 'DATA_QUALITY',
      'EXECUTION_LAG': 'INFRASTRUCTURE',
      'INDICATOR_ANOMALY': 'DATA_QUALITY',
    }

    return categories.get(event_type, 'UNKNOWN')

  def _requires_expert_analysis(
    self,
    event: Dict[str, Any],
    previous_attempts: List[Dict[str, Any]],
    duplicate_count: int = 0
  ) -> bool:
    """Determine if this event needs Expert Agent analysis."""
    severity = event.get('severity', 'INFO')

    # ==============================================================================
    # FIX ATTEMPT [2025-12-11 20:20:00]
    # ==============================================================================
    # ISSUE: Recurring errors (79+ occurrences) still being sent to Expert Agent
    # ISSUE_HASH: recurring_fundamental_001
    #
    # SOLUTION: Block Expert dispatch when same issue occurs > 50 times
    #           These are ARCHITECTURAL problems, not code bugs
    # ==============================================================================
    if duplicate_count > 50:
      logger.warning(f"⛔ Expert dispatch BLOCKED: {duplicate_count} duplicates exceeds threshold (50)")
      return False

    # CRITICAL always needs expert
    if severity == 'CRITICAL':
      return True

    # Position desync always needs expert (known critical bug)
    if event.get('event_type') == 'POSITION_DESYNC':
      return True

    # Errors need expert
    if event.get('event_type') == 'ERROR':
      return True

    # Performance drops need expert
    if event.get('event_type') == 'PERFORMANCE_DROP':
      return True

    # If no previous attempts, probably needs expert
    if not previous_attempts:
      return True

    # Losing trades need expert analysis
    if event.get('event_type') == 'TRADE':
      pnl = event.get('pnl')
      try:
        if pnl is not None and float(pnl) < 0:
          return True
      except (TypeError, ValueError):
        pass

    return False

  async def _generate_context_summary(
    self,
    event: Dict[str, Any],
    previous_attempts: List[Dict[str, Any]]
  ) -> str:
    """Generate a context summary for the Expert Agent using Claude Agent SDK."""

    if not self._sdk_available:
      return self._generate_simple_summary(event, previous_attempts)

    try:
      from claude_agent_sdk import query, ClaudeAgentOptions

      prompt = f"""
{SUPERVISOR_SYSTEM_PROMPT}

Summarize this event for analysis by the Expert Agent:

EVENT DATA:
{json.dumps(event, indent=2, default=str)}

PREVIOUS FIX ATTEMPTS:
{json.dumps(previous_attempts, indent=2) if previous_attempts else "None found"}

Provide a brief summary including:
1. What happened
2. What files are likely involved
3. Key constraints (don't repeat failed solutions)
4. Suggested analysis approach
"""

      options = ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob"],
        cwd=self.cwd,
        permission_mode="acceptEdits"  # bypassPermissions not allowed as root
      )

      summary = ""
      async for message in query(prompt=prompt, options=options):
        # Extract text from AssistantMessage (correct SDK structure)
        msg_type = type(message).__name__
        if msg_type == 'AssistantMessage' and hasattr(message, 'content'):
          for block in message.content:
            if hasattr(block, 'text'):
              summary += block.text

      return summary[:1000] if summary else self._generate_simple_summary(event, previous_attempts)

    except Exception as e:
      logger.error(f"Error generating context summary: {e}")
      # Fallback to simple summary
      return self._generate_simple_summary(event, previous_attempts)

  def _generate_simple_summary(
    self,
    event: Dict[str, Any],
    previous_attempts: List[Dict[str, Any]]
  ) -> str:
    """Generate a simple context summary without Claude."""
    event_type = event.get('event_type', 'UNKNOWN')
    severity = event.get('severity', 'INFO')

    summary = f"Event: {event_type} (Severity: {severity})\n"

    if event_type == 'ERROR':
      summary += f"Error: {event.get('error_message', 'Unknown')}\n"
      summary += f"File: {event.get('source_file', 'Unknown')}\n"

    elif event_type == 'TRADE':
      summary += f"Direction: {event.get('direction', 'UNKNOWN')}\n"
      summary += f"P&L: {event.get('pnl', 'N/A')}\n"
      summary += f"Strategy: {event.get('strategy', 'unknown')}\n"

    elif event_type == 'POSITION_DESYNC':
      summary += f"Local: {event.get('local_position_count', '?')} positions\n"
      summary += f"Broker: {event.get('broker_position_count', '?')} positions\n"
      summary += "CRITICAL: This may indicate the dealId/dealReference bug\n"

    if previous_attempts:
      summary += f"\nPrevious attempts: {len(previous_attempts)} found\n"
      summary += "Review these before proposing new solutions.\n"

    return summary

  def _identify_relevant_files(self, event: Dict[str, Any]) -> List[str]:
    """Identify files relevant to this event."""
    event_type = event.get('event_type', '')
    source_file = event.get('source_file', '')

    files = []

    # Add source file if specified
    if source_file and source_file != 'unknown':
      full_path = source_file
      if not source_file.startswith('/'):
        full_path = f"{ALLOWED_BASE_PATH}/trading/{source_file}"
      files.append(full_path)

    # Add relevant files based on event type
    type_files = {
      'POSITION_DESYNC': [
        'trading/position_manager.py',
        'trading/capital_trader.py',
      ],
      'TRADE': [
        'trading/scalping_strategies.py',
        'trading/enhanced_scalper.py',
      ],
      'ERROR': [
        'trading/position_manager.py',
        'trading/capital_trader.py',
      ],
      'PERFORMANCE_DROP': [
        'trading/scalping_strategies.py',
        'trading/indicator_utils.py',
      ],
      'INDICATOR_ANOMALY': [
        'trading/indicator_utils.py',
      ],
    }

    for rel_path in type_files.get(event_type, []):
      full_path = f"{ALLOWED_BASE_PATH}/{rel_path}"
      if full_path not in files:
        files.append(full_path)

    return files

  async def dispatch_to_expert(self, triage_result: TriageResult) -> Dict[str, Any]:
    """
    Dispatch a triaged event to the Expert Agent.

    Args:
      triage_result: Result of triage

    Returns:
      Response from Expert Agent
    """
    if not triage_result.requires_expert:
      return {
        "status": "skipped",
        "reason": "Does not require expert analysis",
        "triage_result": asdict(triage_result)
      }

    # Add to pending errors list
    self.add_pending_error(
      issue_hash=triage_result.issue_hash,
      description=triage_result.context_summary[:200],
      severity=triage_result.severity,
      source_file=triage_result.recommended_files[0] if triage_result.recommended_files else "unknown"
    )

    # Build dispatch payload
    payload = {
      "triage_result": asdict(triage_result),
      "instructions": [
        "Analyze root cause",
        "Check FIX ATTEMPT history - don't repeat failed solutions",
        "Build on previous fixes - move forward, not in circles",
        "Generate patch with FIX ATTEMPT block",
      ]
    }

    logger.info(f"Dispatching to Expert Agent: {triage_result.event_id}")

    # In full implementation, this would call the Expert Agent
    # For now, return the payload for later processing
    return {
      "status": "dispatched",
      "payload": payload,
      "timestamp": datetime.now().isoformat()
    }


# Mock classes removed - SDK now handles fallback internally via _sdk_available check


async def main():
  """Test the Supervisor Agent."""
  agent = SupervisorAgent()

  # Test event
  test_event = {
    "event_id": "test123",
    "event_type": "POSITION_DESYNC",
    "severity": "CRITICAL",
    "timestamp": datetime.now().isoformat(),
    "symbol": "NVDA",
    "source_file": "position_manager.py",
    "source_line": 332,
    "local_position_count": 1,
    "broker_position_count": 9,
  }

  try:
    result = await agent.triage_event(test_event)
    print(f"Triage Result:")
    print(f"  Issue Hash: {result.issue_hash}")
    print(f"  Category: {result.category}")
    print(f"  Requires Expert: {result.requires_expert}")
    print(f"  Context Summary: {result.context_summary[:200]}...")

    dispatch_result = await agent.dispatch_to_expert(result)
    print(f"\nDispatch Status: {dispatch_result['status']}")

  finally:
    await agent.close()


if __name__ == "__main__":
  asyncio.run(main())
