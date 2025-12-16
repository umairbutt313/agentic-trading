#!/usr/bin/env python3
"""
Expert Agent - Claude Agent SDK Implementation
Performs deep analysis using Liang Wenfeng reasoning and generates patches.

Uses the official query() pattern from claude-agent-sdk for AI-powered analysis.
"""

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

from expert.reasoning_engine import LiangWenfengReasoner, ReasoningLoop
from expert.patch_generator import PatchGenerator, Patch, PatchStatus

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ExpertAgent')


@dataclass
class AnalysisResult:
  """Result of expert analysis."""
  issue_id: str
  issue_hash: str
  reasoning: ReasoningLoop
  patch: Optional[Patch]
  status: str  # analyzed, patch_generated, needs_human_review
  timestamp: str
  summary: str


class ExpertAgent:
  """
  Expert Trading Developer Agent.

  Performs systematic analysis using Liang Wenfeng 5-step reasoning
  and generates patches with FIX ATTEMPT documentation.

  Uses the official Claude Agent SDK query() pattern.
  """

  SYSTEM_PROMPT = """
You are an Expert Trading Developer Agent. Your task is to:
1. Analyze trading issues thoroughly
2. Read and understand code in 250-line chunks
3. Identify root causes using systematic reasoning
4. Generate code changes - CAN be architectural (no line limit)
5. Document all fixes with FIX ATTEMPT blocks

CRITICAL: All patches require HUMAN APPROVAL before application.
CRITICAL: Always check for previous FIX ATTEMPTS before proposing solutions.
CRITICAL: Always include ISSUE_HASH in your comments.
CRITICAL: You CAN propose large architectural changes - no line limit.
"""

  def __init__(
    self,
    model: str = "claude-sonnet-4-20250514",
    cwd: str = "/root/arslan-chart/agentic-trading-dec2025/stocks",
  ):
    """
    Initialize the Expert Agent.

    Args:
      model: Claude model to use
      cwd: Working directory for file operations

    NOTE: No line limit - agents can propose architectural changes.
          All patches require HUMAN APPROVAL before application.
    """
    self.model = model
    self.cwd = cwd

    self.reasoner = LiangWenfengReasoner(model=model, cwd=cwd)
    self.patch_generator = PatchGenerator(base_path=cwd)

    self._analysis_history: List[AnalysisResult] = []
    self._sdk_available = self._check_sdk_available()

  def _check_sdk_available(self) -> bool:
    """Check if Claude Agent SDK is available."""
    try:
      from claude_agent_sdk import query, ClaudeAgentOptions
      logger.info("Claude Agent SDK available - using real AI")
      return True
    except ImportError:
      logger.warning("Claude Agent SDK not installed - using mock responses")
      return False

  async def close(self):
    """Close all resources."""
    await self.reasoner.close()
    # No persistent client to close with query() pattern

  async def analyze_issue(
    self,
    triage_result: Dict[str, Any]
  ) -> AnalysisResult:
    """
    Perform complete analysis of an issue from Supervisor.

    Args:
      triage_result: Triage result from Supervisor Agent

    Returns:
      AnalysisResult with reasoning, patch, and status
    """
    issue_id = triage_result.get('event_id', 'unknown')
    issue_hash = triage_result.get('issue_hash', '')
    context_summary = triage_result.get('context_summary', '')
    previous_attempts = triage_result.get('previous_attempts', [])
    recommended_files = triage_result.get('recommended_files', [])

    logger.info(f"Starting analysis for issue: {issue_id}")

    # Build issue description
    issue_description = self._build_issue_description(triage_result)

    # Step 1: Perform Liang Wenfeng reasoning
    logger.info("Performing Liang Wenfeng 5-step reasoning...")
    reasoning = await self.reasoner.analyze(
      issue_id=issue_id,
      issue_description=issue_description,
      context=triage_result,
      previous_attempts=previous_attempts
    )

    # Step 2: Read relevant code
    logger.info("Reading relevant code files...")
    code_context = await self._read_relevant_code(recommended_files)

    # Step 3: Determine if patch is needed
    needs_patch = self._needs_code_fix(reasoning, triage_result)

    patch = None
    status = "analyzed"

    if needs_patch:
      # Step 4: Generate code changes
      logger.info("Generating code changes...")
      code_changes = await self._generate_code_changes(
        reasoning,
        triage_result,
        code_context
      )

      if code_changes:
        # Step 5: Create patch
        logger.info("Creating patch...")
        try:
          patch = self.patch_generator.generate_patch(
            issue_hash=issue_hash,
            issue_description=issue_description,
            reasoning=reasoning,
            code_changes=code_changes,
            previous_attempts=previous_attempts
          )
          status = "patch_generated"
          logger.info(f"Patch generated: {patch.patch_id}")
        except ValueError as e:
          logger.error(f"Failed to generate patch: {e}")
          status = "patch_generation_failed"

    # Create result
    result = AnalysisResult(
      issue_id=issue_id,
      issue_hash=issue_hash,
      reasoning=reasoning,
      patch=patch,
      status=status,
      timestamp=datetime.now().isoformat(),
      summary=self._generate_summary(reasoning, patch)
    )

    self._analysis_history.append(result)
    return result

  def _build_issue_description(self, triage_result: Dict[str, Any]) -> str:
    """Build a concise issue description from triage result."""
    category = triage_result.get('category', 'UNKNOWN')
    severity = triage_result.get('severity', 'INFO')
    context = triage_result.get('context_summary', '')

    # Extract key information
    parts = [f"[{severity}] {category}"]

    if context:
      # Get first sentence
      first_sentence = context.split('.')[0]
      parts.append(first_sentence[:100])

    return ': '.join(parts)

  async def _read_relevant_code(
    self,
    files: List[str]
  ) -> Dict[str, str]:
    """Read relevant code files directly (no SDK needed for file reading)."""
    code_context = {}

    for file_path in files[:5]:  # Limit to 5 files
      if not os.path.exists(file_path):
        continue

      try:
        # Read in 250-line chunks (Claude Code constraint)
        with open(file_path, 'r') as f:
          content = f.read()

        # Store first 500 lines for context
        lines = content.split('\n')[:500]
        code_context[file_path] = '\n'.join(lines)
        logger.info(f"Read {len(lines)} lines from {file_path}")

      except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")

    return code_context

  def _needs_code_fix(
    self,
    reasoning: ReasoningLoop,
    triage_result: Dict[str, Any]
  ) -> bool:
    """Determine if the issue requires a code fix."""
    # Categories that always need code fixes
    code_fix_categories = [
      'CODE_BUG',
      'CRITICAL_BUG',
    ]

    category = triage_result.get('category', '')
    if category in code_fix_categories:
      return True

    # Check reasoning recommendation
    if reasoning.recommended_action == 'CODE_FIX_REQUIRED':
      return True

    # Check if reflection step suggests code changes
    for step in reasoning.steps:
      if step.step_type.value == 'POST_TRADE_REFLECTION':
        if 'bug' in step.analysis.lower() or 'fix' in step.analysis.lower():
          return True

    return False

  async def _generate_code_changes(
    self,
    reasoning: ReasoningLoop,
    triage_result: Dict[str, Any],
    code_context: Dict[str, str]
  ) -> List[Dict[str, Any]]:
    """Generate specific code changes using Claude Agent SDK query()."""

    # Build prompt for code generation
    prompt = f"""
{self.SYSTEM_PROMPT}

Based on the following analysis, generate specific code changes:

ISSUE: {reasoning.issue_description}

REASONING CONCLUSION: {reasoning.conclusion}

RECOMMENDED ACTION: {reasoning.recommended_action}

RELEVANT CODE FILES:
{json.dumps(list(code_context.keys()), indent=2)}

Previous fix attempts to AVOID:
{json.dumps(triage_result.get('previous_attempts', []), indent=2)}

Please identify the specific code changes needed.
Format your response as JSON with this structure:
{{
  "changes": [
    {{
      "file_path": "path/to/file.py",
      "start_line": 123,
      "end_line": 125,
      "old_code": "the code to replace",
      "new_code": "the new code",
      "description": "why this change is needed"
    }}
  ]
}}

CONSTRAINTS:
- NO LINE LIMIT - you CAN propose architectural changes
- All patches require HUMAN APPROVAL before application
- Do not repeat previously failed solutions
- Be thorough - fix root causes, not just symptoms
"""

    if not self._sdk_available:
      # Return mock changes for testing
      return [{
        "file_path": "trading/position_manager.py",
        "start_line": 332,
        "end_line": 332,
        "old_code": "position.order_id = order_result.get('dealId')",
        "new_code": "position.order_id = order_result.get('dealReference')",
        "description": "[Mock] Fix dealId/dealReference mismatch"
      }]

    try:
      from claude_agent_sdk import query, ClaudeAgentOptions

      options = ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob"],
        cwd=self.cwd,
        permission_mode="acceptEdits"  # bypassPermissions not allowed as root
      )

      response_text = ""
      async for message in query(prompt=prompt, options=options):
        # Extract text from AssistantMessage (correct SDK structure)
        msg_type = type(message).__name__
        if msg_type == 'AssistantMessage' and hasattr(message, 'content'):
          for block in message.content:
            if hasattr(block, 'text'):
              response_text += block.text

      # Parse JSON from response
      return self._parse_code_changes(response_text)

    except Exception as e:
      logger.error(f"Error generating code changes: {e}")
      return []

  def _parse_code_changes(self, response_text: str) -> List[Dict[str, Any]]:
    """Parse code changes from Claude's response."""
    import re

    # Try to find JSON in response
    json_match = re.search(r'\{[\s\S]*"changes"[\s\S]*\}', response_text)
    if not json_match:
      logger.warning("No JSON found in response")
      return []

    try:
      data = json.loads(json_match.group())
      changes = data.get('changes', [])

      # Validate changes - NO LINE LIMIT (architectural changes allowed)
      # All patches require HUMAN APPROVAL before application
      valid_changes = []
      total_lines = 0

      for change in changes:
        if all(k in change for k in ['file_path', 'start_line', 'end_line', 'old_code', 'new_code']):
          lines = max(
            len(change['old_code'].split('\n')),
            len(change['new_code'].split('\n'))
          )
          valid_changes.append(change)
          total_lines += lines

      logger.info(f"Parsed {len(valid_changes)} code changes ({total_lines} total lines)")
      return valid_changes

    except json.JSONDecodeError as e:
      logger.error(f"Failed to parse JSON: {e}")
      return []

  def _generate_summary(
    self,
    reasoning: ReasoningLoop,
    patch: Optional[Patch]
  ) -> str:
    """Generate a human-readable summary of the analysis."""
    lines = []

    lines.append(f"Issue: {reasoning.issue_description}")
    lines.append("")

    lines.append("Reasoning Summary:")
    for step in reasoning.steps:
      summary = step.analysis[:100] + "..." if len(step.analysis) > 100 else step.analysis
      lines.append(f"  {step.step_number}. {step.title}: {summary}")

    lines.append("")
    lines.append(f"Conclusion: {reasoning.conclusion[:200]}")
    lines.append(f"Recommended Action: {reasoning.recommended_action}")

    if patch:
      lines.append("")
      lines.append(f"Patch Generated: {patch.patch_id}")
      lines.append(f"  Files: {[b.file_path for b in patch.blocks]}")
      lines.append(f"  Lines Changed: {patch.total_lines_changed}")
      lines.append(f"  Status: {patch.status.value}")

    return '\n'.join(lines)

  async def review_and_apply_patch(
    self,
    patch_id: str,
    approved: bool = False,
    rejection_reason: str = ""
  ) -> Dict[str, Any]:
    """
    Review and optionally apply a generated patch.

    Args:
      patch_id: ID of the patch to review
      approved: Whether the patch is approved
      rejection_reason: Reason for rejection (if not approved)

    Returns:
      Result of the operation
    """
    if approved:
      success = self.patch_generator.approve_patch(patch_id)
      if success:
        # Apply the patch
        patches = [p for p in self.patch_generator._patches if p.patch_id == patch_id]
        if patches:
          return self.patch_generator.apply_patch(patches[0], dry_run=False)
        return {"success": False, "error": "Patch not found after approval"}
      return {"success": False, "error": "Failed to approve patch"}
    else:
      success = self.patch_generator.reject_patch(patch_id, rejection_reason)
      return {
        "success": success,
        "status": "rejected",
        "reason": rejection_reason
      }

  def get_pending_patches(self) -> List[Dict[str, Any]]:
    """Get all patches pending human review."""
    patches = self.patch_generator.get_pending_patches()
    return [p.to_dict() for p in patches]

  def get_analysis_history(self, limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent analysis history."""
    results = []
    for r in self._analysis_history[-limit:]:
      results.append({
        "issue_id": r.issue_id,
        "issue_hash": r.issue_hash,
        "status": r.status,
        "timestamp": r.timestamp,
        "summary": r.summary[:500],
        "patch_id": r.patch.patch_id if r.patch else None,
      })
    return results


# Mock classes removed - SDK now handles fallback internally


async def main():
  """Test the Expert Agent."""
  agent = ExpertAgent()

  # Test triage result (simulating input from Supervisor)
  test_triage = {
    "event_id": "test_001",
    "issue_hash": "abc123def456",
    "severity": "CRITICAL",
    "category": "CRITICAL_BUG",
    "requires_expert": True,
    "previous_attempts": [],
    "context_summary": """
Position desync detected: System tracking 1 position, broker has 9.
This indicates the dealId/dealReference bug where positions are not
actually being closed on the broker side.
""",
    "recommended_files": [
      "/root/arslan-chart/agentic-trading-dec2025/stocks/trading/position_manager.py",
      "/root/arslan-chart/agentic-trading-dec2025/stocks/trading/capital_trader.py",
    ],
  }

  try:
    result = await agent.analyze_issue(test_triage)

    print("\n" + "=" * 60)
    print("ANALYSIS RESULT")
    print("=" * 60)
    print(f"Issue ID: {result.issue_id}")
    print(f"Status: {result.status}")
    print(f"\nSummary:\n{result.summary}")

    if result.patch:
      print(f"\n" + "=" * 60)
      print("GENERATED PATCH")
      print("=" * 60)
      print(f"Patch ID: {result.patch.patch_id}")
      print(f"Lines Changed: {result.patch.total_lines_changed}")
      print(f"\nFIX ATTEMPT Block:")
      print(result.patch.fix_attempt_block.format()[:1000])

  finally:
    await agent.close()


if __name__ == "__main__":
  asyncio.run(main())
