"""
Liang Wenfeng-Style Reasoning Engine for Expert Agent
Implements systematic 5-step reasoning loop for trading issue analysis.
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ReasoningStepType(Enum):
  """Types of reasoning steps in the Liang Wenfeng loop."""
  MARKET_CONTEXT = "MARKET_CONTEXT_ANALYSIS"
  SIGNAL_INTERPRETATION = "SIGNAL_INTERPRETATION"
  ALTERNATIVE_EVALUATION = "ALTERNATIVE_EVALUATION"
  RISK_MANAGEMENT = "RISK_MANAGEMENT_REASONING"
  POST_TRADE_REFLECTION = "POST_TRADE_REFLECTION"


@dataclass
class ReasoningStep:
  """A single step in the reasoning loop."""
  step_type: ReasoningStepType
  step_number: int
  title: str
  questions: List[str]
  analysis: str = ""
  findings: Dict[str, Any] = field(default_factory=dict)
  completed: bool = False
  timestamp: Optional[datetime] = None

  def complete(self, analysis: str, findings: Optional[Dict[str, Any]] = None):
    """Mark this step as complete with analysis."""
    self.analysis = analysis
    if findings:
      self.findings = findings
    self.completed = True
    self.timestamp = datetime.now()

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for serialization."""
    return {
      "step_type": self.step_type.value,
      "step_number": self.step_number,
      "title": self.title,
      "questions": self.questions,
      "analysis": self.analysis,
      "findings": self.findings,
      "completed": self.completed,
      "timestamp": self.timestamp.isoformat() if self.timestamp else None
    }


@dataclass
class ReasoningLoop:
  """
  Complete 5-step Liang Wenfeng reasoning loop.

  The loop ensures systematic analysis of trading issues:
  1. Market Context Analysis
  2. Signal Interpretation
  3. Alternative Evaluation
  4. Risk Management Reasoning
  5. Post-Trade Reflection
  """
  issue_id: str
  issue_description: str
  steps: List[ReasoningStep] = field(default_factory=list)
  conclusion: str = ""
  recommended_action: str = ""
  started_at: Optional[datetime] = None
  completed_at: Optional[datetime] = None

  def __post_init__(self):
    """Initialize the 5 reasoning steps."""
    if not self.steps:
      self.steps = self._create_steps()
    self.started_at = datetime.now()

  def _create_steps(self) -> List[ReasoningStep]:
    """Create the 5 reasoning steps with their questions."""
    return [
      ReasoningStep(
        step_type=ReasoningStepType.MARKET_CONTEXT,
        step_number=1,
        title="Market Context Analysis",
        questions=[
          "What is the current market regime? (trending/ranging based on ADX)",
          "What are the relevant indicator readings? (ATR, RSI, MACD)",
          "Is sentiment aligned with the technical setup?",
          "What is the current volatility environment?",
        ]
      ),
      ReasoningStep(
        step_type=ReasoningStepType.SIGNAL_INTERPRETATION,
        step_number=2,
        title="Signal Interpretation",
        questions=[
          "Why did this signal trigger?",
          "What was the confidence level and how was it calculated?",
          "Was the signal aligned with the dominant trend direction?",
          "What was the expected probability of success?",
        ]
      ),
      ReasoningStep(
        step_type=ReasoningStepType.ALTERNATIVE_EVALUATION,
        step_number=3,
        title="Alternative Evaluation",
        questions=[
          "What other strategies/signals were considered?",
          "Why were alternatives rejected?",
          "What would have been a better approach?",
          "What edge case was missed?",
        ]
      ),
      ReasoningStep(
        step_type=ReasoningStepType.RISK_MANAGEMENT,
        step_number=4,
        title="Risk Management Reasoning",
        questions=[
          "Was the stop loss appropriate for current ATR?",
          "Was position sizing correct for signal confidence?",
          "What was the expected risk:reward ratio?",
          "Did risk parameters match the strategy type?",
        ]
      ),
      ReasoningStep(
        step_type=ReasoningStepType.POST_TRADE_REFLECTION,
        step_number=5,
        title="Post-Trade Reflection",
        questions=[
          "What assumption failed?",
          "What can be improved?",
          "What parameter needs adjustment?",
          "What pattern should be added to recognition?",
        ]
      ),
    ]

  @property
  def is_complete(self) -> bool:
    """Check if all steps are completed."""
    return all(step.completed for step in self.steps)

  @property
  def current_step(self) -> Optional[ReasoningStep]:
    """Get the current incomplete step."""
    for step in self.steps:
      if not step.completed:
        return step
    return None

  def complete_step(
    self,
    step_number: int,
    analysis: str,
    findings: Optional[Dict[str, Any]] = None
  ):
    """Complete a specific step."""
    for step in self.steps:
      if step.step_number == step_number:
        step.complete(analysis, findings)
        break

  def finalize(self, conclusion: str, recommended_action: str):
    """Finalize the reasoning loop with conclusions."""
    self.conclusion = conclusion
    self.recommended_action = recommended_action
    self.completed_at = datetime.now()

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for serialization."""
    return {
      "issue_id": self.issue_id,
      "issue_description": self.issue_description,
      "steps": [s.to_dict() for s in self.steps],
      "conclusion": self.conclusion,
      "recommended_action": self.recommended_action,
      "is_complete": self.is_complete,
      "started_at": self.started_at.isoformat() if self.started_at else None,
      "completed_at": self.completed_at.isoformat() if self.completed_at else None,
    }

  def format_for_comment(self) -> str:
    """Format the reasoning loop for inclusion in FIX ATTEMPT comment."""
    lines = []
    lines.append("# LIANG WENFENG REASONING:")

    for step in self.steps:
      lines.append(f"#   {step.step_number}. {step.title}:")
      if step.analysis:
        # Indent and wrap analysis
        for line in step.analysis.split('\n'):
          lines.append(f"#      {line[:70]}")
      else:
        lines.append("#      (Not analyzed)")

    if self.conclusion:
      lines.append("#")
      lines.append("# CONCLUSION:")
      for line in self.conclusion.split('\n'):
        lines.append(f"#   {line[:70]}")

    return '\n'.join(lines)


class LiangWenfengReasoner:
  """
  Expert reasoner using Liang Wenfeng methodology.

  Provides systematic analysis through 5 steps with Claude Agent SDK integration.
  Uses the official query() pattern from claude-agent-sdk.
  """

  # System prompt for the Expert Agent
  SYSTEM_PROMPT = """
You are the Expert Trading Developer Agent using Liang Wenfeng-style systematic reasoning.

For every trading issue, you MUST complete this 5-step reasoning loop:

## Step 1: Market Context Analysis
- What is the current market regime? (trending/ranging based on ADX)
- What are the relevant indicator readings? (ATR, RSI, MACD)
- Is sentiment aligned with the technical setup?
- What is the current volatility environment?

## Step 2: Signal Interpretation
- Why did this signal trigger?
- What was the confidence level and how was it calculated?
- Was the signal aligned with the dominant trend direction?
- What was the expected probability of success?

## Step 3: Alternative Evaluation
- What other strategies/signals were considered?
- Why were alternatives rejected?
- What would have been a better approach?
- What edge case was missed?

## Step 4: Risk Management Reasoning
- Was the stop loss appropriate for current ATR?
- Was position sizing correct for signal confidence?
- What was the expected risk:reward ratio?
- Did risk parameters match the strategy type?

## Step 5: Post-Trade Reflection
- What assumption failed?
- What can be improved?
- What parameter needs adjustment?
- What pattern should be added to recognition?

After completing all 5 steps, generate a patch with:
1. Code changes (minimal, focused)
2. FIX ATTEMPT comment block with full reasoning
3. CHANGELOG entry
4. Validation test steps

CRITICAL CONSTRAINTS:
- Never repeat a fix that was previously attempted (check FIX ATTEMPT blocks)
- Never delete previous FIX ATTEMPT comments
- Always include ISSUE_HASH in comments for deduplication
- All fixes must preserve existing risk management rules
- Maximum 50 lines changed per patch
"""

  def __init__(self, model: str = "claude-sonnet-4-20250514", cwd: str = "/root/arslan-chart/agentic-trading-dec2025/stocks"):
    """
    Initialize the reasoner.

    Args:
      model: Claude model to use
      cwd: Working directory for file operations
    """
    self.model = model
    self.cwd = cwd
    self._reasoning_history: List[ReasoningLoop] = []
    self._sdk_available = self._check_sdk_available()

  def _check_sdk_available(self) -> bool:
    """Check if Claude Agent SDK is available."""
    try:
      from claude_agent_sdk import query, ClaudeAgentOptions
      return True
    except ImportError:
      return False

  async def close(self):
    """Close any resources (no persistent client in new SDK pattern)."""
    pass  # No persistent client to close with query() pattern

  async def analyze(
    self,
    issue_id: str,
    issue_description: str,
    context: Dict[str, Any],
    previous_attempts: List[Dict[str, Any]] = None
  ) -> ReasoningLoop:
    """
    Perform full Liang Wenfeng reasoning analysis using Claude Agent SDK.

    Args:
      issue_id: Unique identifier for the issue
      issue_description: Description of the issue to analyze
      context: Additional context (triage result, event data, etc.)
      previous_attempts: Previous FIX ATTEMPTS to avoid

    Returns:
      Completed ReasoningLoop with analysis
    """
    import logging
    logger = logging.getLogger('LiangWenfengReasoner')

    loop = ReasoningLoop(
      issue_id=issue_id,
      issue_description=issue_description
    )

    # Process each step
    for step in loop.steps:
      logger.info(f"Analyzing step {step.step_number}: {step.title}")
      analysis = await self._analyze_step(
        step,
        issue_description,
        context,
        previous_attempts
      )

      # Extract findings from analysis
      findings = self._extract_findings(analysis, step.step_type)
      loop.complete_step(step.step_number, analysis, findings)

    # Generate conclusion
    conclusion = await self._generate_conclusion(loop, context)
    recommended_action = self._determine_action(loop, context)

    loop.finalize(conclusion, recommended_action)

    self._reasoning_history.append(loop)
    return loop

  async def _analyze_step(
    self,
    step: ReasoningStep,
    issue_description: str,
    context: Dict[str, Any],
    previous_attempts: List[Dict[str, Any]] = None
  ) -> str:
    """Analyze a single reasoning step using Claude Agent SDK query()."""
    # Build prompt for this step
    prompt = f"""
{self.SYSTEM_PROMPT}

## {step.title} (Step {step.step_number}/5)

Issue: {issue_description}

Context:
{self._format_context(context)}

{f"Previous FIX ATTEMPTS (DO NOT REPEAT):{chr(10)}{self._format_previous_attempts(previous_attempts)}" if previous_attempts else ""}

Please analyze this step by answering these questions:
{chr(10).join(f"- {q}" for q in step.questions)}

Provide a concise but thorough analysis.
"""

    if not self._sdk_available:
      return f"[Mock Analysis] Step {step.step_number}: {step.title} - Analysis placeholder"

    try:
      from claude_agent_sdk import query, ClaudeAgentOptions

      options = ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob"],
        cwd=self.cwd,
        permission_mode="acceptEdits"  # bypassPermissions not allowed as root
      )

      analysis = ""
      async for message in query(prompt=prompt, options=options):
        # Extract text from AssistantMessage (correct SDK structure)
        msg_type = type(message).__name__
        if msg_type == 'AssistantMessage' and hasattr(message, 'content'):
          for block in message.content:
            if hasattr(block, 'text'):
              analysis += block.text

      return analysis[:2000] if analysis else f"Step {step.step_number} analyzed"

    except Exception as e:
      return f"Analysis error: {str(e)}"

  async def _generate_conclusion(
    self,
    loop: ReasoningLoop,
    context: Dict[str, Any]
  ) -> str:
    """Generate conclusion from completed reasoning loop using Claude Agent SDK."""
    prompt = f"""
Based on the 5-step Liang Wenfeng analysis for issue: {loop.issue_description}

Step summaries:
{chr(10).join(f"{s.step_number}. {s.title}: {s.analysis[:200]}..." for s in loop.steps)}

Please provide a concise conclusion:
1. What was the root cause?
2. What needs to be fixed?
3. What are the key constraints?
"""

    if not self._sdk_available:
      return f"[Mock Conclusion] Root cause identified. Code fix required."

    try:
      from claude_agent_sdk import query, ClaudeAgentOptions

      options = ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob"],
        cwd=self.cwd,
        permission_mode="acceptEdits"  # bypassPermissions not allowed as root
      )

      conclusion = ""
      async for message in query(prompt=prompt, options=options):
        # Extract text from AssistantMessage (correct SDK structure)
        msg_type = type(message).__name__
        if msg_type == 'AssistantMessage' and hasattr(message, 'content'):
          for block in message.content:
            if hasattr(block, 'text'):
              conclusion += block.text

      return conclusion[:1000] if conclusion else "Conclusion generated"

    except Exception as e:
      return f"Could not generate conclusion: {str(e)}"

  def _determine_action(
    self,
    loop: ReasoningLoop,
    context: Dict[str, Any]
  ) -> str:
    """Determine recommended action from analysis."""
    # Simple heuristics for common issues
    if 'dealId' in loop.issue_description or 'position' in loop.issue_description.lower():
      return "CODE_FIX_REQUIRED"

    if 'performance' in loop.issue_description.lower():
      return "PARAMETER_ADJUSTMENT"

    if 'error' in loop.issue_description.lower():
      return "CODE_FIX_REQUIRED"

    # Check analysis for keywords
    all_analysis = ' '.join(s.analysis.lower() for s in loop.steps)
    if 'bug' in all_analysis or 'fix' in all_analysis:
      return "CODE_FIX_REQUIRED"

    # Default
    return "FURTHER_INVESTIGATION"

  def _format_context(self, context: Dict[str, Any]) -> str:
    """Format context for prompt."""
    import json
    return json.dumps(context, indent=2, default=str)[:1500]

  def _format_previous_attempts(self, attempts: List[Dict[str, Any]]) -> str:
    """Format previous attempts for prompt."""
    if not attempts:
      return "None"

    lines = []
    for i, attempt in enumerate(attempts[:5], 1):
      lines.append(f"{i}. {attempt.get('summary', str(attempt)[:200])}")
    return '\n'.join(lines)

  def _extract_findings(
    self,
    analysis: str,
    step_type: ReasoningStepType
  ) -> Dict[str, Any]:
    """Extract structured findings from analysis text."""
    findings = {
      "raw_analysis": analysis,
      "key_points": [],
    }

    # Extract key points (lines starting with - or *)
    import re
    points = re.findall(r'[-*]\s*(.+)', analysis)
    findings["key_points"] = points[:10]

    # Step-specific extraction
    if step_type == ReasoningStepType.MARKET_CONTEXT:
      # Look for regime indication
      if 'trending' in analysis.lower():
        findings["regime"] = "trending"
      elif 'ranging' in analysis.lower():
        findings["regime"] = "ranging"

    elif step_type == ReasoningStepType.RISK_MANAGEMENT:
      # Look for risk indicators
      if 'too tight' in analysis.lower() or 'too close' in analysis.lower():
        findings["stop_loss_issue"] = "too_tight"
      elif 'too wide' in analysis.lower():
        findings["stop_loss_issue"] = "too_wide"

    elif step_type == ReasoningStepType.POST_TRADE_REFLECTION:
      # Look for specific recommendations
      if 'parameter' in analysis.lower():
        findings["recommendation_type"] = "parameter_adjustment"
      elif 'fix' in analysis.lower() or 'bug' in analysis.lower():
        findings["recommendation_type"] = "code_fix"

    return findings


# Mock classes removed - SDK now handles fallback internally via _sdk_available check
