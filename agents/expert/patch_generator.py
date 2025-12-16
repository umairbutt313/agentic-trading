"""
Patch Generator for Expert Agent
Generates code patches with FIX ATTEMPT blocks and CHANGELOG entries.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

from .reasoning_engine import ReasoningLoop


class PatchStatus(Enum):
  """Status of a generated patch."""
  PENDING_REVIEW = "PENDING_REVIEW"
  APPROVED = "APPROVED"
  REJECTED = "REJECTED"
  APPLIED = "APPLIED"


@dataclass
class PatchBlock:
  """
  Represents a code change in a patch.

  Contains the old code to replace and the new code to insert.
  """
  file_path: str
  start_line: int
  end_line: int
  old_code: str
  new_code: str
  description: str

  def validate(self) -> bool:
    """Validate the patch block."""
    if not self.file_path:
      return False
    if self.start_line < 1:
      return False
    if self.end_line < self.start_line:
      return False
    if self.old_code == self.new_code:
      return False
    return True

  @property
  def lines_changed(self) -> int:
    """Calculate number of lines changed."""
    old_lines = len(self.old_code.split('\n'))
    new_lines = len(self.new_code.split('\n'))
    return max(old_lines, new_lines)

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""
    return {
      "file_path": self.file_path,
      "start_line": self.start_line,
      "end_line": self.end_line,
      "old_code": self.old_code,
      "new_code": self.new_code,
      "description": self.description,
      "lines_changed": self.lines_changed,
    }


@dataclass
class FixAttemptBlock:
  """
  FIX ATTEMPT comment block to be added to code.

  Contains full documentation of the fix for future reference.
  """
  timestamp: datetime
  issue_description: str
  issue_hash: str
  previous_attempts: List[Dict[str, str]] = field(default_factory=list)
  reasoning: Optional[ReasoningLoop] = None
  solution_description: str = ""
  validation_steps: List[str] = field(default_factory=list)

  def format(self) -> str:
    """Format the FIX ATTEMPT block as a comment."""
    lines = []

    # Header
    lines.append("# " + "=" * 78)
    lines.append(f"# FIX ATTEMPT [{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]")
    lines.append("# " + "=" * 78)

    # Issue description
    lines.append(f"# ISSUE: {self.issue_description}")
    lines.append(f"# ISSUE_HASH: {self.issue_hash}")
    lines.append("#")

    # Previous attempts
    if self.previous_attempts:
      lines.append("# PREVIOUS ATTEMPTS:")
      for attempt in self.previous_attempts:
        date = attempt.get('date', 'unknown')
        desc = attempt.get('description', 'unknown')[:50]
        outcome = attempt.get('outcome', 'UNKNOWN')
        lines.append(f"#   - [{date}] {desc} - {outcome}")
      lines.append("#")

    # Reasoning (if provided)
    if self.reasoning:
      lines.append(self.reasoning.format_for_comment())
      lines.append("#")

    # Solution
    lines.append("# SOLUTION:")
    for line in self.solution_description.split('\n'):
      lines.append(f"#   {line[:70]}")
    lines.append("#")

    # Validation steps
    if self.validation_steps:
      lines.append("# VALIDATION:")
      for i, step in enumerate(self.validation_steps, 1):
        lines.append(f"#   {i}. {step}")
    else:
      lines.append("# VALIDATION: See test steps in patch description")

    # Footer
    lines.append("# " + "=" * 78)

    return '\n'.join(lines)


@dataclass
class Patch:
  """
  Complete patch with all changes and documentation.
  """
  patch_id: str
  issue_hash: str
  title: str
  description: str
  blocks: List[PatchBlock] = field(default_factory=list)
  fix_attempt_block: Optional[FixAttemptBlock] = None
  changelog_entry: str = ""
  status: PatchStatus = PatchStatus.PENDING_REVIEW
  created_at: datetime = field(default_factory=datetime.now)
  reviewed_at: Optional[datetime] = None
  applied_at: Optional[datetime] = None

  @property
  def total_lines_changed(self) -> int:
    """Calculate total lines changed across all blocks."""
    return sum(b.lines_changed for b in self.blocks)

  def validate(self) -> List[str]:
    """
    Validate the patch.

    Returns:
      List of validation errors (empty if valid)
    """
    errors = []

    if not self.patch_id:
      errors.append("Patch ID is required")

    if not self.issue_hash:
      errors.append("Issue hash is required")

    if not self.blocks:
      errors.append("At least one patch block is required")

    for i, block in enumerate(self.blocks):
      if not block.validate():
        errors.append(f"Invalid patch block {i + 1}")

    # NO LINE LIMIT - Agents can propose architectural changes of any size
    # Human approval is ALWAYS required before applying patches

    if not self.fix_attempt_block:
      errors.append("FIX ATTEMPT block is required")

    return errors

  def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary."""
    return {
      "patch_id": self.patch_id,
      "issue_hash": self.issue_hash,
      "title": self.title,
      "description": self.description,
      "blocks": [b.to_dict() for b in self.blocks],
      "fix_attempt_block": self.fix_attempt_block.format() if self.fix_attempt_block else None,
      "changelog_entry": self.changelog_entry,
      "status": self.status.value,
      "total_lines_changed": self.total_lines_changed,
      "created_at": self.created_at.isoformat(),
      "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
      "applied_at": self.applied_at.isoformat() if self.applied_at else None,
    }


class PatchGenerator:
  """
  Generates patches from Expert Agent analysis.

  Features:
  - Creates minimal, focused code changes
  - Generates FIX ATTEMPT comment blocks
  - Creates CHANGELOG entries
  - NO line limit - agents can propose architectural changes
  - ALL patches require HUMAN APPROVAL before application
  """

  # NO LINE LIMIT - Removed to allow architectural changes
  # Human approval is ALWAYS required before applying any patch

  def __init__(self, base_path: str = "/root/arslan-chart/agentic-trading-dec2025/stocks"):
    """
    Initialize patch generator.

    Args:
      base_path: Base path for code files
    """
    self.base_path = base_path
    self._patches: List[Patch] = []

  def generate_patch(
    self,
    issue_hash: str,
    issue_description: str,
    reasoning: ReasoningLoop,
    code_changes: List[Dict[str, Any]],
    previous_attempts: List[Dict[str, str]] = None,
  ) -> Patch:
    """
    Generate a complete patch from analysis results.

    Args:
      issue_hash: Unique hash for the issue
      issue_description: Description of the issue
      reasoning: Completed reasoning loop
      code_changes: List of code changes to make
      previous_attempts: Previous fix attempts for reference

    Returns:
      Generated Patch object
    """
    patch_id = f"PATCH_{issue_hash}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Create patch blocks from code changes
    blocks = []
    for change in code_changes:
      block = PatchBlock(
        file_path=change.get('file_path', ''),
        start_line=change.get('start_line', 0),
        end_line=change.get('end_line', 0),
        old_code=change.get('old_code', ''),
        new_code=change.get('new_code', ''),
        description=change.get('description', ''),
      )
      blocks.append(block)

    # Create FIX ATTEMPT block
    fix_attempt = FixAttemptBlock(
      timestamp=datetime.now(),
      issue_description=issue_description,
      issue_hash=issue_hash,
      previous_attempts=previous_attempts or [],
      reasoning=reasoning,
      solution_description=reasoning.conclusion if reasoning else "See patch description",
      validation_steps=self._generate_validation_steps(reasoning, code_changes),
    )

    # Generate CHANGELOG entry
    changelog = self._generate_changelog_entry(issue_hash, issue_description, reasoning)

    # Create patch
    patch = Patch(
      patch_id=patch_id,
      issue_hash=issue_hash,
      title=f"Fix: {issue_description[:50]}",
      description=reasoning.conclusion if reasoning else issue_description,
      blocks=blocks,
      fix_attempt_block=fix_attempt,
      changelog_entry=changelog,
    )

    # Validate
    errors = patch.validate()
    if errors:
      raise ValueError(f"Invalid patch: {', '.join(errors)}")

    self._patches.append(patch)
    return patch

  def _generate_validation_steps(
    self,
    reasoning: Optional[ReasoningLoop],
    code_changes: List[Dict[str, Any]]
  ) -> List[str]:
    """Generate validation steps for the patch."""
    steps = [
      "Run existing test suite to ensure no regressions",
    ]

    # Add file-specific validation
    files = set(c.get('file_path', '') for c in code_changes)
    for file_path in files:
      if 'position_manager' in file_path:
        steps.extend([
          "Test position open/close lifecycle",
          "Verify order_id is not None after position open",
          "Confirm positions close successfully on broker",
        ])
      elif 'scalping_strategies' in file_path:
        steps.extend([
          "Run strategy backtests",
          "Verify signal generation with test data",
        ])
      elif 'indicator_utils' in file_path:
        steps.extend([
          "Test indicator calculations with known values",
          "Verify indicator ranges are within expected bounds",
        ])

    # Add reasoning-based validation
    if reasoning and reasoning.recommended_action:
      if reasoning.recommended_action == "CODE_FIX_REQUIRED":
        steps.append("Verify the specific bug is fixed")
      elif reasoning.recommended_action == "PARAMETER_ADJUSTMENT":
        steps.append("Test with new parameters in demo mode")

    steps.append("Monitor for 10 trades before production deployment")

    return steps

  def _generate_changelog_entry(
    self,
    issue_hash: str,
    issue_description: str,
    reasoning: Optional[ReasoningLoop]
  ) -> str:
    """Generate a CHANGELOG entry for the patch."""
    date = datetime.now().strftime('%Y-%m-%d')

    # Determine entry type
    entry_type = "FIX"
    if reasoning:
      if reasoning.recommended_action == "PARAMETER_ADJUSTMENT":
        entry_type = "TUNE"

    # Create summary
    summary = issue_description[:60]
    if len(issue_description) > 60:
      summary += "..."

    return f"# [{date}] {entry_type}: {summary} (ISSUE_HASH: {issue_hash})"

  def apply_patch(self, patch: Patch, dry_run: bool = True) -> Dict[str, Any]:
    """
    Apply a patch to the codebase.

    Args:
      patch: Patch to apply
      dry_run: If True, don't actually modify files

    Returns:
      Result of application
    """
    if patch.status != PatchStatus.APPROVED and not dry_run:
      return {
        "success": False,
        "error": "Patch must be APPROVED before application"
      }

    results = {
      "success": True,
      "files_modified": [],
      "errors": [],
      "dry_run": dry_run,
    }

    for block in patch.blocks:
      try:
        result = self._apply_block(block, patch.fix_attempt_block, dry_run)
        results["files_modified"].append({
          "file": block.file_path,
          "lines": f"{block.start_line}-{block.end_line}",
          "success": result["success"],
        })
        if not result["success"]:
          results["errors"].append(result.get("error", "Unknown error"))
          results["success"] = False

      except Exception as e:
        results["errors"].append(f"{block.file_path}: {str(e)}")
        results["success"] = False

    if results["success"] and not dry_run:
      patch.status = PatchStatus.APPLIED
      patch.applied_at = datetime.now()

    return results

  def _apply_block(
    self,
    block: PatchBlock,
    fix_attempt: FixAttemptBlock,
    dry_run: bool
  ) -> Dict[str, Any]:
    """Apply a single patch block."""
    file_path = block.file_path
    if not file_path.startswith('/'):
      file_path = os.path.join(self.base_path, file_path)

    if not os.path.exists(file_path):
      return {"success": False, "error": f"File not found: {file_path}"}

    with open(file_path, 'r') as f:
      lines = f.readlines()

    # Verify old code matches
    old_lines = ''.join(lines[block.start_line - 1:block.end_line])
    if old_lines.strip() != block.old_code.strip():
      return {
        "success": False,
        "error": f"Old code doesn't match at lines {block.start_line}-{block.end_line}"
      }

    # Build new content
    new_lines = lines[:block.start_line - 1]

    # Add FIX ATTEMPT block before the change
    new_lines.append(fix_attempt.format() + '\n')

    # Add new code
    for line in block.new_code.split('\n'):
      new_lines.append(line + '\n')

    # Add remaining lines
    new_lines.extend(lines[block.end_line:])

    # Add CHANGELOG entry at top of file (after any existing comments)
    # Find first non-comment line
    insert_pos = 0
    for i, line in enumerate(new_lines):
      if not line.strip().startswith('#') and line.strip():
        insert_pos = i
        break

    # Look for existing CHANGELOG section
    changelog_pos = None
    for i, line in enumerate(new_lines[:insert_pos + 10]):
      if '# CHANGELOG:' in line:
        changelog_pos = i + 1
        break

    if changelog_pos is None:
      # Add CHANGELOG section
      changelog_section = "# CHANGELOG:\n" + fix_attempt.format().split('CHANGELOG:')[-1].split('\n')[0] + '\n#\n'
      new_lines.insert(insert_pos, changelog_section)
    else:
      # Add entry to existing CHANGELOG
      changelog_entry = self._generate_changelog_entry(
        fix_attempt.issue_hash,
        fix_attempt.issue_description,
        None
      ) + '\n'
      new_lines.insert(changelog_pos, changelog_entry)

    if dry_run:
      return {
        "success": True,
        "preview": ''.join(new_lines[max(0, block.start_line - 5):block.start_line + 15])
      }

    # Write file
    with open(file_path, 'w') as f:
      f.writelines(new_lines)

    return {"success": True}

  def get_pending_patches(self) -> List[Patch]:
    """Get all patches pending review."""
    return [p for p in self._patches if p.status == PatchStatus.PENDING_REVIEW]

  def approve_patch(self, patch_id: str) -> bool:
    """Approve a patch for application."""
    for patch in self._patches:
      if patch.patch_id == patch_id:
        patch.status = PatchStatus.APPROVED
        patch.reviewed_at = datetime.now()
        return True
    return False

  def reject_patch(self, patch_id: str, reason: str = "") -> bool:
    """Reject a patch."""
    for patch in self._patches:
      if patch.patch_id == patch_id:
        patch.status = PatchStatus.REJECTED
        patch.reviewed_at = datetime.now()
        patch.description += f"\n\nREJECTED: {reason}"
        return True
    return False
