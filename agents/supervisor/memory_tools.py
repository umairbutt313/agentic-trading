"""
Memory Tools for Supervisor Agent
Custom MCP tools for looking up FIX ATTEMPT history and computing issue hashes.
"""

import re
import hashlib
import os
from typing import Dict, Any, List, Optional
from datetime import datetime


# Simulated tool decorator for documentation
# In actual Claude SDK usage, use: from claude_agent_sdk import tool
def tool(name: str, description: str, parameters: Dict[str, type]):
  """Decorator for defining MCP tools (placeholder for Claude SDK)."""
  def decorator(func):
    func._tool_name = name
    func._tool_description = description
    func._tool_parameters = parameters
    return func
  return decorator


# Constants
ALLOWED_BASE_PATH = "/root/arslan-chart/agentic-trading-dec2025/stocks"
MAX_READ_LINES = 250


@tool("lookup_fix_history", "Search code for previous fix attempts", {
  "issue_pattern": str,
  "file_paths": list
})
async def lookup_fix_history(args: Dict[str, Any]) -> Dict[str, Any]:
  """
  Search specified files for FIX ATTEMPT blocks matching the issue pattern.
  Returns previous attempts to prevent repeating failed solutions.

  Args (via args dict):
    issue_pattern: Pattern to search for in FIX ATTEMPT blocks
    file_paths: List of file paths to search

  Returns:
    Dict with content containing found FIX ATTEMPT blocks
  """
  issue_pattern = args.get("issue_pattern", "")
  file_paths = args.get("file_paths", [])

  if not issue_pattern:
    return {
      "content": [{"type": "text", "text": "Error: issue_pattern is required"}],
      "is_error": True
    }

  results = []
  files_searched = 0

  for file_path in file_paths:
    # Security: ensure path is within allowed base
    if not file_path.startswith(ALLOWED_BASE_PATH):
      continue

    if not os.path.exists(file_path):
      continue

    files_searched += 1

    try:
      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

      # Find FIX ATTEMPT blocks
      # Pattern: # FIX ATTEMPT [...] followed by content until # ===...
      pattern = r'# ={10,}\n# FIX ATTEMPT \[(.*?)\].*?# ={10,}'
      matches = re.findall(pattern, content, re.DOTALL)

      for match in matches:
        # Check if issue pattern is in this FIX ATTEMPT block
        if issue_pattern.lower() in match.lower():
          # Extract key information
          attempt_info = _parse_fix_attempt_block(match, file_path)
          results.append(attempt_info)

      # Also search for inline FIX ATTEMPT comments
      inline_pattern = r'# FIX ATTEMPT: (.+)'
      inline_matches = re.findall(inline_pattern, content)
      for inline_match in inline_matches:
        if issue_pattern.lower() in inline_match.lower():
          results.append({
            "file": file_path,
            "type": "inline",
            "description": inline_match[:200],
            "date": "unknown"
          })

    except Exception as e:
      continue

  if not results:
    return {
      "content": [{
        "type": "text",
        "text": f"No previous fix attempts found for pattern '{issue_pattern}' "
               f"(searched {files_searched} files)"
      }]
    }

  # Format results
  result_text = f"Found {len(results)} previous fix attempts:\n\n"
  for i, r in enumerate(results, 1):
    result_text += f"--- Attempt {i} ---\n"
    result_text += f"File: {r.get('file', 'unknown')}\n"
    result_text += f"Date: {r.get('date', 'unknown')}\n"
    if r.get('issue_hash'):
      result_text += f"Issue Hash: {r.get('issue_hash')}\n"
    if r.get('description'):
      result_text += f"Description: {r.get('description')[:300]}\n"
    if r.get('outcome'):
      result_text += f"Outcome: {r.get('outcome')}\n"
    result_text += "\n"

  return {
    "content": [{
      "type": "text",
      "text": result_text[:2000]  # Limit response size
    }]
  }


def _parse_fix_attempt_block(block_content: str, file_path: str) -> Dict[str, Any]:
  """Parse a FIX ATTEMPT block and extract structured information."""
  result = {
    "file": file_path,
    "type": "block",
    "date": "unknown",
    "description": "",
    "issue_hash": None,
    "outcome": None,
  }

  # Extract date from block header
  date_match = re.search(r'\[(\d{4}-\d{2}-\d{2}[^\]]*)\]', block_content)
  if date_match:
    result["date"] = date_match.group(1)

  # Extract ISSUE line
  issue_match = re.search(r'# ISSUE: (.+)', block_content)
  if issue_match:
    result["description"] = issue_match.group(1)[:200]

  # Extract ISSUE_HASH
  hash_match = re.search(r'ISSUE_HASH: (\w+)', block_content)
  if hash_match:
    result["issue_hash"] = hash_match.group(1)

  # Extract outcome (SUCCESS/FAILED)
  if "FAILED" in block_content.upper():
    result["outcome"] = "FAILED"
  elif "SUCCESS" in block_content.upper() or "RESOLVED" in block_content.upper():
    result["outcome"] = "SUCCESS"

  return result


@tool("compute_issue_hash", "Generate deterministic hash for issue deduplication", {
  "issue_description": str,
  "affected_file": str,
  "error_type": str
})
async def compute_issue_hash(args: Dict[str, Any]) -> Dict[str, Any]:
  """
  Generate a hash to identify duplicate issues.
  Uses file, error type, and description to create unique identifier.

  Args (via args dict):
    issue_description: Description of the issue
    affected_file: File where the issue occurs
    error_type: Type of error (e.g., "TypeError", "PositionDesync")

  Returns:
    Dict with the computed issue hash
  """
  issue_description = args.get("issue_description", "")
  affected_file = args.get("affected_file", "")
  error_type = args.get("error_type", "")

  # Normalize inputs
  # Extract just the filename for more stable hashing
  if affected_file:
    affected_file = os.path.basename(affected_file)

  # Normalize description (lowercase, remove extra whitespace)
  normalized_desc = ' '.join(issue_description.lower().split())[:100]

  # Create hash input
  hash_input = f"{affected_file}:{error_type}:{normalized_desc}"
  issue_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

  return {
    "content": [{
      "type": "text",
      "text": f"Issue hash: {issue_hash}\n"
             f"Components:\n"
             f"  - File: {affected_file}\n"
             f"  - Error Type: {error_type}\n"
             f"  - Description (normalized): {normalized_desc[:50]}..."
    }]
  }


@tool("read_trading_file", "Read a trading system source file", {
  "file_path": str,
  "start_line": int,
  "end_line": int
})
async def read_trading_file(args: Dict[str, Any]) -> Dict[str, Any]:
  """
  Read trading system files in 250-line chunks.
  Enforces Claude Code constraints for file reading.

  Args (via args dict):
    file_path: Path to the file to read
    start_line: Starting line number (1-indexed)
    end_line: Ending line number (optional, defaults to start_line + 250)

  Returns:
    Dict with file content
  """
  file_path = args.get("file_path", "")
  start_line = args.get("start_line", 1)
  end_line = args.get("end_line", start_line + MAX_READ_LINES)

  # Security check
  if not file_path.startswith(ALLOWED_BASE_PATH):
    return {
      "content": [{
        "type": "text",
        "text": f"Error: Path must be within {ALLOWED_BASE_PATH}"
      }],
      "is_error": True
    }

  if not os.path.exists(file_path):
    return {
      "content": [{
        "type": "text",
        "text": f"Error: File not found: {file_path}"
      }],
      "is_error": True
    }

  try:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
      lines = f.readlines()

    total_lines = len(lines)

    # Validate line numbers
    start_line = max(1, start_line)
    end_line = min(end_line, total_lines)

    # Enforce chunk size limit
    chunk_size = min(end_line - start_line + 1, MAX_READ_LINES)
    end_line = start_line + chunk_size - 1

    # Extract lines (convert to 0-indexed)
    selected_lines = lines[start_line - 1:end_line]

    # Format with line numbers
    content = ""
    for i, line in enumerate(selected_lines, start_line):
      content += f"{i:4d} | {line}"

    return {
      "content": [{
        "type": "text",
        "text": f"File: {file_path}\n"
               f"Lines: {start_line}-{end_line} of {total_lines}\n"
               f"{'='*60}\n"
               f"{content}"
      }]
    }

  except Exception as e:
    return {
      "content": [{
        "type": "text",
        "text": f"Error reading file: {str(e)}"
      }],
      "is_error": True
    }


@tool("search_fix_attempts_by_hash", "Find FIX ATTEMPT blocks by issue hash", {
  "issue_hash": str,
  "file_paths": list
})
async def search_fix_attempts_by_hash(args: Dict[str, Any]) -> Dict[str, Any]:
  """
  Search for FIX ATTEMPT blocks that have a specific issue hash.
  Used to check if an issue has already been addressed.

  Args (via args dict):
    issue_hash: The issue hash to search for
    file_paths: List of file paths to search

  Returns:
    Dict with found FIX ATTEMPT blocks
  """
  issue_hash = args.get("issue_hash", "")
  file_paths = args.get("file_paths", [])

  if not issue_hash:
    return {
      "content": [{"type": "text", "text": "Error: issue_hash is required"}],
      "is_error": True
    }

  results = []

  for file_path in file_paths:
    if not file_path.startswith(ALLOWED_BASE_PATH):
      continue

    if not os.path.exists(file_path):
      continue

    try:
      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

      # Search for the issue hash
      if issue_hash in content:
        # Find the FIX ATTEMPT block containing this hash
        pattern = rf'# ={10,}.*?ISSUE_HASH: {issue_hash}.*?# ={10,}'
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
          results.append({
            "file": file_path,
            "block": match[:500]
          })

    except Exception:
      continue

  if not results:
    return {
      "content": [{
        "type": "text",
        "text": f"No FIX ATTEMPT blocks found with hash: {issue_hash}"
      }]
    }

  result_text = f"Found {len(results)} FIX ATTEMPT blocks with hash {issue_hash}:\n\n"
  for r in results:
    result_text += f"File: {r['file']}\n"
    result_text += f"{r['block']}\n\n"

  return {
    "content": [{
      "type": "text",
      "text": result_text[:2000]
    }]
  }


def create_memory_tools_server():
  """
  Create an MCP server with all memory tools.

  Returns:
    MCP server configuration for use with Claude SDK

  Usage with Claude SDK:
    from claude_agent_sdk import create_sdk_mcp_server

    memory_server = create_sdk_mcp_server(
      name="memory_tools",
      version="1.0.0",
      tools=[
        lookup_fix_history,
        compute_issue_hash,
        read_trading_file,
        search_fix_attempts_by_hash,
      ]
    )
  """
  # This is a placeholder - actual implementation uses claude_agent_sdk
  tools = [
    lookup_fix_history,
    compute_issue_hash,
    read_trading_file,
    search_fix_attempts_by_hash,
  ]

  return {
    "name": "memory_tools",
    "version": "1.0.0",
    "tools": [
      {
        "name": t._tool_name,
        "description": t._tool_description,
        "parameters": t._tool_parameters,
        "handler": t
      }
      for t in tools
    ]
  }


# Utility functions for non-async contexts (truly synchronous implementations)
def lookup_fix_history_sync(issue_pattern: str, file_paths: List[str]) -> Dict[str, Any]:
  """Synchronous version of lookup_fix_history."""
  if not issue_pattern:
    return {
      "content": [{"type": "text", "text": "Error: issue_pattern is required"}],
      "is_error": True
    }

  results = []
  files_searched = 0

  for file_path in file_paths:
    if not file_path.startswith(ALLOWED_BASE_PATH):
      continue
    if not os.path.exists(file_path):
      continue

    files_searched += 1
    try:
      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

      # Find FIX ATTEMPT blocks
      pattern = r'# ={10,}\n# FIX ATTEMPT \[(.*?)\].*?# ={10,}'
      matches = re.findall(pattern, content, re.DOTALL)

      for match in matches:
        if issue_pattern.lower() in match.lower():
          results.append({
            "file": file_path,
            "type": "block",
            "description": match[:200]
          })
    except Exception:
      continue

  if not results:
    return {
      "content": [{
        "type": "text",
        "text": f"No previous fix attempts found for pattern '{issue_pattern}' (searched {files_searched} files)"
      }]
    }

  result_text = f"Found {len(results)} previous fix attempts:\n\n"
  for i, r in enumerate(results, 1):
    result_text += f"--- Attempt {i} ---\nFile: {r.get('file', 'unknown')}\n{r.get('description', '')[:300]}\n\n"

  return {"content": [{"type": "text", "text": result_text[:2000]}]}


def compute_issue_hash_sync(
  issue_description: str,
  affected_file: str,
  error_type: str
) -> str:
  """Synchronous version of compute_issue_hash that returns just the hash."""
  # Normalize inputs
  if affected_file:
    affected_file = os.path.basename(affected_file)

  # Normalize description
  normalized_desc = ' '.join(issue_description.lower().split())[:100]

  # Create hash
  hash_input = f"{affected_file}:{error_type}:{normalized_desc}"
  return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
