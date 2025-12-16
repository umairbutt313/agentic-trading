# Autonomous Trading Developer System

A production-grade autonomous trading developer system using Claude Agent SDK with Liang Wenfeng-style reasoning, real-time log monitoring, comment-based memory, and human-reviewed code patching.

## Architecture Overview

```
    VPS (Local)                              Claude Agent SDK (Cloud)
+------------------+                     +--------------------------------+
|                  |                     |                                |
| +-------------+  |    REST/Webhook     | +---------------------------+  |
| | WATCHER     |  |    <1KB Events      | | SUPERVISOR AGENT          |  |
| | AGENT       +------------------------->|                           |  |
| |             |  |                     | | - Event Triage            |  |
| | - watchdog  |  |                     | | - Memory Lookup           |  |
| | - Flask API |  |                     | | - Deduplication           |  |
| +------+------+  |                     | +------------+--------------+  |
|        |         |                     |              |                 |
|        | tail    |                     |              | dispatch        |
|        v         |                     |              v                 |
| +-------------+  |                     | +---------------------------+  |
| | TRADING     |  |                     | | EXPERT TRADING DEV AGENT  |  |
| | SYSTEM LOGS |  |                     | |                           |  |
| +-------------+  |                     | | - Liang Wenfeng Reasoning |  |
|                  |                     | | - Root Cause Analysis     |  |
| +-------------+  |    Code Patch       | | - Patch Generation        |  |
| | SOURCE CODE |<-------------------------+ +---------------------------+  |
| +-------------+  |  (Human Review)     |                                |
+------------------+                     +--------------------------------+

                    IN-CODE MEMORY (FIX ATTEMPT blocks + CHANGELOG)
```

## Quick Start

### Prerequisites

1. Python 3.10+
2. Claude Code CLI installed:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```
3. Anthropic API Key set:
   ```bash
   export ANTHROPIC_API_KEY=<your-api-key>
   ```

### Installation

```bash
# Install dependencies
pip install watchdog flask requests pyyaml

# Optional: Install Claude Agent SDK (when available)
pip install claude-agent-sdk
```

### Running the System

```bash
# Start all agents
python agents/run_agents.py

# Start in test mode (no file modifications)
python agents/run_agents.py --test-mode

# Start individual agents
python agents/run_agents.py --agent watcher
python agents/run_agents.py --agent supervisor
python agents/run_agents.py --agent expert
```

## Component Details

### Phase 1: Watcher Agent

**Location**: `agents/watcher/`

Monitors trading logs and emits structured events.

**Features**:
- Watchdog-based file monitoring
- Log parsing with regex patterns
- Event batching and rate limiting
- Webhook emission with retry logic
- Flask API for health checks

**API Endpoints** (port 5000):
- `GET /health` - Health check
- `GET /stats` - Current statistics
- `POST /trigger` - Manual event trigger

### Phase 2: Supervisor Agent

**Location**: `agents/supervisor/`

Receives events, triages them, and routes to Expert Agent.

**Features**:
- Event triage by severity and type
- FIX ATTEMPT history lookup
- Issue hash computation for deduplication
- Context preparation for Expert Agent
- ClaudeSDKClient integration

**API Endpoints** (port 5001):
- `GET /health` - Health check
- `POST /event` - Receive events from Watcher
- `GET /event/<id>` - Get event processing result
- `GET /results` - Get all processing results
- `GET /stats` - Processing statistics

### Phase 3: Expert Agent

**Location**: `agents/expert/`

Performs deep analysis using Liang Wenfeng 5-step reasoning.

**Reasoning Steps**:
1. **Market Context Analysis** - Regime, indicators, sentiment
2. **Signal Interpretation** - Why signal triggered, confidence
3. **Alternative Evaluation** - Other approaches considered
4. **Risk Management Reasoning** - Stops, sizing, R:R
5. **Post-Trade Reflection** - What failed, improvements

**Features**:
- Systematic reasoning loop
- Code reading in 250-line chunks
- Minimal patch generation (max 50 lines)
- FIX ATTEMPT block creation
- CHANGELOG entry generation

## Event Types

| Event Type | Description | Severity |
|------------|-------------|----------|
| TRADE | Every trade execution | INFO/WARNING |
| ERROR | Error log entries | WARNING/CRITICAL |
| POSITION_DESYNC | Local/broker mismatch | CRITICAL |
| CIRCUIT_BREAKER | Safety triggers | CRITICAL |
| PERFORMANCE_DROP | Threshold breaches | WARNING/CRITICAL |
| SENTIMENT | Sentiment changes | INFO |
| EXECUTION_LAG | High API latency | WARNING |

## Memory System

The system uses **in-code comments** as memory:

### FIX ATTEMPT Blocks

```python
# ==============================================================================
# FIX ATTEMPT [2025-12-08 12:00:00]
# ==============================================================================
# ISSUE: Position desync - dealId vs dealReference
# ISSUE_HASH: abc123def456
#
# PREVIOUS ATTEMPTS:
#   - [2025-12-07] Tried X - FAILED (reason)
#
# LIANG WENFENG REASONING:
#   1. Market Context: Trending market, ADX 45
#   2. Signal Interpretation: False signal due to...
#   3. Alternative Evaluation: Could have used...
#   4. Risk Management: Stop was too tight
#   5. Reflection: Need to adjust...
#
# SOLUTION: Changed dealId to dealReference
# VALIDATION: Test position lifecycle
# ==============================================================================
```

### CHANGELOG Entries

```python
# CHANGELOG:
# [2025-12-08] FIX: Position desync issue (ISSUE_HASH: abc123def456)
```

## Configuration

Edit `agents/config.yaml`:

```yaml
watcher:
  logs_dir: "/root/arslan-chart/agentic-trading-dec2025/stocks/logs"
  webhook_url: "http://localhost:5001/event"
  api_port: 5000

triggers:
  max_daily_loss_threshold: 50.0
  min_win_rate_threshold: 0.40
  max_drawdown_threshold: 0.05

supervisor:
  api_port: 5001
  model: "claude-sonnet-4-20250514"

expert:
  max_patch_lines: 50
  reasoning_steps: 5
```

## Human Review Workflow

1. System detects issue and generates patch
2. Patch appears in pending review queue
3. Human reviews patch via API:
   ```bash
   # Get pending patches
   curl http://localhost:5001/patches/pending

   # Approve patch
   curl -X POST http://localhost:5001/patches/PATCH_ID/approve

   # Reject patch
   curl -X POST http://localhost:5001/patches/PATCH_ID/reject \
     -H "Content-Type: application/json" \
     -d '{"reason": "Better approach exists"}'
   ```
4. Approved patches are applied automatically
5. Validation tests run

## File Structure

```
agents/
├── config.yaml           # Agent configuration
├── run_agents.py         # Main entry point
├── README.md             # This file
│
├── watcher/
│   ├── __init__.py
│   ├── event_schemas.py  # Event dataclasses
│   ├── log_parser.py     # Log parsing
│   ├── trigger_rules.py  # Event emission rules
│   └── watcher_service.py # Main service
│
├── supervisor/
│   ├── __init__.py
│   ├── memory_tools.py   # FIX ATTEMPT lookup tools
│   ├── supervisor_agent.py # ClaudeSDKClient agent
│   └── api_server.py     # Webhook receiver
│
├── expert/
│   ├── __init__.py
│   ├── reasoning_engine.py # Liang Wenfeng loop
│   ├── patch_generator.py  # Patch creation
│   └── expert_agent.py     # Main expert agent
│
└── improvement/
    └── strategy_optimizer.py # Parameter tuning
```

## Development

### Testing Individual Components

```python
# Test Watcher
from agents.watcher import LogParser, TriggerRules
parser = LogParser()
events = parser.parse_to_events(log_line, "test.log")

# Test Supervisor
from agents.supervisor import SupervisorAgent
agent = SupervisorAgent()
result = await agent.triage_event(event)

# Test Expert
from agents.expert import ExpertAgent
expert = ExpertAgent()
analysis = await expert.analyze_issue(triage_result)
```

### Adding New Event Types

1. Add to `event_schemas.py`
2. Update `log_parser.py` patterns
3. Add trigger rules in `trigger_rules.py`
4. Update Supervisor routing

## Constraints

- Events must serialize to **<1KB JSON**
- Files read in **250-line chunks** (Claude Code limit)
- Maximum **50 lines** changed per patch
- All patches require **human review**
- Never delete previous **FIX ATTEMPT** comments
- Always include **ISSUE_HASH** for deduplication

## Environment Variables

```bash
ANTHROPIC_API_KEY=<your-api-key>
SUPERVISOR_WEBHOOK_URL=http://localhost:5001/event
LOGS_DIR=/root/arslan-chart/agentic-trading-dec2025/stocks/logs
WATCHER_PORT=5000
SUPERVISOR_PORT=5001
CLAUDE_MODEL=claude-sonnet-4-20250514
```
