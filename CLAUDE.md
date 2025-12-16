# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-frequency scalping trading system combining sentiment analysis, multi-strategy scalping, and real-time market data via Capital.com API. Target: 50-100+ trades/day with millisecond precision.

**Active Configuration**: NVIDIA (NVDA) only, demo account trading, WebSocket real-time data.

## Quick Commands

```bash
# Trading system (test first, always)
python3 test_scalping_system.py                     # Test all components
python3 trading/enhanced_scalper.py --conservative  # Demo trading (safe)
python3 trading/enhanced_scalper.py --aggressive    # Demo trading (WebSocket)

# Sentiment analysis
python3 news/weighted_sentiment_aggregator.py --force-refresh  # Fresh data
python3 news/weighted_sentiment_aggregator.py --quiet          # Use existing

# Real-time prices
python3 start_realtime_prices.py                    # WebSocket + REST fallback

# Web dashboard
python3 serve_charts.py                             # Opens browser automatically

# Full test suite
./test_all.sh

# Docker
make build && make up                               # Production deployment
make automated                                      # Run sentiment in container
```

## Architecture

### Three-Layer Trading System
1. **Sentiment Bias** (10s updates) - News analysis → directional bias via GPT-4
2. **Scalping Strategies** (sub-second) - Momentum, mean reversion, order book, spread
3. **Execution & Risk** (microsecond) - Position management via Capital.com API

### Data Pipelines
```
Sentiment: NewsAPI → news_dump.py → sentiment_analyzer.py → weighted_sentiment_aggregator.py
                  → final-weighted-scores.json → enhanced_scalper.py → Capital.com API

Trading:   WebSocket/REST → scalping_strategies.py → position_manager.py → capital_trader.py
```

### Autonomous Developer Agents (`agents/`)
- **Watcher Agent** (port 5000): Monitors logs, emits events via watchdog
- **Supervisor Agent** (port 5001): Event triage, memory lookup, deduplication
- **Expert Agent**: Liang Wenfeng 5-step reasoning, patch generation
- **Research Agent**: Trading system research (read-only, no code changes)

**Run agents**: `python agents/run_agents.py`

**Claude Agent SDK Integration** (2025-12-13):
- Agents use official `query()` pattern from `claude-agent-sdk`
- No separate API key needed - uses Claude Code authentication
- SDK auto-installed: `pip install claude-agent-sdk --break-system-packages`

```python
# Agent SDK pattern used internally
from claude_agent_sdk import query, ClaudeAgentOptions
async for message in query(prompt="...", options=ClaudeAgentOptions(...)):
    # Process AI response
```

## Key Files

| File | Purpose |
|------|---------|
| `trading/enhanced_scalper.py` | PRIMARY: High-frequency scalper |
| `trading/position_manager.py` | Position lifecycle management |
| `trading/scalping_strategies.py` | Strategy implementations |
| `trading/capital_trader.py` | Capital.com API integration |
| `news/weighted_sentiment_aggregator.py` | Main sentiment pipeline |
| `news/sentiment_analyzer.py` | GPT-4 sentiment analysis |
| `companies.yaml` | Company configuration |

## Configuration

### Environment (.env)
```bash
NEWS_API_KEY=xxx          # NewsAPI
OPENAI_API_KEY=xxx        # GPT-4 sentiment
CAPITAL_API_KEY=xxx       # Capital.com trading
CAPITAL_PASSWORD=xxx      # Capital.com password
CAPITAL_EMAIL=xxx         # Capital.com email
```

### Trading Thresholds
- **Buy**: Sentiment >= 7.0
- **Sell/Short**: Sentiment <= 4.0
- **Position Sizing**: 5-20% of balance (sentiment-based)
- **Market Hours**: Extended 4 AM - 8 PM

### Capital.com API
- Demo: `https://demo-api-capital.backend-capital.com`
- Live: `https://api-capital.backend-capital.com`
- Rate Limits: 10 req/sec, 1 req per 0.1s for orders
- Session: 10-min with auto-refresh

## Development Conventions

- **Indentation**: 2 spaces for Python
- **Sentiment Scale**: 1-10 (1=very bearish, 10=very bullish)
- **File Naming**: `raw-{type}_YYYYMMDD_HHMMSS.json` for scraped data
- **Testing**: Run `./test_all.sh` before commits

## Output Locations

- `container_output/final_score/final-weighted-scores.json` - Main sentiment output
- `container_output/realtime_data/price_history.json` - Real-time prices
- `logs/enhanced_scalper_*.log` - Trading logs
- `web/nvidia_score_price_dump.txt` - Dashboard CSV

## Position Management

Position lifecycle is managed through SQLite persistence with broker as single source of truth.

**Key Files:**
- `trading/position_manager.py` - Position lifecycle, open/close logic
- `trading/position_database.py` - SQLite persistence layer
- `trading/capital_trader.py` - Capital.com API integration

**Verification Commands:**
```python
# Check broker positions
trader.get_positions()

# Check local database
from trading.position_database import PositionDatabase
db = PositionDatabase()
db.get_open_positions()
```

## Agent Memory System

The autonomous agents use **in-code comments** as memory:

### FIX ATTEMPT Blocks
```python
# ==============================================================================
# FIX ATTEMPT [2025-12-08 12:00:00]
# ==============================================================================
# ISSUE: Description
# ISSUE_HASH: unique_hash_for_deduplication
# LIANG WENFENG REASONING:
#   1. Market Context: ...
#   2. Signal Interpretation: ...
#   3. Alternative Evaluation: ...
#   4. Risk Management: ...
#   5. Reflection: ...
# SOLUTION: What was changed
# ==============================================================================
```

**Rules**:
- Never delete previous FIX ATTEMPT comments
- Always include ISSUE_HASH for deduplication
- **NO line limit** - agents can propose architectural changes
- **ALL patches require human approval** before application

## Disabled Features

- **Image Sentiment**: Chart screenshots collected but not analyzed
- **TradingView OHLC**: Collected but unused
- **Reddit Data**: Returns empty arrays
- **Multi-company Trading**: Only NVDA active
- **Stop Loss System**: Capital.com minimum distance issues

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No trades executing | Check sentiment >= 7.0 or <= 4.0 |
| Chrome/Playwright issues | `apt install chromium-browser` or `cd playwright_scrapers && npx playwright install` |
| Trading login fails | Verify Capital.com credentials + enable 2FA |
| Position desync | Compare system tracking vs broker API |
| Missing data files | Run `python3 news/weighted_sentiment_aggregator.py --force-refresh` |
| Agent ports in use | `pkill -f flask` then restart agents |
| SDK permission error | Use `acceptEdits` mode (not `bypassPermissions` when running as root) |

## Agent Configuration (`.claude/agents/`)

| Agent | Purpose | Tools |
|-------|---------|-------|
| `expert-agent` | Bug fixes, patch generation | Read, Edit, Grep, Glob |
| `supervisor-agent` | Event triage, routing | Read, Grep, Glob |
| `watcher-agent` | Log monitoring | Read, Grep |
| `research-trading-agent` | Research only (no code) | Read, Glob, Grep, WebSearch, WebFetch, Task |
| `meta-analyst-agent` | Architecture review | All tools |

## Skills Plan (`.claude/skillplan.md`)

Planned Claude Code skills for automation:
- `refresh-sentiment` - Update sentiment pipeline
- `run-tests` - Execute test suite
- `check-positions` - Verify broker sync
- `start-trading` - Launch with pre-checks
- `analyze-logs` - Parse trading history
