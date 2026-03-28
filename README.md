# 🤖 Sentinel — Autonomous News-Driven Polymarket Trading Agent

> **An AI-powered autonomous agent that monitors breaking news, estimates predictive edge on Polymarket prediction markets, and executes trades using Bayesian confidence scoring and Kelly criterion position sizing.**

Built for the [Penn Blockchain Conference 2026 Hackathon](https://www.pennblockchain.com/) — Polymarket Bounty.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Data Ingestion Strategy](#data-ingestion-strategy)
- [Confidence Scoring Framework](#confidence-scoring-framework)
- [Trade Execution Logic](#trade-execution-logic)
- [Getting Started](#getting-started)
- [Running the Backtest](#running-the-backtest)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Current Status & Roadmap](#current-status--roadmap)

---

## Overview

**Sentinel** is an autonomous trading agent prototype designed for Polymarket. It operates in a continuous loop:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  NEWS        │────▶│  AI          │────▶│  CONFIDENCE  │────▶│  TRADE       │
│  INGESTION   │     │  ANALYSIS    │     │  SCORING     │     │  EXECUTION   │
│              │     │              │     │              │     │              │
│ • RSS feeds  │     │ • LLM-based  │     │ • Bayesian   │     │ • Orderbook  │
│ • Multi-src  │     │ • Structured │     │ • Kelly      │     │ • Slippage   │
│ • Dedup      │     │   prompting  │     │   criterion  │     │   modeling   │
│ • Latency    │     │ • Causal     │     │ • Time decay │     │ • Paper mode │
│   tracking   │     │   reasoning  │     │ • Multi-src  │     │ • Risk mgmt  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Key differentiators:**
1. **Mathematically rigorous edge estimation** — Not just "AI thinks this will go up." We compute a precise Bayesian posterior probability, calculate expected value, and size positions using fractional Kelly criterion.
2. **Latency-optimized ingestion** — Every article is timestamped at discovery with source-level latency tracking. In prediction markets, information speed is alpha.
3. **Orderbook-aware execution** — We walk the book to estimate slippage, adjust position size based on depth, and factor spread costs into edge calculation.

---

## Architecture

```
PolymarketPBC/
├── src/
│   ├── ingestion/
│   │   ├── news.py          # Multi-source RSS ingestion with dedup & latency tracking
│   │   └── markets.py       # Polymarket Gamma API + CLOB orderbook client
│   ├── analysis/
│   │   └── engine.py        # LLM-powered news → market impact analysis
│   ├── scoring/
│   │   └── confidence.py    # Bayesian scoring + Kelly criterion sizing
│   ├── execution/
│   │   └── executor.py      # Paper trading simulator with orderbook-based fills
│   ├── agent.py             # Main autonomous loop
│   ├── backtest.py          # Backtesting simulation
│   └── config.py            # Configuration management
├── logs/                    # Trade logs, backtest results, cycle logs
├── requirements.txt
└── README.md
```

---

## Data Ingestion Strategy

### Sources (Diversity & Coverage)

| Source Type | Feeds | Latency Target | Purpose |
|-------------|-------|----------------|---------|
| **Wire Services** | Reuters, BBC | < 60s from publication | Breaking news with global coverage |
| **Political** | Politico | < 120s | Policy and election-relevant events |
| **Financial** | Bloomberg | < 60s | Market-moving economic data |
| **Crypto-native** | CoinDesk | < 30s | Crypto-specific events |
| **Quality Press** | NYT | < 300s | In-depth analysis and polls |

### Latency Optimization

```python
# Every article gets a timestamp at discovery
article.discovered_at = datetime.now(UTC)
article.ingestion_latency_ms = (discovered_at - published_at).total_seconds() * 1000
```

We track ingestion latency per source to identify and prioritize the fastest feeds. The agent reports latency statistics after each cycle:

```json
{
  "reuters": {"avg_ms": 45200, "min_ms": 12000, "max_ms": 180000, "count": 15},
  "coindesk": {"avg_ms": 28000, "min_ms": 8000, "max_ms": 95000, "count": 8}
}
```

### Deduplication

Articles are deduplicated using a SHA-256 hash of `(title + source)`. This prevents the same story from different feeds from generating duplicate signals, which would artificially inflate confidence.

### Concurrency

All RSS feeds are fetched concurrently using `asyncio` + `aiohttp` with a 10-second timeout. A single ingestion cycle typically completes in < 2 seconds regardless of the number of feeds.

---

## Confidence Scoring Framework

This is the mathematical core of Sentinel. The scoring pipeline transforms raw AI analysis into precise, tradeable signals.

### Step 1: AI-Powered Impact Analysis

Each news article is analyzed against active Polymarket markets using structured LLM prompting. The model acts as a **superforecaster**, trained to:
- Distinguish signal from noise (most news is noise)
- Consider base rates and existing market pricing
- Identify causal chains from news to market outcomes
- Self-report confidence calibrated to epistemic uncertainty

The LLM returns structured JSON:
```json
{
  "direction": "UP",
  "probability_shift": 0.07,
  "confidence": 0.72,
  "is_new_information": true,
  "causal_chain": "Fed rate pause → lower borrowing costs → bullish for crypto markets"
}
```

### Step 2: Bayesian Probability Update

Given the market's current price `p_market` and the AI's estimated shift `Δp`:

```
p_agent = p_market + Δp × w_evidence × w_confidence × w_freshness × w_novelty
```

Where:
- **`w_evidence = 0.15`** — Evidence weight. We intentionally dampen the AI's raw estimates because LLMs tend toward overconfidence. This parameter was tuned for calibration.
- **`w_confidence`** — The LLM's self-reported confidence (0–1).
- **`w_freshness = exp(-t × ln(2) / τ)`** — Exponential time decay with half-life τ = 4 hours. Older signals carry less weight because the market adapts.
- **`w_novelty`** — 1.5× multiplier if the AI identifies the information as genuinely new (not already known/priced in).

### Step 3: Multi-Source Aggregation

When ≥2 independent news sources point in the same direction with no conflicting signals:

```
p_combined = p_market + Σ(Δp_i × w_i) × (1 + bonus)
```

The agreement bonus (`bonus = 0.10`) rewards convergent evidence from independent sources. Importantly, if sources disagree on direction, no bonus is applied and the signals partially cancel — this acts as a natural hedge against false signals.

### Step 4: Edge Calculation & Minimum Threshold

```
edge = p_agent - p_market
```

We only trade when `|edge| > 5%`. This threshold accounts for:
- Spread crossing costs
- Execution slippage
- Model uncertainty
- Transaction fees

Below this threshold, the expected profit doesn't justify the risk.

### Step 5: Kelly Criterion Position Sizing

The Kelly criterion gives the mathematically optimal fraction of bankroll to wager:

```
f* = (p × b - q) / b
```

Where:
- `p` = our estimated probability of the outcome
- `q = 1 - p`
- `b` = payout odds = `(1 / entry_price) - 1`

**We use quarter-Kelly (`f = 0.25 × f*`)** for several reasons:
1. Kelly assumes perfect probability estimates; ours are imperfect
2. Quarter-Kelly preserves ~90% of the growth rate at ~50% of the variance
3. It provides meaningful protection against ruin from estimation error

### Step 6: Orderbook-Adjusted Sizing

The final position size is adjusted based on orderbook quality:

```
position_size = bankroll × kelly_fraction × book_adjustment

book_adjustment = depth_score × max(0.3, 1 - spread × 5)
```

Where:
- **`depth_score`** — Logarithmic normalization of available depth: `min(1, log(1 + depth) / log(1 + 10000))`. This smoothly scales from 0 (no depth) to 1 ($10K+ depth).
- **`spread_penalty`** — Wide spreads eat into edge. A 5¢ spread reduces sizing by 25%.

This prevents the agent from placing large orders into thin books where slippage would destroy the edge.

### Full Pipeline Summary

```
News Article
    ↓
AI Analysis (LLM) → direction, magnitude, confidence
    ↓
Bayesian Update → p_agent = p_market + weighted_shift
    ↓
Edge Check → |p_agent - p_market| > 5%?
    ↓  (yes)
Kelly Sizing → f* × 0.25 (quarter-Kelly)
    ↓
Orderbook Adjustment → reduce for thin books / wide spreads
    ↓
Expected Value Check → EV > 0?
    ↓  (yes)
EXECUTE TRADE
```

---

## Trade Execution Logic

### Paper Trading Mode (Default)

The agent currently operates in paper trading mode, simulating realistic fills by walking the orderbook:

```python
def simulate_fill(signal, orderbook):
    remaining = signal.position_size_usdc
    for level in orderbook.asks:  # Walk the book
        if remaining <= level.price * level.size:
            # Fill at this level
            shares = remaining / level.price
            break
        else:
            # Consume this level, continue to next
            remaining -= level.price * level.size
```

This captures the key slippage dynamics that a naive simulation would miss.

### Live Trading (Roadmap)

The execution engine is architected with a clear interface for live trading via the official `py-clob-client` SDK. The paper trading module validates the full signal → order flow, and the live execution path would plug in directly using:

```python
from py_clob_client.client import ClobClient

client = ClobClient(HOST, key=PRIVATE_KEY, chain_id=137)
client.set_api_creds(client.create_or_derive_api_creds())
```

This is not yet wired for live execution — the focus of this hackathon build was on the **data ingestion strategy, confidence scoring logic, and overall architecture**, which are the primary judging criteria.

### Risk Management

| Control | Value | Rationale |
|---------|-------|-----------|
| Max position % of bankroll | 10% | Prevents concentration risk |
| Min edge threshold | 5% | Ensures trades overcome friction |
| Max slippage tolerance | 2% | Rejects poor fills |
| Kelly fraction | 25% | Reduces variance vs full Kelly |
| Max trades per cycle | 3 | Prevents overtrading |
| Min order size | $5 USDC | Below this, fees dominate |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/LeiGMI/PolymarketPBC.git
cd PolymarketPBC
pip install -r requirements.txt
```

### Environment Variables (Optional)

```bash
# For live AI analysis (optional — mock analysis works without keys)
export ANTHROPIC_API_KEY="sk-ant-..."

# For paper trading with real market data (no keys needed for backtest)
export EXECUTION_MODE="paper"
```

### Run the Agent

```bash
# Paper trading mode (default) — runs 5 cycles
python -m src.agent --cycles 5 --mode paper --bankroll 10000

# Or run the backtest simulation
python -m src.backtest
```

---

## Running the Backtest

The backtest module generates a simulated trading log that demonstrates the agent's full decision-making pipeline:

```bash
python -m src.backtest
```

This will:
1. Attempt to fetch real active markets from Polymarket's Gamma API (falls back to synthetic markets if offline)
2. Simulate 8 realistic news events across politics, economics, crypto, and tech
3. Run the full scoring pipeline for each event
4. Generate paper trades with orderbook-simulated fills
5. Track portfolio P&L through each cycle
6. Output a detailed JSON log to `logs/backtest_log.json`

### Sample Backtest Output

```
═══ BACKTEST SIMULATION START ═══
Using 10 markets for backtest

─── Cycle 1: Federal Reserve signals potential rate pause... ───
  TRADE: BUY Yes on 'fed-rate-cut-next-meeting' | $127.69 @ 0.4581 | edge=+0.056 | EV=$13.34
  TRADE: BUY Yes on 'ecb-rate-cut-surprise' | $86.85 @ 0.3573 | edge=+0.046 | EV=$9.48
  Portfolio: $10,001.62 (PnL: +$1.62 / +0.0%)

─── Cycle 3: Senate advances cryptocurrency regulation bill... ───
  TRADE: BUY Yes on 'btc-150k-july' | $79.38 @ 0.3289 | edge=+0.046 | EV=$9.82
  TRADE: BUY Yes on 'us-crypto-regulation' | $223.29 @ 0.5575 | edge=+0.073 | EV=$26.19
  REJECTED: crypto-exchange-hack (slippage 58.6% exceeded 3% tolerance)
  Portfolio: $10,008.80 (PnL: +$8.80 / +0.1%)

═══ BACKTEST COMPLETE ═══
Final Portfolio: $10,010.19
Total PnL: +$10.19 (+0.1%)
Total Trades: 8 executed, 5 rejected by risk management
```

---

## Configuration

All parameters are configurable via `src/config.py`:

```python
@dataclass
class ScoringConfig:
    prior_confidence: float = 0.5      # Start at max uncertainty
    evidence_weight: float = 0.15      # AI shift dampening factor
    kelly_fraction: float = 0.25       # Quarter-Kelly for safety
    min_edge_threshold: float = 0.05   # 5% minimum edge to trade
    max_position_pct: float = 0.10     # 10% max per position
    decay_half_life_hours: float = 4.0 # Signal freshness decay
    source_agreement_bonus: float = 0.10  # Multi-source bonus
```

---

## Project Structure

```
PolymarketPBC/
├── src/
│   ├── config.py                 # All configurable parameters
│   ├── agent.py                  # Main autonomous agent loop
│   ├── backtest.py               # Backtesting simulation
│   ├── ingestion/
│   │   ├── news.py               # Multi-source RSS ingestion
│   │   └── markets.py            # Polymarket API client
│   ├── analysis/
│   │   └── engine.py             # LLM-based news analysis
│   ├── scoring/
│   │   └── confidence.py         # Bayesian scoring + Kelly sizing
│   └── execution/
│       └── executor.py           # Paper trade execution with orderbook simulation
├── logs/
│   ├── backtest_log.json         # Full backtest output
│   ├── trade_log.jsonl           # Trade-by-trade log
│   ├── cycle_log.jsonl           # Per-cycle summaries
│   └── portfolio_state.json      # Current portfolio snapshot
├── requirements.txt
└── README.md
```

---

## Current Status & Roadmap

This repo is a hackathon prototype and demo artifact. The core decision engine, scoring logic, and paper execution are fully implemented; live capital deployment, production hardening, and comprehensive testing are next steps.

### ✅ What's Built & Working
- **News ingestion pipeline** — Concurrent multi-source RSS with deduplication and latency tracking
- **Polymarket market data client** — Fetches active markets via Gamma API and live orderbooks via CLOB API
- **AI analysis engine** — Structured LLM prompting with auto-detection of Claude or OpenAI via environment variables (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`); includes deterministic mock mode for demos without API keys
- **Bayesian confidence scoring** — Full mathematical pipeline: evidence weighting, time decay, multi-source aggregation, Kelly criterion sizing, orderbook adjustment
- **Paper trading execution** — Realistic fill simulation by walking the orderbook with slippage modeling and risk management (5 trades rejected in backtest due to slippage tolerance)
- **Agent orchestration loop** — Ties all modules together in a continuous autonomous cycle
- **Backtest simulation** — End-to-end demo with 8 news cycles, 8 executed trades, and 5 risk-rejected trades

### ⚠️ Limitations
- **No live execution yet** — The agent runs in paper trading mode only; `py-clob-client` SDK integration is architected but not wired
- **No historical evaluation** — Backtest uses simulated news events against real market structure, not a large real-world news dataset
- **CI scaffolded but not yet active** — GitHub Actions workflow is committed (.github/workflows/ci.yml) but Actions is not yet enabled on this account; no unit test coverage yet
- **Latency claims are architectural** — Ingestion latency targets are design goals based on prototype measurements, not benchmarked on production infrastructure
- **LLM analysis is probabilistic** — The "superforecaster" framing describes structured constrained-output prompting, not a calibrated forecasting system

### 🔜 Roadmap (Post-Hackathon)
- **Live execution** — Wire `py-clob-client` SDK into the execution engine for real Polymarket trades
- **WebSocket streaming** — Replace RSS polling with Polymarket WebSocket for sub-second market data
- **Historical backtesting** — Replay against archived news + historical Polymarket price data
- **Dashboard UI** — Web-based monitoring dashboard for live agent status
- **CI & testing** — Automated test suite and GitHub Actions workflow

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Team

**Leo Song** · **Victor Song** · **Aadit Jerfy** · **Tharun Ekambaran**

Built at Penn Blockchain Conference 2026 Hackathon.
