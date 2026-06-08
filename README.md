# AI Trade Agent

Autonomous AI-powered trading system with dual agents for stocks and options, featuring self-improving capabilities through daily model fine-tuning. Runs 24/7 on an NVIDIA GB10 Grace Blackwell (128 GB unified memory).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Shared vLLM Server                         │
│           Qwen 2.5 32B (BF16 merged + 4-bit quantized)          │
│           Serves all agents over OpenAI-compatible API          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐─────────────┐
          ▼             ▼             ▼              ▼
   Stock Bot       Options Bot    Stock Bot       Options Bot
   (paper)         (paper)        (live)          (live*)
   training        training       exec only       exec only
   fine-tunes      fine-tunes     no training     no training

  * live options bot gated behind $5,000 equity threshold
```

The system runs **four trading agents** sharing one vLLM inference server:
- Paper agents trade unlimited, generate training data, and drive daily fine-tuning
- Live agents execute only — no fine-tuning, no analysis, no weekend strategist
- All trade data is tagged `source=paper|live` and `bot=stock|options` in SQLite

## Features

- **Dual-Agent System**: Separate stock and options bots, each with paper and live instances
- **Shared vLLM Inference**: One Qwen 2.5 32B instance serves all four agents (3-5× faster than per-process inference)
- **Self-Improving Loop**: Daily fine-tuning → eval → A/B promotion → model merge → vLLM reload
- **PDT-Safe Small Account Mode**: Stock bot trades live at any equity; protects against day-trade violations via `no_same_day_close` and conservative position sizing
- **Fractional Share Trading**: $39 position sizes work on stocks priced above $39 via Alpaca fractional shares
- **Equity-Gated Live Options**: Options bot stays on paper until account reaches $5,000
- **Portfolio Overseer**: Sector caps (30% max), max correlated positions (4)
- **Allocation Tiers**: Tier 1/2/3 sizing (1%/3%/5%) unlocked by trade count, Sharpe, and drawdown thresholds
- **Economic Calendar Guard**: Halts trading ±30–60 min around FOMC, NFP, and CPI releases
- **Unusual Options Flow**: Scanner detects institutional positioning signals
- **Congressional Trade Tracking**: Monitors politician disclosures for sentiment signals
- **LLM News Sentiment**: Real-time news scoring via the same Qwen 32B model
- **24/7 Operation**: systemd services with auto-restart and full journald logging
- **Health Endpoint**: `/health` /`status` `/metrics` (Prometheus) on port 8765
- **Email Alerts**: Circuit breaker, trade execution, account upgrade thresholds

## Project Structure

```
ai-trade-agent/
├── scripts/
│   ├── agents/
│   │   ├── autonomous_agent.py       # Stock trading logic (paper + live)
│   │   └── options_agent.py          # Options trading logic (paper + live)
│   ├── analysis/
│   │   ├── backtester.py             # Historical backtesting
│   │   ├── outcome_tracker.py        # Stock trade outcome recording
│   │   ├── options_outcome_tracker.py
│   │   ├── performance_analyzer.py   # Daily stock P&L analysis
│   │   ├── options_performance_analyzer.py
│   │   ├── weekend_strategist.py     # Saturday deep research (stocks)
│   │   ├── options_weekend_strategist.py
│   │   └── weekly_report.py          # Saturday summary email
│   ├── data/
│   │   ├── stock_discovery.py        # Candidate scanner (technicals)
│   │   ├── news_fetcher.py           # News with LLM sentiment
│   │   ├── congressional_tracker.py  # Politician trade disclosures
│   │   ├── unusual_flow_scanner.py   # Options flow anomaly detection
│   │   ├── market_researcher.py      # Macro / sector context
│   │   ├── economic_calendar.py      # FOMC/NFP/CPI guard
│   │   └── data_utils.py
│   ├── training/
│   │   ├── finetune_model.py         # SFT + DPO daily fine-tuning
│   │   ├── training_data_builder.py  # Builds training_data.json from SQLite
│   │   ├── eval_model.py             # A/B evaluation before promotion
│   │   ├── model_promoter.py         # Merges LoRA → BF16, updates symlinks
│   │   └── online_trainer.py        # In-process online learning
│   ├── tools/
│   │   ├── close_all_positions.py    # Emergency: close everything
│   │   ├── liquidate_to_cash.py      # Liquidate to cash
│   │   ├── fix_pdt.py                # PDT flag remediation
│   │   ├── cover_cash_deficit.py     # Cash deficit helper
│   │   └── morning_model_check.py    # Pre-market model health check
│   ├── trading_daemon.py             # Stock agent orchestration loop (24/7)
│   ├── options_daemon.py             # Options agent orchestration loop (24/7)
│   ├── inference_client.py           # vLLM / Ollama abstraction
│   ├── model_inference_lora.py       # Direct LoRA inference (fallback)
│   ├── db.py                         # SQLite schema + queries (WAL mode)
│   ├── account_config.py             # Equity-based param tiers + upgrade alerts
│   ├── portfolio_overseer.py         # Sector caps, correlation guards
│   ├── allocation_controller.py      # Tier 1→3 position sizing
│   ├── alerts.py                     # Email + JSONL alert dispatch
│   ├── health_server.py              # HTTP /health /status /metrics
│   └── start_vllm_server.sh          # vLLM startup script (called by systemd)
├── finetune/
│   ├── fine_tune_llm.py              # Manual fine-tune entry point
│   ├── data_collection.py            # Pre-training data collector
│   ├── data/finance_tuning/
│   │   ├── training_data.json
│   │   └── validation_report.json
│   └── finance_qwen_32b_lora_*/      # Timestamped LoRA checkpoints
│   └── finance_qwen_32b_merged_*/    # Merged BF16 models (vLLM-ready)
│   └── finance_qwen_32b_merged_latest -> ...  # Symlink vLLM always reads
├── services/
│   ├── ai-inference-server.service   # Shared vLLM server
│   ├── ai-trading-bot.service        # Paper stock bot
│   ├── ai-options-bot.service        # Paper options bot
│   ├── ai-trading-bot-live.service   # Live stock bot
│   ├── ai-options-bot-live.service   # Live options bot (equity-gated)
│   └── ai-health-server.service      # Health + metrics server
├── tests/                            # pytest suite
├── logs/                             # Runtime: trading.db, *.jsonl (gitignored)
├── .env                              # Paper trading credentials (gitignored)
├── .env.live                         # Live trading credentials (gitignored)
├── .env.example                      # Template
└── requirements.txt
```

## Hardware Requirements

This system is designed for and tested on an **NVIDIA GB10 Grace Blackwell** (NVIDIA ZGX Nano):

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 40 GB | 128 GB unified |
| RAM | 64 GB | 128 GB unified |
| Disk | 150 GB | 500 GB+ |
| OS | Ubuntu 22.04 | Ubuntu 24.04 |

**Why 128 GB?** Qwen 32B fine-tuning needs ~130 GB. The GB10's unified CPU/GPU memory pool handles this. vLLM is stopped before fine-tuning begins and restarted after — they never run concurrently.

Smaller GPUs can run the system but must lower `--gpu-memory-utilization` and may need a smaller base model (e.g., Qwen 7B).

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/WereAllWinners/ai-trade-agent.git
cd ai-trade-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For fine-tuning (unsloth + bitsandbytes):
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env with your Alpaca paper credentials and model paths
```

Key variables in `.env`:
```env
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
PAPER_TRADING=true

INFERENCE_BACKEND=vllm          # or ollama for lighter-weight testing
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct
INFERENCE_TIMEOUT=300

LORA_ADAPTER_PATH=finetune/finance_qwen_32b_lora_latest

FINNHUB_API_KEY=your_key        # Economic calendar guard
POLYGON_API_KEY=your_key        # Options flow data

# Email alerts (optional — Gmail App Password)
ALERT_EMAIL_FROM=bot@gmail.com
ALERT_EMAIL_TO=you@gmail.com
ALERT_SMTP_HOST=smtp.gmail.com
ALERT_SMTP_PORT=587
ALERT_SMTP_USER=bot@gmail.com
ALERT_SMTP_PASSWORD=xxxx_xxxx_xxxx_xxxx
```

### 3. Initial model setup

**Option A — vLLM (recommended, production)**

vLLM serves a merged BF16 model. After the first fine-tuning run, `model_promoter.py` merges the LoRA into the base weights and creates the `finance_qwen_32b_merged_latest` symlink that vLLM reads.

```bash
# Kick off the first fine-tune to bootstrap the merged model
python finetune/fine_tune_llm.py \
  --data finetune/data/finance_tuning/training_data.json \
  --epochs 1
# model_promoter.py runs automatically and creates the merged model + symlink
```

Then start the inference server:
```bash
sudo systemctl start ai-inference-server
```

**Option B — Ollama (simpler, lower hardware requirement)**

```bash
ollama pull qwen2.5:32b
# Set INFERENCE_BACKEND=ollama in .env
```

### 4. Start the paper trading bots

```bash
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-inference-server ai-trading-bot ai-options-bot ai-health-server
```

Monitor:
```bash
sudo journalctl -u ai-trading-bot -u ai-options-bot -f
curl http://localhost:8765/health
```

## Daily Workflow

| Time (EST) | Event |
|---|---|
| 9:30 AM | Market opens — agents start trading sessions |
| Every 30 min | Stock bot: discover candidates, LLM decision, bracket order |
| Every 60 min | Options bot: screen for calls/puts, LLM decision, order |
| 4:00 PM | Market closes |
| 5:00 PM | Stock bot: performance analysis + online training |
| 5:30 PM | Options bot: performance analysis |
| 8:00 PM | Stock bot: fine-tune (1 epoch) → eval → promote → merge → vLLM reload |
| 9:00 PM | Options bot: fine-tune (1 epoch) → eval → promote → merge → vLLM reload |
| Saturday 10 AM | Weekly report + deep research |
| Saturday 11 AM | Weekend strategist + major fine-tune (2 epochs) |

**vLLM restart sequence after fine-tuning:**
1. Daemon calls `stop_for_finetuning()` → vLLM stops
2. Fine-tuning subprocess runs (Qwen 32B loads into the freed memory)
3. Fine-tuning exits → 90-second drain wait (CUDA memory consolidation)
4. `model_promoter.py` merges LoRA → BF16, updates symlink
5. Daemon calls `start_after_finetuning()` → vLLM restarts on new merged model

## Risk Controls

### Stock Bot

| Control | Standard account (≥$5k) | Small account (<$5k) |
|---|---|---|
| Stop loss | -7% | -7% |
| Take profit | +15% | +15% |
| Min confidence | 60% | 70% |
| Max position size | 5% | 5% |
| Max daily trades | 10 | 10 |
| Same-day close | Allowed | **Blocked** (PDT guard) |
| Cooldown between buys | None | 60 min |
| Daily loss circuit breaker | -5% | -10% |

### Options Bot

| Control | Standard account (≥$25k) | Small account (<$25k) |
|---|---|---|
| Portfolio allocation | 10–15% | Up to 70% |
| Max position size | 3% | 20% |
| Min confidence | 75% | 82% |
| Max daily trades | 5 | 2 |
| DTE range | 7–45 days | 7–30 days |
| Target delta | 0.30 | 0.30 |
| Stop loss | -50% | -50% |
| Take profit | +50% | +50% |
| Same-day close | Allowed | **Blocked** (PDT guard) |

### Portfolio-Level Controls

- **Sector cap**: 30% max in any single GICS sector (portfolio overseer)
- **Correlation guard**: max 4 open positions in the same sector
- **Allocation tiers**: position sizing unlocked progressively as the model proves itself

| Tier | Min trades | Sharpe | Max DD | Position size |
|---|---|---|---|---|
| 1 (Exploration) | 0 | — | — | 1% |
| 2 (Ramping) | 20 | ≥1.0 | ≤8% | 3% |
| 3 (Production) | 50 | ≥1.5 | ≤5% | 5% |

## Live Trading Setup

The live and paper environments run simultaneously. Paper trading always continues — it provides training data and model improvement regardless of live status.

### Equity Thresholds

| Threshold | Default | Effect |
|---|---|---|
| `OPTIONS_LIVE_THRESHOLD` | $5,000 | Options bot switches from paper to live |
| `LIVE_EQUITY_THRESHOLD` | $25,000 | Full allocation params unlock; stock PDT guard relaxes |

The stock live bot trades at **any equity level**. PDT compliance is enforced by `no_same_day_close=True` (never sell a position opened the same day) rather than blocking buying.

### Live Credentials

Create `.env.live` (never commit this file):
```env
ALPACA_API_KEY=your_live_api_key
ALPACA_SECRET_KEY=your_live_secret_key
PAPER_TRADING=false
OPTIONS_LIVE_THRESHOLD=5000
MIN_CASH_RESERVE=75
```

### Enable Live Services

```bash
sudo systemctl enable --now ai-trading-bot-live
# Options live bot — starts paper internally until $5k equity is reached:
sudo systemctl enable --now ai-options-bot-live
```

The live bot reads `.env` first (shared infra: model paths, vLLM URL, email alerts) then `.env.live` overrides credentials and `PAPER_TRADING=false`.

### Order Types

- **Whole shares**: bracket order (limit entry + stop-loss + take-profit legs)
- **Fractional shares**: market order only (Alpaca requirement — bracket orders reject fractional quantities)

### Live Trading Checklist

Before enabling the live services, verify:

- [ ] Paper bot has traded for at least 30 days without crashes
- [ ] Win rate and Sharpe ratio are acceptable over a full market cycle
- [ ] Circuit breaker fires correctly (tested manually or observed in paper)
- [ ] Health endpoint at `localhost:8765/health` returns `"ok"`
- [ ] You have reviewed Alpaca's TOS for automated trading
- [ ] You understand PDT rules: buying is unrestricted; selling a position the same day it was opened counts as a day trade
- [ ] Emergency kill-switch procedure is documented: `sudo systemctl stop ai-trading-bot-live ai-options-bot-live`

## Model Pipeline

```
Daily trading → SQLite (decisions + outcomes)
                     ↓
        training_data_builder.py
        (builds training_data.json)
                     ↓
          fine_tune_llm.py (SFT + DPO)
          [vLLM stopped, full GPU free]
                     ↓
            eval_model.py (A/B test)
            candidate vs. current adapter
                     ↓
          model_promoter.py (if promoted)
          merge LoRA → BF16 full weights
          update finance_qwen_32b_merged_latest symlink
                     ↓
       vLLM restarts, loads new merged model
       (90s drain wait before restart)
```

LoRA adapters and merged models are timestamped:
- `finetune/finance_qwen_32b_lora_YYYYMMDD_HHMMSS/`
- `finetune/finance_qwen_32b_merged_YYYYMMDD_HHMMSS/`
- `finetune/finance_qwen_32b_merged_latest` → symlink vLLM reads

## Managing Services

```bash
# Status
sudo systemctl status ai-inference-server ai-trading-bot ai-options-bot

# Logs (live tail)
sudo journalctl -u ai-trading-bot -f
sudo journalctl -u ai-trading-bot-live -f
sudo journalctl -u ai-inference-server -f

# Restart
sudo systemctl restart ai-trading-bot
sudo systemctl restart ai-inference-server   # reloads the merged model

# Health check
curl http://localhost:8765/health
curl http://localhost:8765/status
curl http://localhost:8765/metrics    # Prometheus format

# Emergency stop (all trading)
sudo systemctl stop ai-trading-bot ai-options-bot ai-trading-bot-live ai-options-bot-live
```

## Troubleshooting

### System crashes / unexpected reboots

The most common cause is vLLM starting immediately after fine-tuning on a fragmented CUDA memory pool. The fixes are in place:
- 90-second drain wait in `inference_client.start_after_finetuning()`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` in `start_vllm_server.sh`

If crashes continue, reduce `--gpu-memory-utilization` in `start_vllm_server.sh` (try 0.55).

Check the BIOS "Restore on AC Power Loss" setting to minimize downtime after hard crashes.

### vLLM connection refused

```bash
sudo systemctl status ai-inference-server
sudo journalctl -u ai-inference-server -n 50
# Model takes ~6-8 minutes to load after restart — agents will retry automatically
```

### Live bot reading paper account

Verify `.env.live` has the correct variable names and `PAPER_TRADING=false`:
```bash
grep -E "ALPACA_API_KEY|ALPACA_SECRET_KEY|PAPER_TRADING" /path/to/.env.live
```

### Circuit breaker fires immediately

The `daily_state` table stores the start-of-day equity per `(bot, source)` pair. If a stale paper equity is contaminating the live circuit breaker, clear it:
```bash
sqlite3 logs/trading.db "DELETE FROM daily_state WHERE source='live' AND trade_date=date('now');"
sudo systemctl restart ai-trading-bot-live
```

### "Position too small" / fractional order rejected

Alpaca fractional shares require simple market orders — bracket orders (stop-loss + take-profit) are rejected for non-integer quantities. The agent handles this automatically: fractional quantities use `MarketOrderRequest`, whole shares use `LimitOrderRequest` with bracket legs.

### Model eval always grades F

The evaluator runs while vLLM is down (fine-tuning stops it). If vLLM isn't back up when eval runs, all inference attempts fail and the grade is 0%. Both candidate and current score 0%, so the model is promoted (0% ≥ 0%). This is expected behavior during the shutdown window — the new model is still used.

### Fine-tuning OOM

Fine-tuning Qwen 32B at 4-bit needs ~130 GB. The vLLM server must be stopped first (the daemon does this automatically). If you trigger fine-tuning manually, stop vLLM first:
```bash
sudo systemctl stop ai-inference-server
python finetune/fine_tune_llm.py --data finetune/data/finance_tuning/training_data.json --epochs 1
sudo systemctl start ai-inference-server
```

### Run tests

```bash
pytest tests/ -v
```

## Alerts

The `alerts.py` module dispatches to two channels:

| Channel | Config | Events |
|---|---|---|
| File log (`logs/alerts.jsonl`) | Always active | All events |
| Email (SMTP) | `ALERT_EMAIL_*` in `.env` | All events |

Alert events: `circuit_breaker` (CRITICAL), `trade_executed` (INFO), `account_upgrade` (INFO — equity crossed threshold), `trade_failed` (WARNING).

```bash
# View recent alerts
tail -20 logs/alerts.jsonl | python -m json.tool
```

## Disclaimer

This is an experimental trading system. **Use at your own risk.**

- Start with paper trading only (`PAPER_TRADING=true`)
- Never invest more than you can afford to lose
- Past paper trading performance does not guarantee live results
- The author is not responsible for any financial losses
- This is not financial advice
- **PDT Rule**: Selling a US stock position the same day you bought it counts as a day trade. More than 3 day trades in 5 business days in an account under $25,000 triggers a 90-day trading restriction. The small-account mode enforces `no_same_day_close` to prevent this.
- **Options Approval**: Buying options requires Level 2 options approval from Alpaca. Apply in your account settings before running the options bot live.
- Review Alpaca's Terms of Service regarding automated trading before going live.

## Resources

- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Unsloth Fine-tuning](https://github.com/unslothai/unsloth)
- [Qwen 2.5 Model](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
