# CLAUDE.md — ai-trade-agent

Autonomous AI-powered trading system using a self-improving Qwen 32B LLM. Makes real trades on Alpaca Markets, learns from daily outcomes via fine-tuning.

## Project Layout

```
scripts/
  agents/          # Core trading logic (autonomous_agent.py, options_agent.py)
  analysis/        # Post-market analysis, backtesting, weekend strategist
  data/            # Market discovery, news, congressional tracker, options flow
  training/        # Training data builder, online trainer, model evaluator
  tools/           # Operational utilities (liquidate, close all, PDT fix)
  trading_daemon.py       # Stock agent 24/7 orchestration loop
  options_daemon.py       # Options agent 24/7 orchestration loop
  model_inference_lora.py # Qwen 32B + LoRA inference (4-bit quantized, cached)
  inference_client.py     # Abstraction: ollama (default) or vLLM backend
  db.py                   # SQLite schema + queries (8 tables, WAL mode)
  portfolio_overseer.py   # Sector caps, correlation guards
  allocation_controller.py# Tiered position sizing (Tier 1→3: 1%→3%→5%)
  health_server.py        # HTTP /health /status /metrics on port 8765
  alerts.py               # Email (SMTP) + JSONL file alerts
finetune/
  fine_tune_llm.py        # Main training script — SFT + DPO, LoRA adapters
  data_collection.py      # Pre-training data from Alpaca, yfinance, Polygon
  data/finance_tuning/    # training_data.json, validation_report.json
  finance_qwen_32b_lora_*/ # Timestamped LoRA adapter checkpoints
services/          # systemd service definitions (stock, options, health)
tests/             # pytest suite (9+ test files)
logs/              # Runtime: trading.db, *.jsonl logs, heartbeats (not in repo)
```

## Tech Stack

- **LLM**: Qwen 2.5 32B (4-bit via bitsandbytes) + LoRA adapters (unsloth/peft/trl)
- **Broker**: Alpaca (alpaca-py) — paper trading default, live behind safety gates
- **Market data**: yfinance, Alpaca historical data
- **Database**: SQLite (WAL, 8 tables: decisions, orders, positions, outcomes, cash_reservations, training_records, portfolio_changes, risk_events)
- **Inference backends**: ollama (default) or vLLM (3-5x faster, set via env)
- **Orchestration**: systemd services (24/7 daemons)
- **Python**: 3.10+ (tested 3.12), venv at `venv/`

## Daily Workflow

1. **9:30 AM EST** — Market opens
2. **Every 30 min (stock) / 60 min (options)** — Trading session:
   - Discover candidates (technicals + news + congressional trades)
   - LLM decision with structured output: action, symbol, confidence, stop/target
   - Portfolio overseer veto (sector cap 30%, max 4 correlated positions)
   - Allocation controller sizing (tier-based 1–5%)
   - Debate mechanism for confidence >90%
   - Bracket order via Alpaca, log to SQLite + JSONL
3. **5:00 / 5:30 PM** — Performance analysis (Sharpe, win rate, max drawdown)
4. **8:00 / 9:00 PM** — Daily fine-tuning (~15 min, 1 epoch, saves LoRA checkpoint)
5. **Saturday 10:00 AM** — Deep weekend analysis + major fine-tune (2 epochs, ~45 min)

## Key Risk Controls

| Control | Stock | Options |
|---|---|---|
| Stop loss | -7% | -50% |
| Take profit | +15% | +50% |
| Min confidence | 60% | 75% |
| Max position size | 5% | 3% |
| Max daily trades | 10 | 5 |
| Daily drawdown circuit breaker | -5% | -5% |
| Options portfolio allocation | — | 10–15% |
| Options DTE | — | 7–45 days |

## Allocation Tiers

- **Tier 1** (Exploration): 1% per position
- **Tier 2** (Ramping): 3% — requires ≥20 trades, Sharpe ≥1.0, max DD ≤8%
- **Tier 3** (Production): 5% — requires ≥50 trades, Sharpe ≥1.5, max DD ≤5%

## Environment Config

Copy `.env.example` → `.env`. Key variables:

```bash
APCA_API_KEY_ID=...
APCA_API_SECRET_KEY=...
PAPER_TRADING=true          # false only after live trading checklist passed
BASE_MODEL=Qwen/Qwen2.5-32B-Instruct
LORA_ADAPTER_PATH=finetune/finance_qwen_32b_lora_YYYYMMDD_HHMMSS
INFERENCE_BACKEND=ollama    # or vllm
MAX_SECTOR_PCT=0.30
MAX_CORR_POSITIONS=4
ALLOC_TIER1_PCT=0.01
HEALTH_PORT=8765
```

## Running

```bash
# Start daemons (systemd)
sudo systemctl start ai-trading-bot
sudo systemctl start ai-options-bot
sudo systemctl start ai-health-server

# Manual run (dev)
python scripts/trading_daemon.py
python scripts/options_daemon.py

# Health check
curl http://localhost:8765/health
curl http://localhost:8765/status
curl http://localhost:8765/metrics   # Prometheus format

# Tests
pytest tests/ -v

# Manual fine-tune
python finetune/fine_tune_llm.py

# Emergency
python scripts/tools/close_all_positions.py
python scripts/tools/liquidate_to_cash.py
```

## Live Trading Checklist (before PAPER_TRADING=false)

- [ ] Account equity ≥ $25,000 (PDT rule)
- [ ] Options Level 2+ approval on Alpaca
- [ ] ≥1 month continuous paper trading
- [ ] Acceptable win rate and Sharpe ratio
- [ ] Circuit breaker tested
- [ ] Manual kill-switch procedure documented

## Hardware Requirements

- GPU: 8GB+ VRAM minimum (32GB+ recommended for 32B model)
- RAM: 64GB+
- Disk: 100GB+ (models + checkpoints + data)
- Tested on GB10 with 128GB unified memory (runs both agents concurrently)

## Important Notes

- `model_inference_lora.py` caps GPU memory at 45GB per process to allow two agents on shared GPU
- SQLite uses WAL mode — safe for concurrent daemon writes
- LoRA adapters are timestamped; update `LORA_ADAPTER_PATH` in `.env` after fine-tuning
- `logs/` directory is gitignored — contains the live DB, trade logs, heartbeats
- `training_data_builder.py` reads from SQLite `decisions` + `outcomes` tables, outputs to `finetune/data/finance_tuning/training_data.json`
