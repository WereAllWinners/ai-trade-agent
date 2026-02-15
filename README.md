# AI Trade Agent 🤖📈

An autonomous AI-powered trading system that discovers stocks, analyzes market data, executes trades, and continuously improves itself through reinforcement learning.

## 🌟 Features

- **Dual-Agent System**: Separate bots for stock trading and options trading
- **Autonomous Discovery**: Automatically finds trading opportunities using technical analysis
- **AI-Powered Decisions**: Uses fine-tuned Qwen 32B LLM for trade analysis
- **Self-Improving**: Daily performance analysis and model fine-tuning
- **Risk Management**: Strict position sizing, stop losses, and portfolio limits
- **24/7 Operation**: Runs continuously with systemd services

## 📊 Architecture

### Stock Trading Bot
- Trades every 30 minutes during market hours
- Max 10 trades per day
- 5% position sizing
- Analyzes portfolio at 5:00 PM EST
- Fine-tunes model at 8:00 PM EST

### Options Trading Bot
- Trades every 60 minutes during market hours
- Limited to 10-15% of total portfolio
- Max 5 options trades per day
- 3% max per position
- Analyzes portfolio at 5:30 PM EST
- Fine-tunes model at 9:00 PM EST

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (for model inference)
- Alpaca API account (paper or live trading)
- ~20GB disk space for model files

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ai-trade-agent.git
cd ai-trade-agent
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Download and fine-tune your model (see Model Setup below)

### Configuration

Create a `.env` file with your credentials:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

### Running the Bots

**Manual testing:**
```bash
# Test stock discovery
python3 scripts/stock_discovery.py

# Test trading agent
python3 scripts/autonomous_agent.py

# Test options agent
python3 scripts/options_agent.py
```

**Production (systemd services):**
```bash
# Install services
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start bots
sudo systemctl start ai-trading-bot.service
sudo systemctl start ai-options-bot.service

# Enable auto-start
sudo systemctl enable ai-trading-bot.service
sudo systemctl enable ai-options-bot.service

# Monitor logs
sudo journalctl -u ai-trading-bot.service -f
```

## 🧠 Model Setup

This project uses a fine-tuned Qwen 32B model. Due to size constraints, the model weights are not included in this repo.

**Option 1: Use your own fine-tuned model**
1. Fine-tune Qwen using your trading data
2. Place adapter in `finetune/finance_qwen_32b_lora/`

**Option 2: Use base model**
1. The system will auto-download the base model
2. Performance will be lower without fine-tuning

## 📁 Project Structure
```
ai-trade-agent/
├── scripts/
│   ├── stock_discovery.py          # Finds trading opportunities
│   ├── autonomous_agent.py          # Stock trading logic
│   ├── options_agent.py             # Options trading logic
│   ├── model_inference_lora.py      # LLM inference
│   ├── performance_analyzer.py      # Stock performance analysis
│   ├── options_performance_analyzer.py
│   ├── trading_daemon.py            # Stock trading daemon
│   └── options_daemon.py            # Options trading daemon
├── finetune/
│   ├── data/                        # Training data
│   └── finance_qwen_32b_lora/       # Model weights (not in repo)
├── logs/
│   ├── trade_log.jsonl              # Stock trade history
│   ├── options_trade_log.jsonl      # Options trade history
│   └── discovered_opportunities.json
├── services/
│   ├── ai-trading-bot.service       # Systemd service for stocks
│   └── ai-options-bot.service       # Systemd service for options
└── README.md
```

## 🛡️ Risk Management

- **Position Sizing**: Max 5% per stock, 3% per option
- **Stop Losses**: -7% for stocks, -50% for options
- **Take Profits**: +15% for stocks, +50% for options
- **Daily Limits**: 10 stock trades, 5 options trades
- **Portfolio Allocation**: Options limited to 15% max

## 📈 Performance Tracking

Both bots generate daily performance reports:
- Win/loss rates
- Average returns
- Best/worst trades
- Portfolio allocation
- Risk metrics

View metrics:
```bash
cat logs/performance_metrics.json
cat logs/options_performance_metrics.json
```

## 🔄 Self-Improvement Loop

1. **Trade Execution**: Bots make trades during market hours
2. **Analysis (5:00 PM)**: Analyze day's performance
3. **Training Data Generation**: Create examples from winners/losers
4. **Fine-tuning (8:00 PM)**: Update model with new patterns
5. **Next Day**: Trade with improved model

## ⚠️ Disclaimer

This is an experimental trading system. Use at your own risk:
- Start with paper trading
- Never invest more than you can afford to lose
- Past performance doesn't guarantee future results
- The author is not responsible for any financial losses

## 🤝 Contributing

Pull requests welcome! Please:
1. Test thoroughly with paper trading
2. Document any new features
3. Follow existing code style
4. Add appropriate error handling

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Alpaca API for trading infrastructure
- Qwen team for the base language model
- unsloth for efficient fine-tuning
- yfinance for market data

---

**Built with ❤️ by autonomous AI traders**
