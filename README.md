# AI Trade Agent 🤖📈

Autonomous AI-powered trading system with dual agents for stocks and options, featuring self-improving capabilities through daily model fine-tuning.

## 🌟 Features

- **Dual-Agent System**: Separate bots for stock trading and options trading
- **Autonomous Discovery**: Automatically finds trading opportunities using technical analysis
- **AI-Powered Decisions**: Uses fine-tuned Qwen LLM for trade analysis
- **Self-Improving**: Daily performance analysis and model fine-tuning
- **Risk Management**: Strict position sizing, stop losses, and portfolio limits
- **24/7 Operation**: Runs continuously with systemd services

## 📊 System Architecture

### Stock Trading Bot
- **Trading Frequency**: Every 30 minutes during market hours (9:30 AM - 4:00 PM EST)
- **Position Sizing**: Max 5% of portfolio per position
- **Daily Trade Limit**: 10 trades maximum
- **Risk Controls**: 
  - Stop loss: -7%
  - Take profit: +15%
  - Min confidence: 60%
- **Schedule**:
  - 5:00 PM EST: Daily performance analysis
  - 8:00 PM EST: Model fine-tuning

### Options Trading Bot
- **Trading Frequency**: Every 60 minutes during market hours
- **Portfolio Allocation**: 10-15% of total portfolio
- **Position Sizing**: Max 3% per options position
- **Daily Trade Limit**: 5 options trades maximum
- **Risk Controls**:
  - Stop loss: -50%
  - Take profit: +50%
  - Min confidence: 75%
  - DTE range: 7-45 days
- **Schedule**:
  - 5:30 PM EST: Daily performance analysis
  - 9:00 PM EST: Model fine-tuning

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **GPU**: CUDA-capable GPU recommended (8GB+ VRAM)
- **Disk Space**: ~20GB for model files
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **API**: Alpaca trading account (paper or live)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ai-trade-agent.git
cd ai-trade-agent
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
nano .env  # Edit with your API keys
```

Your `.env` file should look like:
```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

Get your API keys from [Alpaca Markets](https://alpaca.markets/)

## 🧠 Model Setup & Fine-tuning

This project uses a fine-tuned Qwen 2.5 model. You have two options:

### Option 1: Use Pre-trained Base Model (Recommended for Testing)

The system will automatically download the base Qwen model on first run:
```bash
# The model will be cached in ~/.cache/huggingface/
# No additional setup needed
```

### Option 2: Fine-tune Your Own Model (Recommended for Production)

#### Step 1: Collect Training Data
```bash
# Collect financial data from various sources
python3 finetune/data_collection.py
```

This creates `finetune/data/finance_tuning/training_data.json` with examples from:
- Stock price movements
- Technical indicators
- Market sentiment
- Trading patterns

#### Step 2: Fine-tune the Model
```bash
# Install unsloth for efficient fine-tuning
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Run fine-tuning (requires GPU)
python3 finetune/fine_tune_llm.py \
  --data finetune/data/finance_tuning/training_data.json \
  --model unsloth/qwen2.5-0.5b-instruct-bnb-4bit \
  --output finetune/finance_qwen_32b_lora \
  --epochs 3
```

**Fine-tuning Options:**
```bash
--data      Path to training data JSON file
--model     Base model to fine-tune (default: qwen2.5-0.5b)
--output    Output directory for LoRA adapter
--epochs    Number of training epochs (default: 3)
```

**Training Time**: ~30 minutes on modern GPU (RTX 3090, A100, etc.)

**Output**: LoRA adapter saved to `finetune/finance_qwen_32b_lora/`

#### Step 3: Verify Model
```bash
# Test the fine-tuned model
python3 scripts/model_inference_lora.py
```

### Understanding the Model Architecture

The system uses **LoRA (Low-Rank Adaptation)** for efficient fine-tuning:
```
Base Model (Qwen 2.5 - 0.5B parameters)
    ↓
+ LoRA Adapter (~2GB - your custom trading knowledge)
    ↓
Fine-tuned Trading Model
```

**Why LoRA?**
- ✅ Train in minutes instead of hours
- ✅ Small adapter size (2GB vs 50GB full model)
- ✅ Easy to update daily with new trading data
- ✅ Can run on consumer GPUs

## 📁 Project Structure
```
ai-trade-agent/
├── scripts/
│   ├── autonomous_agent.py          # Stock trading logic
│   ├── options_agent.py              # Options trading logic
│   ├── trading_daemon.py             # Stock trading daemon
│   ├── options_daemon.py             # Options trading daemon
│   ├── model_inference_lora.py       # LLM inference with LoRA
│   ├── performance_analyzer.py       # Stock performance analysis
│   ├── options_performance_analyzer.py
│   ├── stock_discovery.py            # Find trading opportunities
│   ├── weekend_strategist.py         # Weekend analysis
│   └── finetune_model.py            # Daily fine-tuning script
├── finetune/
│   ├── data_collection.py           # Collect training data
│   ├── fine_tune_llm.py             # Fine-tune model
│   ├── data/                        # Training data storage
│   │   └── finance_tuning/
│   │       └── training_data.json
│   └── finance_qwen_32b_lora/       # Fine-tuned model (not in repo)
├── services/
│   ├── ai-trading-bot.service       # Systemd service for stocks
│   └── ai-options-bot.service       # Systemd service for options
├── logs/
│   ├── trade_log.jsonl              # Stock trade history
│   ├── options_trade_log.jsonl      # Options trade history
│   ├── performance_metrics.json
│   └── options_performance_metrics.json
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🎮 Running the Bots

### Manual Testing
```bash
# Test stock discovery
python3 scripts/stock_discovery.py

# Test stock trading
python3 scripts/autonomous_agent.py

# Test options trading
python3 scripts/options_agent.py

# Test model inference
python3 scripts/model_inference_lora.py
```

### Production Deployment (Systemd)

**Install services:**
```bash
# Copy service files
sudo cp services/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start bots
sudo systemctl start ai-trading-bot.service
sudo systemctl start ai-options-bot.service

# Enable auto-start on boot
sudo systemctl enable ai-trading-bot.service
sudo systemctl enable ai-options-bot.service
```

**Monitor logs:**
```bash
# Stock trading bot
sudo journalctl -u ai-trading-bot.service -f

# Options trading bot
sudo journalctl -u ai-options-bot.service -f
```

**Control services:**
```bash
# Stop
sudo systemctl stop ai-trading-bot.service

# Restart
sudo systemctl restart ai-trading-bot.service

# Check status
sudo systemctl status ai-trading-bot.service
```

## 📊 Performance Tracking

### View Metrics
```bash
# Stock performance
cat logs/performance_metrics.json

# Options performance
cat logs/options_performance_metrics.json
```

### Sample Metrics Output
```json
{
  "total_trades": 45,
  "closed_trades": 38,
  "open_trades": 7,
  "winners": 24,
  "losers": 14,
  "win_rate": 0.63,
  "avg_return": 0.042,
  "best_trade": 0.18,
  "worst_trade": -0.07
}
```

## 🔄 Self-Improvement Loop

The bots continuously learn from their own performance:

1. **Trade Execution** (Market hours)
   - Bots make trades based on AI analysis
   - All trades logged with reasoning

2. **Daily Analysis** (5:00 PM / 5:30 PM)
   - Analyze day's trades
   - Identify winners and losers
   - Extract patterns

3. **Training Data Generation**
   - Winners (>30% profit) → Positive examples
   - Losers (<-30% loss) → Negative examples
   - Strong positions (>40% unrealized) → Current learnings

4. **Model Fine-tuning** (8:00 PM / 9:00 PM)
   - Update model with new examples
   - Reinforce successful patterns
   - Learn to avoid losing patterns

5. **Next Day Trading**
   - Trade with improved model
   - Repeat cycle

## 🛡️ Risk Management

### Stock Trading Controls
- Max 5% per position
- Max 10 trades per day
- Stop loss: -7%
- Take profit: +15%
- Min confidence: 60%

### Options Trading Controls
- Max 15% portfolio allocation
- Max 3% per position
- Max 5 trades per day
- Stop loss: -50%
- Take profit: +50%
- Min confidence: 75%
- DTE: 7-45 days
- 1-hour cooldown per symbol

## 🔧 Configuration

Edit trading parameters in the respective agent files:

**Stock Trading** (`scripts/autonomous_agent.py`):
```python
self.params = {
    'max_position_size': 0.05,  # 5% per position
    'min_confidence': 0.60,     # 60% minimum
    'max_daily_trades': 10,
}
```

**Options Trading** (`scripts/options_agent.py`):
```python
self.params = {
    'max_portfolio_allocation': 0.15,  # 15% max
    'max_position_size': 0.03,          # 3% per position
    'min_confidence': 0.75,             # 75% minimum
    'max_daily_trades': 5,
}
```

## 🐛 Troubleshooting

### Issue: Model not loading
```bash
# Check if model directory exists
ls -la finetune/finance_qwen_32b_lora/

# If empty, either:
# 1. Download base model (auto on first run)
# 2. Fine-tune your own model (see Model Setup section)
```

### Issue: API authentication failed
```bash
# Verify .env file exists and has correct keys
cat .env

# Test Alpaca connection
python3 -c "from alpaca.trading.client import TradingClient; import os; from dotenv import load_dotenv; load_dotenv(); client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True); print(client.get_account())"
```

### Issue: Service won't start
```bash
# Check service status and logs
sudo systemctl status ai-trading-bot.service
sudo journalctl -u ai-trading-bot.service -n 50

# Common fixes:
# 1. Ensure logs directory exists
mkdir -p logs

# 2. Verify Python path in service file
which python3

# 3. Check file permissions
chmod +x scripts/*.py
```

### Issue: Out of memory during fine-tuning
```bash
# Use smaller model or reduce batch size in fine_tune_llm.py:
per_device_train_batch_size=1  # Reduce from 2 to 1
gradient_accumulation_steps=8  # Increase from 4 to 8
```

## ⚠️ Disclaimer

This is an experimental trading system. **Use at your own risk.**

- ⚠️ Start with **paper trading** only
- ⚠️ Never invest more than you can afford to lose
- ⚠️ Past performance does not guarantee future results
- ⚠️ The author is not responsible for any financial losses
- ⚠️ This is not financial advice
- ⚠️ Always do your own research

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly with paper trading
4. Submit a pull request

## 📚 Resources

- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [Unsloth Fine-tuning](https://github.com/unslothai/unsloth)
- [Qwen Model](https://huggingface.co/Qwen)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🙏 Acknowledgments

- Alpaca API for trading infrastructure
- Qwen team for the base language model
- Unsloth for efficient fine-tuning
- yfinance for market data

---

**Built with ❤️ by autonomous AI traders**

*Trade smart. Trade safe. Let AI do the heavy lifting.* 🚀
