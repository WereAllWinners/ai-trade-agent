# AI Trade Agent 🤖📈

Autonomous AI-powered trading system with dual agents for stocks and options, featuring self-improving capabilities through daily model fine-tuning.

## 🌟 Features

- **Dual-Agent System**: Separate bots for stock trading and options trading
- **Autonomous Discovery**: Automatically finds trading opportunities using technical analysis
- **AI-Powered Decisions**: Uses fine-tuned Qwen 2.5 32B LLM for trade analysis
- **Self-Improving**: Daily performance analysis and model fine-tuning
- **Weekend Deep Research**: Comprehensive market analysis and model updates on Saturdays
- **Risk Management**: Strict position sizing, stop losses, and portfolio limits
- **24/7 Operation**: Runs continuously with systemd services
- **World-Class Learning**: Learns from congressional trades, 13F filings, and proven strategies

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
  - Target delta: 0.30
- **Schedule**:
  - 5:30 PM EST: Daily performance analysis
  - 9:00 PM EST: Model fine-tuning

### Weekend Strategy (Saturday 10:00 AM EST)
1. 🔬 Deep market research and trend analysis
2. 📊 Collect world-class training data (congressional trades, 13F filings, proven strategies)
3. 🎓 Fine-tune model with new insights (2 epochs, ~45 minutes)
4. ✅ Model ready and optimized before Monday market open!

## 🚀 Quick Start

### Prerequisites

- **Hardware**: 
  - CUDA-capable GPU with 8GB+ VRAM (recommended: 32GB+ for Qwen 32B)
  - 64GB+ RAM
  - 100GB+ free disk space (for model files and data)
- **Software**:
  - Python 3.10 or higher
  - CUDA 12.1+ (for GPU acceleration)
  - Linux (Ubuntu 22.04+ recommended)
- **API**: Alpaca trading account (paper or live)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/WereAllWinners/ai-trade-agent.git
cd ai-trade-agent
```

2. **Create and activate conda environment (recommended):**
```bash
conda create -n trading python=3.12
conda activate trading
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
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

This project uses **Qwen 2.5 32B Instruct** model, fine-tuned with LoRA (Low-Rank Adaptation) for efficient training.

### Model Details

- **Base Model**: `unsloth/qwen2.5-32b-instruct-bnb-4bit`
- **Model Size**: ~20GB (4-bit quantized)
- **Parameters**: 32 billion
- **Architecture**: Qwen 2.5 with instruction tuning
- **Quantization**: 4-bit via bitsandbytes (for memory efficiency)
- **Fine-tuning Method**: LoRA adapters (~2GB)

### Option 1: Automatic Download (Recommended for First-Time Users)

The model will automatically download when you run fine-tuning:
```bash
python3 finetune/fine_tune_llm.py \
  --data finetune/data/finance_tuning/training_data.json \
  --epochs 3
```

The model will be cached in `~/.cache/huggingface/hub/` and reused for future runs.

### Option 2: Manual Download (Recommended for Advanced Users)

**Using Hugging Face CLI:**
```bash
# Install Hugging Face Hub
pip install huggingface-hub[cli]

# Login (optional - only needed for gated models)
huggingface-cli login

# Download the model
huggingface-cli download unsloth/qwen2.5-32b-instruct-bnb-4bit \
  --local-dir ~/.cache/huggingface/hub/models--unsloth--qwen2.5-32b-instruct-bnb-4bit \
  --local-dir-use-symlinks False
```

**Using Python:**
```python
from huggingface_hub import snapshot_download

# Download model
model_path = snapshot_download(
    repo_id="unsloth/qwen2.5-32b-instruct-bnb-4bit",
    cache_dir="~/.cache/huggingface/hub",
    resume_download=True  # Resume if interrupted
)

print(f"Model downloaded to: {model_path}")
```

**Download Progress:**
- Total size: ~20GB
- Time: 10-30 minutes (depending on internet speed)
- Location: `~/.cache/huggingface/hub/models--unsloth--qwen2.5-32b-instruct-bnb-4bit/`

### Verify Model Download
```bash
# Check if model exists
ls -lh ~/.cache/huggingface/hub/models--unsloth--qwen2.5-32b-instruct-bnb-4bit/

# Should show files like:
# - model-00001-of-00004.safetensors
# - model-00002-of-00004.safetensors
# - model-00003-of-00004.safetensors
# - model-00004-of-00004.safetensors
# - config.json
# - tokenizer.json
# - etc.

# Check total size
du -sh ~/.cache/huggingface/hub/models--unsloth--qwen2.5-32b-instruct-bnb-4bit/
# Should be ~20GB
```

### Fine-tuning the Model

#### Step 1: Collect Training Data
```bash
# Collect financial data from multiple sources
python3 finetune/data_collection.py
```

This creates `finetune/data/finance_tuning/training_data.json` with examples from:
- Your Alpaca portfolio performance
- High-volume and trending stocks
- Technical indicators and market conditions
- Dynamic symbol discovery

**For world-class training data (congressional trades, 13F filings, proven strategies):**
```bash
# Edit data_collection.py to use WorldClassDataCollector
# Then run it to get elite training data
python3 finetune/data_collection.py
```

#### Step 2: Fine-tune the Model

**Fresh Training (First Time):**
```bash
python3 finetune/fine_tune_llm.py \
  --data finetune/data/finance_tuning/training_data.json \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 2e-4
```

**Continue Training (Add New Knowledge to Existing Model):**
```bash
python3 finetune/fine_tune_llm.py \
  --data finetune/data/finance_tuning/training_data.json \
  --continue-from finetune/finance_qwen_32b_lora \
  --epochs 2
```

**Fine-tuning Options:**
```bash
--data           Path to training data JSON file (required)
--continue-from  Path to existing LoRA adapter (optional, for continued training)
--base-model     Base model to use (default: qwen2.5-32b-instruct-bnb-4bit)
--output         Output directory for LoRA adapter
--epochs         Number of training epochs (default: 3)
--batch-size     Per-device batch size (default: 2)
--learning-rate  Learning rate (default: 2e-4)
```

**Training Time**: 
- Fresh training: ~1-2 hours (3 epochs)
- Continue training: ~30-45 minutes (2 epochs)
- Depends on: dataset size, GPU, batch size

**Output**: LoRA adapter saved to `finetune/finance_qwen_32b_lora_[timestamp]/`

#### Step 3: Verify Model
```bash
# Test the fine-tuned model
python3 scripts/model_inference_lora.py
```

### Understanding the Model Architecture
```
┌─────────────────────────────────────────┐
│   Qwen 2.5 32B Base Model (20GB)        │
│   - Pre-trained on general knowledge    │
│   - Instruction-tuned                   │
│   - 4-bit quantized for efficiency      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   LoRA Adapter (~2GB)                   │
│   - Your custom trading knowledge       │
│   - Portfolio performance lessons       │
│   - Congressional trades                │
│   - 13F filings & proven strategies     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Fine-tuned Trading Model              │
│   - Ready for autonomous trading        │
│   - Self-improving daily                │
└─────────────────────────────────────────┘
```

**Why LoRA?**
- ✅ Train in minutes instead of hours
- ✅ Small adapter size (2GB vs 50GB full model)
- ✅ Easy to update daily with new trading data
- ✅ Can run on consumer GPUs
- ✅ Multiple adapters for different strategies

### Model Update Schedule

**Daily (Weekdays at 8:00 PM EST):**
- Quick fine-tuning with day's trading results
- 1 epoch, ~15 minutes
- Incremental learning from wins/losses

**Weekly (Saturday at 10:00 AM EST):**
- Comprehensive training with world-class data
- 2 epochs, ~45 minutes
- Major knowledge updates before Monday

## 📁 Project Structure
```
ai-trade-agent/
├── scripts/                          # Trading agents and utilities
│   ├── autonomous_agent.py           # Stock trading logic
│   ├── options_agent.py              # Options trading logic
│   ├── trading_daemon.py             # Stock trading daemon (24/7)
│   ├── options_daemon.py             # Options trading daemon (24/7)
│   ├── model_inference_lora.py       # LLM inference with LoRA
│   ├── performance_analyzer.py       # Stock performance analysis
│   ├── options_performance_analyzer.py
│   ├── stock_discovery.py            # Find trading opportunities
│   ├── weekend_strategist.py         # Weekend deep analysis
│   ├── finetune_model.py            # Daily fine-tuning script
│   └── data_utils.py                # Utility functions
├── finetune/                        # Model training system
│   ├── data_collection.py           # Collect training data
│   ├── fine_tune_llm.py             # Fine-tune model
│   ├── data/                        # Training data storage
│   │   └── finance_tuning/
│   │       ├── training_data.json
│   │       ├── validation_report.json
│   │       └── portfolio_analysis.json
│   └── finance_qwen_32b_lora/       # Fine-tuned model (not in repo)
├── services/                        # Systemd service files
│   ├── ai-trading-bot.service       # Stock trading service
│   └── ai-options-bot.service       # Options trading service
├── logs/                            # Trade history (not in repo)
│   ├── trade_log.jsonl              # Stock trade history
│   ├── options_trade_log.jsonl      # Options trade history
│   ├── performance_metrics.json
│   └── options_performance_metrics.json
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
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

# Test weekend analysis
python3 scripts/weekend_strategist.py
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

# Both bots
sudo journalctl -u ai-trading-bot.service -u ai-options-bot.service -f
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
cat logs/performance_metrics.json | jq

# Options performance
cat logs/options_performance_metrics.json | jq

# Recent trades
tail -20 logs/trade_log.jsonl | jq
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
  "worst_trade": -0.07,
  "sharpe_ratio": 1.34,
  "max_drawdown": -0.12
}
```

## 🔄 Self-Improvement Loop

The bots continuously learn from their own performance:

1. **Trade Execution** (Market hours)
   - Bots make trades based on AI analysis
   - All trades logged with reasoning and confidence

2. **Daily Analysis** (5:00 PM / 5:30 PM EST)
   - Analyze day's trades
   - Identify winners (>30% profit) and losers (<-30% loss)
   - Extract patterns and lessons

3. **Training Data Generation**
   - Winners → Positive examples (what worked)
   - Losers → Negative examples (what to avoid)
   - Strong positions (>40% unrealized) → Current best practices

4. **Model Fine-tuning** (8:00 PM / 9:00 PM EST)
   - Update model with new examples
   - Reinforce successful patterns
   - Learn to avoid losing patterns
   - Takes 15-30 minutes

5. **Weekend Deep Learning** (Saturday 10:00 AM EST)
   - Comprehensive market analysis
   - Collect data from elite sources (congressional trades, 13F filings)
   - Major model update (45 minutes)
   - Ready for Monday market open

6. **Next Day Trading**
   - Trade with improved model
   - Better pattern recognition
   - Repeat cycle

**Result**: The model gets smarter every day! 📈

## 🛡️ Risk Management

### Stock Trading Controls
- Max 5% per position
- Max 10 trades per day
- Stop loss: -7%
- Take profit: +15%
- Min confidence: 60%
- Max portfolio heat: 50%

### Options Trading Controls
- Max 15% portfolio allocation to options
- Max 3% per options position
- Max 5 trades per day
- Stop loss: -50%
- Take profit: +50%
- Min confidence: 75%
- DTE: 7-45 days
- Target delta: 0.30
- 1-hour cooldown per symbol

## 🔧 Configuration

Edit trading parameters in the respective agent files:

**Stock Trading** (`scripts/autonomous_agent.py`):
```python
self.params = {
    'max_position_size': 0.05,      # 5% per position
    'min_confidence': 0.60,         # 60% minimum
    'max_daily_trades': 10,
    'stop_loss': -0.07,             # -7%
    'take_profit': 0.15,            # +15%
}
```

**Options Trading** (`scripts/options_agent.py`):
```python
self.params = {
    'max_portfolio_allocation': 0.15,  # 15% max in options
    'max_position_size': 0.03,         # 3% per position
    'min_confidence': 0.75,            # 75% minimum
    'max_daily_trades': 5,
    'stop_loss': -0.50,                # -50%
    'take_profit': 0.50,               # +50%
    'dte_min': 7,                      # Minimum days to expiration
    'dte_max': 45,                     # Maximum days to expiration
    'target_delta': 0.30,              # Target option delta
}
```

## 🐛 Troubleshooting

### Issue: Model not loading
```bash
# Check if model directory exists
ls -la ~/.cache/huggingface/hub/models--unsloth--qwen2.5-32b-instruct-bnb-4bit/

# Check if LoRA adapter exists
ls -la finetune/finance_qwen_32b_lora/

# If empty, download base model or fine-tune
python3 finetune/fine_tune_llm.py --data finetune/data/finance_tuning/training_data.json
```

### Issue: CUDA not available
```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
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
sudo journalctl -u ai-trading-bot.service -n 100

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
# Reduce batch size in fine_tune_llm.py:
python3 finetune/fine_tune_llm.py \
  --data finetune/data/finance_tuning/training_data.json \
  --batch-size 1 \
  --epochs 2
```

## ⚠️ Disclaimer

This is an experimental trading system. **Use at your own risk.**

- ⚠️ Start with **paper trading** only
- ⚠️ Never invest more than you can afford to lose
- ⚠️ Past performance does not guarantee future results
- ⚠️ The author is not responsible for any financial losses
- ⚠️ This is not financial advice
- ⚠️ Always do your own research
- ⚠️ Test thoroughly before using real money
- ⚠️ Markets are inherently risky

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
- [Hugging Face Hub](https://huggingface.co/docs/hub)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 🙏 Acknowledgments

- Alpaca API for trading infrastructure
- Qwen team for the base language model
- Unsloth for efficient fine-tuning tools
- Hugging Face for model hosting
- yfinance for market data

---

**Built with ❤️ by your fellow hobbyist**