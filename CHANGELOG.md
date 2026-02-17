# Changelog

## 2026-02-17 - Data Fetching & Fine-tuning Improvements

### Fixed
- **Alpaca API compatibility**: Updated from old `bars_data[symbol].df` to new `bars_data.df` format
- **Data availability**: Added automatic yfinance fallback for symbols not on Alpaca free tier
- **Options bot timeout**: Increased from 5min to 20min to allow model loading
- **Fine-tuning checkpoints**: Now save every 500 steps (~4hr) instead of per epoch (~78hr)

### Added
- **Multi-source data pipeline**: Stock bot tries Alpaca hourly → Alpaca daily → yfinance
- **yfinance fallback**: Can now trade ANY symbol (AMZN, JNJ, etc.) not just Alpaca's free tier

### System Status
- ✅ Stock trading bot: Fully operational
- ✅ Options trading bot: Operational (needs GPU free from fine-tuning)
- ✅ Fine-tuning: In progress (25,169 examples, saving checkpoints every 500 steps)
- ✅ Both bots restart automatically on server reboot
- ✅ GitHub repo synced

### Next Steps
- Wait for fine-tuning to complete
- Monitor first trades tomorrow (Wednesday 9:30 AM EST)
- Consider scheduling fine-tuning for off-hours to free GPU for options trading
