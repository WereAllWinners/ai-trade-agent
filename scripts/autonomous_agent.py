#!/usr/bin/env python3
"""
Autonomous Trading Agent - Continuously runs during market hours
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

import ollama
from stock_discovery import StockDiscovery
from decision_parser import parse_decision
from alerts import alert_circuit_breaker, alert_trade_executed, alert_trade_failed
import news_fetcher

OLLAMA_MODEL = "qwen3:8b"
_DEBATE_CONFIDENCE_THRESHOLD = 0.75   # Only debate high-conviction trades

def _json_default(obj):
    """JSON serializer for numpy scalar types that the stdlib encoder can't handle."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def get_trading_decision(prompt):
    """Get trading decision via Ollama (model stays resident in memory)."""
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        think=False,
        options={"temperature": 0.7, "top_p": 0.9, "num_predict": 200}
    )
    return response['response']


def debate_trade(symbol: str, action: str, confidence: float, reasoning: str) -> dict:
    """
    Contrarian second LLM call — challenges a high-conviction trade.
    Returns {'verdict': 'PROCEED'|'ABORT', 'reason': str}.
    On failure defaults to PROCEED (safe fallback: don't block the trade).
    """
    debate_prompt = (
        f"You are a skeptical risk analyst reviewing a proposed trade.\n"
        f"Proposal: {action.upper()} {symbol}  (confidence: {confidence:.0%})\n"
        f"Bull argument: {reasoning}\n\n"
        f"List 2 specific risks or counter-arguments for this trade. "
        f"Then state your verdict on a NEW LINE as exactly one of:\n"
        f"VERDICT: PROCEED\n"
        f"VERDICT: ABORT\n"
        f"Abort only if the risks clearly outweigh the opportunity."
    )
    try:
        resp = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=debate_prompt,
            think=False,
            options={"temperature": 0.5, "top_p": 0.9, "num_predict": 150}
        )
        text = resp['response']
        verdict = 'PROCEED'
        for line in text.splitlines():
            if 'VERDICT:' in line.upper():
                verdict = 'ABORT' if 'ABORT' in line.upper() else 'PROCEED'
                break
        return {'verdict': verdict, 'reason': text.strip()}
    except Exception as e:
        logging.warning(f"  Debate call failed for {symbol}: {e}")
        return {'verdict': 'PROCEED', 'reason': 'debate unavailable'}

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AutonomousAgent:
    def __init__(self):
        """Initialize the autonomous trading agent."""
        # Alpaca clients
        _paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        if not _paper:
            logging.warning("LIVE TRADING MODE ENABLED - real money is at risk!")
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=_paper
        )
        self.data_client = StockHistoricalDataClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY')
        )
        
        # Stock discovery
        self.discovery = StockDiscovery()
        
        # Load adaptive parameters
        self.params = self.load_parameters()
        
        # Cooldown tracking
        self.cooldowns = {}
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.daily_start_equity = None  # Set at start of each trading day

        # Decision logging — unique ID per process run for replay correlation
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._decision_log = Path('logs/decision_log.jsonl')
        self._decision_log.parent.mkdir(exist_ok=True)

        logging.info("🤖 Autonomous Agent Initialized")
    
    def load_parameters(self):
        """Load adaptive parameters (Monday params take precedence)."""
        # Default parameters
        defaults = {
            'max_position_size': 0.05,  # 5% per trade
            'min_confidence': 0.60,     # AI confidence threshold
            'max_stocks_to_analyze': 25,
            'max_daily_trades': 10,
            'cooldown_minutes': 15,
            'stop_loss': -0.07,         # -7% stop loss
            'take_profit': 0.15,        # +15% take profit
            'max_daily_loss_pct': 0.05, # Circuit breaker: stop trading after -5% day
        }
        
        # Try Monday params first (from weekend analysis)
        monday_params = {}
        monday_path = Path('logs/monday_params.json')
        if monday_path.exists():
            with open(monday_path) as f:
                monday_params = json.load(f)
                defaults.update(monday_params)
                logging.info("📅 Loaded Monday parameters")

        # Try adaptive params (from nightly analysis)
        # Only apply keys that Monday params didn't already set
        adaptive_path = Path('logs/adaptive_params.json')
        if adaptive_path.exists():
            with open(adaptive_path) as f:
                adaptive_params = json.load(f)
                for k, v in adaptive_params.items():
                    if k not in monday_params:
                        defaults[k] = v
                logging.info("🔧 Loaded adaptive parameters")
        
        return defaults
    
    def _log_decision(self, symbol: str, prompt: str, raw_response: str,
                      decision: dict, indicators: dict, executed: bool,
                      debate: dict | None = None) -> None:
        """Append one decision record to decision_log.jsonl.

        Every call to the LLM is logged — including HOLDs — so replay mode
        has a complete history of what the model saw and decided.
        """
        record = {
            'timestamp':    datetime.now().isoformat(),
            'session_id':   self.session_id,
            'bot':          'stock',
            'model':        OLLAMA_MODEL,
            'symbol':       symbol,
            'indicators':   indicators,
            'prompt':       prompt,
            'raw_response': raw_response,
            'decision':     decision.get('decision'),
            'confidence':   decision.get('confidence'),
            'reasoning':    decision.get('reasoning', ''),
            'executed':     executed,
            'debate':       debate,
        }
        try:
            with open(self._decision_log, 'a') as f:
                f.write(json.dumps(record, default=_json_default) + '\n')
        except Exception as e:
            logging.warning(f"⚠️  Could not write decision log: {e}")

    def check_cooldown(self, symbol):
        """Check if symbol is on cooldown."""
        if symbol not in self.cooldowns:
            return False
        
        elapsed = (datetime.now() - self.cooldowns[symbol]).total_seconds() / 60
        return elapsed < self.params['cooldown_minutes']
    
    def get_market_data(self, symbol, bars=100):
        """Get recent market data for a symbol with Alpaca -> yfinance fallback."""
        # Try Alpaca hourly first (during market hours)
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            
            bars_data = self.data_client.get_stock_bars(request_params)
            df = bars_data.df
            if not df.empty and len(df) >= 14:
                logging.info(f"✅ Got {len(df)} hourly bars for {symbol} (Alpaca)")
                return df
        
        except Exception as e:
            logging.debug(f"Alpaca hourly data failed for {symbol}: {e}")
        
        # Fallback to Alpaca daily bars
        try:
            end = datetime.now()
            start = end - timedelta(days=365)
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            bars_data = self.data_client.get_stock_bars(request_params)
            df = bars_data.df
            if not df.empty and len(df) >= 14:
                logging.info(f"✅ Got {len(df)} daily bars for {symbol} (Alpaca)")
                return df
            
        except Exception as e:
            logging.debug(f"Alpaca daily data failed for {symbol}: {e}")
        
        # Final fallback to yfinance
        try:
            import yfinance as yf
            logging.info(f"🔄 Trying yfinance for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y')
            
            if not df.empty and len(df) >= 14:
                # Rename columns to match Alpaca format
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                logging.info(f"✅ Got {len(df)} bars for {symbol} (yfinance)")
                return df
            
        except Exception as e:
            logging.debug(f"yfinance failed for {symbol}: {e}")
        
        logging.warning(f"⚠️  No data available for {symbol} from any source")
        return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        if df is None or len(df) < 14:
            return None
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        
        # Volume ratio
        avg_volume = df['volume'].rolling(window=20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        # Price change
        price_change_pct = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        
        return {
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'volume_ratio': volume_ratio,
            'price_change_pct': price_change_pct,
            'current_price': df['close'].iloc[-1]
        }
    
    def get_position(self, symbol):
        """Return the current position for symbol, or None if not held."""
        try:
            return self.trading_client.get_open_position(symbol)
        except Exception:
            return None

    def execute_trade(self, symbol, decision, equity):
        """Execute a trade based on AI decision."""
        try:
            side = OrderSide.BUY if decision['decision'] == 'buy' else OrderSide.SELL
            current_price = decision.get('current_price', 0)

            if current_price == 0:
                logging.error(f"❌ Cannot execute trade for {symbol}: no price data")
                return False

            # For sells, verify we actually hold the position
            if side == OrderSide.SELL:
                position = self.get_position(symbol)
                if position is None:
                    logging.warning(f"⚠️  Skipping SELL {symbol}: no open position held")
                    return False
                shares = int(float(position.qty))
            else:
                position_value = equity * self.params['max_position_size']
                shares = int(position_value / current_price)

            if shares == 0:
                logging.warning(f"⚠️  Position too small for {symbol}")
                return False

            # Build order — bracket for buys, plain market for sells
            if side == OrderSide.BUY:
                stop_price = round(current_price * (1 + self.params['stop_loss']), 2)
                take_profit_price = round(current_price * (1 + self.params['take_profit']), 2)
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class='bracket',
                    take_profit=TakeProfitRequest(limit_price=take_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_price),
                )
                logging.info(f"  Bracket: SL=${stop_price:.2f} / TP=${take_profit_price:.2f}")
            else:
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )

            # Submit order
            submitted_order = self.trading_client.submit_order(order)

            # Log trade
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': decision['decision'],
                'shares': shares,
                'confidence': decision['confidence'],
                'reasoning': decision['reasoning'],
                'order_id': str(submitted_order.id)
            }

            with open('logs/trade_log.jsonl', 'a') as f:
                f.write(json.dumps(trade_log) + '\n')

            logging.info(f"✅ Executed {decision['decision'].upper()} {shares} shares of {symbol}")
            alert_trade_executed(
                'StockAgent', symbol, decision['decision'],
                shares, current_price, str(submitted_order.id)
            )

            # Update cooldown
            self.cooldowns[symbol] = datetime.now()
            self.daily_trades += 1

            return True

        except Exception as e:
            logging.error(f"❌ Trade execution failed for {symbol}: {e}")
            alert_trade_failed('StockAgent', symbol, str(e))
            return False
    
    def run_trading_session(self):
        """Run a single trading session - analyze and potentially trade."""
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = today
            self.daily_start_equity = None
            logging.info("🔄 Daily trade counter reset")

        # Get account info
        account = self.trading_client.get_account()
        equity = float(account.equity)
        logging.info(f"💰 Account Equity: ${equity:,.2f}")

        # Record starting equity for circuit breaker
        if self.daily_start_equity is None:
            self.daily_start_equity = equity
            logging.info(f"📌 Daily starting equity set: ${self.daily_start_equity:,.2f}")

        # Circuit breaker: stop trading if daily loss exceeds threshold
        daily_pnl_pct = (equity - self.daily_start_equity) / self.daily_start_equity
        max_loss = self.params['max_daily_loss_pct']
        if daily_pnl_pct <= -max_loss:
            logging.warning(
                f"🛑 CIRCUIT BREAKER: daily P&L is {daily_pnl_pct:.1%} "
                f"(limit: -{max_loss:.1%}). Halting trading for the day."
            )
            alert_circuit_breaker('StockAgent', daily_pnl_pct, equity)
            return
        
        # Discover opportunities
        max_stocks = self.params['max_stocks_to_analyze']
        logging.info(f"🔍 Discovering top {max_stocks} opportunities...")
        
        opportunities = self.discovery.discover_opportunities(max_stocks=max_stocks)
        logging.info(f"✅ Found {len(opportunities)} opportunities")
        
        if not opportunities:
            logging.warning("⚠️  No opportunities found")
            return
        
        # Analyze each opportunity
        trades_executed = 0
        
        # Correct version (inside a loop)
        for opp in opportunities:
            try:
                if isinstance(opp, str):
                    symbol = opp
                else:
                    symbol = opp['symbol']
            except Exception as e:
                logging.error(f"Error processing opportunity: {opp}. Error: {str(e)}")
                continue
            
            # Check limits
            if self.daily_trades >= self.params['max_daily_trades']:
                logging.info(f"⛔ Daily trade limit reached ({self.params['max_daily_trades']})")
                break
            
            if self.check_cooldown(symbol):
                logging.info(f"⏳ {symbol} on cooldown")
                continue
            
            # Get market data
            df = self.get_market_data(symbol)
            if df is None:
                continue
            
            # Calculate indicators
            indicators = self.calculate_indicators(df)
            if indicators is None:
                continue
            
            # Fetch recent news snippet (no-op if POLYGON_API_KEY not set)
            news_snippet = news_fetcher.format_for_prompt(symbol)

            # Create AI prompt
            prompt = f"""Analyze {symbol} for trading:

            Current Price: ${indicators['current_price']:.2f}
            RSI (14): {indicators['rsi']:.1f}
            MACD: {indicators['macd']:.2f}
            Volume Ratio: {indicators['volume_ratio']:.1f}x average
            Price Change (100 bars): {indicators['price_change_pct']:+.1f}%

            Discovery Signals: {', '.join(self.discovery.opportunities.get(symbol, []))}{news_snippet}

            Based on this data, should we BUY, SELL, or HOLD?
            Provide your decision, confidence (0-1), and reasoning."""

            # Get AI decision
            logging.info(f"🤔 Analyzing {symbol}...")
            response = get_trading_decision(prompt)
            decision = parse_decision(response)
            decision['current_price'] = indicators['current_price']

            logging.info(f"📊 {symbol}: {decision['decision'].upper()} (confidence: {decision['confidence']:.2f})")

            # Multi-agent debate for high-conviction trades
            debate_result = None
            will_execute = (
                decision['decision'] in ['buy', 'sell']
                and decision['confidence'] >= self.params['min_confidence']
            )
            if will_execute and decision['confidence'] >= _DEBATE_CONFIDENCE_THRESHOLD:
                logging.info(f"⚖️  Debating {symbol} ({decision['decision'].upper()} @ {decision['confidence']:.0%})...")
                debate_result = debate_trade(
                    symbol, decision['decision'],
                    decision['confidence'], decision.get('reasoning', '')
                )
                if debate_result['verdict'] == 'ABORT':
                    logging.info(f"🚫 Debate ABORTED {symbol} trade: {debate_result['reason'][:80]}...")
                    will_execute = False
                else:
                    logging.info(f"✅ Debate confirmed {symbol}: PROCEED")

            executed = False
            if will_execute:
                executed = self.execute_trade(symbol, decision, equity)
                if executed:
                    trades_executed += 1
            elif not will_execute and decision['decision'] in ['buy', 'sell']:
                logging.info(f"⏭️  Skipping {symbol}: confidence {decision['confidence']:.2f} < {self.params['min_confidence']:.2f}")

            # Log every decision (including HOLDs) for replay and validation
            log_indicators = {**indicators,
                              'discovery_signals': self.discovery.opportunities.get(symbol, [])}
            self._log_decision(symbol, prompt, response, decision, log_indicators, executed,
                               debate=debate_result)
        
        logging.info(f"✅ Session complete: {trades_executed} trades executed")

if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.run_trading_session()