#!/usr/bin/env python3
"""
Options Trading Agent - Trades options contracts (puts/calls)
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
import yfinance as yf
import pandas as pd
import numpy as np

import ollama
sys.path.append(str(Path(__file__).resolve().parent))
from model_inference_lora import parse_decision
from alerts import alert_circuit_breaker, alert_trade_executed, alert_trade_failed

OLLAMA_MODEL = "qwen3:8b"

def get_trading_decision(prompt, max_new_tokens=200):
    """Get trading decision via Ollama (model stays resident in memory)."""
    response = ollama.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        think=False,
        options={"temperature": 0.7, "top_p": 0.9, "num_predict": max_new_tokens}
    )
    return response['response']

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptionsAgent:
    def __init__(self):
        """Initialize the options trading agent."""
        _paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        if not _paper:
            logging.warning("LIVE TRADING MODE ENABLED - real money is at risk!")
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=_paper
        )
        self.option_data_client = OptionHistoricalDataClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY')
        )
        
        # Options-specific parameters
        self.params = {
            'max_portfolio_allocation': 0.15,  # 15% max
            'min_portfolio_allocation': 0.10,  # 10% min
            'max_position_size': 0.03,          # 3% per position
            'min_confidence': 0.75,             # Higher than stocks (75%)
            'max_daily_trades': 5,              # Max options trades per day
            'max_dte': 45,                      # Max days to expiration
            'min_dte': 7,                       # Min days to expiration
            'target_delta': 0.30,               # Target delta for options
            'max_loss_per_trade': 0.02,         # Max 2% loss per trade
            'take_profit': 0.50,                # Take profit at 50%
            'stop_loss': -0.50,                 # Stop loss at -50%
            'max_daily_loss_pct': 0.05,         # Circuit breaker: halt after -5% day
        }

        # Watchlist for options trading
        self.watchlist = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']

        # Path to research params written by market_researcher / options_weekend_strategist
        self._params_file = Path(__file__).resolve().parent.parent / 'logs' / 'monday_params_options.json'

        # Trading state
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.trade_cooldown = {}  # Symbol -> last trade time
        self.daily_start_equity = None  # Set at start of each trading day
        
        logging.info("🎯 Options Agent Initialized")
    
    def load_research_params(self) -> dict:
        """Load latest research params from market_researcher or options_weekend_strategist."""
        try:
            if self._params_file.exists():
                with open(self._params_file) as f:
                    data = json.load(f)
                logging.info(
                    f"📋 Research params loaded — source: {data.get('source', 'unknown')} | "
                    f"bias: {data.get('market_bias')} | "
                    f"VIX regime: {data.get('vix_regime')}"
                )
                return data
        except Exception as e:
            logging.warning(f"⚠️  Could not load research params: {e}")
        return {}

    def get_available_capital(self):
        """Calculate available capital for options (10-15% of portfolio)."""
        try:
            account = self.trading_client.get_account()
            total_equity = float(account.equity)
            
            # Get current options positions value
            positions = self.trading_client.get_all_positions()
            options_value = sum(
                float(p.market_value) 
                for p in positions 
                if hasattr(p, 'symbol') and len(p.symbol) > 10  # Options symbols are long
            )
            
            # Calculate target allocation (12.5% midpoint)
            target_allocation = total_equity * 0.125
            available = target_allocation - options_value
            
            # Ensure we stay within 10-15% bounds
            max_allocation = total_equity * self.params['max_portfolio_allocation']
            min_allocation = total_equity * self.params['min_portfolio_allocation']
            
            if options_value >= max_allocation:
                available = 0
            elif available < 0:
                available = 0
            
            logging.info(f"💰 Total Equity: ${total_equity:,.2f}")
            logging.info(f"💰 Options Value: ${options_value:,.2f} ({options_value/total_equity:.1%})")
            logging.info(f"💰 Available for Options: ${available:,.2f}")
            
            return available
            
        except Exception as e:
            logging.error(f"❌ Failed to get available capital: {e}")
            return 0
    
    def get_market_data(self, symbol):
        """Get market data for underlying stock."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='3mo', interval='1d')
            
            if df.empty:
                return None
            
            df.columns = df.columns.str.lower()
            return df
            
        except Exception as e:
            logging.error(f"❌ Failed to get data for {symbol}: {e}")
            return None
    
    def analyze_stock_for_options(self, symbol):
        """Analyze stock for options trading opportunity."""
        df = self.get_market_data(symbol)
        if df is None:
            return None
        
        # Calculate indicators
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volatility (20-day)
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            
            # Momentum
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            return {
                'symbol': symbol,
                'current_price': df['close'].iloc[-1],
                'rsi': rsi.iloc[-1],
                'volatility': volatility.iloc[-1],
                'momentum': momentum,
                'volume': df['volume'].iloc[-1]
            }
            
        except Exception as e:
            logging.error(f"❌ Analysis failed for {symbol}: {e}")
            return None
    
    def get_ai_options_decision(self, analysis, market_bias: str = 'neutral', symbol_bias: str = 'neutral'):
        """Get AI decision for options trading."""
        prompt = f"""Analyze {analysis['symbol']} for OPTIONS trading:

Current Price: ${analysis['current_price']:.2f}
RSI (14): {analysis['rsi']:.1f}
Volatility (20d): {analysis['volatility']:.1%}
Momentum (20d): {analysis['momentum']:.1%}

Market Context (from overnight research):
- Overall market bias: {market_bias}
- Symbol-specific bias: {symbol_bias}

Based on this data, should we:
- BUY_CALL (bullish, expect price to rise)
- BUY_PUT (bearish, expect price to fall)
- HOLD (wait for better opportunity)

Provide your decision, confidence (0-1), and reasoning."""
        
        response = get_trading_decision(prompt, max_new_tokens=150)
        decision = parse_decision(response)
        
        # Map to options-specific decisions
        # Explicit model keywords take priority; fallback uses both AI action AND momentum
        if 'call' in response.lower() or (decision['decision'] == 'buy' and analysis['momentum'] > 0):
            decision['decision'] = 'buy_call'
        elif 'put' in response.lower() or (decision['decision'] == 'sell' and analysis['momentum'] < 0):
            decision['decision'] = 'buy_put'
        else:
            decision['decision'] = 'hold'
        
        return decision
    
    def find_optimal_option(self, symbol, contract_type, strike_price):
        """Find the best option contract."""
        try:
            # Calculate expiration range
            min_exp = (datetime.now() + timedelta(days=self.params['min_dte'])).strftime('%Y-%m-%d')
            max_exp = (datetime.now() + timedelta(days=self.params['max_dte'])).strftime('%Y-%m-%d')
            
            # Get option contracts
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status='active',
                expiration_date_gte=min_exp,
                expiration_date_lte=max_exp,
                type=contract_type
            )
            
            resp = self.trading_client.get_option_contracts(request)
            contracts = resp.option_contracts if resp and resp.option_contracts else []

            if not contracts:
                return None

            # Filter by strike price and DTE
            best_contract = None
            best_score = float('inf')

            for contract in contracts:
                strike = float(contract.strike_price)
                dte = (datetime.strptime(contract.expiration_date, '%Y-%m-%d') - datetime.now()).days
                
                # Score based on proximity to target delta and DTE
                strike_diff = abs(strike - strike_price) / strike_price
                dte_score = abs(dte - 30) / 30  # Target ~30 DTE
                
                score = strike_diff + dte_score
                
                if score < best_score:
                    best_score = score
                    best_contract = contract
            
            return best_contract
            
        except Exception as e:
            logging.error(f"❌ Failed to find option contract: {e}")
            return None
    
    def calculate_position_size(self, option_price, available_capital, size_multiplier: float = 1.0):
        """Calculate position size (number of contracts), scaled by research multiplier."""
        max_position_value = available_capital * self.params['max_position_size'] * size_multiplier

        # Each contract controls 100 shares
        contract_cost = option_price * 100

        # Calculate max contracts
        max_contracts = int(max_position_value / contract_cost)

        # Limit to reasonable size
        max_contracts = min(max_contracts, 10)

        return max(1, max_contracts)
    
    def get_option_price(self, contract_symbol):
        """Get current mid-price for an option contract from Alpaca market data."""
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol)
            quotes = self.option_data_client.get_option_latest_quote(request)
            if quotes and contract_symbol in quotes:
                q = quotes[contract_symbol]
                bid = float(q.bid_price) if q.bid_price else 0.0
                ask = float(q.ask_price) if q.ask_price else 0.0
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
                elif ask > 0:
                    return ask
                elif bid > 0:
                    return bid
        except Exception as e:
            logging.warning(f"⚠️ Could not fetch option price for {contract_symbol}: {e}")
        return None

    def execute_options_trade(self, symbol, decision, analysis, available_capital, size_multiplier: float = 1.0):
        """Execute options trade."""
        try:
            # Determine contract type
            contract_type = ContractType.CALL if decision['decision'] == 'buy_call' else ContractType.PUT
            
            # Find optimal strike
            current_price = analysis['current_price']
            if contract_type == ContractType.CALL:
                strike_price = current_price * 1.02  # 2% OTM call
            else:
                strike_price = current_price * 0.98  # 2% OTM put
            
            # Find contract
            contract = self.find_optimal_option(symbol, contract_type, strike_price)
            
            if not contract:
                logging.warning(f"⚠️  No suitable option contract found for {symbol}")
                return False
            
            # Get real option price from market data (bid/ask midpoint)
            option_price = self.get_option_price(contract.symbol)
            if option_price is None:
                logging.warning(f"⚠️ No market price for {contract.symbol}, skipping trade")
                return False

            # Calculate position size (scaled by research multiplier)
            quantity = self.calculate_position_size(option_price, available_capital, size_multiplier)
            
            if quantity == 0:
                logging.warning(f"⚠️  Position size too small for {symbol}")
                return False
            
            # Create order
            order = MarketOrderRequest(
                symbol=contract.symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            submitted_order = self.trading_client.submit_order(order)
            
            # Log trade
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'underlying': symbol,
                'contract': contract.symbol,
                'type': contract_type.value,
                'strike': float(contract.strike_price),
                'expiration': contract.expiration_date,
                'action': 'buy',
                'quantity': quantity,
                'entry_price': option_price,
                'confidence': decision['confidence'],
                'reasoning': decision['reasoning'],
                'order_id': str(submitted_order.id)
            }
            
            with open('logs/options_trade_log.jsonl', 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
            
            logging.info(f"✅ Executed OPTIONS trade: {quantity} contracts of {symbol} {contract_type.value}")
            alert_trade_executed(
                'OptionsAgent', contract.symbol, 'buy',
                quantity, option_price, str(submitted_order.id)
            )
            return True

        except Exception as e:
            logging.error(f"❌ Options trade execution failed: {e}")
            alert_trade_failed('OptionsAgent', symbol, str(e))
            return False
    
    def manage_existing_positions(self):
        """Monitor and manage existing options positions."""
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                # Check if it's an option (options have long symbols)
                if len(position.symbol) <= 10:
                    continue
                
                qty = float(position.qty)
                unrealized_plpc = float(position.unrealized_plpc)
                
                logging.info(f"📊 Option: {position.symbol[:20]}... P&L: {unrealized_plpc:+.1%}")
                
                # Exit criteria
                should_exit = False
                exit_reason = None
                
                # Take profit at 50%
                if unrealized_plpc >= self.params['take_profit']:
                    should_exit = True
                    exit_reason = "take_profit_50%"
                
                # Stop loss at -50%
                elif unrealized_plpc <= self.params['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss_50%"
                
                if should_exit:
                    # Create sell order
                    order = MarketOrderRequest(
                        symbol=position.symbol,
                        qty=int(qty),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    submitted_order = self.trading_client.submit_order(order)
                    
                    trade_log = {
                        'timestamp': datetime.now().isoformat(),
                        'contract': position.symbol,
                        'action': 'sell',
                        'quantity': int(qty),
                        'reason': exit_reason,
                        'exit_pl_pct': unrealized_plpc,
                        'order_id': str(submitted_order.id)
                    }
                    
                    with open('logs/options_trade_log.jsonl', 'a') as f:
                        f.write(json.dumps(trade_log) + '\n')
                    
                    logging.info(f"✅ Closed option position: {exit_reason} ({unrealized_plpc:+.1%})")
                    
        except Exception as e:
            logging.error(f"❌ Position management failed: {e}")
    
    def run_options_session(self):
        """Run a single options trading session."""
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = today
            self.daily_start_equity = None
            logging.info("🔄 Daily options trade counter reset")

        # Circuit breaker: check daily P&L before doing anything
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            if self.daily_start_equity is None:
                self.daily_start_equity = equity
                logging.info(f"📌 Daily starting equity set: ${self.daily_start_equity:,.2f}")
            daily_pnl_pct = (equity - self.daily_start_equity) / self.daily_start_equity
            max_loss = self.params['max_daily_loss_pct']
            if daily_pnl_pct <= -max_loss:
                logging.warning(
                    f"🛑 CIRCUIT BREAKER: daily P&L is {daily_pnl_pct:.1%} "
                    f"(limit: -{max_loss:.1%}). Halting options trading for the day."
                )
                alert_circuit_breaker('OptionsAgent', daily_pnl_pct, equity)
                return
        except Exception as e:
            logging.error(f"❌ Failed to check circuit breaker: {e}")

        # Load overnight research params (graceful — defaults used if file absent)
        research       = self.load_research_params()
        min_conf       = research.get('min_confidence_override', self.params['min_confidence'])
        size_mult      = research.get('position_size_multiplier', 1.0)
        market_bias    = research.get('market_bias', 'neutral')
        avoid_earnings = set(research.get('avoid_earnings_risk', []))
        symbol_bias    = research.get('symbol_bias', {})

        logging.info(f"📊 Daily options trades: {self.daily_trades}/{self.params['max_daily_trades']}")

        # Step 1: Manage existing positions
        self.manage_existing_positions()

        # Step 2: Check if we can trade more
        if self.daily_trades >= self.params['max_daily_trades']:
            logging.info(f"⛔ Daily options trade limit reached ({self.params['max_daily_trades']})")
            return

        # Step 3: Get available capital
        available_capital = self.get_available_capital()

        if available_capital < 100:
            logging.info("⚠️  Insufficient capital for options trading")
            return

        # Step 4: Scan watchlist
        trades_executed = 0

        for symbol in self.watchlist:
            if self.daily_trades >= self.params['max_daily_trades']:
                break

            # Skip tickers flagged for earnings risk (IV crush danger)
            if symbol in avoid_earnings:
                logging.info(f"⏭️  {symbol}: skipping — earnings within 2 days (IV crush risk)")
                continue

            # Check cooldown
            if symbol in self.trade_cooldown:
                time_since_trade = (datetime.now() - self.trade_cooldown[symbol]).total_seconds()
                if time_since_trade < 3600:  # 1 hour cooldown
                    continue

            logging.info(f"🤔 Analyzing {symbol} for options...")

            # Analyze
            analysis = self.analyze_stock_for_options(symbol)
            if not analysis:
                continue

            # Get AI decision — pass research context into the prompt
            sym_bias = symbol_bias.get(symbol, 'neutral')
            decision = self.get_ai_options_decision(analysis, market_bias=market_bias, symbol_bias=sym_bias)

            logging.info(f"📊 {symbol}: {decision['decision'].upper()} "
                         f"(confidence: {decision['confidence']:.0%}, "
                         f"bias: {sym_bias}, min_conf: {min_conf:.0%})")

            # Execute if confidence meets (possibly research-adjusted) threshold
            if decision['decision'] in ['buy_call', 'buy_put'] and decision['confidence'] >= min_conf:
                if self.execute_options_trade(symbol, decision, analysis, available_capital,
                                              size_multiplier=size_mult):
                    self.daily_trades += 1
                    trades_executed += 1
                    self.trade_cooldown[symbol] = datetime.now()

                    # Recalculate available capital
                    available_capital = self.get_available_capital()
                    if available_capital < 100:
                        break
            else:
                logging.info(f"⏭️  Skipping {symbol}: {decision['decision']} "
                             f"(confidence: {decision['confidence']:.0%})")
        
        logging.info(f"✅ Options session complete: {trades_executed} trades executed")

if __name__ == "__main__":
    agent = OptionsAgent()
    agent.run_options_session()
