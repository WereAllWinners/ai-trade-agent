#!/usr/bin/env python3
"""
Options Trading Agent - Trades options contracts (puts/calls)
"""
import os
import sys
import json
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, GetOrdersRequest, MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, ContractType
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
import yfinance as yf
import pandas as pd
import numpy as np

try:
    from zoneinfo import ZoneInfo
    _EST = ZoneInfo('America/New_York')
except Exception:
    _EST = None  # fallback: use local system time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401
from model_inference_lora import get_trading_decision, parse_decision, _generate_base_model
from alerts import alert_circuit_breaker, alert_trade_executed, alert_trade_failed
import news_fetcher
import unusual_flow_scanner
import economic_calendar
import db as _db
from portfolio_overseer import PortfolioOverseer
from allocation_controller import AllocationController
from paper_market_simulator import PaperMarketSimulator
import account_config as _acfg

_DEBATE_CONFIDENCE_THRESHOLD = 0.85   # Options trades require higher bar for debate
_debate_unavailable_count: int = 0    # E2: tracks base-model debate failures
_MIN_CASH_RESERVE = 500.0             # Never spend the account's last $500 of cash
_DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'  # log orders but never submit
# Near-zero option exit: skip closing if total contract value is below this threshold
# (the broker's minimum notional to avoid a pointless trade + commission)
_MIN_EXIT_VALUE_USD = 10.0

# Options liquidity thresholds — override via environment variables
_MIN_OPEN_INTEREST  = int(os.getenv('MIN_OPEN_INTEREST_OPTIONS', '200'))
_MIN_OPTION_VOLUME  = int(os.getenv('MIN_OPTION_VOLUME', '50'))
# E3: Hard liquidity gates (code-level, not just prompt instructions)
_OPT_MIN_OI         = int(os.getenv('OPT_MIN_OI', '250'))           # OI gate (stricter than _MIN_OPEN_INTEREST)
_OPT_MAX_SPREAD_PCT = float(os.getenv('OPT_MAX_SPREAD_PCT', '0.12'))  # (ask-bid)/mid > 12% → skip
_OPT_MAX_FRICTION   = float(os.getenv('OPT_MAX_FRICTION_PCT', '0.15')) # (ask-mid)/mid > 15% → skip

def _json_default(obj):
    """JSON serializer for types that the stdlib encoder can't handle."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _settled_cash(account) -> float:
    """Return fully settled cash (non_marginable_buying_power), excluding T+1 unsettled proceeds."""
    return float(account.non_marginable_buying_power)


def _load_macro_context_block() -> str:
    """Read the most recent options macro context entry and format it for prompts.

    Reads logs/market_context_options.json (written nightly by market_researcher).
    Returns '' if file is missing or entry is stale (>36 h old).
    """
    try:
        path = Path(__file__).resolve().parent.parent / 'logs' / 'market_context_options.json'
        if not path.exists():
            return ''
        with open(path) as f:
            records = json.load(f)
        if not records:
            return ''
        latest = records[-1]
        ts = datetime.fromisoformat(latest['timestamp'])
        if (datetime.now() - ts).total_seconds() > 36 * 3600:
            return ''

        macro     = latest.get('macro', {})
        vix_d     = macro.get('vix', {})
        spy_d     = macro.get('spy', {})
        vix_val   = vix_d.get('current', '?')
        vix_reg   = vix_d.get('regime', '?')
        spy_trend = spy_d.get('trend', '?')
        spy_ma    = 'above 50MA' if spy_d.get('above_50ma') else 'below 50MA'

        line1 = f"Regime: {spy_trend} | VIX: {vix_val} ({vix_reg}) | SPY: {spy_ma}"

        params = latest.get('recommended_params', {})
        bias   = params.get('market_bias', '')
        iv_elv = params.get('iv_elevated_tickers', [])[:4]

        parts = []
        if bias:
            parts.append(f"Market bias: {bias}")
        if iv_elv:
            parts.append(f"IV elevated on: {', '.join(iv_elv)}")
        line2 = ' | '.join(parts)

        lines = ['Market Context (as of last night):', f'  {line1}']
        if line2:
            lines.append(f'  {line2}')
        return '\n'.join(lines)
    except Exception as e:
        logging.debug(f"Could not load options macro context block: {e}")
        return ''


def _load_weekend_context_block() -> str:
    """Read the most recent weekend strategist entry and format it for prompts.

    Reads logs/weekend_context_options.json (written Saturday).
    Only injected Mon–Fri before 2 PM and when the file is ≤5 days old.
    Returns '' if conditions are not met or on any error.
    """
    try:
        now = datetime.now()
        if now.weekday() >= 5 or now.hour >= 14:
            return ''

        path = Path(__file__).resolve().parent.parent / 'logs' / 'weekend_context_options.json'
        if not path.exists():
            return ''
        with open(path) as f:
            records = json.load(f)
        if not records:
            return ''
        latest = records[-1]
        ts = datetime.fromisoformat(latest['timestamp'])
        if (now - ts).total_seconds() > 5 * 86400:
            return ''

        params      = latest.get('recommended_params', {})
        sector_data = latest.get('sector_analysis', {})
        iv_analysis = latest.get('iv_analysis', {})
        earnings    = latest.get('earnings_next_7_days', [])[:5]

        top_sectors  = params.get('top_sectors', [])[:3]
        weak_sectors = params.get('weak_sectors', [])[:2]
        sector_parts = []
        if top_sectors:
            top_str = ', '.join(
                f"{s} ({sector_data.get(s, {}).get('return_1m_pct', 0):+.1f}%)"
                for s in top_sectors
            )
            sector_parts.append(f"Top sectors: {top_str}")
        if weak_sectors:
            weak_str = ', '.join(
                f"{s} ({sector_data.get(s, {}).get('return_1m_pct', 0):+.1f}%)"
                for s in weak_sectors
            )
            sector_parts.append(f"Avoid: {weak_str}")
        sectors_line = ' | '.join(sector_parts)

        cong_bull  = params.get('congressional_bullish', [])[:5]
        cong_bear  = params.get('congressional_bearish', [])[:5]
        cong_parts = []
        if cong_bull:
            cong_parts.append(f"bullish {', '.join(cong_bull)}")
        if cong_bear:
            cong_parts.append(f"bearish {', '.join(cong_bear)}")
        cong_line = f"Congressional: {' | '.join(cong_parts)}" if cong_parts else ''

        elevated = [s for s, v in iv_analysis.items() if v.get('vol_regime') == 'elevated'][:4]
        iv_line  = (f"IV regime: elevated on {', '.join(elevated)} — premium buying expensive this week"
                    if elevated else '')

        earn_parts = [f"{e['symbol']} ({e.get('earnings_in_days', '?')}d)" for e in earnings]
        earn_line  = (f"Earnings this week: {', '.join(earn_parts)} — avoid new positions day-before"
                      if earn_parts else '')

        content_lines = [l for l in [sectors_line, cong_line, iv_line, earn_line] if l]
        if not content_lines:
            return ''
        lines = ['Weekend Research Context:'] + [f'  {l}' for l in content_lines]
        return '\n'.join(lines)
    except Exception as e:
        logging.debug(f"Could not load weekend context block: {e}")
        return ''


def _compute_regime() -> str | None:
    """Return current market regime string e.g. 'bull_midvol'.

    Fetches SPY vs its 200-day MA (bull/bear) and VIX level (lowvol/midvol/highvol).
    Returns None silently on any data-fetch failure so trading never blocks.
    """
    try:
        spy = yf.download('SPY', period='1y', auto_adjust=True, progress=False)['Close']
        if spy.empty or len(spy) < 200:
            return None
        spy_price  = float(spy.iloc[-1])
        spy_200ma  = float(spy.tail(200).mean())
        trend = 'bull' if spy_price > spy_200ma else 'bear'

        vix = yf.download('^VIX', period='5d', auto_adjust=True, progress=False)['Close']
        vix_val = float(vix.iloc[-1]) if not vix.empty else 20.0
        vol = 'highvol' if vix_val > 25 else ('lowvol' if vix_val < 15 else 'midvol')
        return f"{trend}_{vol}"
    except Exception as e:
        logging.debug("_compute_regime failed (non-fatal): %s", e)
        return None


def debate_trade(symbol: str, action: str, confidence: float, reasoning: str) -> dict:
    """
    Contrarian second LLM call — challenges a high-conviction options trade.
    Returns {'verdict': 'PROCEED'|'ABORT', 'reason': str}.
    Defaults to PROCEED on failure so the trade isn't silently blocked.
    """
    debate_prompt = (
        f"You are a balanced options risk/reward analyst reviewing a proposed trade.\n"
        f"Proposal: {action.upper()} on {symbol}  (confidence: {confidence:.0%})\n"
        f"Argument: {reasoning}\n\n"
        f"Briefly state ONE supporting factor and ONE risk (IV crush, event risk, liquidity, wrong direction) "
        f"for this trade.\n"
        f"Then state your verdict on a NEW LINE as exactly one of:\n"
        f"VERDICT: PROCEED\n"
        f"VERDICT: ABORT\n"
        f"Choose ABORT only if the risk clearly and specifically outweighs the opportunity. "
        f"When in doubt, PROCEED."
    )
    try:
        # E2: route through the BASE (un-fine-tuned) model for diverse perspective.
        text = _generate_base_model(debate_prompt)
        verdict = 'PROCEED'
        for line in text.splitlines():
            if 'VERDICT:' in line.upper():
                verdict = 'ABORT' if 'ABORT' in line.upper() else 'PROCEED'
                break
        return {'verdict': verdict, 'reason': text.strip()}
    except Exception as e:
        global _debate_unavailable_count
        _debate_unavailable_count += 1
        logging.warning(
            f"  Debate call failed for {symbol}: {e}  (debate_unavailable #{_debate_unavailable_count})"
        )
        return {'verdict': 'PROCEED', 'reason': 'debate unavailable'}

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
        if not _paper:
            _api_key    = os.getenv('ALPACA_API_KEY_LIVE') or os.getenv('ALPACA_API_KEY')
            _api_secret = os.getenv('ALPACA_API_SECRET_LIVE') or os.getenv('ALPACA_SECRET_KEY')
        else:
            _api_key    = os.getenv('ALPACA_API_KEY')
            _api_secret = os.getenv('ALPACA_SECRET_KEY')
        self.trading_client = TradingClient(_api_key, _api_secret, paper=_paper)
        from utils.alpaca_retry import retry_on_rate_limit
        for _m in ('submit_order', 'get_account', 'get_all_positions', 'get_orders', 'get_order_by_id'):
            if hasattr(self.trading_client, _m):
                method = getattr(self.trading_client, _m)
                if not hasattr(method, '_mock_name'):  # skip wrapping unittest.mock stubs
                    setattr(self.trading_client, _m, retry_on_rate_limit(method))
        self.option_data_client = OptionHistoricalDataClient(_api_key, _api_secret)
        
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
            'exit_dte_threshold': 2,            # Close positions within N calendar days of expiry
            'max_daily_loss_pct': 0.05,         # Circuit breaker: halt after -5% day
        }

        # Watchlist for options trading
        self.watchlist = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD']

        # Path to research params written by market_researcher / options_weekend_strategist
        self._params_file = Path(__file__).resolve().parent.parent / 'logs' / 'monday_params_options.json'

        # Decision logging — unique ID per process run for replay correlation
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._decision_log = Path(__file__).resolve().parent.parent / 'logs' / 'options_decision_log.jsonl'
        self._decision_log.parent.mkdir(exist_ok=True)

        # Trading state
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.trade_cooldown = {}  # Symbol -> last trade time
        self.daily_start_equity = None  # Set at start of each trading day
        self._last_order_id: str | None = None
        self._session_regime: str | None = None

        if _DRY_RUN:
            logging.warning("🧪 DRY RUN MODE: orders will be logged but NOT submitted to Alpaca")

        self.paper_sim = PaperMarketSimulator(paper=_paper)
        self._paper    = _paper

        # Portfolio-level guards (sector cap + correlation limit)
        self.overseer = PortfolioOverseer(self.trading_client)

        # Gradual allocation ramp (Tier 1→3 based on live Sharpe/DD)
        self.alloc_controller = AllocationController(_db)
        self.alloc_controller.refresh()
        logging.info(
            f"📊 Allocation tier: {self.alloc_controller.current_tier()} "
            f"({self.alloc_controller.get_position_size_pct():.0%} per position)"
        )

        # Init DB (creates tables including cash_reservations if needed)
        _db.init_db()

        self._prev_equity: float | None = None
        logging.info("🎯 Options Agent Initialized")
    
    def _log_decision(self, symbol: str, prompt: str, raw_response: str,
                      decision: dict, indicators: dict, market_context: dict,
                      executed: bool, debate: dict | None = None,
                      order_id: str | None = None) -> None:
        """Append one options decision record to options_decision_log.jsonl."""
        record = {
            'timestamp':      datetime.now().isoformat(),
            'session_id':     self.session_id,
            'bot':            'options',
            'model':          os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-32B-Instruct') + '+LoRA',
            'symbol':         symbol,
            'indicators':     indicators,
            'market_context': market_context,
            'prompt':         prompt,
            'raw_response':   raw_response,
            'decision':       decision.get('decision'),
            'confidence':     decision.get('confidence'),
            'reasoning':      decision.get('reasoning', ''),
            'executed':       executed,
            'debate':         debate,
            'order_id':       order_id,
            'regime':         self._session_regime,
        }
        try:
            with open(self._decision_log, 'a') as f:
                f.write(json.dumps(record, default=_json_default) + '\n')
        except Exception as e:
            logging.warning(f"⚠️  Could not write options decision log: {e}")
        try:
            _db.insert_decision(record, source='paper' if getattr(self, '_paper', True) else 'live')
        except Exception as e:
            logging.warning(f"⚠️  Could not write options decision to DB: {e}")

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
        """Calculate available capital for options (10-15% of portfolio, cash only)."""
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

            cash = float(account.cash)
            settled = _settled_cash(account)

            # Hard gate: no buys if cash is at or below the minimum reserve
            if cash <= _MIN_CASH_RESERVE:
                logging.warning(
                    f"🛑 Cash is ${cash:,.2f} ≤ reserve ${_MIN_CASH_RESERVE:,.2f} "
                    f"— no options buys until cash recovers."
                )
                return 0

            # Calculate target allocation (12.5% midpoint), capped strictly to spendable cash
            target_allocation = total_equity * 0.125
            available = target_allocation - options_value

            # Ensure we stay within 10-15% bounds
            max_allocation = total_equity * self.params['max_portfolio_allocation']

            if options_value >= max_allocation:
                available = 0
            elif available < 0:
                available = 0

            # Hard cap: never exceed settled cash (T+1 proceeds excluded) minus the reserve buffer
            spendable_cash = settled - _MIN_CASH_RESERVE
            available = min(available, spendable_cash)
            available = max(available, 0)  # floor at zero

            logging.info(
                f"💰 Total Equity: ${total_equity:,.2f} | Cash: ${cash:,.2f} | "
                f"Settled: ${settled:,.2f} (T+1 pending: ${cash - settled:,.2f})"
            )
            logging.info(f"💰 Options Value: ${options_value:,.2f} ({options_value/total_equity:.1%})")
            logging.info(f"💰 Available for Options: ${available:,.2f}")

            return available
            
        except Exception as e:
            logging.error(f"❌ Failed to get available capital: {e}")
            return 0
    
    def get_market_data(self, symbol):
        """Get market data for underlying stock.

        Primary: Alpaca historical bars (same API key already configured).
        Fallback: yfinance (kept for resilience but rate-limited at scale).
        """
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # --- Alpaca primary ---
        try:
            data_client = StockHistoricalDataClient(
                os.getenv('ALPACA_API_KEY'),
                os.getenv('ALPACA_SECRET_KEY')
            )
            end = datetime.now()
            start = end - timedelta(days=90)
            bars = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            ))
            df = bars.df
            if not df.empty and len(df) >= 14:
                df.index = df.index.get_level_values('timestamp') if isinstance(df.index, pd.MultiIndex) else df.index
                logging.debug(f"✅ Got {len(df)} daily bars for {symbol} (Alpaca)")
                return df
        except Exception as e:
            logging.debug(f"Alpaca data failed for {symbol}: {e}")

        # --- yfinance fallback ---
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='3mo', interval='1d')
            if not df.empty:
                df.columns = df.columns.str.lower()
                return df
        except Exception as e:
            logging.error(f"❌ Failed to get data for {symbol} (yfinance fallback): {e}")

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
    
    def get_ai_options_decision(self, analysis, market_bias: str = 'neutral',
                               symbol_bias: str = 'neutral',
                               news_snippet: str = '', flow_snippet: str = '',
                               greeks: dict | None = None,
                               iv_rank: float | None = None,
                               macro_context_block: str = '',
                               weekend_context_block: str = '',
                               macro_guard_block: str = ''):
        """Get AI decision for options trading.

        Returns (decision, prompt, raw_response) so callers can log all three
        without making a second LLM call.

        greeks (optional): dict with delta, theta, gamma, vega, iv keys from
            get_atm_option_greeks().  When provided they are included in the
            prompt so the model can factor in IV crush risk and time decay.
        macro_context_block: nightly macro summary (Fix D).
        weekend_context_block: Saturday research summary (Fix E).
        macro_guard_block: today's high-impact economic events (Fix C).
        """
        extra_context = (news_snippet + flow_snippet).strip()
        context_block = f"\n{extra_context}" if extra_context else ''

        # Weekend context appended to context_block (Fix E)
        if weekend_context_block:
            context_block += f"\n{weekend_context_block}"

        # Greeks block — only appended when data is available
        greeks_block = ''
        if greeks:
            parts = []
            if greeks.get('delta') is not None:
                parts.append(f"Delta: {greeks['delta']:+.3f}  (directional exposure per $1 move)")
            if greeks.get('theta') is not None:
                parts.append(f"Theta: {greeks['theta']:+.4f}/day  (time decay cost)")
            if greeks.get('gamma') is not None:
                parts.append(f"Gamma: {greeks['gamma']:.4f}  (delta acceleration)")
            iv = greeks.get('iv')
            if iv is not None:
                iv_rank_str = f"{iv_rank:.0%}" if iv_rank is not None else "N/A"
                parts.append(
                    f"IV: {iv:.1%}  |  IV Rank: {iv_rank_str}  "
                    f"(high IV rank = expensive premium, high IV crush risk)"
                )
                # E3: expected move (approximate log-normal ATM formula)
                dte = greeks.get('dte')
                current_price = analysis.get('current_price')
                if dte and current_price:
                    expected_move = current_price * iv * (dte / 365) ** 0.5
                    parts.append(
                        f"Market-implied move by expiry: ±${expected_move:.2f}  "
                        f"({expected_move / current_price:.1%})"
                    )
            if parts:
                greeks_block = '\n\nLive Option Greeks (ATM contract):\n' + '\n'.join(f'- {p}' for p in parts)

        # Fix D: macro context block inserted before Market Context section
        _macro_ctx = f"\n{macro_context_block}\n" if macro_context_block else ''
        # Fix C: macro guard block inserted before the decision instructions
        _guard = f"\n{macro_guard_block}\n" if macro_guard_block else ''

        prompt = f"""You are a decisive options trader. Analyze {analysis['symbol']} and give a clear options decision.

Current Price: ${analysis['current_price']:.2f}
RSI (14): {analysis['rsi']:.1f}  (>70 overbought, <30 oversold)
Volatility (20d): {analysis['volatility']:.1%}  (historical vol)
Momentum (20d): {analysis['momentum']:.1%}  (positive = bullish trend){greeks_block}
{_macro_ctx}
Market Context (from overnight research):
- Overall market bias: {market_bias}
- Symbol-specific bias: {symbol_bias}{context_block}
{_guard}
Make a decisive call based on the signals above:
- BUY_CALL if momentum and bias are bullish (avoid if IV is extremely high — IV crush risk)
- BUY_PUT if momentum and bias are bearish (avoid if IV is extremely high — IV crush risk)
- HOLD only if signals are genuinely conflicting with no clear edge

Use confidence above 0.75 when signals align clearly, below 0.75 when signals are mixed.

Respond in this exact format:
Decision: <BUY_CALL|BUY_PUT|HOLD>
Confidence: <0.00-1.00>
Reasoning: <one sentence explaining the key signal>"""

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

        return decision, prompt, response
    
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
                exp = contract.expiration_date
                if isinstance(exp, str):
                    exp = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp - datetime.now().date()).days

                # E3: Hard OI gate (uses stricter _OPT_MIN_OI for new entries)
                oi = int(contract.open_interest or 0)
                if oi < _OPT_MIN_OI:
                    logging.debug(
                        f"  Skipping {contract.symbol}: OI {oi} < {_MIN_OPEN_INTEREST}"
                    )
                    continue

                # Score based on proximity to target strike and DTE
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
    
    def parse_dte_from_symbol(self, symbol: str, _today=None) -> int | None:
        """
        Extract days-to-expiration from an OCC option symbol.

        OCC format: <TICKER><YYMMDD><C|P><8-digit-strike>
        The expiry date is always at symbol[-15:-9] (6 chars, YYMMDD).

        Examples:
            SPY250328P00560000  -> symbol[-15:-9] = '250328' -> 2025-03-28
            AAPL260319C00207000 -> symbol[-15:-9] = '260319' -> 2026-03-19

        _today: date override for unit tests; defaults to current EST date.
        """
        try:
            date_str = symbol[-15:-9]
            expiry = datetime.strptime(date_str, '%y%m%d').date()
            if _today is None:
                _today = datetime.now(_EST).date() if _EST else datetime.now().date()
            return (expiry - _today).days
        except (ValueError, IndexError):
            logging.warning(f"Could not parse DTE from symbol: {symbol}")
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
        """Get current mid-price for an option contract from Alpaca market data.

        Returns None if the contract has no tradeable price OR if bid+ask size
        is below _MIN_OPTION_VOLUME (illiquid contract gate).
        """
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=contract_symbol)
            quotes = self.option_data_client.get_option_latest_quote(request)
            if quotes and contract_symbol in quotes:
                q = quotes[contract_symbol]
                bid = float(q.bid_price) if q.bid_price else 0.0
                ask = float(q.ask_price) if q.ask_price else 0.0

                # Volume gate: sum of bid+ask sizes is a proxy for quote depth
                bid_size = int(q.bid_size) if getattr(q, 'bid_size', None) else 0
                ask_size = int(q.ask_size) if getattr(q, 'ask_size', None) else 0
                quote_depth = bid_size + ask_size
                if quote_depth > 0 and quote_depth < _MIN_OPTION_VOLUME:
                    logging.debug(
                        f"  Skipping {contract_symbol}: quote depth {quote_depth} < {_MIN_OPTION_VOLUME}"
                    )
                    return None

                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    # E3: Hard spread gate — wide spreads signal illiquid contracts
                    # where slippage destroys edge before the position is filled.
                    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                    if spread_pct > _OPT_MAX_SPREAD_PCT:
                        logging.debug(
                            "  Skipping %s: spread %.0f%% > max %.0f%%",
                            contract_symbol, spread_pct * 100, _OPT_MAX_SPREAD_PCT * 100,
                        )
                        return None
                    # E3: Friction gate — (ask-mid)/mid captures the immediate cost-to-entry
                    friction = (ask - mid) / mid if mid > 0 else 1.0
                    if friction > _OPT_MAX_FRICTION:
                        logging.debug(
                            "  Skipping %s: friction %.0f%% > max %.0f%%",
                            contract_symbol, friction * 100, _OPT_MAX_FRICTION * 100,
                        )
                        return None
                    return mid
                elif ask > 0:
                    return ask
                elif bid > 0:
                    return bid
        except Exception as e:
            logging.warning(f"⚠️ Could not fetch option price for {contract_symbol}: {e}")
        return None

    def get_atm_option_greeks(self, symbol: str, contract_type) -> dict | None:
        """
        Fetch live Greeks (delta, theta, gamma, IV) for the nearest ATM contract.

        Uses Alpaca OptionSnapshotRequest.  Returns a dict on success, None on
        any failure (so callers can silently skip Greeks enrichment).
        """
        try:
            from alpaca.data.requests import OptionSnapshotRequest

            # Find ATM contract (reuse find_optimal_option with current price)
            analysis_price = None
            try:
                mdata = self.get_market_data(symbol)
                if mdata is not None and not mdata.empty:
                    analysis_price = float(mdata['close'].iloc[-1])
            except Exception:
                pass
            if analysis_price is None:
                return None

            contract = self.find_optimal_option(symbol, contract_type, analysis_price)
            if not contract:
                return None

            snapshots = self.option_data_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=contract.symbol)
            )
            snap = snapshots.get(contract.symbol) if snapshots else None
            if not snap:
                return None

            greeks = getattr(snap, 'greeks', None)
            iv     = getattr(snap, 'implied_volatility', None)
            if not greeks and iv is None:
                return None

            return {
                'contract':  contract.symbol,
                'delta':     round(float(greeks.delta), 3)  if greeks and greeks.delta  is not None else None,
                'gamma':     round(float(greeks.gamma), 4)  if greeks and greeks.gamma  is not None else None,
                'theta':     round(float(greeks.theta), 4)  if greeks and greeks.theta  is not None else None,
                'vega':      round(float(greeks.vega),  4)  if greeks and greeks.vega   is not None else None,
                'iv':        round(float(iv), 4)            if iv is not None else None,
            }
        except Exception as e:
            logging.debug(f"get_atm_option_greeks({symbol}): {e}")
            return None

    def execute_options_trade(self, symbol, decision, analysis, available_capital, size_multiplier: float = 1.0):
        """Execute options trade."""
        reservation_id = None
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

            # Override max_position_size with allocation-controller tier
            alloc_pct = self.alloc_controller.get_position_size_pct()
            effective_max_position = alloc_pct * size_multiplier
            original_max = self.params['max_position_size']
            self.params['max_position_size'] = effective_max_position

            # Calculate position size (scaled by research multiplier)
            quantity = self.calculate_position_size(option_price, available_capital, size_multiplier)

            # Restore original param so other code paths are unaffected
            self.params['max_position_size'] = original_max

            if quantity == 0:
                logging.warning(f"⚠️  Position size too small for {symbol}")
                return False

            # Each contract = 100 shares, so estimated cost = price × 100 × qty
            estimated_cost = option_price * 100 * quantity

            # Clean up stale cross-bot reservations before re-checking cash
            _db.cleanup_stale_reservations(max_age_seconds=120)

            # Re-fetch live cash + subtract any active cross-bot reservations.
            # available_capital was computed earlier and can be stale if the stock
            # bot spent cash concurrently.
            live_cash = float(self.trading_client.get_account().cash)
            already_reserved = _db.get_total_reserved()
            effective_cash = live_cash - already_reserved
            if effective_cash - estimated_cost <= _MIN_CASH_RESERVE:
                logging.warning(
                    f"🛑 Aborting options buy for {symbol}: effective cash "
                    f"${effective_cash:,.2f} (live=${live_cash:,.2f} "
                    f"− reserved=${already_reserved:,.2f}) − cost ${estimated_cost:,.2f} "
                    f"would breach ${_MIN_CASH_RESERVE:,.2f} reserve. Skipping."
                )
                return False

            # Atomically reserve this cash so the stock bot can't double-spend it
            reservation_id = _db.reserve_cash('options', estimated_cost)

            # Paper-trading realism: simulate options spread, partial fills, non-fills
            indicators = decision.get('indicators', {})
            fill_sim = self.paper_sim.simulate_options_fill(
                side='buy',
                mid_price=option_price,
                contracts=quantity,
                open_interest=int(indicators.get('open_interest', 0)),
                daily_volume=int(indicators.get('volume', 0)),
            )
            self.paper_sim.log_fill(contract.symbol, fill_sim)

            if not fill_sim.filled:
                logging.warning(f"⚠️  {contract.symbol} options non-fill simulated — skipping trade")
                return False

            quantity   = fill_sim.fill_qty
            fill_price = fill_sim.fill_price
            estimated_cost = fill_price * 100 * quantity

            if quantity == 0:
                logging.warning(f"⚠️  Options partial fill resulted in 0 contracts for {symbol}")
                return False

            order = LimitOrderRequest(
                symbol=contract.symbol,
                qty=quantity,
                side=OrderSide.BUY,
                limit_price=round(fill_price, 2),
                time_in_force=TimeInForce.DAY,
            )

            # DRY RUN: log the order without submitting it
            if _DRY_RUN:
                dry_id = f"dry-run-{datetime.now().strftime('%H%M%S%f')}"
                logging.info(
                    f"[DRY RUN] Would submit BUY {quantity} contracts of "
                    f"{contract.symbol} @ ${fill_price:.4f} limit "
                    f"(slippage {fill_sim.slippage_bps:.0f}bps, est. cost=${estimated_cost:.2f})"
                )
                submitted_order = type('_DryOrder', (), {'id': dry_id})()
            else:
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
                f.write(json.dumps(trade_log, default=_json_default) + '\n')
            try:
                _db.insert_trade(trade_log, bot='options', source='paper' if getattr(self, '_paper', True) else 'live')
            except Exception as e:
                logging.warning(f"⚠️  Could not write options trade to DB: {e}")
            
            self._last_order_id = str(submitted_order.id)

            # C2: submit a GTC stop-limit SELL immediately after entry so the position
            # is protected even if the daemon goes offline.  The stop fires at the same
            # percentage threshold as the software stop_loss param.
            if os.getenv('OPTIONS_GTC_STOP', 'true').lower() == 'true':
                stop_trigger = round(fill_price * (1 + self.params['stop_loss']), 2)
                stop_limit   = round(stop_trigger * 0.90, 2)  # 10% room for wide option spreads
                if _DRY_RUN:
                    logging.info(
                        f"[DRY RUN] Would submit GTC stop-limit SELL: {contract.symbol[:20]} "
                        f"stop=${stop_trigger:.4f} limit=${stop_limit:.4f}"
                    )
                else:
                    try:
                        stop_order = StopLimitOrderRequest(
                            symbol=contract.symbol,
                            qty=quantity,
                            side=OrderSide.SELL,
                            stop_price=stop_trigger,
                            limit_price=stop_limit,
                            time_in_force=TimeInForce.GTC,
                        )
                        self.trading_client.submit_order(stop_order)
                        logging.info(
                            f"🛡️  GTC stop-limit SELL placed: {contract.symbol[:20]} "
                            f"stop=${stop_trigger:.4f} limit=${stop_limit:.4f}"
                        )
                    except Exception as _se:
                        logging.warning(
                            f"⚠️  GTC stop submission failed for {contract.symbol[:20]}: {_se}"
                        )

            logging.info(f"✅ Executed OPTIONS trade: {quantity} contracts of {symbol} {contract_type.value}")
            alert_trade_executed(
                'OptionsAgent', contract.symbol, 'buy',
                quantity, option_price, str(submitted_order.id),
                target_price=round(option_price * (1 + self.params['take_profit']), 4),
                stop_price=round(option_price * (1 + self.params['stop_loss']), 4),
            )
            return True

        except Exception as e:
            logging.error(f"❌ Options trade execution failed: {e}")
            alert_trade_failed('OptionsAgent', symbol, str(e))
            return False
        finally:
            # Always release the cash reservation, whether the order succeeded or failed
            if reservation_id is not None:
                try:
                    _db.release_cash(reservation_id)
                except Exception as rel_err:
                    logging.warning(f"⚠️  Could not release options cash reservation {reservation_id}: {rel_err}")

    def _has_open_exit_order(self, symbol: str) -> bool:
        """
        Return True if a live SELL order already exists for this option symbol.
        Prevents duplicate exit submissions when a GTC order is still pending
        from a prior session.  Defaults to False on API error so the exit is
        never silently blocked.
        """
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status='open')
            )
            return any(
                order.symbol == symbol and order.side == OrderSide.SELL
                for order in open_orders
            )
        except Exception as e:
            logging.warning(f"Could not check open orders for {symbol}: {e}")
            return False

    def _cancel_gtc_stops(self, symbol: str) -> None:
        """Cancel pending GTC stop-limit SELL orders for symbol.

        Called before agent-initiated exits so the passive broker stop does not
        block _has_open_exit_order.  Only cancels orders with a stop_price set
        (stop-limit orders); plain GTC limit orders (DTE exits) are untouched.
        """
        try:
            open_orders = self.trading_client.get_orders(
                GetOrdersRequest(status='open')
            )
            for order in open_orders:
                if (order.symbol == symbol
                        and order.side == OrderSide.SELL
                        and order.time_in_force == TimeInForce.GTC
                        and getattr(order, 'stop_price', None) is not None):
                    self.trading_client.cancel_order_by_id(str(order.id))
                    logging.info(
                        f"🗑️  Cancelled GTC stop order for {symbol[:20]}: {order.id}"
                    )
        except Exception as e:
            logging.warning(f"⚠️  Could not cancel GTC stops for {symbol[:20]}: {e}")

    def _opened_today(self, symbol: str) -> bool:
        """True if a filled BUY order for this options contract was placed today (same-day PDT guard)."""
        try:
            today = datetime.now(_EST).date() if _EST else datetime.now().date()
            today_start = datetime(today.year, today.month, today.day, 0, 0, 0)
            orders = self.trading_client.get_orders(
                GetOrdersRequest(status='closed', after=today_start, symbols=[symbol])
            )
            return any(o.side == OrderSide.BUY for o in orders)
        except Exception as e:
            logging.debug(f"_opened_today check failed for {symbol}: {e}")
            return False

    def _fill_timeout_retry(self, submitted_order, position, qty: int, limit_price: float):
        """Wait 5 minutes; cancel and resubmit at 99% of limit_price if not filled."""
        time.sleep(300)
        try:
            order_status = self.trading_client.get_order_by_id(str(submitted_order.id))
            if order_status.status.value not in ('filled', 'partially_filled'):
                self.trading_client.cancel_order_by_id(str(submitted_order.id))
                aggressive_price = round(limit_price * 0.99, 2)
                retry_order = LimitOrderRequest(
                    symbol=position.symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    limit_price=aggressive_price,
                    time_in_force=TimeInForce.DAY,
                )
                submitted_order = self.trading_client.submit_order(retry_order)
                logging.info(
                    f"🔄 Resubmitted exit at ${aggressive_price:.2f} (5-min timeout retry)"
                )
        except Exception as e:
            logging.warning(
                f"⚠️  Fill timeout check failed for {position.symbol[:20]}...: {e}"
            )
        return submitted_order

    def manage_existing_positions(self):
        """Monitor and manage existing options positions."""
        try:
            positions = self.trading_client.get_all_positions()

            for position in positions:
                # Options have long OCC symbols (e.g. SPY250328P00560000); stocks are short
                if len(position.symbol) <= 10:
                    continue

                qty = float(position.qty)
                unrealized_plpc = float(position.unrealized_plpc)

                logging.info(f"📊 Option: {position.symbol[:20]}... P&L: {unrealized_plpc:+.1%}")

                should_exit = False
                exit_reason = None
                time_in_force = TimeInForce.DAY  # default for P&L exits

                # Take profit at +50%
                if unrealized_plpc >= self.params['take_profit']:
                    should_exit = True
                    exit_reason = "take_profit_50%"

                # Stop loss at -50%
                elif unrealized_plpc <= self.params['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss_50%"

                # DTE exit — only evaluated when P&L thresholds are not triggered
                if not should_exit:
                    dte = self.parse_dte_from_symbol(position.symbol)
                    threshold = self.params['exit_dte_threshold']
                    if dte is not None and dte <= threshold:
                        should_exit = True
                        exit_reason = f"dte_exit_{dte}d_remaining"
                        # GTC so the order persists if the intraday fill fails —
                        # we must exit before the contract expires worthless
                        time_in_force = TimeInForce.GTC

                if not should_exit:
                    continue

                # no_same_day_close guard: small accounts must not round-trip the same
                # option contract on the same calendar day — counts as a PDT day trade.
                if self.params.get('no_same_day_close') and self._opened_today(position.symbol):
                    logging.info(
                        f"⏭️  no_same_day_close: {position.symbol[:20]}... opened today — "
                        f"skipping {exit_reason} (will retry tomorrow)"
                    )
                    continue

                # Cancel any pending GTC stop orders before the duplicate-exit check.
                # Agent-initiated exits (P&L/DTE thresholds) take priority over passive stops.
                self._cancel_gtc_stops(position.symbol)

                # Skip if an open sell order still exists after stop cancellation
                # (e.g. a manually-submitted exit or a DTE GTC from a prior session).
                if self._has_open_exit_order(position.symbol):
                    logging.info(
                        f"⏭️  Skipping exit for {position.symbol[:20]}... — "
                        f"open sell order already exists ({exit_reason})"
                    )
                    continue

                # Near-zero contract check: if the total recovery value is below the
                # minimum exit threshold, skip the close order to avoid a wasted
                # commission (the contract will expire worthless, which is the same
                # financial outcome but without the order rejection risk on illiquid
                # near-zero options).
                current_price = self.get_option_price(position.symbol)
                if current_price is not None:
                    total_value = current_price * 100 * int(qty)  # 100 shares per contract
                    if total_value < _MIN_EXIT_VALUE_USD:
                        logging.warning(
                            f"⚠️  Skipping exit for {position.symbol[:20]}... — "
                            f"total value ${total_value:.2f} < min ${_MIN_EXIT_VALUE_USD:.2f}. "
                            f"Will expire worthless (same outcome, no commission)."
                        )
                        continue
                    if current_price < 0.05:
                        logging.warning(
                            f"⚠️  Near-zero contract: {position.symbol[:20]}... "
                            f"(${current_price:.2f}/share, total=${total_value:.2f}). "
                            f"Exiting to recover remaining value."
                        )

                # Prefer a limit order at the mid-price to avoid wide-spread
                # market-order slippage on illiquid contracts.  Fall back to
                # market if we genuinely cannot determine a price.
                try:
                    limit_price = current_price if current_price is not None else (
                        float(position.current_price) if position.current_price else 0.0
                    )
                except (TypeError, ValueError):
                    limit_price = 0.0
                if limit_price > 0:
                    order = LimitOrderRequest(
                        symbol=position.symbol,
                        qty=int(qty),
                        side=OrderSide.SELL,
                        limit_price=round(limit_price, 2),
                        time_in_force=time_in_force,
                    )
                else:
                    logging.warning(
                        f"⚠️  No price available for {position.symbol[:20]}...; "
                        f"falling back to market order"
                    )
                    order = MarketOrderRequest(
                        symbol=position.symbol,
                        qty=int(qty),
                        side=OrderSide.SELL,
                        time_in_force=time_in_force,
                    )

                if _DRY_RUN:
                    dry_id = f"dry-run-{datetime.now().strftime('%H%M%S%f')}"
                    logging.info(
                        f"[DRY RUN] Would submit SELL {int(qty)} contracts of "
                        f"{position.symbol[:20]}... ({exit_reason}, order_id={dry_id})"
                    )
                    submitted_order = type('_DryOrder', (), {'id': dry_id})()
                else:
                    submitted_order = self.trading_client.submit_order(order)

                # 5-minute fill timeout for intraday (DAY) limit exits only.
                # GTC exits must remain open until filled — no retry.
                if not _DRY_RUN and limit_price > 0 and time_in_force == TimeInForce.DAY:
                    submitted_order = self._fill_timeout_retry(
                        submitted_order, position, int(qty), limit_price
                    )

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
                    f.write(json.dumps(trade_log, default=_json_default) + '\n')

                logging.info(f"✅ Closed option position: {exit_reason} ({unrealized_plpc:+.1%})")

                unrealized_pl  = float(position.unrealized_pl)
                avg_entry      = float(position.avg_entry_price)
                exit_price     = float(position.current_price)
                alert_trade_executed(
                    'OptionsAgent', position.symbol, 'sell',
                    int(qty), exit_price, str(submitted_order.id),
                    pnl=unrealized_pl, pnl_pct=unrealized_plpc,
                    avg_entry_price=avg_entry,
                )

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

        # Compute market regime once per session for decision tagging
        self._session_regime = _compute_regime()
        if self._session_regime:
            logging.info(f"📊 Market regime: {self._session_regime}")

        # Fetch equity once at session start — drives account_config param selection
        # and the circuit breaker further below.
        try:
            _session_account = self.trading_client.get_account()
            equity = float(_session_account.equity)
        except Exception as _e:
            logging.error(f"❌ Could not fetch account equity at session start: {_e}")
            equity = 0.0

        # Gate: stay on paper until account reaches OPTIONS_LIVE_THRESHOLD.
        # Allows the stock bot to grow the account before options capital is risked.
        if not getattr(self, '_paper', True) and not _acfg.options_live_allowed(equity):
            logging.warning(
                f"⚠️  OPTIONS PAPER FORCED: equity ${equity:,.0f} is below "
                f"the ${_acfg.OPTIONS_LIVE_THRESHOLD:,.0f} options threshold — "
                f"running on paper endpoint this session."
            )
            from alpaca.trading.client import TradingClient as _TC
            self.trading_client = _TC(
                os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True
            )

        # Apply account-tier params (small <$25k vs full) and fire upgrade alert if crossed.
        self.params.update(_acfg.get_options_params(equity))
        _acfg.check_for_upgrade(equity, self._prev_equity)
        self._prev_equity = equity

        # Load overnight research params first so exit_dte_threshold override is applied
        # before manage_existing_positions() runs.
        research       = self.load_research_params()
        min_conf       = research.get('min_confidence_override', self.params['min_confidence'])
        size_mult      = research.get('position_size_multiplier', 1.0)
        market_bias    = research.get('market_bias', 'neutral')
        avoid_earnings = set(research.get('avoid_earnings_risk', []))
        symbol_bias    = research.get('symbol_bias', {})
        # Gap 3 (research override): allow monday_params_options.json to tighten the
        # DTE exit window during high-volatility weeks
        self.params['exit_dte_threshold'] = research.get(
            'exit_dte_threshold', self.params['exit_dte_threshold']
        )

        # Step 1: Manage existing positions — runs BEFORE the circuit breaker so
        # DTE exits still fire on bad P&L days (defensive, not speculative).
        self.manage_existing_positions()

        # C3: write reconcile status after the position sweep so health monitoring
        # and morning_model_check have an up-to-date unprotected-positions count.
        try:
            from risk_reconciler import write_reconcile_status
            write_reconcile_status(
                self.trading_client,
                Path('logs/reconcile_status.json'),
            )
        except Exception as _re:
            logging.debug("Could not write reconcile_status: %s", _re)

        # Circuit breaker: check daily P&L before opening any new positions
        # (equity was already fetched at session start above)
        try:
            if self.daily_start_equity is None:
                saved = _db.load_daily_start_equity('options', source='paper' if getattr(self, '_paper', True) else 'live')
                if saved is not None:
                    self.daily_start_equity = saved
                    logging.info(f"📌 Restored daily starting equity from DB: ${saved:,.2f}")
                else:
                    self.daily_start_equity = equity
                    _db.save_daily_start_equity('options', equity, source='paper' if getattr(self, '_paper', True) else 'live')
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

        # ── Macro guard (Fix B/C) ──────────────────────────────────────────────
        macro_events = economic_calendar.get_todays_high_impact_events()
        halt, halt_reason = economic_calendar.should_halt_trading(macro_events)
        if halt:
            logging.warning(f"🛑 MACRO GUARD: {halt_reason} — halting options session")
            return

        # For options: tighten min_conf by +0.05 whenever any high-impact event
        # exists today (even outside the halt window) — options are more sensitive
        # to vol spikes than stocks.
        if macro_events:
            min_conf = min(min_conf + 0.05, 0.99)
            logging.info(
                f"📅 Macro events today — tightening options min_conf to {min_conf:.0%}"
            )

        # ── Session-level context blocks (Fix D / Fix E) ─────────────────────
        macro_context_block   = _load_macro_context_block()
        weekend_context_block = _load_weekend_context_block()
        _options_earnings_list = economic_calendar.get_earnings_today(self.watchlist)
        live_earnings_symbols  = set(_options_earnings_list)  # hard BUY_CALL/BUY_PUT block below
        macro_guard_block      = economic_calendar.format_macro_guard_block(
            macro_events,
            _options_earnings_list,
        )
        if macro_guard_block:
            logging.info(f"📅 Options macro guard:\n{macro_guard_block}")
        if live_earnings_symbols:
            logging.info(f"🛑 Options earnings hard-block active for: {sorted(live_earnings_symbols)}")

        logging.info(f"📊 Daily options trades: {self.daily_trades}/{self.params['max_daily_trades']}")

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

            # Fetch enrichment snippets (both are no-ops if APIs unavailable)
            news_snippet = news_fetcher.format_for_prompt(symbol)
            flow_result  = unusual_flow_scanner.scan_symbol(symbol)
            flow_snippet = unusual_flow_scanner.format_for_prompt(symbol, flow_result)

            # Fetch live Greeks for ATM contract (best-effort; None if unavailable)
            # Used to enrich the AI prompt with IV crush / theta context.
            preliminary_ctype = (
                __import__('alpaca.trading.enums', fromlist=['ContractType']).ContractType.CALL
            )
            greeks = self.get_atm_option_greeks(symbol, preliminary_ctype)

            # E3: compute IV rank and upsert snapshot for historical tracking
            iv_rank = None
            if greeks and greeks.get('iv') is not None:
                iv_val = float(greeks['iv'])
                iv_rank = _db.get_iv_rank(symbol, iv_val)
                try:
                    _db.upsert_iv_snapshot(symbol, iv_val)
                except Exception as _iv_err:
                    logging.debug("Could not upsert IV snapshot for %s: %s", symbol, _iv_err)

            # Get AI decision — returns (decision, prompt, raw_response) in one LLM call
            sym_bias = symbol_bias.get(symbol, 'neutral')
            decision, prompt, raw_response = self.get_ai_options_decision(
                analysis, market_bias=market_bias, symbol_bias=sym_bias,
                news_snippet=news_snippet, flow_snippet=flow_snippet,
                greeks=greeks, iv_rank=iv_rank,
                macro_context_block=macro_context_block,
                weekend_context_block=weekend_context_block,
                macro_guard_block=macro_guard_block,
            )

            logging.info(f"📊 {symbol}: {decision['decision'].upper()} "
                         f"(confidence: {decision['confidence']:.0%}, "
                         f"bias: {sym_bias}, min_conf: {min_conf:.0%})")

            # Multi-agent debate for high-conviction trades
            debate_result = None
            will_execute = decision['decision'] in ['buy_call', 'buy_put'] and decision['confidence'] >= min_conf
            # Hard-block new BUY_CALL/BUY_PUT on live earnings (in addition to research file check above)
            if will_execute and symbol in live_earnings_symbols:
                logging.warning(f"🛑 EARNINGS BLOCK: {symbol} has earnings today/tomorrow — {decision['decision'].upper()} blocked")
                will_execute = False
            if will_execute and decision['confidence'] >= _DEBATE_CONFIDENCE_THRESHOLD:
                logging.info(f"⚖️  Debating {symbol} ({decision['decision'].upper()} @ {decision['confidence']:.0%})...")
                debate_result = debate_trade(
                    symbol, decision['decision'],
                    decision['confidence'], decision.get('reasoning', '')
                )
                if debate_result['verdict'] == 'ABORT':
                    logging.info(f"🚫 Debate ABORTED {symbol} options trade: {debate_result['reason'][:80]}...")
                    will_execute = False
                else:
                    logging.info(f"✅ Debate confirmed {symbol}: PROCEED")

            executed = False
            market_ctx = {'market_bias': market_bias, 'symbol_bias': sym_bias}
            if will_execute:
                # Portfolio concentration guard before executing
                try:
                    position_pct = self.alloc_controller.get_position_size_pct()
                    spend_usd = available_capital * position_pct
                    allowed, veto_reason = self.overseer.is_buy_allowed(symbol, spend_usd, equity)
                    if not allowed:
                        logging.warning(f"🚫 Portfolio overseer vetoed {symbol}: {veto_reason}")
                        will_execute = False
                except Exception as ov_err:
                    logging.warning(f"overseer check failed for {symbol}: {ov_err}")
            if will_execute:
                executed = self.execute_options_trade(symbol, decision, analysis, available_capital,
                                                      size_multiplier=size_mult)
                if executed:
                    self.daily_trades += 1
                    trades_executed += 1
                    self.trade_cooldown[symbol] = datetime.now()
                    available_capital = self.get_available_capital()
                    if available_capital < 100:
                        self._log_decision(symbol, prompt, raw_response, decision, analysis,
                                           market_ctx, executed, debate=debate_result,
                                           order_id=self._last_order_id)
                        break
            else:
                if debate_result and debate_result['verdict'] == 'ABORT':
                    logging.info(f"⏭️  Skipping {symbol}: debate aborted (confidence was {decision['confidence']:.0%})")
                else:
                    logging.info(f"⏭️  Skipping {symbol}: {decision['decision']} "
                                 f"(confidence: {decision['confidence']:.0%} below threshold {min_conf:.0%})")

            # Log every decision (including HOLDs) for replay and validation
            self._log_decision(symbol, prompt, raw_response, decision, analysis,
                               market_ctx, executed, debate=debate_result,
                               order_id=self._last_order_id if executed else None)
        
        logging.info(f"✅ Options session complete: {trades_executed} trades executed")

if __name__ == "__main__":
    agent = OptionsAgent()
    agent.run_options_session()
