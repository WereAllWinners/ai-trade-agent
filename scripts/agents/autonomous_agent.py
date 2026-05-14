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
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# Add scripts/ root and all subdirs to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

from model_inference_lora import get_trading_decision, parse_decision, load_model_once
from stock_discovery import StockDiscovery
from alerts import alert_circuit_breaker, alert_trade_executed, alert_trade_failed
import news_fetcher
import economic_calendar
import db as _db
from fee_simulator import FeeSimulator
from paper_market_simulator import PaperMarketSimulator
from alpaca.trading.requests import LimitOrderRequest
from portfolio_overseer import PortfolioOverseer
from allocation_controller import AllocationController
import account_config as _acfg

_DEBATE_CONFIDENCE_THRESHOLD = 0.90   # Only debate very high-conviction trades
_MIN_CASH_RESERVE = 500.0             # Never spend the account's last $500 of cash
_DRY_RUN = os.getenv('DRY_RUN', 'false').lower() == 'true'  # log orders but never submit
_PDT_DAY_TRADE_LIMIT = 3              # Max same-day round trips before PDT block kicks in
_EXPLORATION_SHARPE_THRESHOLD = 1.0   # Inject novel symbols when recent Sharpe is below this
_EXPLORATION_COUNT = 2                # Number of random symbols to inject per session
_ALLOC_CONFIG = Path(__file__).resolve().parent.parent / 'logs' / 'allocation_config.json'

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


def _load_macro_context_block() -> str:
    """Read the most recent trading macro context entry and format it for prompts.

    Reads logs/market_context_trading.json (written nightly by market_researcher).
    Returns '' if the file is missing or its newest entry is stale (>36 h old).
    Cached per call — callers should invoke once per session and reuse.
    """
    try:
        path = Path('logs/market_context_trading.json')
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

        macro    = latest.get('macro', {})
        vix_d    = macro.get('vix', {})
        spy_d    = macro.get('spy', {})
        vix_val  = vix_d.get('current', '?')
        vix_reg  = vix_d.get('regime', '?')
        spy_trend = spy_d.get('trend', '?')
        spy_ma   = 'above 50MA' if spy_d.get('above_50ma') else 'below 50MA'

        line1 = f"Regime: {spy_trend} | VIX: {vix_val} ({vix_reg}) | SPY: {spy_ma}"

        # Additional context: top momentum + oversold tickers from nightly scan
        momentum = latest.get('top_momentum_tickers', [])[:5]
        oversold = latest.get('oversold_tickers', [])[:3]
        parts = []
        if momentum:
            parts.append(f"Momentum leaders: {', '.join(momentum)}")
        if oversold:
            parts.append(f"Oversold: {', '.join(oversold)}")
        line2 = ' | '.join(parts)

        lines = ['Market Context (as of last night):', f'  {line1}']
        if line2:
            lines.append(f'  {line2}')
        return '\n'.join(lines)
    except Exception as e:
        logging.debug(f"Could not load macro context block: {e}")
        return ''


def _load_weekend_context_block() -> str:
    """Read the most recent weekend strategist entry and format it for prompts.

    Reads logs/weekend_context_options.json (written Saturday by the weekend
    strategist).  Only injected Mon–Fri before 2 PM and when the file is ≤5
    days old — by Friday the weekend research is stale.
    Returns '' if conditions are not met or on any error.
    """
    try:
        now = datetime.now()
        # Only valid on weekdays (Mon=0 … Fri=4) before 2 PM local time
        if now.weekday() >= 5 or now.hour >= 14:
            return ''

        path = Path('logs/weekend_context_options.json')
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

        # Sector summary
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

        # Congressional signals
        cong_bull = params.get('congressional_bullish', [])[:5]
        cong_bear = params.get('congressional_bearish', [])[:5]
        cong_parts = []
        if cong_bull:
            cong_parts.append(f"bullish {', '.join(cong_bull)}")
        if cong_bear:
            cong_parts.append(f"bearish {', '.join(cong_bear)}")
        cong_line = f"Congressional: {' | '.join(cong_parts)}" if cong_parts else ''

        # IV elevated tickers
        elevated = [s for s, v in iv_analysis.items() if v.get('vol_regime') == 'elevated'][:4]
        iv_line = (f"IV regime: elevated on {', '.join(elevated)} — premium buying expensive"
                   if elevated else '')

        # Earnings this week
        earn_parts = [f"{e['symbol']} ({e.get('earnings_in_days', '?')}d)"
                      for e in earnings]
        earn_line = (f"Earnings this week: {', '.join(earn_parts)} — avoid new positions day-before"
                     if earn_parts else '')

        content_lines = [l for l in [sectors_line, cong_line, iv_line, earn_line] if l]
        if not content_lines:
            return ''
        lines = ['Weekend Research Context:'] + [f'  {l}' for l in content_lines]
        return '\n'.join(lines)
    except Exception as e:
        logging.debug(f"Could not load weekend context block: {e}")
        return ''


def _write_rotation_log(sell_sym, buy_sym, sell_oid, buy_oid, outcome):
    """Append a rotation event record to logs/rotation_log.jsonl."""
    record = {
        'timestamp': datetime.now().isoformat(),
        'sell_symbol': sell_sym,
        'buy_symbol':  buy_sym,
        'sell_order_id': sell_oid,
        'buy_order_id':  buy_oid,
        'outcome':       outcome,
    }
    try:
        os.makedirs('logs', exist_ok=True)
        with open('logs/rotation_log.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logging.warning(f"⚠️  Could not write rotation log: {e}")


def _settled_cash(account) -> float:
    """Return actual cash balance from Alpaca."""
    return float(account.cash)


def debate_trade(symbol: str, action: str, confidence: float, reasoning: str) -> dict:
    """
    Contrarian second LLM call — challenges a high-conviction trade.
    Returns {'verdict': 'PROCEED'|'ABORT', 'reason': str}.
    On failure defaults to PROCEED (safe fallback: don't block the trade).
    """
    debate_prompt = (
        f"You are a balanced risk/reward analyst reviewing a proposed trade.\n"
        f"Proposal: {action.upper()} {symbol}  (confidence: {confidence:.0%})\n"
        f"Argument: {reasoning}\n\n"
        f"Briefly state ONE supporting factor and ONE risk for this trade.\n"
        f"Then state your verdict on a NEW LINE as exactly one of:\n"
        f"VERDICT: PROCEED\n"
        f"VERDICT: ABORT\n"
        f"Choose ABORT only if the risk clearly and specifically outweighs the opportunity. "
        f"When in doubt, PROCEED."
    )
    try:
        text = get_trading_decision(debate_prompt)   # Now uses new LoRA + DPO model
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

        # PDT gate: stock bot must stay on the paper endpoint until equity >= $25,000.
        # If PAPER_TRADING=false was set prematurely, re-create the client as paper and warn.
        if not _paper:
            _equity = _acfg.fetch_equity(self.trading_client)
            if not _acfg.stock_live_allowed(_equity):
                logging.warning(
                    f"⚠️  PAPER TRADING FORCED: account equity ${_equity:,.0f} is below "
                    f"the ${_acfg.LIVE_EQUITY_THRESHOLD:,.0f} PDT threshold. "
                    f"Stock bot will run on the paper endpoint until the threshold is cleared."
                )
                _paper = True
                self.trading_client = TradingClient(
                    os.getenv('ALPACA_API_KEY'),
                    os.getenv('ALPACA_SECRET_KEY'),
                    paper=True,
                )

        from utils.alpaca_retry import retry_on_rate_limit
        for _m in ('submit_order', 'get_account', 'get_all_positions', 'get_orders', 'get_order_by_id'):
            if hasattr(self.trading_client, _m):
                method = getattr(self.trading_client, _m)
                if not hasattr(method, '_mock_name'):  # skip wrapping unittest.mock stubs
                    setattr(self.trading_client, _m, retry_on_rate_limit(method))
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
        self.pdt_blocked = False        # Set True if PDT restriction detected
        self.fee_simulator      = FeeSimulator(paper=_paper)
        self.paper_sim          = PaperMarketSimulator(paper=_paper)
        self._last_order_id: str | None = None
        self._paper             = _paper

        # Decision logging — unique ID per process run for replay correlation
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._decision_log = Path('logs/decision_log.jsonl')
        self._decision_log.parent.mkdir(exist_ok=True)

        if _DRY_RUN:
            logging.warning("🧪 DRY RUN MODE: orders will be logged but NOT submitted to Alpaca")

        # Pre-load the LoRA model so the first trading decision has no cold-start delay
        load_model_once()

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

        logging.info("🤖 Autonomous Agent Initialized")
    
    def load_parameters(self):
        """Load adaptive parameters (Monday params take precedence)."""
        # Default parameters
        defaults = {
            'max_position_size': 0.05,  # 5% per trade
            'min_confidence': 0.60,     # AI confidence threshold
            'max_stocks_to_analyze': 35,
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
            'model':        os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-32B-Instruct') + '+LoRA',
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
        try:
            _db.insert_decision(record)
        except Exception as e:
            logging.warning(f"⚠️  Could not write decision to DB: {e}")

    def _count_todays_roundtrips(self) -> int:
        """
        Count completed same-day SELL orders in the DB (proxy for round trips).
        Used for proactive PDT enforcement: if we've already done 3 day-trades
        today, block new buys before the broker rejects them.
        """
        today = datetime.now().date().isoformat()
        try:
            with _db.get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE bot='stock' AND action='sell' "
                    "AND timestamp LIKE ?",
                    [f"{today}%"]
                ).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

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
            'current_price': df['close'].iloc[-1],
            'avg_volume': float(avg_volume.iloc[-1]),
            'session_volume': float(current_volume),
        }
    
    def get_position(self, symbol):
        """Return the current position for symbol, or None if not held."""
        try:
            return self.trading_client.get_open_position(symbol)
        except Exception:
            return None

    def execute_trade(self, symbol, decision, equity, available_cash):
        """Execute a trade based on AI decision."""
        reservation_id = None
        try:
            side = OrderSide.BUY if decision['decision'] == 'buy' else OrderSide.SELL
            current_price = decision.get('current_price', 0)

            if current_price == 0:
                logging.error(f"❌ Cannot execute trade for {symbol}: no price data")
                return False

            # For sells, verify we actually hold the position
            avg_entry_price = None
            if side == OrderSide.SELL:
                position = self.get_position(symbol)
                if position is None:
                    logging.warning(f"⚠️  Skipping SELL {symbol}: no open position held")
                    return False
                shares = int(float(position.qty))
                avg_entry_price = float(position.avg_entry_price)
            else:
                # Proactive PDT check — count today's round trips before another buy
                if not self.pdt_blocked:
                    todays_roundtrips = self._count_todays_roundtrips()
                    if todays_roundtrips >= _PDT_DAY_TRADE_LIMIT:
                        logging.warning(
                            f"🚫 PDT pre-check: {todays_roundtrips} day-trades today "
                            f"(limit {_PDT_DAY_TRADE_LIMIT}) — blocking buy {symbol}"
                        )
                        self.pdt_blocked = True
                        return False

                # Clean up any stale cross-bot reservations (crash survivors)
                _db.cleanup_stale_reservations(max_age_seconds=120)

                # Re-fetch live cash + subtract any active cross-bot reservations.
                # This is the single source of truth for available cash.
                live_cash = float(self.trading_client.get_account().cash)
                already_reserved = _db.get_total_reserved()
                effective_cash = live_cash - already_reserved
                if effective_cash <= _MIN_CASH_RESERVE:
                    logging.warning(
                        f"🛑 BUY {symbol} blocked: effective cash ${effective_cash:,.2f} "
                        f"(live=${live_cash:,.2f} − reserved=${already_reserved:,.2f}) "
                        f"≤ reserve ${_MIN_CASH_RESERVE:,.2f}"
                    )
                    return False
                # Size against spendable effective cash only — never touch the reserve
                spendable = effective_cash - _MIN_CASH_RESERVE
                if spendable < current_price:
                    logging.warning(
                        f"⚠️  Skipping BUY {symbol}: spendable cash ${spendable:,.2f} "
                        f"< share price ${current_price:.2f}"
                    )
                    return False

                # Use allocation-controller tier to determine position size
                alloc_pct = self.alloc_controller.get_position_size_pct()
                position_value = spendable * alloc_pct
                shares = int(position_value / current_price)

                # Portfolio concentration guard
                allowed, veto_reason = self.overseer.is_buy_allowed(symbol, position_value, equity)
                if not allowed:
                    logging.warning(f"🚫 Portfolio overseer vetoed BUY {symbol}: {veto_reason}")
                    return False

                # Atomically reserve this cash so the options bot can't double-spend it
                reservation_id = _db.reserve_cash('stock', position_value)

            if shares == 0:
                logging.warning(f"⚠️  Position too small for {symbol}")
                return False

            # Paper-trading realism: simulate slippage, partial fills, non-fills
            indicators = decision.get('indicators', {})
            avg_volume     = indicators.get('avg_volume', 1_000_000)
            session_volume = indicators.get('session_volume', avg_volume)
            fill_sim = self.paper_sim.simulate_stock_fill(
                side=side.value,
                price=current_price,
                qty=shares,
                adv=avg_volume,
                session_volume=session_volume,
            )
            self.paper_sim.log_fill(symbol, fill_sim)

            if not fill_sim.filled:
                logging.warning(f"⚠️  {symbol} order non-fill simulated — skipping trade")
                return False

            # Apply partial fill to share count
            shares = fill_sim.fill_qty
            fill_price = fill_sim.fill_price

            if shares == 0:
                logging.warning(f"⚠️  Partial fill resulted in 0 shares for {symbol}")
                return False

            # Bracket order at the simulated fill price with broker-native
            # stop-loss and take-profit child orders. Alpaca manages the exits
            # automatically so we don't rely on the LLM to re-examine positions.
            stop_price   = round(fill_price * (1 + self.params['stop_loss']), 2)
            target_price = round(fill_price * (1 + self.params['take_profit']), 2)
            order = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=side,
                limit_price=round(fill_price, 2),
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=target_price),
            )

            # DRY RUN: log the order without submitting it
            if _DRY_RUN:
                dry_id = f"dry-run-{datetime.now().strftime('%H%M%S%f')}"
                logging.info(
                    f"[DRY RUN] Would submit {side.value.upper()} {shares} {symbol} "
                    f"@ ${fill_price:.4f} limit (slippage {fill_sim.slippage_bps:.1f}bps)"
                )
                submitted_order = type('_DryOrder', (), {'id': dry_id})()
            else:
                submitted_order = self.trading_client.submit_order(order)

            self._last_order_id = str(submitted_order.id)

            # Log trade
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': decision['decision'],
                'shares': shares,
                'confidence': decision['confidence'],
                'reasoning': decision['reasoning'],
                'order_id': str(submitted_order.id),
                'bracket_stop': stop_price,
                'bracket_take_profit': target_price,
            }

            with open('logs/trade_log.jsonl', 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
            try:
                _db.insert_trade(trade_log, bot='stock')
            except Exception as e:
                logging.warning(f"⚠️  Could not write trade to DB: {e}")

            logging.info(f"✅ Executed {decision['decision'].upper()} {shares} shares of {symbol}")
            if side == OrderSide.SELL and avg_entry_price is not None:
                realized_pnl = (fill_price - avg_entry_price) * shares
                realized_pnl_pct = (fill_price - avg_entry_price) / avg_entry_price
                alert_trade_executed(
                    'StockAgent', symbol, decision['decision'],
                    shares, fill_price, str(submitted_order.id),
                    pnl=realized_pnl, pnl_pct=realized_pnl_pct,
                    avg_entry_price=avg_entry_price,
                )
            else:
                alert_trade_executed(
                    'StockAgent', symbol, decision['decision'],
                    shares, fill_price, str(submitted_order.id),
                    stop_price=stop_price, target_price=target_price,
                )

            # Update cooldown
            self.cooldowns[symbol] = datetime.now()
            if side == OrderSide.BUY:
                self.daily_trades += 1

            return True

        except Exception as e:
            err = str(e)
            if '40310000' in err or 'daytrading_buying_power' in err or 'insufficient day trading' in err.lower():
                logging.warning(f"🚫 PDT restriction active — halting buy orders for today. Will resume tomorrow.")
                self.pdt_blocked = True
            logging.error(f"❌ Trade execution failed for {symbol}: {e}")
            alert_trade_failed('StockAgent', symbol, err)
            return False
        finally:
            # Always release the cash reservation, whether the order succeeded or failed
            if reservation_id is not None:
                try:
                    _db.release_cash(reservation_id)
                except Exception as rel_err:
                    logging.warning(f"⚠️  Could not release cash reservation {reservation_id}: {rel_err}")
    
    # ── Position rotation helpers ──────────────────────────────────────────────

    def _score_position(self, symbol: str, position) -> dict | None:
        """Re-analyse an existing holding with current indicators and AI.

        Returns a dict merging the AI decision with live position metrics,
        or None if data is unavailable.
        """
        try:
            df = self.get_market_data(symbol)
            if df is None:
                return None
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return None

            news_snippet = news_fetcher.format_for_prompt(symbol)
            prompt = f"""You are a decisive short-term stock trader. Analyze {symbol} and give a clear trading decision.

            Current Price: ${indicators['current_price']:.2f}
            RSI (14): {indicators['rsi']:.1f}  (>70 overbought, <30 oversold)
            MACD: {indicators['macd']:.2f}  (positive = bullish momentum)
            Volume Ratio: {indicators['volume_ratio']:.1f}x average  (>1.5 = elevated activity)
            Price Change (100 bars): {indicators['price_change_pct']:+.1f}%

            Discovery Signals: {', '.join(self.discovery.opportunities.get(symbol, []))}{news_snippet}

            Make a decisive call. Only use HOLD if the indicators are genuinely mixed with no clear edge.
            Use confidence above 0.70 when signals align, below 0.60 when signals conflict.

            Respond in this exact format:
            Decision: <BUY|SELL|HOLD>
            Confidence: <0.00-1.00>
            Reasoning: <one sentence explaining the key signal>"""

            response = get_trading_decision(prompt)
            decision = parse_decision(response)

            unrealized_pl  = float(position.unrealized_pl)
            market_value   = float(position.market_value)
            unrealized_pct = unrealized_pl / (market_value - unrealized_pl) if (market_value - unrealized_pl) else 0

            return {
                'symbol':         symbol,
                'decision':       decision['decision'],
                'confidence':     decision['confidence'],
                'reasoning':      decision.get('reasoning', ''),
                'current_price':  indicators['current_price'],
                'indicators':     indicators,
                'market_value':   market_value,
                'unrealized_pl':  unrealized_pl,
                'unrealized_pct': unrealized_pct,
                'qty':            int(float(position.qty)),
            }
        except Exception as e:
            logging.warning(f"⚠️  Could not score position {symbol}: {e}")
            return None

    def _find_weakest_position(self, scored_positions: list) -> dict | None:
        """Return the weakest holding from a list of scored position dicts.

        Weakness index (higher = weaker = better candidate to exit):
          sell_signal_weight    — 1.0 if AI says SELL, 0.5 if HOLD, 0.0 if BUY
          low_confidence_weight — (1.0 - AI confidence)
          unrealized_loss_weight — capped 0-1, where -10% loss → 1.0
        """
        if not scored_positions:
            return None

        for p in scored_positions:
            sell_w  = 1.0 if p['decision'] == 'sell' else (0.5 if p['decision'] == 'hold' else 0.0)
            conf_w  = 1.0 - p['confidence']
            loss_w  = min(max(-p['unrealized_pct'] / 0.10, 0.0), 1.0)
            p['weakness_score'] = sell_w + conf_w + loss_w

        return max(scored_positions, key=lambda p: p['weakness_score'])

    def _recent_avg_win_pct(self, lookback_days: int = 30, db_path=None) -> float:
        """Mean pnl_pct of winning trades in the last lookback_days days; fallback 0.05 if < 5 wins."""
        _FALLBACK = 0.05
        _MIN_WINS = 5
        try:
            cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
            with _db.get_conn(db_path or _db.DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT pnl_pct FROM outcomes WHERE won = 1 AND exit_timestamp > ?",
                    [cutoff],
                ).fetchall()
            if len(rows) >= _MIN_WINS:
                return float(sum(r['pnl_pct'] for r in rows) / len(rows))
        except Exception as e:
            logging.debug("Could not compute recent avg win pct: %s", e)
        return _FALLBACK

    def _attempt_rotation(
        self,
        new_symbol:   str,
        new_decision: dict,
        remaining_cash: float,
        equity: float,
    ) -> bool:
        """Sell the weakest existing holding to fund a new opportunity.

        Only executes if the net expected value of the swap exceeds a minimum
        threshold after accounting for simulated round-trip trading fees.

        Returns True if both the sell and buy completed successfully.
        """
        # Minimum net EV required to justify a rotation (1%)
        MIN_NET_EV     = 0.01
        assumed_return = self._recent_avg_win_pct()

        try:
            positions = self.trading_client.get_all_positions()
        except Exception as e:
            logging.warning(f"⚠️  Could not fetch positions for rotation: {e}")
            return False

        if not positions:
            return False

        # Score all holdings that are not on cooldown.
        # Skip options positions (long OCC symbols e.g. SPY250328P00560000) —
        # those belong to the options bot and cannot be rotated as stocks.
        scored = []
        for p in positions:
            if len(p.symbol) > 10:  # OCC option symbol — not a stock
                continue
            if self.check_cooldown(p.symbol):
                continue
            if p.symbol == new_symbol:
                continue
            score = self._score_position(p.symbol, p)
            if score:
                scored.append(score)

        if not scored:
            logging.info("⚙️  Rotation: no scoreable positions available")
            return False

        weakest = self._find_weakest_position(scored)
        if weakest is None:
            return False

        # Estimate full round-trip fee cost
        exit_price   = weakest['current_price']
        exit_shares  = weakest['qty']
        entry_price  = new_decision['current_price']
        entry_shares = max(int((exit_price * exit_shares) / entry_price), 1)

        fee_breakdown    = self.fee_simulator.estimate_round_trip_cost(
            exit_price, exit_shares, entry_price, entry_shares
        )
        total_fee_pct    = fee_breakdown['total_cost_pct']

        # Net expected value calculation
        new_ev            = new_decision['confidence'] * assumed_return
        weakest_hold_cost = (weakest['confidence'] * assumed_return
                             if weakest['decision'] == 'sell' else 0.0)
        net_ev            = new_ev + weakest_hold_cost - total_fee_pct

        logging.info(
            f"⚙️  Rotation eval: SELL {weakest['symbol']} "
            f"(weakness={weakest['weakness_score']:.2f}, "
            f"unrealized={weakest['unrealized_pct']:+.1%}) "
            f"→ BUY {new_symbol} | "
            f"assumed_return={assumed_return:.3f}, new_ev={new_ev:.3f}, "
            f"avoided_loss={weakest_hold_cost:.3f}, fees={total_fee_pct:.4%}, net_ev={net_ev:.3f}"
        )

        if net_ev < MIN_NET_EV:
            logging.info(
                f"⚙️  Rotation rejected: net_ev {net_ev:.3f} < "
                f"threshold {MIN_NET_EV:.3f}"
            )
            return False

        if new_decision['confidence'] <= weakest['confidence']:
            logging.info(
                f"⚙️  Rotation rejected: new confidence "
                f"{new_decision['confidence']:.2f} ≤ weakest "
                f"{weakest['confidence']:.2f}"
            )
            return False

        # Execute — sell first, then buy with the freed cash
        logging.info(
            f"🔄 Rotating: SELL {weakest['symbol']} → BUY {new_symbol} "
            f"(est. fees ${fee_breakdown['total_cost']:.2f})"
        )
        sell_decision = {
            'decision':      'sell',
            'confidence':    weakest['confidence'],
            'reasoning':     f"rotation out — weakest holding vs {new_symbol}",
            'current_price': exit_price,
        }
        sold = self.execute_trade(weakest['symbol'], sell_decision, equity, 0)
        if not sold:
            logging.warning(f"⚙️  Rotation abandoned: could not sell {weakest['symbol']}")
            return False

        sell_order_id = self._last_order_id

        # Re-fetch settled cash — sale proceeds may not be immediately reflected.
        try:
            fresh_account = self.trading_client.get_account()
            settled_cash = float(fresh_account.cash)
        except Exception as e:
            logging.warning(f"⚙️  Could not re-fetch cash after rotation sell: {e}")
            settled_cash = remaining_cash + (exit_price * exit_shares)

        entry_cost = entry_price * entry_shares
        if settled_cash < entry_cost:
            logging.warning(
                f"⚙️  Rotation buy skipped: settled cash ${settled_cash:,.2f} "
                f"< required ${entry_cost:,.2f} — sale proceeds not yet settled"
            )
            _write_rotation_log(
                weakest['symbol'], new_symbol, sell_order_id, None,
                'buy_skipped_insufficient_cash'
            )
            return False

        bought = self.execute_trade(new_symbol, new_decision, equity, settled_cash)
        buy_order_id = self._last_order_id
        _write_rotation_log(
            weakest['symbol'], new_symbol, sell_order_id, buy_order_id,
            'completed' if bought else 'buy_failed'
        )
        if not bought:
            logging.warning(
                f"⚙️  Rotation incomplete: sold {weakest['symbol']} "
                f"but failed to buy {new_symbol}"
            )
        return bought

    # ── Main session loop ──────────────────────────────────────────────────────

    def run_trading_session(self):
        """Run a single trading session - analyze and potentially trade."""
        # Reset daily counter if new day
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = today
            self.daily_start_equity = None
            self.pdt_blocked = False
            logging.info("🔄 Daily trade counter reset")

        # Get account info
        account = self.trading_client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        settled = _settled_cash(account)
        logging.info(
            f"💰 Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | "
            f"Settled: ${settled:,.2f} (T+1 pending: ${cash - settled:,.2f})"
        )

        # Hard gate: refuse all buys if cash is at or below the minimum reserve
        if cash <= _MIN_CASH_RESERVE:
            logging.warning(
                f"🛑 Cash is ${cash:,.2f} ≤ reserve ${_MIN_CASH_RESERVE:,.2f} "
                f"— no buys allowed until cash recovers."
            )
            return

        # Record starting equity for circuit breaker — restore from DB first so
        # a daemon restart mid-day doesn't reset the baseline to current equity.
        if self.daily_start_equity is None:
            saved = _db.load_daily_start_equity('stock')
            if saved is not None:
                self.daily_start_equity = saved
                logging.info(f"📌 Restored daily starting equity from DB: ${saved:,.2f}")
            else:
                self.daily_start_equity = equity
                _db.save_daily_start_equity('stock', equity)
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

        # ── Macro guard (Fix B/C) ──────────────────────────────────────────────
        macro_events = economic_calendar.get_todays_high_impact_events()
        halt, halt_reason = economic_calendar.should_halt_trading(macro_events)
        if halt:
            logging.warning(f"🛑 MACRO GUARD: {halt_reason} — halting session")
            return

        # ── Session-level context blocks (Fix D / Fix E) ─────────────────────
        # Loaded once; reused for every symbol in the opportunity loop.
        macro_context_block  = _load_macro_context_block()
        weekend_context_block = _load_weekend_context_block()

        # Format macro guard prompt fragment (empty when no events)
        macro_guard_block = economic_calendar.format_macro_guard_block(
            macro_events,
            economic_calendar.get_earnings_today([]),  # symbols added below after discovery
        )

        # Discover opportunities
        max_stocks = self.params['max_stocks_to_analyze']
        logging.info(f"🔍 Discovering top {max_stocks} opportunities...")
        
        opportunities = self.discovery.discover_opportunities(max_stocks=max_stocks)
        logging.info(f"✅ Found {len(opportunities)} opportunities")

        # Exploration: inject random symbols when recent Sharpe is below target
        try:
            alloc_data = json.loads(_ALLOC_CONFIG.read_text())
            current_sharpe = alloc_data.get('sharpe')
            if current_sharpe is not None and current_sharpe < _EXPLORATION_SHARPE_THRESHOLD:
                current_symbols = {
                    (opp['symbol'] if isinstance(opp, dict) else opp)
                    for opp in opportunities
                }
                explore = self.discovery.get_exploration_symbols(
                    n=_EXPLORATION_COUNT, exclude=current_symbols
                )
                if explore:
                    logging.info(
                        f"🔭 Sharpe={current_sharpe:.2f} < {_EXPLORATION_SHARPE_THRESHOLD} "
                        f"— injecting exploration symbols: {explore}"
                    )
                    opportunities = list(opportunities) + [
                        {'symbol': s, 'signals': ['[EXPLORATION]']} for s in explore
                    ]
        except Exception as e:
            logging.debug(f"Exploration check skipped: {e}")

        if not opportunities:
            logging.warning("⚠️  No opportunities found")
            return

        # Refresh earnings guard now that we have the symbol list
        opp_symbols = [
            (opp['symbol'] if isinstance(opp, dict) else opp)
            for opp in opportunities
        ]
        macro_guard_block = economic_calendar.format_macro_guard_block(
            macro_events,
            economic_calendar.get_earnings_today(opp_symbols),
        )
        if macro_guard_block:
            logging.info(f"📅 Macro guard active:\n{macro_guard_block}")

        # Analyze each opportunity
        trades_executed = 0
        remaining_cash = settled - _MIN_CASH_RESERVE  # track spendable cash from settled funds only

        for opp in opportunities:
            try:
                if isinstance(opp, str):
                    symbol = opp
                else:
                    symbol = opp['symbol']
            except Exception as e:
                logging.error(f"Error processing opportunity: {opp}. Error: {str(e)}")
                continue

            # Skip any OCC option symbols that leaked into the opportunity list
            # (options bot's positions have long symbols e.g. SPY250328P00560000)
            if len(symbol) > 10:
                logging.debug(f"⏭️  Skipping {symbol} — OCC option symbol, not a stock")
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

            # Build optional context segments (empty string = no contribution)
            _macro_ctx = f"\n{macro_context_block}\n" if macro_context_block else ''
            _weekend   = f"\n{weekend_context_block}" if weekend_context_block else ''
            _guard     = f"\n{macro_guard_block}\n" if macro_guard_block else ''

            # Create AI prompt (Fix D: macro context before Discovery Signals;
            # Fix E: weekend context after news; Fix C: guard before decision line)
            prompt = f"""You are a decisive short-term stock trader. Analyze {symbol} and give a clear trading decision.

            Current Price: ${indicators['current_price']:.2f}
            RSI (14): {indicators['rsi']:.1f}  (>70 overbought, <30 oversold)
            MACD: {indicators['macd']:.2f}  (positive = bullish momentum)
            Volume Ratio: {indicators['volume_ratio']:.1f}x average  (>1.5 = elevated activity)
            Price Change (100 bars): {indicators['price_change_pct']:+.1f}%
{_macro_ctx}
            Discovery Signals: {', '.join(self.discovery.opportunities.get(symbol, []))}{news_snippet}{_weekend}
{_guard}
            Make a decisive call. Only use HOLD if the indicators are genuinely mixed with no clear edge.
            Use confidence above 0.70 when signals align, below 0.60 when signals conflict.

            Respond in this exact format:
            Decision: <BUY|SELL|HOLD>
            Confidence: <0.00-1.00>
            Reasoning: <one sentence explaining the key signal>"""

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
            # Short-circuit SELL when we have no position — avoids an ~80s debate call
            if will_execute and decision['decision'] == 'sell' and self.get_position(symbol) is None:
                logging.warning(f"⚠️  Skipping SELL {symbol}: no open position held")
                will_execute = False
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
            if will_execute and self.pdt_blocked and decision['decision'] == 'buy':
                logging.info(f"⏭️  Skipping {symbol}: PDT restriction active for today")
                will_execute = False
            elif will_execute:
                is_buy = decision['decision'] == 'buy'
                needs_rotation = is_buy and remaining_cash < decision.get('current_price', 0)

                if needs_rotation:
                    logging.info(
                        f"💡 Cash ${remaining_cash:,.2f} insufficient for {symbol} "
                        f"@ ${decision.get('current_price', 0):.2f} — attempting rotation..."
                    )
                    executed = self._attempt_rotation(symbol, decision, remaining_cash, equity)
                    if executed:
                        trades_executed += 1
                        remaining_cash = max(
                            remaining_cash - equity * self.params['max_position_size'], 0
                        )
                        logging.info(f"💵 Remaining cash this session: ${remaining_cash:,.2f}")
                else:
                    executed = self.execute_trade(symbol, decision, equity, remaining_cash)
                    if executed:
                        trades_executed += 1
                        if is_buy:
                            cost = remaining_cash * self.params['max_position_size']
                            remaining_cash = max(remaining_cash - cost, 0)
                            logging.info(f"💵 Remaining cash this session: ${remaining_cash:,.2f}")
            elif not will_execute and decision['decision'] in ['buy', 'sell']:
                if debate_result and debate_result['verdict'] == 'ABORT':
                    logging.info(f"⏭️  Skipping {symbol}: debate aborted (confidence was {decision['confidence']:.2f})")
                else:
                    logging.info(f"⏭️  Skipping {symbol}: confidence {decision['confidence']:.2f} below threshold {self.params['min_confidence']:.2f}")

            # Log every decision (including HOLDs) for replay and validation
            log_indicators = {**indicators,
                              'discovery_signals': self.discovery.opportunities.get(symbol, [])}
            self._log_decision(symbol, prompt, response, decision, log_indicators, executed,
                               debate=debate_result)
        
        logging.info(f"✅ Session complete: {trades_executed} trades executed")

if __name__ == "__main__":
    agent = AutonomousAgent()
    agent.run_trading_session()