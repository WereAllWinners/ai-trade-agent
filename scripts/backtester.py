#!/usr/bin/env python3
"""
Backtester - Replays historical OHLCV data to validate strategy logic.

Simulates the autonomous agent's indicator calculation and rule-based
decision process against historical bars, then computes key metrics:
  - Win rate, profit factor
  - Sharpe ratio (annualised)
  - Max drawdown
  - Total return

The LLM is intentionally NOT called during backtesting; instead a
simple rule-based decision function mirrors the heuristics the agent
was designed around (RSI + MACD + momentum thresholds).  This lets you
run the backtest without a GPU and still validate the risk/sizing logic.

Modes:
  backtest  (default) — rule-based simulation over historical OHLCV
  replay              — simulate using real LLM decisions logged by the agent
                        (loads logs/decision_log.jsonl)

Usage:
  python3 scripts/backtester.py --symbols AAPL MSFT SPY --days 365
  python3 scripts/backtester.py --symbols NVDA --days 180 --stop-loss -0.07 --take-profit 0.15
  python3 scripts/backtester.py --mode replay
  python3 scripts/backtester.py --mode replay --log logs/decision_log.jsonl --bot stock
  python3 scripts/backtester.py --mode replay --rerun  # re-send saved prompts to Ollama
"""
import argparse
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = compute_rsi(df['close'])
    df['macd'] = compute_macd(df['close'])
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['mom_20'] = df['close'].pct_change(20) * 100
    return df.dropna()


# ---------------------------------------------------------------------------
# Rule-based decision (mirrors heuristics; no LLM required)
# ---------------------------------------------------------------------------

def rule_decision(row: pd.Series, min_confidence: float = 0.60):
    """
    Lightweight rule set that mirrors what a well-calibrated LLM is
    expected to decide on average for this indicator set.

    Returns (action, confidence) where action is 'buy' | 'sell' | 'hold'.
    """
    score = 0.0
    reasons = 0

    # RSI oversold/overbought
    if row['rsi'] < 35:
        score += 0.3
        reasons += 1
    elif row['rsi'] > 65:
        score -= 0.3
        reasons += 1

    # MACD direction
    if row['macd'] > 0:
        score += 0.2
        reasons += 1
    else:
        score -= 0.2
        reasons += 1

    # Momentum
    if row['mom_20'] > 5:
        score += 0.2
        reasons += 1
    elif row['mom_20'] < -5:
        score -= 0.2
        reasons += 1

    # Volume confirmation
    if row['vol_ratio'] > 1.5:
        score += 0.1 * (1 if score > 0 else -1)

    confidence = min(0.5 + abs(score), 0.95)

    if score > 0 and confidence >= min_confidence:
        return 'buy', confidence
    elif score < 0 and confidence >= min_confidence:
        return 'sell', confidence
    return 'hold', confidence


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(
        self,
        stop_loss: float = -0.07,
        take_profit: float = 0.15,
        max_position_pct: float = 0.05,
        min_confidence: float = 0.60,
        starting_capital: float = 100_000.0,
    ):
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_pct = max_position_pct
        self.min_confidence = min_confidence
        self.starting_capital = starting_capital

    # ------------------------------------------------------------------
    def _fetch(self, symbol: str, days: int) -> pd.DataFrame:
        logging.info(f"  Fetching {days}d of daily data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f'{days}d', interval='1d')
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    # ------------------------------------------------------------------
    def run_symbol(self, symbol: str, days: int) -> dict:
        """Backtest a single symbol. Returns a metrics dict."""
        df = self._fetch(symbol, days)
        df = compute_indicators(df)

        equity = self.starting_capital
        equity_curve = [equity]
        trades = []
        position = None  # {'entry_price', 'shares', 'entry_date'}

        for i, (ts, row) in enumerate(df.iterrows()):
            price = row['close']

            # --- manage open position ---
            if position is not None:
                pnl_pct = (price - position['entry_price']) / position['entry_price']

                if pnl_pct <= self.stop_loss or pnl_pct >= self.take_profit:
                    exit_reason = 'stop_loss' if pnl_pct <= self.stop_loss else 'take_profit'
                    pnl_dollar = pnl_pct * position['shares'] * position['entry_price']
                    equity += pnl_dollar
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': ts,
                        'entry_price': position['entry_price'],
                        'exit_price': price,
                        'shares': position['shares'],
                        'pnl_pct': pnl_pct,
                        'pnl_dollar': pnl_dollar,
                        'exit_reason': exit_reason,
                    })
                    position = None

            equity_curve.append(equity)

            # --- look for new entry (only if flat) ---
            if position is None:
                action, confidence = rule_decision(row, self.min_confidence)
                if action == 'buy':
                    pos_value = equity * self.max_position_pct
                    shares = pos_value / price
                    position = {
                        'entry_price': price,
                        'shares': shares,
                        'entry_date': ts,
                    }

        # Close any open position at the last bar
        if position is not None:
            price = df['close'].iloc[-1]
            pnl_pct = (price - position['entry_price']) / position['entry_price']
            pnl_dollar = pnl_pct * position['shares'] * position['entry_price']
            equity += pnl_dollar
            trades.append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'shares': position['shares'],
                'pnl_pct': pnl_pct,
                'pnl_dollar': pnl_dollar,
                'exit_reason': 'end_of_period',
            })

        return self._metrics(symbol, trades, equity_curve)

    # ------------------------------------------------------------------
    def _metrics(self, symbol: str, trades: list, equity_curve: list) -> dict:
        if not trades:
            return {'symbol': symbol, 'total_trades': 0, 'note': 'no trades generated'}

        closed = [t for t in trades if t['exit_reason'] != 'end_of_period']
        winners = [t for t in closed if t['pnl_dollar'] > 0]
        losers = [t for t in closed if t['pnl_dollar'] <= 0]

        win_rate = len(winners) / len(closed) if closed else 0.0
        avg_win = np.mean([t['pnl_pct'] for t in winners]) if winners else 0.0
        avg_loss = np.mean([t['pnl_pct'] for t in losers]) if losers else 0.0
        gross_profit = sum(t['pnl_dollar'] for t in winners)
        gross_loss = abs(sum(t['pnl_dollar'] for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        eq = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        max_drawdown = float(drawdown.min())

        # Annualised Sharpe (using daily returns from equity curve)
        daily_returns = np.diff(eq) / eq[:-1]
        sharpe = (
            (np.mean(daily_returns) / np.std(daily_returns)) * math.sqrt(252)
            if np.std(daily_returns) > 0 else 0.0
        )

        total_return = (eq[-1] - eq[0]) / eq[0]

        return {
            'symbol': symbol,
            'total_trades': len(trades),
            'closed_trades': len(closed),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': round(win_rate, 4),
            'avg_win_pct': round(avg_win, 4),
            'avg_loss_pct': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 3),
            'max_drawdown': round(max_drawdown, 4),
            'sharpe_ratio': round(sharpe, 3),
            'total_return': round(total_return, 4),
            'final_equity': round(eq[-1], 2),
        }

    # ------------------------------------------------------------------
    def run(self, symbols: list, days: int) -> list:
        results = []
        for sym in symbols:
            try:
                r = self.run_symbol(sym, days)
                results.append(r)
            except Exception as e:
                logging.error(f"  Backtest failed for {sym}: {e}")
                results.append({'symbol': sym, 'error': str(e)})
        return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: list, params: dict):
    print('\n' + '=' * 70)
    print('BACKTEST RESULTS')
    print('=' * 70)
    print(
        f"Parameters: SL={params['stop_loss']:.0%}  TP={params['take_profit']:.0%}  "
        f"pos_size={params['max_position_pct']:.0%}  min_conf={params['min_confidence']:.0%}  "
        f"capital=${params['starting_capital']:,.0f}"
    )
    print('-' * 70)

    fmt = '{:<8} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'
    print(fmt.format('Symbol', 'Trades', 'WinRate', 'PFactor', 'Sharpe', 'MaxDD', 'Return', 'FinalEq'))
    print('-' * 70)

    for r in results:
        if 'error' in r:
            print(f"{r['symbol']:<8}  ERROR: {r['error']}")
        elif r.get('total_trades', 0) == 0:
            print(f"{r['symbol']:<8}  No trades generated")
        else:
            print(fmt.format(
                r['symbol'],
                r['closed_trades'],
                f"{r['win_rate']:.1%}",
                f"{r['profit_factor']:.2f}",
                f"{r['sharpe_ratio']:.2f}",
                f"{r['max_drawdown']:.1%}",
                f"{r['total_return']:.1%}",
                f"${r['final_equity']:,.0f}",
            ))

    print('=' * 70)

    # Go-live readiness gate (mirrors weekly_report.py thresholds)
    tradeable = [r for r in results if r.get('closed_trades', 0) >= 10]
    if tradeable:
        avg_win_rate = np.mean([r['win_rate'] for r in tradeable])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in tradeable])
        worst_dd = min(r['max_drawdown'] for r in tradeable)
        avg_pf = np.mean([r['profit_factor'] for r in tradeable if r['profit_factor'] != float('inf')])

        print('\nAGGREGATE (symbols with ≥10 closed trades):')
        gates = [
            ('Win rate ≥ 55%', avg_win_rate >= 0.55, f'{avg_win_rate:.1%}'),
            ('Sharpe ≥ 1.0', avg_sharpe >= 1.0, f'{avg_sharpe:.2f}'),
            ('Max drawdown ≤ 15%', worst_dd >= -0.15, f'{worst_dd:.1%}'),
            ('Profit factor ≥ 1.4', avg_pf >= 1.4, f'{avg_pf:.2f}'),
        ]
        all_pass = all(g[1] for g in gates)
        for name, passed, val in gates:
            icon = '✅' if passed else '❌'
            print(f"  {icon} {name}: {val}")
        print()
        if all_pass:
            print('  >>> Strategy PASSES all go-live gates for this period <<<')
        else:
            print('  >>> Strategy FAILS one or more go-live gates — keep paper trading <<<')
    print('=' * 70 + '\n')


def save_results(results: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': results}, f, indent=2)
    logging.info(f'Results saved to {output_path}')


# ---------------------------------------------------------------------------
# Replay mode — simulate using real LLM decisions from decision_log.jsonl
# ---------------------------------------------------------------------------

def _load_decision_log(log_path: Path, bot_filter: str | None) -> list[dict]:
    """Load and optionally filter decision log entries."""
    records = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if bot_filter and r.get('bot') != bot_filter:
                        continue
                    records.append(r)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        logging.error(f"Decision log not found: {log_path}")
    logging.info(f"  Loaded {len(records)} decision records from {log_path.name}"
                 + (f" (bot={bot_filter})" if bot_filter else ""))
    return records


def _rerun_prompt(prompt: str, model: str = 'qwen3:8b') -> dict:
    """Re-run a saved prompt through Ollama and return a parsed decision."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        import ollama
        from model_inference_lora import parse_decision
        response = ollama.generate(model=model, prompt=prompt, think=False,
                                   options={'num_predict': 150})
        raw = response.get('response', '')
        d = parse_decision(raw)
        return {'decision': d.get('decision', 'hold'),
                'confidence': d.get('confidence', 0.5),
                'raw_response': raw}
    except Exception as e:
        logging.warning(f"  Rerun failed: {e}")
        return {'decision': 'hold', 'confidence': 0.5, 'raw_response': ''}


class ReplayBacktester(Backtester):
    """
    Backtester that uses real LLM decisions from decision_log.jsonl instead
    of the rule-based proxy.  Each logged 'buy' decision becomes a simulated
    trade entry; price history is fetched from the decision timestamp forward,
    and the same SL/TP rules are applied.
    """

    def run_replay(
        self,
        log_path: Path,
        bot_filter: str | None = None,
        rerun: bool = False,
    ) -> list:
        """
        Load decision log, group buy decisions by symbol, simulate each trade,
        then compute per-symbol and aggregate metrics.
        """
        records = _load_decision_log(log_path, bot_filter)
        if not records:
            logging.warning("No records to replay — run the agent for a few weeks first.")
            return []

        # Group only actionable buy decisions by symbol
        buys_by_symbol: dict[str, list] = {}
        for r in records:
            decision = r.get('decision', 'hold')
            confidence = float(r.get('confidence') or 0.0)

            if rerun and r.get('prompt'):
                fresh = _rerun_prompt(r['prompt'], model=r.get('model', 'qwen3:8b'))
                decision = fresh['decision']
                confidence = fresh['confidence']

            if decision not in ('buy', 'buy_call'):
                continue
            if confidence < self.min_confidence:
                continue

            sym = r.get('symbol', '').upper()
            if not sym:
                continue
            buys_by_symbol.setdefault(sym, []).append({
                'timestamp': r['timestamp'],
                'confidence': confidence,
            })

        if not buys_by_symbol:
            logging.warning("No buy decisions above min_confidence in the log.")
            return []

        logging.info(f"  Replaying {sum(len(v) for v in buys_by_symbol.values())} "
                     f"buy decisions across {len(buys_by_symbol)} symbols...")

        results = []
        for sym, entries in buys_by_symbol.items():
            try:
                r = self._replay_symbol(sym, entries)
                results.append(r)
            except Exception as e:
                logging.error(f"  Replay failed for {sym}: {e}")
                results.append({'symbol': sym, 'error': str(e)})

        return results

    def _replay_symbol(self, symbol: str, entries: list) -> dict:
        """
        For each logged buy entry, fetch forward price data and apply SL/TP.
        Returns metrics dict matching the standard backtest format.
        """
        trades = []
        equity = self.starting_capital
        equity_curve = [equity]

        for entry in sorted(entries, key=lambda x: x['timestamp']):
            ts_str = entry['timestamp']
            try:
                entry_dt = datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            # Fetch price from entry date forward (up to 60 trading days)
            start_str = entry_dt.strftime('%Y-%m-%d')
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_str, interval='1d')
                if hist.empty or len(hist) < 2:
                    continue
                hist.columns = [c.lower() for c in hist.columns]
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
            except Exception as e:
                logging.warning(f"    Price fetch failed for {symbol} from {start_str}: {e}")
                continue

            entry_price = float(hist['close'].iloc[0])
            pos_value = equity * self.max_position_pct
            shares = pos_value / entry_price
            entry_date = hist.index[0]
            exit_price = None
            exit_date = None
            exit_reason = 'end_of_period'

            for ts, row in hist.iloc[1:].iterrows():
                price = float(row['close'])
                pnl_pct = (price - entry_price) / entry_price
                if pnl_pct <= self.stop_loss:
                    exit_price = price
                    exit_date = ts
                    exit_reason = 'stop_loss'
                    break
                if pnl_pct >= self.take_profit:
                    exit_price = price
                    exit_date = ts
                    exit_reason = 'take_profit'
                    break

            if exit_price is None:
                exit_price = float(hist['close'].iloc[-1])
                exit_date = hist.index[-1]

            pnl_pct = (exit_price - entry_price) / entry_price
            pnl_dollar = pnl_pct * shares * entry_price
            equity += pnl_dollar
            equity_curve.append(equity)

            trades.append({
                'symbol':       symbol,
                'entry_date':   entry_date,
                'exit_date':    exit_date,
                'entry_price':  entry_price,
                'exit_price':   exit_price,
                'shares':       shares,
                'pnl_pct':      pnl_pct,
                'pnl_dollar':   pnl_dollar,
                'exit_reason':  exit_reason,
                'confidence':   entry['confidence'],
            })

        return self._metrics(symbol, trades, equity_curve)


def print_replay_report(results: list, log_path: Path, rerun: bool):
    print('\n' + '=' * 70)
    mode_tag = 'REPLAY (re-run prompts via Ollama)' if rerun else 'REPLAY (stored LLM decisions)'
    print(f'BACKTEST RESULTS — {mode_tag}')
    print(f'Log: {log_path}')
    print('=' * 70)

    fmt = '{:<8} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'
    print(fmt.format('Symbol', 'Trades', 'WinRate', 'PFactor', 'Sharpe', 'MaxDD', 'Return', 'FinalEq'))
    print('-' * 70)

    tradeable = []
    for r in results:
        if 'error' in r:
            print(f"{r['symbol']:<8}  ERROR: {r['error']}")
        elif r.get('total_trades', 0) == 0:
            print(f"{r['symbol']:<8}  No trades replayed")
        else:
            print(fmt.format(
                r['symbol'],
                r.get('closed_trades', r['total_trades']),
                f"{r['win_rate']:.1%}",
                f"{r['profit_factor']:.2f}",
                f"{r['sharpe_ratio']:.2f}",
                f"{r['max_drawdown']:.1%}",
                f"{r['total_return']:.1%}",
                f"${r['final_equity']:,.0f}",
            ))
            if r.get('closed_trades', 0) >= 3:
                tradeable.append(r)

    print('=' * 70)

    if tradeable:
        avg_win_rate = float(np.mean([r['win_rate'] for r in tradeable]))
        avg_sharpe   = float(np.mean([r['sharpe_ratio'] for r in tradeable]))
        worst_dd     = min(r['max_drawdown'] for r in tradeable)
        pfs          = [r['profit_factor'] for r in tradeable if r['profit_factor'] != float('inf')]
        avg_pf       = float(np.mean(pfs)) if pfs else float('inf')

        print('\nAGGREGATE (symbols with ≥3 replayed trades):')
        gates = [
            ('Win rate ≥ 55%',       avg_win_rate >= 0.55, f'{avg_win_rate:.1%}'),
            ('Sharpe ≥ 0.8',         avg_sharpe   >= 0.8,  f'{avg_sharpe:.2f}'),
            ('Max drawdown ≤ 15%',   worst_dd     >= -0.15, f'{worst_dd:.1%}'),
            ('Profit factor ≥ 1.4',  avg_pf       >= 1.4,  f'{avg_pf:.2f}'),
        ]
        for name, passed, val in gates:
            print(f"  {'✅' if passed else '❌'} {name}: {val}")
        print()
        if all(g[1] for g in gates):
            print('  >>> LLM decisions PASS all go-live gates for this period <<<')
        else:
            print('  >>> LLM decisions FAIL one or more gates — keep paper trading <<<')
    else:
        print('\n  (not enough closed trades for aggregate gate check)')

    print('=' * 70 + '\n')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest trading strategy on historical data')
    parser.add_argument('--mode', choices=['backtest', 'replay'], default='backtest',
                        help='backtest: rule-based sim | replay: use real LLM decisions from log')
    # --- replay flags ---
    parser.add_argument('--log', type=str, default='logs/decision_log.jsonl',
                        help='Path to decision_log.jsonl (replay mode)')
    parser.add_argument('--bot', type=str, default=None,
                        help='Filter by bot type: stock | options (replay mode)')
    parser.add_argument('--rerun', action='store_true',
                        help='Re-send saved prompts to Ollama instead of using stored decisions')
    # --- backtest / shared flags ---
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA'],
                        help='Ticker symbols to backtest (backtest mode)')
    parser.add_argument('--days', type=int, default=365, help='Historical days to fetch')
    parser.add_argument('--stop-loss', type=float, default=-0.07, help='Stop loss (e.g. -0.07)')
    parser.add_argument('--take-profit', type=float, default=0.15, help='Take profit (e.g. 0.15)')
    parser.add_argument('--position-size', type=float, default=0.05,
                        help='Max position size as fraction of equity (e.g. 0.05)')
    parser.add_argument('--min-confidence', type=float, default=0.60,
                        help='Minimum confidence to replay a decision (0-1)')
    parser.add_argument('--capital', type=float, default=100_000.0, help='Starting capital ($)')
    parser.add_argument('--output', type=str, default='logs/backtest_results.json',
                        help='Output JSON path')
    args = parser.parse_args()

    params = {
        'stop_loss': args.stop_loss,
        'take_profit': args.take_profit,
        'max_position_pct': args.position_size,
        'min_confidence': args.min_confidence,
        'starting_capital': args.capital,
    }

    if args.mode == 'replay':
        log_path = Path(args.log)
        bt = ReplayBacktester(**params)
        logging.info(f"Replaying decisions from {log_path}...")
        results = bt.run_replay(log_path, bot_filter=args.bot, rerun=args.rerun)
        print_replay_report(results, log_path, rerun=args.rerun)
        out = args.output.replace('backtest_results', 'replay_results')
        save_results(results, Path(out))
    else:
        bt = Backtester(**params)
        logging.info(f"Running backtest on {len(args.symbols)} symbols over {args.days} days...")
        results = bt.run(args.symbols, args.days)
        print_report(results, params)
        save_results(results, Path(args.output))
