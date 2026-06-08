#!/usr/bin/env python3
"""
strategy_evolver.py — Weekend strategy evolution + training data enrichment.

Runs on Saturdays after weekend_strategist.py. Each run:
  1. Loads or seeds a population of trading strategies (parameter dicts).
  2. Backtests every strategy against a 90-day universe of 60 stocks.
  3. Mutates the winners to produce the next generation.
  4. Backtests mutants and keeps the top-8 by Sharpe ratio.
  5. Converts all backtest trade results into labeled training examples.
  6. Appends manually curated knowledge-base JSONL entries.
  7. Both batches go into the SQLite DB and are re-exported to training_data.json
     so the Saturday fine-tune learns from them.

Strategies are persisted across weekends in logs/evolved_strategies.json.
The model gradually learns from an ever-improving strategy library.
"""
import hashlib
import json
import logging
import math
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

import db as _db

# Import indicator helpers from the backtester (RSI, MACD, vol_ratio, mom_20)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtester import compute_indicators

# Import tiered labeling from training_data_builder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from training_data_builder import _tiered_label_from_pnl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_EVOLVED_PATH = _PROJECT_ROOT / 'logs' / 'evolved_strategies.json'
_KNOWLEDGE_DIR = _PROJECT_ROOT / 'knowledge'
_DATA_DIR      = _PROJECT_ROOT / 'finetune' / 'data' / 'finance_tuning'

# Generation-1 seeds translated from weekend_strategist.py's hardcoded configs
_SEED_STRATEGIES = [
    {
        'name': 'momentum_only_v1',
        'params': {
            'rsi_max': None, 'rsi_min': None,
            'macd_positive': True,
            'volume_ratio_min': None,
            'mom_20_min': 10.0,
            'stop_loss': -0.07, 'take_profit': 0.15, 'hold_days': 7,
        },
        'generation': 1, 'parent': None,
        'backtest_sharpe': None, 'backtest_win_rate': None,
        'backtest_max_dd': None, 'backtest_total_trades': None,
    },
    {
        'name': 'oversold_bounce_v1',
        'params': {
            'rsi_max': 30.0, 'rsi_min': None,
            'macd_positive': None,
            'volume_ratio_min': None,
            'mom_20_min': None,
            'stop_loss': -0.07, 'take_profit': 0.15, 'hold_days': 5,
        },
        'generation': 1, 'parent': None,
        'backtest_sharpe': None, 'backtest_win_rate': None,
        'backtest_max_dd': None, 'backtest_total_trades': None,
    },
    {
        'name': 'breakout_continuation_v1',
        'params': {
            'rsi_max': None, 'rsi_min': 50.0,
            'macd_positive': True,
            'volume_ratio_min': 1.5,
            'mom_20_min': 5.0,
            'stop_loss': -0.07, 'take_profit': 0.20, 'hold_days': 10,
        },
        'generation': 1, 'parent': None,
        'backtest_sharpe': None, 'backtest_win_rate': None,
        'backtest_max_dd': None, 'backtest_total_trades': None,
    },
    {
        'name': 'multi_signal_v1',
        'params': {
            'rsi_max': 45.0, 'rsi_min': None,
            'macd_positive': True,
            'volume_ratio_min': 1.2,
            'mom_20_min': None,
            'stop_loss': -0.07, 'take_profit': 0.15, 'hold_days': 7,
        },
        'generation': 1, 'parent': None,
        'backtest_sharpe': None, 'backtest_win_rate': None,
        'backtest_max_dd': None, 'backtest_total_trades': None,
    },
]

# Sector proxy universe (mirrors weekend_strategist.py sector_rotation_analysis)
_SECTOR_UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META',
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK',
    'AMZN', 'WMT', 'HD', 'MCD', 'NKE',
    'BA', 'CAT', 'GE', 'UNP', 'HON',
]

# Param mutation ranges
_PARAM_BOUNDS = {
    'rsi_max':          (15.0,  50.0),
    'rsi_min':          (40.0,  70.0),
    'volume_ratio_min': (0.8,   3.0),
    'mom_20_min':       (2.0,   25.0),
    'stop_loss':        (-0.15, -0.03),
    'take_profit':      (0.07,  0.35),
    'hold_days':        (3,     20),
}


# ---------------------------------------------------------------------------
# Strategy decision
# ---------------------------------------------------------------------------

def _strategy_decision(row: pd.Series, params: dict, min_confidence: float = 0.55):
    """
    Apply a strategy's parameter constraints to produce (action, confidence).

    Uses the same indicator columns that compute_indicators() provides:
    rsi, macd, vol_ratio, mom_20.
    """
    score = 0.0

    rsi_max = params.get('rsi_max')
    rsi_min = params.get('rsi_min')
    macd_pos = params.get('macd_positive')
    vol_min = params.get('volume_ratio_min')
    mom_min = params.get('mom_20_min')

    if rsi_max is not None and row['rsi'] < rsi_max:
        score += 0.3
    if rsi_min is not None and row['rsi'] > rsi_min:
        score += 0.2
    if macd_pos is True and row['macd'] > 0:
        score += 0.25
    elif macd_pos is False and row['macd'] < 0:
        score -= 0.25
    if vol_min is not None and row['vol_ratio'] > vol_min:
        score += 0.15
    if mom_min is not None and row['mom_20'] > mom_min:
        score += 0.2

    confidence = min(0.5 + abs(score), 0.95)
    if score > 0 and confidence >= min_confidence:
        return 'buy', confidence
    return 'hold', confidence


# ---------------------------------------------------------------------------
# StrategyEvolver
# ---------------------------------------------------------------------------

class StrategyEvolver:

    def __init__(self):
        self.backtest_days  = 90
        self.universe_size  = 60
        self.min_sharpe     = 0.8
        self.min_win_rate   = 0.55
        self.max_dd         = -0.25
        self.min_trades     = 5
        self.top_n_keep     = 8
        self.starting_cap   = 100_000.0

    # ------------------------------------------------------------------
    # Strategy persistence
    # ------------------------------------------------------------------

    def load_strategies(self) -> list[dict]:
        if _EVOLVED_PATH.exists():
            try:
                strategies = json.loads(_EVOLVED_PATH.read_text())
                if strategies:
                    logging.info(f"📚 Loaded {len(strategies)} evolved strategies from disk")
                    return strategies
            except Exception as e:
                logging.warning(f"Could not load evolved_strategies.json: {e}")
        logging.info("🌱 No evolved strategies found — using generation-1 seeds")
        return [dict(s) for s in _SEED_STRATEGIES]

    def save_strategies(self, strategies: list[dict]) -> None:
        _EVOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EVOLVED_PATH.write_text(json.dumps(strategies, indent=2))
        logging.info(f"💾 Saved {len(strategies)} strategies to {_EVOLVED_PATH}")

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def _get_universe(self) -> list[str]:
        symbols = list(_SECTOR_UNIVERSE)

        # Add recently discovered opportunities if available
        opp_path = _PROJECT_ROOT / 'logs' / 'discovered_opportunities.json'
        if opp_path.exists():
            try:
                data = json.loads(opp_path.read_text())
                if isinstance(data, list):
                    for item in data:
                        sym = item.get('symbol') if isinstance(item, dict) else item
                        if isinstance(sym, str) and sym not in symbols:
                            symbols.append(sym)
            except Exception:
                pass

        # De-duplicate and cap
        seen = set()
        unique = []
        for s in symbols:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique[:self.universe_size]

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def _fetch_daily(self, symbol: str) -> pd.DataFrame | None:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f'{self.backtest_days}d', interval='1d')
            if df.empty:
                return None
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            logging.debug(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _run_simulation(self, df: pd.DataFrame, params: dict, symbol: str) -> list[dict]:
        """Simulate trades for a strategy on a single symbol's daily bars."""
        stop_loss   = params['stop_loss']
        take_profit = params['take_profit']
        hold_days   = int(params['hold_days'])

        equity = self.starting_cap
        equity_curve = [equity]
        trades = []
        position = None  # {'entry_price', 'shares', 'entry_date', 'entry_idx'}

        for i, (ts, row) in enumerate(df.iterrows()):
            price = row['close']

            if position is not None:
                pnl_pct = (price - position['entry_price']) / position['entry_price']
                bars_held = i - position['entry_idx']
                exit_reason = None

                if pnl_pct <= stop_loss:
                    exit_reason = 'stop_loss'
                elif pnl_pct >= take_profit:
                    exit_reason = 'take_profit'
                elif bars_held >= hold_days:
                    exit_reason = 'hold_expired'

                if exit_reason:
                    pnl_dollar = pnl_pct * position['shares'] * position['entry_price']
                    equity += pnl_dollar
                    trades.append({
                        'symbol':      symbol,
                        'entry_date':  position['entry_date'],
                        'exit_date':   ts,
                        'entry_price': position['entry_price'],
                        'exit_price':  price,
                        'pnl_pct':     pnl_pct,
                        'pnl_dollar':  pnl_dollar,
                        'exit_reason': exit_reason,
                    })
                    position = None

            equity_curve.append(equity)

            if position is None:
                action, _ = _strategy_decision(row, params)
                if action == 'buy':
                    pos_value = equity * 0.05  # fixed 5% position size for backtest
                    shares = pos_value / price if price > 0 else 0
                    if shares > 0:
                        position = {
                            'entry_price': price,
                            'shares':      shares,
                            'entry_date':  ts,
                            'entry_idx':   i,
                        }

        # Close open position at last bar
        if position is not None:
            price = df['close'].iloc[-1]
            pnl_pct = (price - position['entry_price']) / position['entry_price']
            pnl_dollar = pnl_pct * position['shares'] * position['entry_price']
            equity += pnl_dollar
            trades.append({
                'symbol':      symbol,
                'entry_date':  position['entry_date'],
                'exit_date':   df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price':  price,
                'pnl_pct':     pnl_pct,
                'pnl_dollar':  pnl_dollar,
                'exit_reason': 'end_of_period',
            })

        return trades, equity_curve

    def _aggregate_metrics(self, all_trades: list[dict], all_equity_curves: list[list]) -> dict:
        """Compute portfolio-level metrics across all symbols."""
        if not all_trades:
            return {'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0, 'max_dd': 0.0}

        closed = [t for t in all_trades if t['exit_reason'] != 'end_of_period']
        if not closed:
            return {'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0, 'max_dd': 0.0}

        winners = [t for t in closed if t['pnl_dollar'] > 0]
        win_rate = len(winners) / len(closed)

        # Combine equity curves to produce a blended daily-return series
        min_len = min(len(ec) for ec in all_equity_curves)
        blended = np.mean([ec[:min_len] for ec in all_equity_curves], axis=0)
        daily_returns = np.diff(blended) / blended[:-1]
        sharpe = (
            (np.mean(daily_returns) / np.std(daily_returns)) * math.sqrt(252)
            if np.std(daily_returns) > 0 else 0.0
        )

        peak = np.maximum.accumulate(blended)
        drawdowns = (blended - peak) / peak
        max_dd = float(drawdowns.min())

        return {
            'total_trades': len(closed),
            'win_rate':     round(win_rate, 4),
            'sharpe':       round(sharpe, 3),
            'max_dd':       round(max_dd, 4),
        }

    def backtest_strategy(self, strategy: dict, universe: list[str]) -> dict:
        """Backtest a strategy across the universe. Returns strategy dict with stats filled."""
        all_trades = []
        all_curves = []

        for symbol in universe:
            time.sleep(0.3)  # yfinance rate-limit courtesy pause
            df = self._fetch_daily(symbol)
            if df is None or len(df) < 30:
                continue
            try:
                df = compute_indicators(df)
            except Exception:
                continue
            trades, equity_curve = self._run_simulation(df, strategy['params'], symbol)
            all_trades.extend(trades)
            if equity_curve:
                all_curves.append(equity_curve)

        metrics = self._aggregate_metrics(all_trades, all_curves)
        result = dict(strategy)
        result['backtest_sharpe']       = metrics['sharpe']
        result['backtest_win_rate']     = metrics['win_rate']
        result['backtest_max_dd']       = metrics['max_dd']
        result['backtest_total_trades'] = metrics['total_trades']
        result['_all_trades']           = all_trades  # kept in-memory for training gen
        return result

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def mutate_strategy(self, parent: dict) -> dict:
        """Produce one mutant by perturbing 1-2 numeric params."""
        mutant = dict(parent)
        mutant['params'] = dict(parent['params'])
        mutant['generation'] = parent['generation'] + 1
        mutant['parent'] = parent['name']
        mutant['backtest_sharpe'] = None
        mutant['backtest_win_rate'] = None
        mutant['backtest_max_dd'] = None
        mutant['backtest_total_trades'] = None
        tag = random.randint(1000, 9999)
        mutant['name'] = f"{parent['name']}_m{tag}"

        # Choose 1-2 numeric params to perturb
        numeric_params = [k for k in _PARAM_BOUNDS if parent['params'].get(k) is not None]
        if not numeric_params:
            return mutant
        targets = random.sample(numeric_params, k=min(2, len(numeric_params)))

        for key in targets:
            lo, hi = _PARAM_BOUNDS[key]
            old_val = mutant['params'][key]
            new_val = old_val * random.uniform(0.88, 1.12)
            if key == 'hold_days':
                new_val = int(round(new_val))
            new_val = max(lo, min(hi, new_val))
            if key == 'hold_days':
                new_val = int(new_val)
            mutant['params'][key] = new_val

        # 20% chance to flip a boolean condition
        if random.random() < 0.2 and mutant['params'].get('macd_positive') is not None:
            mutant['params']['macd_positive'] = not mutant['params']['macd_positive']

        return mutant

    def _is_winner(self, s: dict) -> bool:
        return (
            (s.get('backtest_sharpe') or 0) >= self.min_sharpe
            and (s.get('backtest_win_rate') or 0) >= self.min_win_rate
            and (s.get('backtest_max_dd') or 0) >= self.max_dd
            and (s.get('backtest_total_trades') or 0) >= self.min_trades
        )

    def evolve(self, universe: list[str]) -> tuple[list[dict], list[dict]]:
        """
        Run one evolution cycle.
        Returns (winners, all_candidates) where all_candidates includes backtest trades.
        """
        strategies = self.load_strategies()

        # Phase 1: backtest existing strategies
        logging.info(f"🔬 Backtesting {len(strategies)} strategies on {len(universe)} symbols...")
        candidates = []
        for s in strategies:
            logging.info(f"  Testing {s['name']}...")
            result = self.backtest_strategy(s, universe)
            candidates.append(result)
            logging.info(
                f"  → Sharpe={result['backtest_sharpe']:.2f}  "
                f"WinRate={result['backtest_win_rate']:.0%}  "
                f"MaxDD={result['backtest_max_dd']:.1%}  "
                f"Trades={result['backtest_total_trades']}"
            )

        # Phase 2: generate mutants from winners
        winners = [s for s in candidates if self._is_winner(s)]
        logging.info(f"🏆 {len(winners)}/{len(candidates)} strategies are winners — generating mutants...")
        mutants_backtested = []
        for parent in winners:
            for _ in range(2):  # 2 mutants per winner
                mutant = self.mutate_strategy(parent)
                logging.info(f"  Backtesting mutant {mutant['name']}...")
                result = self.backtest_strategy(mutant, universe)
                mutants_backtested.append(result)
                logging.info(
                    f"  → Sharpe={result['backtest_sharpe']:.2f}  "
                    f"WinRate={result['backtest_win_rate']:.0%}"
                )

        all_candidates = candidates + mutants_backtested

        # Phase 3: select top-N by Sharpe, require min_trades
        tradeable = [s for s in all_candidates if (s.get('backtest_total_trades') or 0) >= self.min_trades]
        tradeable.sort(key=lambda s: (s.get('backtest_sharpe') or 0), reverse=True)
        kept = tradeable[:self.top_n_keep]

        # Strip the in-memory trades before persisting (they can be large)
        to_save = [{k: v for k, v in s.items() if k != '_all_trades'} for s in kept]
        self.save_strategies(to_save)

        kept_winners = [s for s in kept if self._is_winner(s)]
        return kept_winners, all_candidates

    # ------------------------------------------------------------------
    # Training example generation
    # ------------------------------------------------------------------

    def generate_training_examples(self, all_candidates: list[dict]) -> int:
        """Convert backtest trade results into labeled training examples."""
        existing_hashes = _db.get_existing_prompt_hashes()
        added = 0

        for strategy in all_candidates:
            trades = strategy.get('_all_trades', [])
            if not trades:
                continue
            params = strategy['params']
            cond_parts = []
            if params.get('rsi_max') is not None:
                cond_parts.append(f"RSI < {params['rsi_max']:.0f}")
            if params.get('rsi_min') is not None:
                cond_parts.append(f"RSI > {params['rsi_min']:.0f}")
            if params.get('macd_positive') is True:
                cond_parts.append("MACD > 0")
            if params.get('volume_ratio_min') is not None:
                cond_parts.append(f"Volume > {params['volume_ratio_min']:.1f}x avg")
            if params.get('mom_20_min') is not None:
                cond_parts.append(f"20d momentum > {params['mom_20_min']:.0f}%")
            condition_summary = ', '.join(cond_parts) if cond_parts else 'all-weather'

            for trade in trades:
                pnl_pct = trade['pnl_pct']
                symbol  = trade['symbol']
                entry_dt = trade['entry_date']
                if isinstance(entry_dt, pd.Timestamp):
                    entry_date_str = entry_dt.date().isoformat()
                else:
                    entry_date_str = str(entry_dt)[:10]

                label, reward = _tiered_label_from_pnl(pnl_pct, 'buy')

                decision_str = 'BUY' if pnl_pct > 0 else 'HOLD'
                confidence = 0.72 if pnl_pct > 0 else 0.55

                input_text = (
                    f"Analyze {symbol} for trading decision.\n\n"
                    f"Technical Analysis:\n"
                    f"- Entry Conditions: {condition_summary}\n"
                    f"- Strategy: {strategy['name']}\n"
                    f"- Stop Loss: {params['stop_loss']:.0%} | Take Profit: {params['take_profit']:.0%}\n"
                    f"- Max Hold: {int(params['hold_days'])} trading days\n\n"
                    f"Provide trading recommendation with reasoning."
                )
                ph = hashlib.md5(input_text.encode()).hexdigest()
                if ph in existing_hashes:
                    continue

                output_text = (
                    f"Decision: {decision_str}\n"
                    f"Confidence: {confidence:.0%}\n"
                    f"Reasoning: Strategy {strategy['name']} ({condition_summary}) "
                    f"{'triggered entry — ' + trade['exit_reason'] + ' exit at ' + f'{pnl_pct:+.1%}' if pnl_pct >= 0 else 'signal present but outcome was negative — conditions alone insufficient'}"
                )

                example = {
                    'input':  input_text,
                    'output': output_text,
                    'label':  label,
                    'metadata': {
                        'bot':          'stock',
                        'source':       'synthetic_backtest',
                        'symbol':       symbol,
                        'decision':     'buy',
                        'confidence':   confidence,
                        'pnl_pct':      round(pnl_pct, 6),
                        'reward':       round(reward, 6),
                        'entry_date':   entry_date_str,
                        'session_id':   '',
                        'prompt_hash':  ph,
                        'generated_at': datetime.now().isoformat(),
                    },
                }
                if _db.insert_training_example(example, source='synthetic_backtest'):
                    existing_hashes.add(ph)
                    added += 1

        logging.info(f"📊 Added {added} synthetic backtest training examples")
        return added

    def append_knowledge_base_examples(self) -> int:
        """Append curated financial strategy knowledge from knowledge/*.jsonl files."""
        if not _KNOWLEDGE_DIR.exists():
            logging.info("ℹ️  No knowledge/ directory found — skipping knowledge injection")
            return 0

        existing_hashes = _db.get_existing_prompt_hashes()
        added = 0

        for jsonl_file in sorted(_KNOWLEDGE_DIR.glob('*.jsonl')):
            try:
                lines = jsonl_file.read_text().strip().splitlines()
            except Exception as e:
                logging.warning(f"Could not read {jsonl_file}: {e}")
                continue

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(f"{jsonl_file.name}:{line_num} — invalid JSON: {e}")
                    continue

                input_text = example.get('input', '')
                if not input_text:
                    continue

                ph = hashlib.md5(input_text.encode()).hexdigest()
                if ph in existing_hashes:
                    continue

                # Ensure required fields exist
                if 'metadata' not in example:
                    example['metadata'] = {}
                example['metadata'].setdefault('bot', 'stock')
                example['metadata'].setdefault('source', 'knowledge_base')
                example['metadata'].setdefault('symbol', '')
                example['metadata'].setdefault('decision', 'buy')
                example['metadata'].setdefault('confidence', 0.70)
                example['metadata'].setdefault('pnl_pct', None)
                example['metadata'].setdefault('reward', None)
                example['metadata'].setdefault('entry_date', None)
                example['metadata'].setdefault('session_id', '')
                example['metadata']['prompt_hash']  = ph
                example['metadata']['generated_at'] = datetime.now().isoformat()

                if _db.insert_training_example(example, source='knowledge_base'):
                    existing_hashes.add(ph)
                    added += 1

        logging.info(f"📚 Added {added} knowledge-base training examples from {_KNOWLEDGE_DIR}")
        return added

    def rebuild_training_json(self) -> None:
        """Re-export training_data.json from DB so fine-tuner picks up new examples."""
        try:
            sys.path.insert(0, str(_PROJECT_ROOT / 'scripts' / 'training'))
            import training_data_builder
            n_written, _ = training_data_builder.build_and_store('stock')
            logging.info(f"📄 Rebuilt training_data.json — {n_written} total examples")
        except Exception as e:
            logging.error(f"❌ Failed to rebuild training_data.json: {e}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        logging.info("=" * 70)
        logging.info("🧬 STRATEGY EVOLVER — starting weekend evolution cycle")
        logging.info("=" * 70)

        universe = self._get_universe()
        logging.info(f"📈 Universe: {len(universe)} symbols")

        winners, all_candidates = self.evolve(universe)

        n_backtest  = self.generate_training_examples(all_candidates)
        n_knowledge = self.append_knowledge_base_examples()

        if n_backtest + n_knowledge > 0:
            self.rebuild_training_json()

        summary = {
            'winners':         len(winners),
            'total_candidates': len(all_candidates),
            'n_backtest_examples': n_backtest,
            'n_knowledge_examples': n_knowledge,
        }
        logging.info(
            f"✅ Evolution complete — {len(winners)} winner strategies, "
            f"+{n_backtest} backtest examples, +{n_knowledge} knowledge examples"
        )
        return summary


if __name__ == '__main__':
    evolver = StrategyEvolver()
    evolver.run()
