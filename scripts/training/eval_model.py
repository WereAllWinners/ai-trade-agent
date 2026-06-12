#!/usr/bin/env python3
"""
eval_model.py — Post-fine-tune model quality evaluation

Tests the currently loaded LoRA adapter against a set of benchmark scenarios
with known expected trading directions. Produces a quality report with:

  - Format compliance rate  (does the model output Decision/Confidence/Reasoning?)
  - Directional accuracy    (does it pick the right direction for clear setups?)
  - Confidence calibration  (are scores spread across the range vs all 0.5?)
  - Response latency        (seconds per inference)
  - Regression check        (compares against a saved baseline if one exists)

Usage:
    python3 scripts/eval_model.py
    python3 scripts/eval_model.py --save-baseline   # save this run as the new baseline
    python3 scripts/eval_model.py --adapter /path/to/adapter  # eval a specific adapter

The script is GPU-intensive (loads the 32B model). Allow ~5-10 min for cold start.
"""

import hashlib
import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

_SCRIPTS_DIR        = Path(__file__).resolve().parent.parent
_PROJECT_ROOT       = _SCRIPTS_DIR.parent
_EVAL_DIR           = _PROJECT_ROOT / 'logs' / 'eval'
_GOLDEN_SET_PATH    = _EVAL_DIR / 'golden_set.jsonl'
_GOLDEN_BASELINE    = _EVAL_DIR / 'golden_baseline.json'

# Direction class for golden-eval scoring.
# buy_put and sell are both bearish; buy_call and buy are both bullish.
# A correct answer requires the same direction class — buy_call vs buy_put
# must never score as a match.
_DIRECTION = {
    'buy':      'bullish',
    'buy_call': 'bullish',
    'buy_put':  'bearish',
    'sell':     'bearish',
    'hold':     'neutral',
}

# ---------------------------------------------------------------------------
# Benchmark scenarios — clear ground-truth direction for each
# ---------------------------------------------------------------------------
# Each entry: (name, prompt, expected_decision, min_confidence)

_SCENARIOS = [
    (
        "strong_oversold_buy",
        """Analyze NVDA for trading:

Current Price: $142.00
RSI (14): 22.5          <- deeply oversold
MACD: +2.1 (bullish crossover just occurred)
Volume Ratio: 3.2x avg  <- strong unusual volume
Price vs 200-day MA: -8% (approaching support)
Price Change (5d): -12%

Discovery Signals: Oversold bounce candidate, institutional accumulation detected.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "buy",
        0.60,
    ),
    (
        "exit_after_gains",
        """Analyze TSLA for trading:

Current Price: $390.00
RSI (14): 78.4          <- overbought
MACD: -1.2 (bearish divergence developing)
Volume Ratio: 0.6x avg  <- declining volume on price stall
Price vs 200-day MA: +28% (overextended)
Price Change (5d): -2% (stalling after big run)
Current Position: Long TSLA entered at $341 (+14.4% unrealised gain)
Stop loss: -7% | Take profit: +15%

Discovery Signals: Take-profit level approaching, RSI overbought, momentum fading.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "sell",
        0.60,
    ),
    (
        "earnings_hold",
        """Analyze AMZN for trading:

Current Price: $215.00
RSI (14): 58.0          <- moderately bullish
MACD: +1.1 (positive momentum)
Volume Ratio: 0.9x avg
Price vs 200-day MA: +12%
Price Change (5d): +3%
Upcoming: Earnings report in 2 days (IV elevated 40% above historical norm)

Discovery Signals: Moderate bullish setup but high earnings event uncertainty.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "hold",
        0.50,
    ),
    (
        "macro_guard_hold",
        """Analyze NVDA for trading:

Current Price: $130.00
RSI (14): 52.0          <- neutral
MACD: +0.5 (mild positive)
Volume Ratio: 1.0x avg
Price vs 200-day MA: +5%
Price Change (5d): +1%
Market Context: Federal Reserve interest rate decision announcement in 2 hours.

Discovery Signals: Neutral technical setup on a high-volatility macro event day.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "hold",
        0.40,
    ),
    (
        "neutral_hold",
        """Analyze MSFT for trading:

Current Price: $415.00
RSI (14): 51.2          <- neutral
MACD: +0.3 (flat, no clear direction)
Volume Ratio: 1.0x avg  <- average volume
Price vs 200-day MA: +2% (slightly above, normal)
Price Change (5d): +0.3%

Discovery Signals: No significant signals. Normal market conditions.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "hold",
        0.50,
    ),
    (
        "breakout_buy",
        """Analyze META for trading:

Current Price: $605.00
RSI (14): 63.0          <- bullish momentum, not overbought
MACD: +3.5 (strong bullish momentum)
Volume Ratio: 2.8x avg  <- breakout volume confirmation
Price vs 200-day MA: +18% (strong uptrend)
Price Change (5d): +8%
52-week high breakout: YES

Discovery Signals: New 52-week high on strong volume, momentum trade setup.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "buy",
        0.55,
    ),
    (
        "distribution_sell",
        """Analyze AAPL for trading:

Current Price: $198.00
RSI (14): 71.5          <- overbought
MACD: +0.2 (momentum fading, near zero line)
Volume Ratio: 1.8x avg  <- high volume on stall
Price vs 200-day MA: +24%
Price Change (5d): +1% (stalling after big run)

Discovery Signals: High volume stall near resistance, potential distribution.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "sell",
        0.50,
    ),
    (
        "sector_rotation_hold",
        """Analyze JPM for trading:

Current Price: $248.00
RSI (14): 55.0
MACD: +0.8 (mild positive)
Volume Ratio: 1.1x avg
Price vs 200-day MA: +5%
Price Change (5d): +2%

Market Context: Fed rate decision pending this week, bank sector mixed signals.

Discovery Signals: Moderate bullish signals but macro uncertainty elevated.

Should we BUY, SELL, or HOLD? Provide Decision, Confidence (0-1), and Reasoning.""",
        "hold",
        0.40,   # lower bar — this one is genuinely ambiguous
    ),
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _check_format(response: str) -> dict:
    """Check if response contains required fields."""
    r = response.lower()
    has_decision   = any(kw in r for kw in ['decision:', 'decision '])
    has_confidence = 'confidence' in r
    has_reasoning  = any(kw in r for kw in ['reasoning', 'reason', 'because', 'analysis'])
    valid_action   = any(kw in r for kw in ['buy', 'sell', 'hold'])
    return {
        'has_decision':   has_decision,
        'has_confidence': has_confidence,
        'has_reasoning':  has_reasoning,
        'valid_action':   valid_action,
        'score':          sum([has_decision, has_confidence, has_reasoning, valid_action]) / 4,
    }


def _extract_decision(response: str) -> str:
    """Extract decision from response text.

    Lowercases before all substring checks so the model's uppercase output
    (e.g. 'Decision: BUY_CALL') is handled correctly.  Options-specific
    tokens (buy_call, buy_put) are checked before the plain 'buy' prefix
    so they are never swallowed by the shorter match.
    """
    r = response.lower()
    for line in r.splitlines():
        if 'decision:' in line:
            if 'buy_call' in line: return 'buy_call'
            if 'buy_put'  in line: return 'buy_put'
            if 'buy'      in line: return 'buy'
            if 'sell'     in line: return 'sell'
            if 'hold'     in line: return 'hold'
    # Fallback: count mentions
    buys  = r.count('buy')
    sells = r.count('sell')
    holds = r.count('hold')
    if buys == sells == holds == 0:
        return 'unknown'
    return max([('buy', buys), ('sell', sells), ('hold', holds)], key=lambda x: x[1])[0]


def _extract_confidence(response: str) -> float | None:
    import re
    m = re.search(r'confidence[:\s]+([0-9]*\.?[0-9]+)', response.lower())
    if m:
        v = float(m.group(1))
        return v / 100 if v > 1 else v
    return None


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(adapter_path: str | None = None) -> dict:
    # Override adapter path if specified
    if adapter_path:
        os.environ['LORA_ADAPTER_PATH'] = adapter_path
        logging.info(f"Evaluating adapter: {adapter_path}")
    else:
        resolved = os.getenv(
            'LORA_ADAPTER_PATH',
            str(_PROJECT_ROOT / 'finetune' / 'finance_qwen_32b_lora_latest')
        )
        # Resolve symlink
        p = Path(resolved)
        if p.is_symlink():
            resolved = str(p.resolve())
        logging.info(f"Evaluating adapter: {resolved}")
        adapter_path = resolved

    # Verify adapter exists
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logging.error(f"Adapter path does not exist: {adapter_path}")
        return {'error': f'Adapter not found: {adapter_path}'}

    adapter_config = {}
    config_file = adapter_dir / 'adapter_config.json'
    if config_file.exists():
        adapter_config = json.loads(config_file.read_text())

    logging.info(f"Adapter config: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}, "
                 f"base={adapter_config.get('base_model_name_or_path', 'unknown')}")
    logging.info(f"\nRunning {len(_SCENARIOS)} benchmark scenarios...\n{'='*60}")

    from model_inference_lora import get_trading_decision, parse_decision, _MODEL_CACHE
    # Force reload if a different adapter is being tested
    if adapter_path and _MODEL_CACHE.get('loaded'):
        logging.info("Model already cached — using cached weights for this eval run")

    results = []
    latencies = []

    for name, prompt, expected, min_conf in _SCENARIOS:
        logging.info(f"  Running: {name} (expected: {expected.upper()})")
        t0 = time.time()
        try:
            response = get_trading_decision(prompt)
            latency = time.time() - t0
        except Exception as e:
            logging.error(f"  Inference failed: {e}")
            results.append({'scenario': name, 'error': str(e), 'passed': False})
            continue

        latencies.append(latency)
        fmt     = _check_format(response)
        actual  = _extract_decision(response)
        conf    = _extract_confidence(response)
        correct = (actual == expected)

        passed = correct and (conf is not None) and (conf >= min_conf) and fmt['score'] >= 0.75

        logging.info(
            f"    Got: {actual.upper():4s} | Expected: {expected.upper():4s} | "
            f"Conf: {(f'{conf:.2f}' if conf is not None else 'N/A'):5s} | "
            f"Format: {fmt['score']:.0%} | {'✅' if passed else '❌'} | "
            f"{latency:.1f}s"
        )

        results.append({
            'scenario':    name,
            'expected':    expected,
            'actual':      actual,
            'correct':     correct,
            'confidence':  conf,
            'min_conf':    min_conf,
            'format':      fmt,
            'latency_s':   round(latency, 2),
            'passed':      passed,
            'response':    response[:500],  # truncate for storage
        })

    # Aggregate metrics
    n = len(results)
    n_ok = sum(1 for r in results if 'error' not in r)
    directional_acc = sum(1 for r in results if r.get('correct')) / n_ok if n_ok else 0
    format_score    = sum(r.get('format', {}).get('score', 0) for r in results if 'error' not in r) / n_ok if n_ok else 0
    confs           = [r['confidence'] for r in results if r.get('confidence') is not None]
    conf_mean       = sum(confs) / len(confs) if confs else 0
    conf_std        = (sum((c - conf_mean) ** 2 for c in confs) / len(confs)) ** 0.5 if len(confs) > 1 else 0
    pass_rate       = sum(1 for r in results if r.get('passed')) / n if n else 0
    avg_latency     = sum(latencies) / len(latencies) if latencies else 0

    summary = {
        'adapter_path':      adapter_path,
        'adapter_config':    {k: adapter_config.get(k) for k in ('r', 'lora_alpha', 'base_model_name_or_path', 'peft_type')},
        'evaluated_at':      datetime.now().isoformat(),
        'num_scenarios':     n,
        'directional_acc':   round(directional_acc, 4),
        'format_score':      round(format_score, 4),
        'pass_rate':         round(pass_rate, 4),
        'confidence_mean':   round(conf_mean, 4),
        'confidence_std':    round(conf_std, 4),
        'avg_latency_s':     round(avg_latency, 2),
        'scenarios':         results,
    }

    # Grade
    if pass_rate >= 0.80:
        grade = 'A'
    elif pass_rate >= 0.60:
        grade = 'B'
    elif pass_rate >= 0.40:
        grade = 'C'
    else:
        grade = 'F'
    summary['grade'] = grade

    return summary


def _compare_to_baseline(current: dict, baseline: dict) -> None:
    print("\n📊 Regression vs baseline:")
    metrics = ['directional_acc', 'format_score', 'pass_rate', 'confidence_std']
    for m in metrics:
        cur = current.get(m, 0)
        prev = baseline.get(m, 0)
        delta = cur - prev
        arrow = '▲' if delta > 0.01 else ('▼' if delta < -0.01 else '→')
        print(f"  {m:20s}: {prev:.2%} → {cur:.2%}  {arrow} ({delta:+.2%})")


def _print_report(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"🧪 MODEL EVAL REPORT — {summary['evaluated_at'][:16]}")
    print(f"{'='*60}")
    print(f"Adapter:           {Path(summary['adapter_path']).name}")
    print(f"Base model:        {summary['adapter_config'].get('base_model_name_or_path', 'unknown')}")
    print(f"LoRA r / alpha:    {summary['adapter_config'].get('r')} / {summary['adapter_config'].get('lora_alpha')}")
    print(f"")
    print(f"Directional acc:   {summary['directional_acc']:.0%}  (correct direction picked)")
    print(f"Format score:      {summary['format_score']:.0%}  (Decision/Confidence/Reasoning present)")
    print(f"Pass rate:         {summary['pass_rate']:.0%}  (direction + confidence + format)")
    print(f"Confidence mean:   {summary['confidence_mean']:.2f}  ± {summary['confidence_std']:.2f}  (want >0.55, std >0.10)")
    print(f"Avg latency:       {summary['avg_latency_s']:.1f}s per inference")
    print(f"")
    print(f"Overall grade:     {summary['grade']}")
    print(f"{'='*60}")


def _hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:12]


def run_golden_eval(
    adapter_path: str | None,
    golden_path: Path = _GOLDEN_SET_PATH,
    baseline_path: Path = _GOLDEN_BASELINE,
) -> dict:
    """
    Evaluate adapter against the golden set at *golden_path*.

    Uses temperature=0.0 (greedy decoding) for determinism.
    Normalises buy_call / buy_put → buy for direction comparison.
    Compares pass_rate to *baseline_path* and sends a CRITICAL alert when
    the drop exceeds 15 percentage points.

    Returns a result dict with a 'golden_veto' bool key.
    """
    items = [
        json.loads(line)
        for line in golden_path.read_text().splitlines()
        if line.strip()
    ]

    from model_inference_lora import get_trading_decision

    results = []
    for item in items:
        prompt   = item['prompt']
        expected = item['expected']
        try:
            response = get_trading_decision(prompt, temperature=0.0)
        except Exception as exc:
            results.append({
                'prompt_hash': _hash_prompt(prompt),
                'expected':    expected,
                'actual':      'error',
                'passed':      False,
                'error':       str(exc),
            })
            continue

        actual = _extract_decision(response)
        # Direction-class comparison: buy_call/buy share 'bullish';
        # buy_put/sell share 'bearish'. Unknown actual → False, no exception.
        passed = (
            _DIRECTION.get(actual) == _DIRECTION.get(expected)
            and _DIRECTION.get(actual) is not None
        )
        results.append({
            'prompt_hash': _hash_prompt(prompt),
            'expected':    expected,
            'actual':      actual,
            'passed':      passed,
        })

    pass_rate = sum(1 for r in results if r['passed']) / len(results) if results else 0.0

    summary: dict = {
        'adapter_path': adapter_path,
        'evaluated_at': datetime.now().isoformat(),
        'num_items':    len(results),
        'pass_rate':    round(pass_rate, 4),
        'per_item':     results,
        'golden_veto':  False,
    }

    if baseline_path.exists():
        baseline   = json.loads(baseline_path.read_text())
        b_rate     = baseline['pass_rate']
        drop       = b_rate - pass_rate
        summary['baseline_pass_rate'] = b_rate
        summary['drop']               = round(drop, 4)

        if drop > 0.15:
            try:
                from alerts import send_alert, AlertLevel
                send_alert(
                    AlertLevel.CRITICAL,
                    'golden_gate_regression',
                    (
                        f"Golden set pass_rate dropped {drop:.1%} vs baseline "
                        f"({pass_rate:.1%} < {b_rate:.1%}) — promotion vetoed."
                    ),
                    {
                        'pass_rate': pass_rate,
                        'baseline':  b_rate,
                        'drop':      drop,
                        'adapter':   adapter_path,
                    },
                )
            except Exception as alert_err:
                logging.warning("Could not send golden gate alert: %s", alert_err)

            logging.error(
                "❌ Golden gate: %.1f%% dropped %.1f pts vs baseline %.1f%% (threshold 15 pts)",
                pass_rate * 100, drop * 100, b_rate * 100,
            )
            summary['golden_veto'] = True

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned trading model quality')
    parser.add_argument('--adapter', type=str, default=None,
                        help='Path to specific LoRA adapter (default: finance_qwen_32b_lora_latest)')
    parser.add_argument('--save-baseline', action='store_true',
                        help='Save this run as the new baseline for future comparisons')
    parser.add_argument('--golden', action='store_true',
                        help='Run golden set evaluation (requires golden_set.jsonl + golden_baseline.json)')
    parser.add_argument('--build-baseline', action='store_true',
                        help='Build golden_baseline.json from the current adapter (write-once; use --force to overwrite)')
    parser.add_argument('--force', action='store_true',
                        help='Allow overwriting golden_baseline.json with --build-baseline')
    args = parser.parse_args()

    # ── --build-baseline: run golden eval and write baseline (write-once) ──
    if args.build_baseline:
        if not _GOLDEN_SET_PATH.exists():
            print(f"\n❌ Golden set not found: {_GOLDEN_SET_PATH}")
            print("   Run: python scripts/training/build_golden_set.py")
            sys.exit(1)
        if _GOLDEN_BASELINE.exists() and not args.force:
            print(f"\n❌ golden_baseline.json already exists: {_GOLDEN_BASELINE}")
            print("   Use --force to overwrite.")
            sys.exit(1)

        # Run smoke test to load model; golden eval reuses the cache.
        smoke = run_eval(adapter_path=args.adapter)
        golden_result = run_golden_eval(
            adapter_path=args.adapter,
            golden_path=_GOLDEN_SET_PATH,
            baseline_path=Path('/dev/null'),  # no baseline to compare against when building
        )
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        baseline_payload = {
            'pass_rate':   golden_result['pass_rate'],
            'num_items':   golden_result['num_items'],
            'built_at':    datetime.now().isoformat(),
            'adapter_path': args.adapter,
            'per_item':    golden_result['per_item'],
        }
        _GOLDEN_BASELINE.write_text(json.dumps(baseline_payload, indent=2))
        print(f"\n📌 Golden baseline saved → {_GOLDEN_BASELINE}")
        print(f"   pass_rate: {golden_result['pass_rate']:.1%}  ({golden_result['num_items']} items)")
        sys.exit(0)

    # ── Standard smoke-test eval ─────────────────────────────────────────────
    summary = run_eval(adapter_path=args.adapter)

    if 'error' in summary:
        print(f"\n❌ Eval failed: {summary['error']}")
        sys.exit(1)

    _print_report(summary)

    # Save smoke-test results
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = _EVAL_DIR / f'eval_{ts}.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n💾 Full results saved → {out_path}")

    baseline_path = _EVAL_DIR / 'baseline.json'
    if baseline_path.exists() and not args.save_baseline:
        baseline = json.loads(baseline_path.read_text())
        _compare_to_baseline(summary, baseline)

    if args.save_baseline:
        baseline_path.write_text(json.dumps(summary, indent=2))
        print(f"📌 Saved as new baseline → {baseline_path}")

    # ── --golden: run golden set eval after smoke test (shares model load) ───
    if args.golden:
        if not _GOLDEN_SET_PATH.exists():
            logging.warning("Golden set absent (%s) — skipping golden gate", _GOLDEN_SET_PATH)
        else:
            golden_result = run_golden_eval(
                adapter_path=args.adapter,
                golden_path=_GOLDEN_SET_PATH,
                baseline_path=_GOLDEN_BASELINE,
            )
            ts_g = datetime.now().strftime('%Y%m%d_%H%M%S')
            golden_out = _EVAL_DIR / f'golden_eval_{ts_g}.json'
            golden_out.write_text(json.dumps(golden_result, indent=2))
            print(f"\n🥇 Golden eval: {golden_result['pass_rate']:.1%} pass rate "
                  f"({golden_result['num_items']} items) → {golden_out.name}")
            if golden_result.get('drop') is not None:
                drop = golden_result['drop']
                arrow = '▼' if drop > 0.01 else ('▲' if drop < -0.01 else '→')
                print(f"   vs baseline: {arrow} {drop:+.1%}")
            if golden_result['golden_veto']:
                sys.exit(2)

    # Exit 1 if grade is F
    sys.exit(0 if summary['grade'] != 'F' else 1)


if __name__ == '__main__':
    main()
