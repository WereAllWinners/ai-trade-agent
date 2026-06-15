#!/usr/bin/env python3
"""
morning_model_check.py — Pre-market model health check

Run this before the 9:30 AM open to confirm:
  1. The LoRA symlink is valid and points to a real adapter
  2. The adapter config matches expectations (r=64, correct task type)
  3. The adapter training date is recent enough
  4. A test inference returns valid structured output
  5. The correct model version is active in the running service

Usage:
    python3 scripts/morning_model_check.py              # full check (loads GPU model)
    python3 scripts/morning_model_check.py --no-inference  # skip GPU test, metadata only
    python3 scripts/morning_model_check.py --verbose    # show full model response

Exit codes:
    0 — all checks passed, safe to trade
    1 — one or more checks failed, investigate before market open
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.WARNING)   # suppress INFO during check runs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401

_SCRIPTS_DIR  = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_ADAPTER_LATEST = _PROJECT_ROOT / 'finetune' / 'finance_qwen_32b_lora_latest'

# What we expect in a healthy adapter
_EXPECTED_TASK_TYPE = 'CAUSAL_LM'
_EXPECTED_PEFT_TYPE = 'LORA'
_MAX_ADAPTER_AGE_DAYS = 30   # warn if adapter is older than this

# Quick smoke-test prompt — should produce a clear BUY signal
_SMOKE_PROMPT = """Analyze AAPL for trading:

Current Price: $195.00
RSI (14): 28.0       <- oversold
MACD: +1.5 (bullish crossover)
Volume Ratio: 2.5x average <- high volume
Price Change (5d): -9%

Should we BUY, SELL, or HOLD? Reply with: Decision, Confidence (0-1), Reasoning."""


PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


class CheckResult:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []   # (check_name, status, detail)

    def add(self, name: str, passed: bool, detail: str, warn_only: bool = False):
        if passed:
            status = PASS
        elif warn_only:
            status = WARN
        else:
            status = FAIL
        self.items.append((name, status, detail))

    @property
    def all_passed(self) -> bool:
        return all(s != FAIL for _, s, _ in self.items)

    @property
    def has_warnings(self) -> bool:
        return any(s == WARN for _, s, _ in self.items)

    def print_report(self) -> None:
        width = max(len(n) for n, _, _ in self.items) + 2
        for name, status, detail in self.items:
            print(f"  {status}  {name:{width}s}  {detail}")


def check_symlink(results: CheckResult) -> Path | None:
    """Check that the _latest symlink exists and points to a real directory."""
    if not _ADAPTER_LATEST.is_symlink():
        results.add('symlink_exists', False, f'{_ADAPTER_LATEST} is not a symlink')
        return None

    target = _ADAPTER_LATEST.resolve()
    if not target.exists():
        results.add('symlink_valid', False,
                    f'Broken symlink → {target} does not exist')
        return None

    results.add('symlink_valid', True, f'→ {target.name}')
    return target


def check_adapter_config(adapter_path: Path, results: CheckResult) -> dict:
    """Validate adapter_config.json."""
    config_file = adapter_path / 'adapter_config.json'
    if not config_file.exists():
        results.add('adapter_config', False, 'adapter_config.json not found')
        return {}

    try:
        config = json.loads(config_file.read_text())
    except Exception as e:
        results.add('adapter_config', False, f'Could not parse: {e}')
        return {}

    task_ok  = config.get('task_type') == _EXPECTED_TASK_TYPE
    peft_ok  = config.get('peft_type') == _EXPECTED_PEFT_TYPE
    r_val    = config.get('r', 0)
    base     = config.get('base_model_name_or_path', 'unknown')

    results.add('adapter_task_type', task_ok,
                f"{config.get('task_type')} (expected {_EXPECTED_TASK_TYPE})")
    results.add('adapter_peft_type', peft_ok,
                f"{config.get('peft_type')} (expected {_EXPECTED_PEFT_TYPE})")
    results.add('adapter_rank', r_val >= 16,
                f"r={r_val}, alpha={config.get('lora_alpha')}")
    results.add('adapter_base_model', bool(base and base != 'unknown'),
                base[:60])

    return config


def check_adapter_freshness(adapter_path: Path, results: CheckResult) -> None:
    """Check adapter was created recently enough."""
    # Parse date from directory name (format: ..._YYYYMMDD_HHMMSS)
    name = adapter_path.name
    parts = name.rsplit('_', 2)  # ['finance_qwen_32b_lora', 'YYYYMMDD', 'HHMMSS']
    try:
        if len(parts) >= 3:
            trained_at = datetime.strptime(f"{parts[-2]}_{parts[-1]}", '%Y%m%d_%H%M%S')
            age_days = (datetime.now() - trained_at).days
            fresh = age_days <= _MAX_ADAPTER_AGE_DAYS
            results.add('adapter_freshness',
                        passed=fresh,
                        detail=f"Trained {age_days}d ago ({trained_at.strftime('%Y-%m-%d %H:%M')})",
                        warn_only=not fresh)
        else:
            results.add('adapter_freshness', True, f'Could not parse date from {name}', warn_only=True)
    except ValueError:
        results.add('adapter_freshness', True, f'Could not parse date from {name}', warn_only=True)


def check_adapter_files(adapter_path: Path, results: CheckResult) -> None:
    """Check that required adapter weight files are present."""
    required = ['adapter_config.json', 'adapter_model.safetensors']
    for fname in required:
        exists = (adapter_path / fname).exists()
        results.add(f'file_{fname}', exists,
                    'present' if exists else f'MISSING — {adapter_path / fname}')


def check_inference(results: CheckResult, verbose: bool = False) -> None:
    """Run a quick smoke-test inference to verify the model responds correctly."""
    sys.path.insert(0, str(_SCRIPTS_DIR))

    print("\n  ⏳ Loading model for inference check (may take a few minutes on cold start)...")
    t0 = time.time()
    try:
        from model_inference_lora import get_trading_decision, parse_decision
        response = get_trading_decision(_SMOKE_PROMPT, max_new_tokens=200)
        latency = time.time() - t0
    except Exception as e:
        results.add('inference_runs', False, f'Model failed to run: {e}')
        return

    results.add('inference_runs', True, f'Responded in {latency:.1f}s')

    # Check response isn't empty / garbage
    non_empty = len(response.strip()) > 20
    results.add('inference_non_empty', non_empty,
                f'{len(response)} chars' if non_empty else 'Response too short')

    # Check format
    r = response.lower()
    has_decision   = any(kw in r for kw in ['decision:', 'buy', 'sell', 'hold'])
    has_confidence = 'confidence' in r
    results.add('inference_format', has_decision and has_confidence,
                'Decision + Confidence present' if (has_decision and has_confidence)
                else f'Missing: {"decision" if not has_decision else ""} {"confidence" if not has_confidence else ""}')

    # Parse and check for BUY (expected for this oversold setup)
    parsed = parse_decision(response)
    decision  = parsed['decision']
    confidence = parsed['confidence']
    results.add('inference_sensible',
                decision in ('buy', 'hold'),   # sell would be wrong for an oversold setup
                f"Decision={decision.upper()}, Confidence={confidence:.2f}",
                warn_only=(decision == 'hold'))

    if verbose:
        print(f"\n  --- Model response ---\n{response[:600]}\n  ---")


def repair_naked_positions(results: CheckResult) -> None:
    """Run reprotect_positions sweep before counting unprotected positions.

    This ensures the morning count reflects "repair attempted and failed"
    rather than just "unprotected at last session end."  Failures are
    warn_only — the broker API may be unavailable pre-market.
    """
    import os
    try:
        from alpaca.trading.client import TradingClient
        from risk_reconciler import reprotect_positions, write_reconcile_status
        api_key = os.getenv('ALPACA_API_KEY') or os.getenv('APCA_API_KEY_ID', '')
        secret  = os.getenv('ALPACA_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY', '')
        if not api_key or not secret:
            results.add('repair_sweep', False,
                        'Alpaca credentials not set — skipping repair sweep',
                        warn_only=True)
            return
        paper  = os.getenv('PAPER_TRADING', 'true').lower() != 'false'
        client = TradingClient(api_key, secret, paper=paper)
        dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
        summary = reprotect_positions(
            client,
            {'stop_loss': -0.07, 'take_profit': 0.15},
            dry_run=dry_run,
        )
        write_reconcile_status(client, _PROJECT_ROOT / 'logs' / 'reconcile_status.json')
        results.add(
            'repair_sweep', True,
            f"Repair sweep: protected={summary['protected']} "
            f"fractional_skipped={summary['skipped_fractional']} "
            f"deferred_pdt={summary['deferred_pdt']} "
            f"errors={summary['errors']}",
        )
    except Exception as e:
        results.add('repair_sweep', False,
                    f'Repair sweep failed: {e}',
                    warn_only=True)


def check_reconcile_status(results: CheckResult) -> None:
    """Verify no unprotected positions remain after the repair sweep.

    Hard fail: options > 0 or stocks_whole > 0 (these must have broker-side stops).
    Warn-only: stocks_fractional > 0 (GTC stop unsupported for fractional — known limitation).
    Warn-only: any count == -1 (API unavailable at check time).
    """
    status_path = _PROJECT_ROOT / 'logs' / 'reconcile_status.json'
    if not status_path.exists():
        results.add(
            'reconcile_status', False,
            'reconcile_status.json not found (daemons not started yet)',
            warn_only=True,
        )
        return
    try:
        data = json.loads(status_path.read_text())
        checked_at = data.get('checked_at', 'unknown')

        # Prefer the three-way split keys; fall back to legacy for old-format files.
        n_opts  = int(data.get('unprotected_options',
                               data.get('unprotected_positions', -1)))
        n_whole = int(data.get('unprotected_stocks_whole', -1))
        n_frac  = int(data.get('unprotected_stocks_fractional', -1))

        # options: hard fail if > 0
        if n_opts < 0:
            results.add('reconcile_options', False,
                        f'Could not determine options protection status (API error) '
                        f'— checked at {checked_at}', warn_only=True)
        else:
            detail = (
                f'{n_opts} unprotected options position(s) — checked at {checked_at}'
                if n_opts > 0
                else f'all options positions protected — checked at {checked_at}'
            )
            results.add('reconcile_options', n_opts == 0, detail)

        # stocks_whole: hard fail if > 0
        if n_whole < 0:
            results.add('reconcile_stocks_whole', False,
                        f'Could not determine whole-share stock protection status (API error) '
                        f'— checked at {checked_at}', warn_only=True)
        else:
            detail = (
                f'{n_whole} unprotected whole-share stock position(s) — checked at {checked_at}'
                if n_whole > 0
                else f'all whole-share stocks protected — checked at {checked_at}'
            )
            results.add('reconcile_stocks_whole', n_whole == 0, detail)

        # stocks_fractional: always warn-only (Alpaca GTC limitation)
        if n_frac < 0:
            results.add('reconcile_stocks_fractional', False,
                        f'Could not determine fractional stock protection status (API error) '
                        f'— checked at {checked_at}', warn_only=True)
        elif n_frac > 0:
            results.add('reconcile_stocks_fractional', False,
                        f'{n_frac} fractional position(s) unprotected (broker GTC stop '
                        f'unsupported — agent SELL is the backstop) — checked at {checked_at}',
                        warn_only=True)
        else:
            results.add('reconcile_stocks_fractional', True,
                        f'no unprotected fractional positions — checked at {checked_at}')

    except Exception as e:
        results.add('reconcile_status', False,
                    f'Could not read reconcile_status.json: {e}',
                    warn_only=True)


def _check_vllm_staleness(results: CheckResult, merged_latest: Path) -> None:
    """Warn if vLLM was started before merged_latest was last repointed."""
    try:
        out = subprocess.run(
            ['systemctl', 'show', 'ai-inference-server.service',
             '--property=ActiveEnterTimestamp'],
            capture_output=True, text=True, timeout=5,
        )
        ts_str = out.stdout.strip().split('=', 1)[-1].strip()
        if not ts_str or ts_str.lower() == 'n/a':
            return  # service never started — nothing to check
        # Systemd format: "Sun 2026-06-15 01:30:00 UTC" — strip leading weekday
        parts = ts_str.split(' ', 1)
        if len(parts) == 2 and len(parts[0]) == 3:
            ts_str = parts[1]
        from datetime import timezone as _tz
        service_start = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S %Z')
        if service_start.tzinfo is None:
            service_start = service_start.replace(tzinfo=_tz.utc)
        symlink_mtime = datetime.fromtimestamp(
            os.lstat(str(merged_latest)).st_mtime, tz=_tz.utc,
        )
        if service_start < symlink_mtime:
            delta_min = int((symlink_mtime - service_start).total_seconds() // 60)
            results.add(
                'vllm_model_freshness', False,
                f"vLLM started {delta_min}m before symlink was repointed — "
                f"restart ai-inference-server to load the current merged model",
                warn_only=True,
            )
        else:
            results.add('vllm_model_freshness', True,
                        'vLLM started after last symlink repoint — serving current model')
    except Exception as e:
        results.add('vllm_model_freshness', False,
                    f'could not check vLLM staleness: {e}', warn_only=True)


def check_merged_symlink_alignment(results: CheckResult) -> None:
    """
    Verify finance_qwen_32b_merged_latest alignment with the latest promotion record.

    Hard fail:
      - merged_model_promoted=true  AND symlink != recorded path  (promoted model not served)
      - merged_model_promoted=false AND symlink == recorded path  (serving a declined model)
    Warn-only:
      - merged_model is null (no merge to check)
      - merged_model_promoted field absent in record (old format — intent unknown)
      - vLLM service predates symlink repoint (stale serve, needs restart)
    """
    from alerts import send_alert, AlertLevel

    merged_latest = _PROJECT_ROOT / 'finetune' / 'finance_qwen_32b_merged_latest'
    if not merged_latest.is_symlink():
        results.add('merged_symlink', False,
                    f'{merged_latest.name} is missing — vLLM may be serving an outdated model',
                    warn_only=True)
        return

    actual = merged_latest.resolve()

    promo_files = sorted(
        (_PROJECT_ROOT / 'logs' / 'eval').glob('promotion_*.json'),
        key=lambda f: f.stat().st_mtime,
    )
    if not promo_files:
        results.add('merged_symlink_alignment', True,
                    'no promotion record found — cannot verify alignment',
                    warn_only=True)
        return

    try:
        promo = json.loads(promo_files[-1].read_text())
    except Exception as e:
        results.add('merged_symlink_alignment', False,
                    f'could not read promotion record: {e}', warn_only=True)
        return

    recorded = promo.get('merged_model')
    if not recorded:
        results.add('merged_symlink_alignment', True,
                    'promotion record has no merged_model — run after next fine-tune',
                    warn_only=True)
        return

    was_promoted = promo.get('merged_model_promoted')  # None for old-format records
    recorded_resolved = Path(recorded).resolve()
    aligned = actual == recorded_resolved

    if was_promoted is None:
        # Old-format record: merged_model_promoted absent — can't determine intent, warn only
        results.add(
            'merged_symlink_alignment',
            aligned,
            f"readlink={actual.name} {'==' if aligned else '!='} recorded={recorded_resolved.name} "
            f"(old record — merged_model_promoted absent, warn-only)",
            warn_only=True,
        )
    elif was_promoted:
        # Candidate was promoted — symlink MUST point at the recorded (promoted) path
        if aligned:
            results.add('merged_symlink_alignment', True,
                        f"readlink={actual.name} == promoted={recorded_resolved.name}")
        else:
            msg = (
                f"merged_latest → {actual.name} but promoted model is "
                f"{recorded_resolved.name} — vLLM is NOT serving the promoted model"
            )
            send_alert(AlertLevel.CRITICAL, 'merged_symlink_mismatch_promoted', msg)
            results.add('merged_symlink_alignment', False, msg)
    else:
        # Candidate was declined — symlink must NOT point at the declined model
        if aligned:
            msg = (
                f"merged_latest → {actual.name} but this model was DECLINED — "
                f"vLLM is serving a model that failed evaluation"
            )
            send_alert(AlertLevel.CRITICAL, 'merged_symlink_serving_declined', msg)
            results.add('merged_symlink_alignment', False, msg)
        else:
            results.add(
                'merged_symlink_alignment', True,
                f"readlink={actual.name} (declined model {recorded_resolved.name} not served — correct)",
            )

    # Cross-check: warn if vLLM started before symlink was last repointed
    if os.getenv('INFERENCE_BACKEND', 'ollama').lower() == 'vllm':
        _check_vllm_staleness(results, merged_latest)


def check_service_status(results: CheckResult) -> None:
    """Check that the trading daemon services are running."""
    for service in ('ai-trading-bot.service', 'ai-options-bot.service'):
        try:
            out = subprocess.run(
                ['systemctl', 'is-active', service],
                capture_output=True, text=True, timeout=5
            )
            active = out.stdout.strip() == 'active'
            results.add(f'service_{service.replace(".service", "")}',
                        active,
                        out.stdout.strip())
        except Exception as e:
            results.add(f'service_{service.replace(".service", "")}',
                        False, f'Could not check: {e}', warn_only=True)


def main():
    parser = argparse.ArgumentParser(description='Pre-market model health check')
    parser.add_argument('--no-inference', action='store_true',
                        help='Skip GPU inference test (fast metadata-only check)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full model response')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🌅 MORNING MODEL CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    results = CheckResult()

    # 0. Self-heal: ensure vLLM is running before any checks that depend on it.
    #    Uses start-only (never stop/restart) so it is safe to call here.
    if os.getenv('INFERENCE_BACKEND', 'ollama').lower() == 'vllm':
        try:
            sys.path.insert(0, str(_SCRIPTS_DIR))
            from inference_client import ensure_vllm_running
            ensure_vllm_running()
        except Exception as e:
            results.add('vllm_self_heal', False,
                        f'ensure_vllm_running failed: {e}', warn_only=True)

    # 1. LoRA adapter symlink
    adapter_path = check_symlink(results)

    if adapter_path:
        # 2. Config
        check_adapter_config(adapter_path, results)
        # 3. Freshness
        check_adapter_freshness(adapter_path, results)
        # 4. Weight files
        check_adapter_files(adapter_path, results)

    # 5. Merged model symlink vs promotion record alignment
    check_merged_symlink_alignment(results)

    # 6. Services
    check_service_status(results)

    # 7. Repair + detect: reprotect naked positions, then verify all protected.
    # Sequence is deliberate: sweep runs first so a nonzero count after repair
    # means "tried and failed," not just "not checked yet."
    repair_naked_positions(results)
    check_reconcile_status(results)

    # 8. Inference (optional, GPU-heavy)
    if not args.no_inference and adapter_path:
        check_inference(results, verbose=args.verbose)
    elif args.no_inference:
        print("  ⏭️  Skipping inference check (--no-inference)")

    # Report
    print(f"\n{'─'*60}")
    results.print_report()
    print(f"{'─'*60}")

    if results.all_passed and not results.has_warnings:
        print(f"\n🟢 ALL CHECKS PASSED — model is ready for trading\n")
    elif results.all_passed:
        print(f"\n🟡 PASSED WITH WARNINGS — review warnings before market open\n")
    else:
        print(f"\n🔴 CHECKS FAILED — do not trade until issues are resolved\n")
        print("Quick fixes:")
        print("  Broken symlink:  ln -sfn /path/to/adapter finetune/finance_qwen_32b_lora_latest")
        print("  Missing weights: re-run fine_tune_llm.py")
        print("  Service down:    sudo systemctl restart ai-trading-bot.service")

    sys.exit(0 if results.all_passed else 1)


if __name__ == '__main__':
    main()
