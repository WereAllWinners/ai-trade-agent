#!/usr/bin/env python3
"""
backfill_trade_log_source.py — targeted source-backfill for legacy trade_log
rows (sprint02 D4.4).

Problem this closes: D4.3 makes outcome_tracker's FIFO matching bucket by
(symbol, source) instead of symbol alone, to stop paper/live cross-contamination
(confirmed empirically — 2/15 sampled order_ids in the shared trade_log.jsonl
actually belonged to the live account). But every currently-open position's
original BUY row predates the `source` field. Once readers filter by source, a
post-fix SELL for one of those positions finds no matching BUY in its own
source's bucket — producing a fresh wave of missing_entry_fill for the entire
current book, which is worse than doing nothing for exactly the positions
sprint02 D3 is trying to protect.

Scope — deliberately narrow, NOT the full historical file:
  (a) legacy (no 'source' field) BUY rows for symbols currently held open on
      either the paper or live account (read-only get_all_positions() lookup)
  (b) legacy BUY rows within --lookback days that have no matching SELL row
      for the same symbol anywhere in the trade log (orphaned, likely still
      open even if not caught by (a) for some reason)
  Capped at --max-candidates (default 50) — the full 1,305+ line historical
  archaeology stays explicitly out of scope; see docs/sprint02-d7-findings.md.

For each candidate, the actual owning account is determined via the same
read-only get_order_by_id lookup against both TradingClients already proven
correct in the D4.3 empirical spot-check (Sprint01/02). Matches sync_broker_exits.py's
--dry-run/--execute preview-gate pattern — --execute must never run from
automation; preview first, human sign-off, then execute.

Usage (standalone):
    python3 scripts/backfill_trade_log_source.py                    # dry-run preview
    python3 scripts/backfill_trade_log_source.py --lookback 60       # wider preview
    python3 scripts/backfill_trade_log_source.py --execute           # apply — NEVER
                                                                       # run this from
                                                                       # automation
"""
import json
import logging
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pathfix  # noqa: F401

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRADE_LOG_PATH = _PROJECT_ROOT / 'logs' / 'trade_log.jsonl'
_DEFAULT_MAX_CANDIDATES = 50


def _load_lines(trade_log_path: Path) -> list:
    """Return (line_index, parsed_dict_or_None, raw_line) for every line."""
    if not trade_log_path.exists():
        return []
    out = []
    with open(trade_log_path) as f:
        for i, line in enumerate(f):
            raw = line.rstrip('\n')
            if not raw.strip():
                out.append((i, None, raw))
                continue
            try:
                out.append((i, json.loads(raw), raw))
            except json.JSONDecodeError:
                out.append((i, None, raw))
    return out


def _open_symbols(trading_client) -> set:
    """Read-only get_all_positions() — symbols currently held (stock only,
    excludes long OCC option symbols)."""
    try:
        positions = trading_client.get_all_positions()
        return {p.symbol for p in positions if len(p.symbol) <= 10}
    except Exception as e:
        logging.warning("backfill_trade_log_source: could not fetch positions: %s", e)
        return set()


def _bucketed_unmatched_buys(rows: list) -> list:
    """Mirror outcome_tracker.match_and_calculate_pnl's (symbol, source) FIFO
    bucketing exactly, to find buy rows that are unmatched under the ACTUAL
    production matching logic today.

    A source-agnostic replay (ignore 'source', just chronologically pair
    buys/sells per symbol) would hide precisely the gap D4.4 exists to close:
    a legacy (untagged) buy and a new source-tagged sell for the very same
    real-world trade land in different buckets — (SYM, None) vs (SYM,
    'live') — and outcome_tracker never pairs them, even though
    chronologically the sell obviously closed that buy. Replaying the same
    bucketing here is what correctly flags that buy as needing a source
    stamp, while leaving genuinely-already-matched legacy pairs (both sides
    bucketed under None) alone.
    """
    by_bucket = {}
    for idx, rec, _raw in rows:
        if not rec or not rec.get('symbol'):
            continue
        key = (rec['symbol'], rec.get('source'))
        by_bucket.setdefault(key, []).append((idx, rec))

    unmatched = []
    for _key, bucket_rows in by_bucket.items():
        bucket_rows.sort(key=lambda pair: pair[1].get('timestamp', ''))
        open_buys = []
        for idx, rec in bucket_rows:
            action = rec.get('action')
            if action == 'buy':
                open_buys.append((idx, rec))
            elif action == 'sell' and open_buys:
                open_buys.pop(0)
        unmatched.extend(open_buys)
    return unmatched


def find_candidates(trade_log_path: Path, paper_client, live_client,
                     lookback_days: int = 30, max_candidates: int = _DEFAULT_MAX_CANDIDATES) -> list:
    """Return the candidate legacy BUY rows to backfill.

    Scope is FIFO-unmatched buys only (the ones actually contributing to an
    open lot or otherwise unreconciled) — not every historical buy of a
    symbol that happens to be held open today, which for a bot that rotates
    the same tickers repeatedly would sweep in hundreds of already-closed
    buys. Each candidate: {'line_index', 'order_id', 'symbol', 'timestamp',
    'reason'} where reason is 'open_position' or 'orphaned_recent'.

    'open_position' candidates are NEVER capped by max_candidates — that set
    is inherently bounded by the broker's real, current position list, not
    by an arbitrary limit (empirically this can run into the hundreds for an
    account that's been repeatedly buying a name with almost no filled
    sells, which is a legitimate finding, not a scope-creep bug — see
    sprint02 D4.4 findings). max_candidates only bounds 'orphaned_recent',
    which is the genuinely open-ended scope (b) — any legacy buy within the
    lookback window for a symbol NOT currently open — and still needs a
    safety cap.
    """
    rows = _load_lines(trade_log_path)
    open_syms = _open_symbols(paper_client) | _open_symbols(live_client)
    cutoff = datetime.now() - timedelta(days=lookback_days)

    open_position_candidates = []
    orphaned_candidates = []
    for idx, rec in _bucketed_unmatched_buys(rows):
        if 'source' in rec:
            continue
        symbol = rec.get('symbol')
        order_id = rec.get('order_id')
        if not symbol or not order_id:
            continue

        if symbol in open_syms:
            open_position_candidates.append({
                'line_index': idx, 'order_id': order_id, 'symbol': symbol,
                'timestamp': rec.get('timestamp', ''), 'reason': 'open_position',
            })
            continue

        try:
            ts = datetime.fromisoformat(rec.get('timestamp', ''))
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        orphaned_candidates.append({
            'line_index': idx, 'order_id': order_id, 'symbol': symbol,
            'timestamp': rec.get('timestamp', ''), 'reason': 'orphaned_recent',
        })

    open_position_candidates.sort(key=lambda c: c['timestamp'])
    orphaned_candidates.sort(key=lambda c: c['timestamp'])
    return open_position_candidates + orphaned_candidates[:max_candidates]


def determine_source(order_id: str, paper_client, live_client) -> str | None:
    """Read-only get_order_by_id lookup against both accounts. Returns
    'paper', 'live', or None if found in neither (order too old / purged)."""
    try:
        paper_client.get_order_by_id(order_id)
        return 'paper'
    except Exception:
        pass
    try:
        live_client.get_order_by_id(order_id)
        return 'live'
    except Exception:
        pass
    return None


def backfill_source(trade_log_path: Path = _DEFAULT_TRADE_LOG_PATH, *,
                     paper_client, live_client,
                     lookback_days: int = 30, max_candidates: int = _DEFAULT_MAX_CANDIDATES,
                     dry_run: bool = True) -> dict:
    """Stamp `source` onto targeted legacy BUY rows in trade_log.jsonl.

    Returns {'candidates': int, 'stamped': int, 'unresolved': int, 'errors': int,
             'report': [...]}  # report always populated (preview or executed)
    """
    summary = {'candidates': 0, 'stamped': 0, 'unresolved': 0, 'errors': 0, 'report': []}

    try:
        candidates = find_candidates(trade_log_path, paper_client, live_client,
                                      lookback_days, max_candidates)
    except Exception as e:
        logging.warning("backfill_trade_log_source: could not scan candidates: %s", e)
        summary['errors'] += 1
        return summary

    summary['candidates'] = len(candidates)
    if not candidates:
        return summary

    resolved = {}  # line_index -> source
    for c in candidates:
        try:
            source = determine_source(c['order_id'], paper_client, live_client)
        except Exception as e:
            logging.warning("backfill_trade_log_source: lookup failed for %s: %s",
                             c['order_id'], e)
            source = None
        entry = {**c, 'resolved_source': source}
        summary['report'].append(entry)
        if source is None:
            summary['unresolved'] += 1
        else:
            resolved[c['line_index']] = source

    if dry_run or not resolved:
        return summary

    # In-place rewrite: only the targeted line indices are modified; every
    # other line (including malformed ones) is preserved byte-for-byte.
    rows = _load_lines(trade_log_path)
    out_lines = []
    for idx, rec, raw in rows:
        if idx in resolved and rec is not None:
            rec = {**rec, 'source': resolved[idx]}
            out_lines.append(json.dumps(rec))
            summary['stamped'] += 1
        else:
            out_lines.append(raw)

    trade_log_path.write_text('\n'.join(out_lines) + ('\n' if out_lines else ''))
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Targeted source-backfill for legacy trade_log.jsonl rows (sprint02 D4.4)'
    )
    parser.add_argument('--lookback', type=int, default=30,
                         help='Lookback window in days for orphaned-buy scope (default 30)')
    parser.add_argument('--max-candidates', type=int, default=_DEFAULT_MAX_CANDIDATES,
                         help=f'Cap on candidates scanned per run (default {_DEFAULT_MAX_CANDIDATES})')
    parser.add_argument('--execute', action='store_true',
                         help='Apply the backfill. NEVER invoke this from automation — '
                              'preview with the default dry-run first and get explicit sign-off.')
    args = parser.parse_args()

    # Paper and live are separate Alpaca accounts with distinct credentials
    # (not one key pair distinguished by a `paper=` flag) — load each env
    # file into its own dict rather than os.environ, so neither clobbers
    # the other when both define the same variable names.
    from dotenv import dotenv_values
    paper_env = dotenv_values(_PROJECT_ROOT / '.env')
    live_env = dotenv_values(_PROJECT_ROOT / '.env.live')

    paper_key = paper_env.get('ALPACA_API_KEY') or paper_env.get('APCA_API_KEY_ID', '')
    paper_secret = paper_env.get('ALPACA_SECRET_KEY') or paper_env.get('APCA_API_SECRET_KEY', '')
    live_key = live_env.get('ALPACA_API_KEY') or live_env.get('APCA_API_KEY_ID', '')
    live_secret = live_env.get('ALPACA_SECRET_KEY') or live_env.get('APCA_API_SECRET_KEY', '')
    if not paper_key or not paper_secret:
        print(f"Paper credentials not set in {_PROJECT_ROOT / '.env'}")
        sys.exit(1)
    if not live_key or not live_secret:
        print(f"Live credentials not set in {_PROJECT_ROOT / '.env.live'}")
        sys.exit(1)

    from alpaca.trading.client import TradingClient
    paper_client = TradingClient(paper_key, paper_secret, paper=True)
    live_client = TradingClient(live_key, live_secret, paper=False)

    dry_run = not args.execute
    logging.info(
        "🔄 backfill_trade_log_source running in %s mode (lookback=%dd, max=%d)",
        'execute' if args.execute else 'dry-run', args.lookback, args.max_candidates,
    )

    summary = backfill_source(
        paper_client=paper_client, live_client=live_client,
        lookback_days=args.lookback, max_candidates=args.max_candidates,
        dry_run=dry_run,
    )

    print(f"\n{'Preview' if dry_run else 'Backfill'} report "
          f"({summary['candidates']} candidate(s)):\n")
    for entry in summary['report']:
        print(
            f"  {entry['symbol']:12s}  order_id={entry['order_id']}  "
            f"reason={entry['reason']:16s}  resolved_source={entry['resolved_source']}"
        )

    print(
        f"\nSummary: candidates={summary['candidates']}  stamped={summary['stamped']}  "
        f"unresolved={summary['unresolved']}  errors={summary['errors']}"
    )
    if summary['unresolved']:
        print(
            f"\n⚠️  {summary['unresolved']} candidate(s) could not be resolved against "
            f"either account (order too old / purged from Alpaca's history) — left untouched."
        )


if __name__ == '__main__':
    main()
