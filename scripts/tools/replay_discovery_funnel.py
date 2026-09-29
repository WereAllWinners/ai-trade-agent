#!/usr/bin/env python3
"""
replay_discovery_funnel.py — offline discovery-funnel replay (sprint04 F2.2).

Runs the SAME StockDiscovery.build_scan_universe() + run_scan_pipeline() the
live trading session uses, against the CURRENT curated universe with CURRENT
thresholds. Read-only: prints the funnel + nearest-miss table, writes
NOTHING (no discovery_cache.json, no discovered_opportunities*.json, no
alerts, no delisted_cache.json fail-count mutation).

Neutralizing the one persisted side effect a normal scan triggers
(delisted_cache.json fail-count writes via _record_fetch_result) is done by
monkeypatching that method to a no-op on the instance — self._delisted still
reflects today's real denylist (read-only), it's just never *updated* by
this run.

Usage:
    python3 scripts/tools/replay_discovery_funnel.py            # human-readable
    python3 scripts/tools/replay_discovery_funnel.py --deep      # full universe
    python3 scripts/tools/replay_discovery_funnel.py --json      # machine-readable
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data'))
from stock_discovery import StockDiscovery  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep', action='store_true', help='Scan the full universe (slow)')
    parser.add_argument('--json', action='store_true', help='Print raw JSON instead of a table')
    args = parser.parse_args()

    disco = StockDiscovery()
    # Read-only guarantee: neutralize the one persisted side effect scanning
    # normally triggers. self._delisted itself still reflects today's real
    # denylist — it's just never updated by this run.
    disco._record_fetch_result = lambda symbol, success: None

    universe = disco.build_scan_universe(deep_scan=args.deep)
    result = disco.run_scan_pipeline(universe)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    print(f"Universe: {result['universe_size']} symbols")
    print()
    for stage, stats in result['funnel'].items():
        denom = stats.get('universe_in')
        denom_str = f"/ {denom}" if denom is not None else ""
        print(f"  {stage:16s} {stats['passed']:>4d} {denom_str}")
        for sym, val in stats.get('near_misses', [])[:10]:
            print(f"      near-miss: {sym:8s} = {val}")
    print()
    print(f"Final ranked candidates ({len(result['ranked_stocks'])}):")
    print(result['ranked_stocks'][:35])


if __name__ == '__main__':
    main()
