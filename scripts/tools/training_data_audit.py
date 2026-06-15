#!/usr/bin/env python3
"""
training_data_audit.py — READ-ONLY training data quality audit.

Writes /tmp/training_data_audit.txt and prints it.
Safe to run while the system is live; the DB is opened read-only (mode=ro)
so writes are structurally impossible.

Usage:
    python3 scripts/tools/training_data_audit.py
"""

import json
import os
import sqlite3
import sys
import textwrap
from datetime import datetime
from io import StringIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution — use the project's own db module, never a hardcoded path
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _pathfix  # noqa: F401
import db as _db

_PROJECT_ROOT = _db.DB_PATH.parent.parent  # logs/ → project root

DB_PATH = _db.DB_PATH
REPORT_PATH = Path('/tmp/training_data_audit.txt')

# SFT exclude set — must match fine_tune_llm.py:load_training_data exactly
_SFT_EXCLUDE_LABELS = {'weak_loser', 'loser', 'strong_loser'}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _table(rows: list[tuple], headers: list[str]) -> str:
    """Format rows as a fixed-width aligned table."""
    if not rows:
        return '  (no rows)\n'
    all_rows = [headers] + [tuple(str(v) if v is not None else 'NULL' for v in r) for r in rows]
    widths = [max(len(str(c)) for c in col) for col in zip(*all_rows)]
    sep = '  ' + '  '.join('-' * w for w in widths)
    lines = []
    lines.append('  ' + '  '.join(str(h).ljust(w) for h, w in zip(headers, widths)))
    lines.append(sep)
    for row in all_rows[1:]:
        lines.append('  ' + '  '.join(str(v).ljust(w) for v, w in zip(row, widths)))
    return '\n'.join(lines) + '\n'


def _section(buf: StringIO, title: str) -> None:
    buf.write(f'\n{"="*70}\n')
    buf.write(f'  {title}\n')
    buf.write(f'{"="*70}\n')


def _read(buf: StringIO, text: str) -> None:
    buf.write(f'\nREAD: {textwrap.fill(text, width=68, subsequent_indent="       ")}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    buf = StringIO()

    buf.write('=' * 70 + '\n')
    buf.write('  TRAINING DATA AUDIT\n')
    buf.write(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    buf.write(f'  DB (READ-ONLY): {DB_PATH}\n')
    buf.write('=' * 70 + '\n')

    # Open DB strictly read-only — writing is structurally impossible
    conn = sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row

    # ── Schema dump ─────────────────────────────────────────────────────────
    _section(buf, 'SCHEMA — PRAGMA table_info(training_examples)')
    try:
        rows = conn.execute('PRAGMA table_info(training_examples)').fetchall()
        headers = ['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']
        buf.write(_table([tuple(r) for r in rows], headers))
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    # ── Accumulate summary stats for the top paragraph ───────────────────
    _summary_total   = None
    _summary_pct_opt = None
    _summary_missed  = None
    _summary_conf    = '(not computed)'

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Label distribution
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 1 — Label distribution')
    try:
        rows = conn.execute("""
            SELECT label,
                   COUNT(*)                         AS n,
                   ROUND(AVG(pnl_pct), 4)           AS avg_pnl,
                   ROUND(AVG(confidence), 3)        AS avg_conf
            FROM training_examples
            GROUP BY label
            ORDER BY n DESC
        """).fetchall()
        headers = ['label', 'n', 'avg_pnl', 'avg_conf']
        buf.write(_table([tuple(r) for r in rows], headers))
        _summary_total = sum(r['n'] for r in rows)
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    _read(buf,
        "If weak_winner + winner dominate the count, SFT is mostly imitating marginal outcomes. "
        "strong_winner and correct_hold are the labels we WANT dominating. "
        "loser/strong_loser/weak_loser rows should exist but must be EXCLUDED from SFT (Section 6).")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Bot mix
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 2 — Bot mix (stock vs options)')
    try:
        rows = conn.execute("""
            SELECT bot, COUNT(*) AS n FROM training_examples GROUP BY bot
        """).fetchall()
        headers = ['bot', 'n']
        buf.write(_table([tuple(r) for r in rows], headers))
        total = sum(r['n'] for r in rows)
        opt_n = next((r['n'] for r in rows if r['bot'] == 'options'), 0)
        _summary_pct_opt = opt_n / total if total else 0
        buf.write(f'\n  options share: {_summary_pct_opt:.1%} of {total} total\n')
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    buf.write('\n  -- per-bot label breakdown --\n')
    try:
        rows = conn.execute("""
            SELECT bot, label, COUNT(*) AS n
            FROM training_examples
            GROUP BY bot, label
            ORDER BY bot, n DESC
        """).fetchall()
        headers = ['bot', 'label', 'n']
        buf.write(_table([tuple(r) for r in rows], headers))
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    _read(buf,
        "If options is < ~20% of total, an options-specialized candidate trains on a thin, "
        "high-variance slice while likely being judged on a stock-heavy eval set — "
        "a plausible direct cause of the streak. Note per-bot label mix: "
        "options with mostly losers/weak is especially noisy.")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Temporal distribution
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 3 — Temporal distribution (month × label)')
    try:
        rows = conn.execute("""
            SELECT substr(entry_date, 1, 7) AS month,
                   label,
                   COUNT(*) AS n
            FROM training_examples
            WHERE entry_date IS NOT NULL AND entry_date != ''
            GROUP BY month, label
            ORDER BY month
        """).fetchall()
        headers = ['month', 'label', 'n']
        buf.write(_table([tuple(r) for r in rows], headers))
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    _read(buf,
        "With a 45-day half-life, recent months dominate after weighting. "
        "If the most recent 1-2 months are mostly losers/weak_winners, the model is being taught "
        "to imitate a bad stretch and judged against a broader-history golden set. "
        "Look for a recent regime worse than the historical average.")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 4 — missed_opportunity leakage into SFT
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 4 — missed_opportunity leakage into SFT')
    try:
        row = conn.execute("""
            SELECT COUNT(*) AS missed_opp_total FROM training_examples
            WHERE label = 'missed_opportunity'
        """).fetchone()
        _summary_missed = row['missed_opp_total'] if row else 0
        buf.write(f'  missed_opportunity total: {_summary_missed}\n')
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    buf.write('\n  -- sample 5 rows (output head) --\n')
    try:
        rows = conn.execute("""
            SELECT symbol, entry_date,
                   substr(ideal_output, 1, 80) AS output_head
            FROM training_examples
            WHERE label = 'missed_opportunity'
            LIMIT 5
        """).fetchall()
        headers = ['symbol', 'entry_date', 'output_head']
        buf.write(_table([tuple(r) for r in rows], headers))
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    _read(buf,
        "A missed_opportunity is a HOLD that missed a real winner. "
        "If these carry Decision: HOLD and reach SFT as imitation targets, the model is "
        "trained to reproduce the hold that missed the move. "
        "More than a handful is a real quality drag. "
        "Confirm output_head shows Decision: HOLD.")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Confidence calibration
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 5 — Confidence calibration (is conf predictive?)')
    try:
        rows = conn.execute("""
            SELECT
              CASE WHEN confidence < 0.70 THEN '60-70'
                   WHEN confidence < 0.80 THEN '70-80'
                   WHEN confidence < 0.90 THEN '80-90'
                   ELSE '90-100' END                         AS conf_bin,
              COUNT(*)                                        AS n,
              ROUND(AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END), 3) AS win_rate,
              ROUND(AVG(pnl_pct), 4)                          AS avg_pnl
            FROM training_examples
            WHERE pnl_pct IS NOT NULL
            GROUP BY conf_bin
            ORDER BY conf_bin
        """).fetchall()
        headers = ['conf_bin', 'n', 'win_rate', 'avg_pnl']
        buf.write(_table([tuple(r) for r in rows], headers))

        # Determine trend for summary paragraph
        if len(rows) >= 2:
            bin_order = ['60-70', '70-80', '80-90', '90-100']
            rates = {r['conf_bin']: r['win_rate'] for r in rows if r['win_rate'] is not None}
            ordered = [rates[b] for b in bin_order if b in rates]
            if len(ordered) >= 2:
                rising   = all(ordered[i] <= ordered[i+1] for i in range(len(ordered)-1))
                inverted = all(ordered[i] >= ordered[i+1] for i in range(len(ordered)-1))
                spread   = max(ordered) - min(ordered)
                if spread < 0.05:
                    _summary_conf = 'FLAT (spread < 5pp — conf carries no predictive signal)'
                elif rising:
                    _summary_conf = 'RISING (conf is predictive — healthy)'
                elif inverted:
                    _summary_conf = 'INVERTED (high conf → lower win rate — miscalibrated)'
                else:
                    _summary_conf = f'MIXED (spread={spread:.2f}, not monotone)'
        buf.write(f'\n  Win-rate trend: {_summary_conf}\n')
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    _read(buf,
        "If win_rate is roughly FLAT across confidence bins, stated confidence carries no "
        "predictive signal — calibration collapses every target toward ~0.5 and the model "
        "loses the ability to size up on genuine high-conviction setups. "
        "Healthy: win_rate rises with the confidence bin. Flat or inverted is a finding.")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Verify losers excluded from SFT export
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 6 — Loser exclusion from SFT export')

    buf.write('  -- loser counts in DB --\n')
    try:
        rows = conn.execute("""
            SELECT label, COUNT(*) AS n FROM training_examples
            WHERE label IN ('loser','strong_loser','weak_loser')
            GROUP BY label
        """).fetchall()
        headers = ['label', 'n']
        buf.write(_table([tuple(r) for r in rows], headers))
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    buf.write('\n  -- SFT JSON export check (replicates fine_tune_llm.py:load_training_data filter) --\n')
    try:
        # Find data paths from most recent training_metadata.json
        metadata_files = sorted(
            (_PROJECT_ROOT / 'finetune').glob('*/training_metadata.json'),
            key=lambda f: f.stat().st_mtime,
        )
        data_paths: list[Path] = []
        if metadata_files:
            meta = json.loads(metadata_files[-1].read_text())
            data_paths = [Path(p) for p in meta.get('data_paths', [])]
            buf.write(f'  Source: {metadata_files[-1].parent.name}/training_metadata.json\n')
        else:
            # Fallback: canonical paths for both bots
            for name in ('options_training_data.json', 'training_data.json'):
                p = _PROJECT_ROOT / 'finetune' / 'data' / name
                if p.exists():
                    data_paths.append(p)
            buf.write('  Source: fallback canonical paths\n')

        all_examples: list[dict] = []
        for dp in data_paths:
            if not dp.exists():
                buf.write(f'  SKIP (not found): {dp}\n')
                continue
            buf.write(f'  Loading: {dp.name} ({dp.stat().st_size // 1024} KB)\n')
            examples = json.loads(dp.read_text())
            for ex in examples:
                if ex.get('input') and ex.get('output'):
                    all_examples.append(ex)

        before = len(all_examples)
        sft_set = [ex for ex in all_examples if ex.get('label') not in _SFT_EXCLUDE_LABELS]
        filtered_out = before - len(sft_set)

        buf.write(f'\n  Examples in JSON (before filter) : {before}\n')
        buf.write(f'  Removed by SFT label filter      : {filtered_out}\n')
        buf.write(f'  Remaining for SFT training        : {len(sft_set)}\n')

        # Count labels reaching SFT
        from collections import Counter
        sft_labels = Counter(ex.get('label') for ex in sft_set)
        buf.write('\n  Labels reaching SFT (must be ZERO loser/strong_loser/weak_loser):\n')
        label_rows = sorted(sft_labels.items(), key=lambda kv: -kv[1])
        buf.write(_table(label_rows, ['label', 'count']))

        # Assert
        loser_in_sft = sum(sft_labels.get(lab, 0) for lab in _SFT_EXCLUDE_LABELS)
        if loser_in_sft == 0:
            buf.write('  RESULT: PASS — zero loser-labelled examples reach SFT\n')
        else:
            buf.write(f'  RESULT: FAIL — {loser_in_sft} loser-labelled example(s) reach SFT '
                      f'(A2 filter regression)\n')
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')
        import traceback
        buf.write(traceback.format_exc())

    _read(buf,
        "Must be zero losers in SFT. counterfactual rows SHOULD be present "
        "(those teach HOLD from loser prompts). If any raw loser label reaches SFT, "
        "the A2 fix has regressed.")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Reasoning eyeball dump
    # ────────────────────────────────────────────────────────────────────────
    _section(buf, 'SECTION 7 — Reasoning eyeball dump (winner vs loser)')

    buf.write('\n  ── WINNER REASONINGS (20 random winner/strong_winner/weak_winner) ──\n\n')
    try:
        rows = conn.execute("""
            SELECT label, symbol, pnl_pct,
                   substr(ideal_output, instr(ideal_output, 'Reasoning:'), 300) AS reasoning
            FROM training_examples
            WHERE label IN ('winner','strong_winner','weak_winner')
            ORDER BY RANDOM() LIMIT 20
        """).fetchall()
        for r in rows:
            buf.write(f"  [{r['label']}] {r['symbol']}  pnl={r['pnl_pct']}\n")
            reasoning = (r['reasoning'] or '').strip()
            for line in textwrap.wrap(reasoning, width=66, initial_indent='    ',
                                      subsequent_indent='    '):
                buf.write(line + '\n')
            buf.write('\n')
        if not rows:
            buf.write('  (no winner rows found)\n')
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    buf.write('\n  ── LOSER REASONINGS (20 random loser/strong_loser/weak_loser) ──\n\n')
    try:
        rows = conn.execute("""
            SELECT label, symbol, pnl_pct,
                   substr(ideal_output, instr(ideal_output, 'Reasoning:'), 300) AS reasoning
            FROM training_examples
            WHERE label IN ('loser','strong_loser','weak_loser')
            ORDER BY RANDOM() LIMIT 20
        """).fetchall()
        for r in rows:
            buf.write(f"  [{r['label']}] {r['symbol']}  pnl={r['pnl_pct']}\n")
            reasoning = (r['reasoning'] or '').strip()
            for line in textwrap.wrap(reasoning, width=66, initial_indent='    ',
                                      subsequent_indent='    '):
                buf.write(line + '\n')
            buf.write('\n')
        if not rows:
            buf.write('  (no loser rows found)\n')
    except Exception as e:
        buf.write(f'  ERROR: {e}\n')

    _read(buf,
        "Read WINNER and LOSER reasonings cold — ignore pnl_pct. The question is "
        "'does this reasoning describe a real edge?' "
        "If winner and loser reasonings are indistinguishable boilerplate (same generic "
        "'RSI oversold, volume up' language on both), SFT is learning noise dressed as signal.")

    conn.close()

    # ────────────────────────────────────────────────────────────────────────
    # Prepend summary paragraph (we now have the numbers)
    # ────────────────────────────────────────────────────────────────────────
    total_str  = str(_summary_total) if _summary_total is not None else '?'
    pct_opt    = f'{_summary_pct_opt:.1%}' if _summary_pct_opt is not None else '?'
    missed_str = str(_summary_missed) if _summary_missed is not None else '?'

    summary = (
        f"SUMMARY: total_examples={total_str}  "
        f"pct_options={pct_opt}  "
        f"missed_opportunity_in_DB={missed_str}  "
        f"conf_win_rate_trend={_summary_conf}. "
        f"DB opened read-only (mode=ro); no writes were made."
    )

    report = textwrap.fill(summary, width=70) + '\n' + buf.getvalue()

    REPORT_PATH.write_text(report)
    print(report)
    print(f'\n[Report written to {REPORT_PATH}]')


if __name__ == '__main__':
    main()
