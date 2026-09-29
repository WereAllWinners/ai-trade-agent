# Sprint 01 — V0 Verification Findings

Read-only verification probes run before any code changes, per Bugfix Sprint 01
(v2). Both suspected bugs were checked against the live codebase; findings below.

## V0-A: Does `simulate_stock_fill` bypass simulation in live mode?

**Verdict: NOT_A_BUG**

`scripts/paper_market_simulator.py:159-160` — `simulate_stock_fill` has the
identical early-return guard to `simulate_options_fill` (`:197-198`):

```python
if not self.paper:
    return SimulatedFill(price, qty, qty, False, True, 0.0, 'live mode — no simulation')
```

Both construction sites pass a non-hardcoded `paper=` flag derived from the
`PAPER_TRADING` env var:

- `scripts/agents/autonomous_agent.py:284` sets
  `_paper = os.getenv('PAPER_TRADING', 'true').lower() != 'false'`, used at
  `scripts/agents/autonomous_agent.py:329`:
  `self.paper_sim = PaperMarketSimulator(paper=_paper)`.
- `scripts/agents/options_agent.py:265` sets the identical `_paper` expression,
  used at `scripts/agents/options_agent.py:322`:
  `self.paper_sim = PaperMarketSimulator(paper=_paper)`.

Conclusion: the live bot does not skip real trades or price live bracket limits
off injected noise. Stage C4 (conditional live-passthrough fix) is **dropped** —
no code change required.

## V0-B: Is `decision['indicators']` ever attached before `execute_trade`?

**Verdict: CONFIRMED_BUG**

`execute_trade` (`scripts/agents/autonomous_agent.py`, ~line 666-675) reads:

```python
indicators = decision.get('indicators', {})
avg_volume     = indicators.get('avg_volume', 1_000_000)
session_volume = indicators.get('session_volume', avg_volume)
```

A repo-wide grep for `decision['indicators']` / `decision["indicators"]` returns
zero matches — the key is never set anywhere in `scripts/`. Every fill simulation
has run with the hardcoded 1,000,000-share default ADV since inception, regardless
of the real symbol's actual liquidity.

`run_trading_session` builds real `indicators` via `calculate_indicators(df)` and
only merges the single `current_price` field into `decision`
(`scripts/agents/autonomous_agent.py:1308`):

```python
decision['current_price'] = indicators['current_price']
```

**Fix (Stage C1.3):** immediately after line 1308, add
`decision['indicators'] = indicators`. Landed as part of Stage C1 (not run as its
own gated stage) since it's a one-line, low-risk change in the same file/session
as C1's other edits.

### Known residual gap (accepted, not fixed this sprint)

`_attempt_rotation`'s `sell_decision` dict (`scripts/agents/autonomous_agent.py:1013-1018`)
is constructed with only `decision`/`confidence`/`reasoning`/`current_price` — no
`indicators` key. This means the rotation-exit leg's fill simulation still runs on
the 1,000,000-share default ADV even after the V0-B fix lands, since C1.3 only
patches the main opportunity-analysis path at line 1308, not the rotation-sell
path.

Impact assessed as low: sell-side slippage on liquid exits is typically small, and
rotation sells target existing (already-vetted) holdings rather than newly
discovered illiquid candidates. Accepted as a known gap; a `# TODO(sprint02)`
comment is placed at the `sell_decision` construction site
(`scripts/agents/autonomous_agent.py:1013`) so it is not silent. Revisit in a
future sprint if rotation-exit slippage becomes a measurable issue in paper
trading.
