"""
Tests for Fix 5 — B3 golden set.

Covers:
  - build_golden_set.py: winner selection, hold selection, <5-hold refusal, write-once
  - eval_model.py run_golden_eval: pass, small drop, >15pt drop (alert + veto), determinism
  - Always-BUY model fails hold items
  - model_promoter.py golden gate integration
"""
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'training'))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_path(tmp_path):
    """Minimal SQLite DB with decisions, outcomes, and training_examples."""
    p = tmp_path / 'test.db'
    conn = sqlite3.connect(str(p))
    conn.executescript("""
        CREATE TABLE decisions (
            id INTEGER PRIMARY KEY,
            bot TEXT, symbol TEXT, decision TEXT,
            prompt TEXT, confidence REAL, executed INTEGER,
            order_id TEXT, timestamp TEXT
        );
        CREATE TABLE outcomes (
            id INTEGER PRIMARY KEY,
            symbol TEXT, bot TEXT, source TEXT,
            buy_order_id TEXT, sell_order_id TEXT,
            entry_timestamp TEXT, exit_timestamp TEXT,
            entry_price REAL, exit_price REAL,
            realized_pnl REAL, pnl_pct REAL, hold_hours REAL,
            entry_confidence REAL, entry_reasoning TEXT,
            won INTEGER, spy_return_pct REAL,
            excess_return_pct REAL, regime TEXT
        );
        CREATE TABLE training_examples (
            id INTEGER PRIMARY KEY,
            bot TEXT, source TEXT, symbol TEXT,
            prompt TEXT, ideal_output TEXT,
            label TEXT, confidence REAL, pnl_pct REAL,
            entry_date TEXT, session_id TEXT,
            prompt_hash TEXT UNIQUE, generated_at TEXT
        );
    """)
    conn.commit()
    conn.close()
    return p


def _insert_winner(db_path, decision_id=1, order_id='ord-win-1', decision='buy',
                   excess=0.08, regime='bull'):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
        (decision_id, 'stock', 'AAPL', decision,
         'Analyze AAPL: RSI 22 oversold. Should we BUY?', 0.80, 1, order_id,
         '2026-01-15T09:30:00'),
    )
    conn.execute(
        "INSERT INTO outcomes(symbol,bot,source,buy_order_id,excess_return_pct,regime) "
        "VALUES (?,?,?,?,?,?)",
        ('AAPL', 'stock', 'paper', order_id, excess, regime),
    )
    conn.commit()
    conn.close()


def _insert_hold(db_path, row_id=1, label='correct_hold', bot='stock'):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO training_examples "
        "(id,bot,source,symbol,prompt,ideal_output,label,confidence,pnl_pct,"
        "entry_date,session_id,prompt_hash,generated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (row_id, bot, 'paper', 'MSFT',
         f'Analyze MSFT for session {row_id}: RSI 50 neutral. Buy/Sell/Hold?',
         'Decision: HOLD\nConfidence: 0.65\nReasoning: Neutral setup.',
         label, 0.65, 0.005, '2026-01-10', f'sess-{row_id}',
         f'hash-{row_id}', '2026-01-10T18:00:00'),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# build_golden_set.py — selection and guards
# ---------------------------------------------------------------------------

class TestBuildGoldenSet:

    def test_selects_strong_winners(self, db_path):
        """excess_return_pct >= 0.05 → included; < 0.05 → excluded."""
        from build_golden_set import _load_winners
        _insert_winner(db_path, decision_id=1, order_id='w1', excess=0.08)
        _insert_winner(db_path, decision_id=2, order_id='w2', excess=0.03)
        winners = _load_winners(db_path)
        assert len(winners) == 1
        assert winners[0]['excess_return_pct'] == 0.08

    def test_excludes_non_buy_from_winners(self, db_path):
        """HOLDs are not in ('buy','buy_call','buy_put') — excluded from winners."""
        from build_golden_set import _load_winners
        _insert_winner(db_path, decision_id=1, order_id='w1', decision='hold', excess=0.10)
        assert _load_winners(db_path) == []

    def test_selects_correct_holds(self, db_path):
        """label='correct_hold' rows are included; other labels excluded."""
        from build_golden_set import _load_holds
        for i in range(1, 6):
            _insert_hold(db_path, row_id=i, label='correct_hold')
        _insert_hold(db_path, row_id=6, label='missed_opportunity')
        holds = _load_holds(db_path)
        assert len(holds) == 5
        assert all(h['expected'] == 'hold' for h in holds)

    def test_refuses_fewer_than_5_holds(self, tmp_path, db_path):
        """build_golden_set returns 1 when fewer than 5 holds found."""
        from build_golden_set import build_golden_set
        _insert_winner(db_path, decision_id=1, order_id='w1', excess=0.08)
        for i in range(1, 4):  # only 3 holds
            _insert_hold(db_path, row_id=i)
        out = tmp_path / 'golden_set.jsonl'
        assert build_golden_set(db_path, out, force=False) == 1
        assert not out.exists()

    def test_write_once_protection(self, tmp_path, db_path):
        """Second call without --force returns 1 and does NOT overwrite."""
        from build_golden_set import build_golden_set
        _insert_winner(db_path, decision_id=1, order_id='w1', excess=0.08)
        for i in range(1, 8):
            _insert_hold(db_path, row_id=i)
        out = tmp_path / 'golden_set.jsonl'
        assert build_golden_set(db_path, out) == 0
        original = out.read_text()
        assert build_golden_set(db_path, out, force=False) == 1
        assert out.read_text() == original

    def test_force_overwrites(self, tmp_path, db_path):
        """--force allows overwriting an existing file."""
        from build_golden_set import build_golden_set
        _insert_winner(db_path, decision_id=1, order_id='w1', excess=0.08)
        for i in range(1, 8):
            _insert_hold(db_path, row_id=i)
        out = tmp_path / 'golden_set.jsonl'
        assert build_golden_set(db_path, out) == 0
        assert build_golden_set(db_path, out, force=True) == 0

    def test_output_contains_required_keys(self, tmp_path, db_path):
        """Each line of output is valid JSON with prompt, expected, bot keys."""
        from build_golden_set import build_golden_set
        _insert_winner(db_path, decision_id=1, order_id='w1', excess=0.08)
        for i in range(1, 6):
            _insert_hold(db_path, row_id=i)
        out = tmp_path / 'golden_set.jsonl'
        assert build_golden_set(db_path, out) == 0
        items = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        for item in items:
            assert 'prompt'   in item
            assert 'expected' in item
            assert 'bot'      in item


# ---------------------------------------------------------------------------
# eval_model.py run_golden_eval — helpers
# ---------------------------------------------------------------------------

def _golden_path(tmp_path, items):
    p = tmp_path / 'golden_set.jsonl'
    p.write_text('\n'.join(json.dumps(i) for i in items) + '\n')
    return p


def _baseline_path(tmp_path, rate):
    p = tmp_path / 'golden_baseline.json'
    p.write_text(json.dumps({'pass_rate': rate, 'num_items': 4, 'built_at': '2026-01-01T00:00:00'}))
    return p


def _patch_model(responses):
    """Context manager: patch model_inference_lora.get_trading_decision with a queue."""
    queue = list(responses)

    def _fake(prompt, temperature=0.7, **kw):
        return queue.pop(0)

    import model_inference_lora as _mil
    original = _mil.__dict__.get('get_trading_decision')
    _mil.get_trading_decision = _fake
    return _mil, original


def _unpatch_model(mil, original):
    if original is not None:
        mil.get_trading_decision = original
    else:
        mil.__dict__.pop('get_trading_decision', None)


# ---------------------------------------------------------------------------
# eval_model.py run_golden_eval — scoring and veto logic
# ---------------------------------------------------------------------------

class TestRunGoldenEval:

    def test_all_correct_no_veto(self, tmp_path):
        """All correct → golden_veto=False, pass_rate=1.0."""
        from eval_model import run_golden_eval
        items = [
            {'prompt': 'Buy AAPL?', 'expected': 'buy',  'bot': 'stock'},
            {'prompt': 'Hold MSFT?', 'expected': 'hold', 'bot': 'stock'},
        ]
        gp = _golden_path(tmp_path, items)
        bp = _baseline_path(tmp_path, 1.0)
        responses = [
            'Decision: BUY\nConfidence: 0.80\nReasoning: r',
            'Decision: HOLD\nConfidence: 0.65\nReasoning: r',
        ]
        mil, orig = _patch_model(responses)
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)
        assert result['pass_rate'] == 1.0
        assert result['golden_veto'] is False

    def test_5pt_drop_no_veto(self, tmp_path):
        """5pt drop (< 15pt threshold) → golden_veto=False."""
        from eval_model import run_golden_eval
        items = [
            {'prompt': 'Buy?',  'expected': 'buy',  'bot': 'stock'},
            {'prompt': 'Hold?', 'expected': 'hold', 'bot': 'stock'},
            {'prompt': 'Sell?', 'expected': 'sell', 'bot': 'stock'},
            {'prompt': 'Hold?', 'expected': 'hold', 'bot': 'stock'},
        ]
        gp = _golden_path(tmp_path, items)
        bp = _baseline_path(tmp_path, 0.80)  # 80% baseline; 3/4=75% → 5pt drop

        responses = [
            'Decision: BUY\nConfidence: 0.80\nReasoning: r',
            'Decision: HOLD\nConfidence: 0.65\nReasoning: r',
            'Decision: SELL\nConfidence: 0.70\nReasoning: r',
            'Decision: BUY\nConfidence: 0.70\nReasoning: r',  # wrong (expected hold)
        ]
        mil, orig = _patch_model(responses)
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)
        assert result['golden_veto'] is False

    def test_16pt_drop_sets_veto(self, tmp_path):
        """16pt drop → golden_veto=True (alert tested separately via integration)."""
        from eval_model import run_golden_eval
        items = [
            {'prompt': 'Buy?',  'expected': 'buy',  'bot': 'stock'},
            {'prompt': 'Hold?', 'expected': 'hold', 'bot': 'stock'},
            {'prompt': 'Sell?', 'expected': 'sell', 'bot': 'stock'},
            {'prompt': 'Buy?',  'expected': 'buy',  'bot': 'stock'},
        ]
        gp = _golden_path(tmp_path, items)
        bp = _baseline_path(tmp_path, 1.0)  # 100% baseline; 3/4=75% → 25pt drop

        responses = [
            'Decision: BUY\nConfidence: 0.80\nReasoning: r',
            'Decision: HOLD\nConfidence: 0.65\nReasoning: r',
            'Decision: HOLD\nConfidence: 0.60\nReasoning: r',  # wrong (expected sell)
            'Decision: BUY\nConfidence: 0.70\nReasoning: r',
        ]
        mil, orig = _patch_model(responses)
        mock_send = MagicMock()
        try:
            with patch('alerts.send_alert', mock_send):
                result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)
        assert result['golden_veto'] is True
        assert result['pass_rate'] == pytest.approx(0.75)

    def test_16pt_drop_sends_critical_alert(self, tmp_path):
        """16pt drop calls send_alert with AlertLevel.CRITICAL."""
        from eval_model import run_golden_eval
        from alerts import AlertLevel
        items = [
            {'prompt': 'Buy?', 'expected': 'buy',  'bot': 'stock'},
            {'prompt': 'Hold?', 'expected': 'hold', 'bot': 'stock'},
        ]
        gp = _golden_path(tmp_path, items)
        bp = _baseline_path(tmp_path, 1.0)  # 100% baseline; 1/2=50% → 50pt drop

        responses = [
            'Decision: BUY\nConfidence: 0.80\nReasoning: r',
            'Decision: BUY\nConfidence: 0.70\nReasoning: r',  # wrong (expected hold)
        ]
        mil, orig = _patch_model(responses)
        mock_send = MagicMock()
        try:
            with patch('alerts.send_alert', mock_send):
                run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        assert call_args[0] == AlertLevel.CRITICAL
        assert call_args[1] == 'golden_gate_regression'

    def test_always_buy_fails_hold_items(self, tmp_path):
        """Always-BUY model fails hold-expected items → golden_veto=True vs 100% baseline."""
        from eval_model import run_golden_eval
        items = [
            {'prompt': 'Buy AAPL?', 'expected': 'buy',  'bot': 'stock'},
            {'prompt': 'Hold MSFT?', 'expected': 'hold', 'bot': 'stock'},
            {'prompt': 'Hold JPM?',  'expected': 'hold', 'bot': 'stock'},
            {'prompt': 'Buy NVDA?',  'expected': 'buy',  'bot': 'stock'},
        ]
        gp = _golden_path(tmp_path, items)
        bp = _baseline_path(tmp_path, 1.0)

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: BUY\nConfidence: 0.80\nReasoning: always buy'
        try:
            with patch('alerts.send_alert'):
                result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['pass_rate'] == 0.5
        assert result['golden_veto'] is True
        hold_items = [r for r in result['per_item'] if r['expected'] == 'hold']
        assert all(not r['passed'] for r in hold_items)

    def test_determinism_identical_pass_rates(self, tmp_path):
        """Two runs on the same deterministic model produce identical pass_rates."""
        from eval_model import run_golden_eval
        items = [
            {'prompt': 'Analyze AAPL?', 'expected': 'buy',  'bot': 'stock'},
            {'prompt': 'Analyze MSFT?', 'expected': 'hold', 'bot': 'stock'},
            {'prompt': 'Analyze TSLA?', 'expected': 'sell', 'bot': 'stock'},
        ]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'  # absent — no comparison

        def _deterministic(prompt, temperature=0.7, **kw):
            if 'aapl' in prompt.lower():
                return 'Decision: BUY\nConfidence: 0.80\nReasoning: r'
            if 'msft' in prompt.lower():
                return 'Decision: HOLD\nConfidence: 0.65\nReasoning: r'
            return 'Decision: SELL\nConfidence: 0.70\nReasoning: r'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = _deterministic
        try:
            r1 = run_golden_eval(None, gp, bp)
            r2 = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert r1['pass_rate'] == r2['pass_rate']
        assert [i['passed'] for i in r1['per_item']] == [i['passed'] for i in r2['per_item']]

    def test_buy_call_same_direction_class_as_buy(self, tmp_path):
        """expected=buy_call is bullish; model returning BUY (also bullish) → PASS."""
        from eval_model import run_golden_eval
        items = [{'prompt': 'Options call on NVDA?', 'expected': 'buy_call', 'bot': 'options'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: BUY\nConfidence: 0.75\nReasoning: bullish'
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['per_item'][0]['passed'] is True

    # --- Direction-class scoring tests (Session A) ---------------------------

    def test_buy_call_response_matches_buy_call_expected(self, tmp_path):
        """expected=buy_call, model emits Decision: BUY_CALL (uppercase) → PASS."""
        from eval_model import run_golden_eval
        items = [{'prompt': 'Bullish options?', 'expected': 'buy_call', 'bot': 'options'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: BUY_CALL\nConfidence: 0.80\nReasoning: bullish setup'
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['per_item'][0]['passed'] is True

    def test_buy_put_response_matches_buy_put_expected(self, tmp_path):
        """expected=buy_put, model emits Decision: BUY_PUT → PASS."""
        from eval_model import run_golden_eval
        items = [{'prompt': 'Bearish options?', 'expected': 'buy_put', 'bot': 'options'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: BUY_PUT\nConfidence: 0.78\nReasoning: bearish setup'
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['per_item'][0]['passed'] is True

    def test_buy_call_vs_buy_put_is_wrong_direction(self, tmp_path):
        """REGRESSION: expected=buy_put (bearish) but model says BUY_CALL (bullish) → FAIL.

        This is the direction-collapse case the old normalisation silently passed.
        """
        from eval_model import run_golden_eval
        items = [{'prompt': 'Bearish setup?', 'expected': 'buy_put', 'bot': 'options'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: BUY_CALL\nConfidence: 0.80\nReasoning: bullish'
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['per_item'][0]['passed'] is False

    def test_buy_call_expected_with_plain_buy_response_passes(self, tmp_path):
        """expected=buy_call (bullish), model returns BUY (also bullish) → PASS."""
        from eval_model import run_golden_eval
        items = [{'prompt': 'Options call?', 'expected': 'buy_call', 'bot': 'options'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: BUY\nConfidence: 0.75\nReasoning: bullish'
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['per_item'][0]['passed'] is True

    def test_no_parseable_decision_is_false_no_exception(self, tmp_path):
        """Unparseable model output → passed=False, no KeyError or exception."""
        from eval_model import run_golden_eval
        items = [{'prompt': 'Buy AAPL?', 'expected': 'buy', 'bot': 'stock'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'no_baseline.json'

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Sorry, I cannot process this request.'
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        item = result['per_item'][0]
        assert item['passed'] is False
        assert item.get('error') is None  # no exception stored

    def test_extract_decision_lowercases_uppercase_options_input(self):
        """_extract_decision handles the model's uppercase BUY_CALL / BUY_PUT output."""
        from eval_model import _extract_decision
        assert _extract_decision('Decision: BUY_CALL\nConfidence: 0.80\nReasoning: r') == 'buy_call'
        assert _extract_decision('Decision: BUY_PUT\nConfidence: 0.78\nReasoning: r')  == 'buy_put'

    def test_no_baseline_no_veto(self, tmp_path):
        """When baseline is absent, golden_veto is always False regardless of pass_rate."""
        from eval_model import run_golden_eval
        items = [{'prompt': 'Buy?', 'expected': 'buy', 'bot': 'stock'}]
        gp = _golden_path(tmp_path, items)
        bp = tmp_path / 'absent_baseline.json'  # does not exist

        import model_inference_lora as mil
        orig = mil.__dict__.get('get_trading_decision')
        mil.get_trading_decision = lambda p, temperature=0.7, **kw: \
            'Decision: SELL\nConfidence: 0.70\nReasoning: r'  # wrong answer
        try:
            result = run_golden_eval(None, gp, bp)
        finally:
            _unpatch_model(mil, orig)

        assert result['golden_veto'] is False  # no baseline → no veto possible


# ---------------------------------------------------------------------------
# model_promoter.py — golden gate integration
# ---------------------------------------------------------------------------

class TestGoldenGateInPromoter:

    def _make_smoke_result(self, pass_rate=0.9, grade='A'):
        return {
            'pass_rate': pass_rate, 'directional_acc': pass_rate,
            'format_score': 0.95, 'grade': grade,
            'num_scenarios': 8, 'scenarios': [],
        }

    def test_no_golden_veto_proceeds_to_smoke_gate(self):
        """golden_veto=None (gate skipped) allows smoke test to decide."""
        from model_promoter import promote

        candidate_smoke = self._make_smoke_result(pass_rate=0.9)
        current_smoke   = self._make_smoke_result(pass_rate=0.7)

        def _fake_run_eval(adapter_path, with_golden=False):
            return candidate_smoke, None  # golden gate skipped

        promoted_result = []
        with patch('model_promoter._run_eval', side_effect=_fake_run_eval), \
             patch('model_promoter._current_production_adapter', return_value='/fake/current'), \
             patch('model_promoter._write_promotion_record'), \
             patch('model_promoter._update_latest_symlink'), \
             patch('model_promoter._update_env'), \
             patch('model_promoter._update_merged_symlink'), \
             patch('model_promoter._restart_services'), \
             patch('model_promoter.Path') as MockPath:
            # candidate_path != current_path
            MockPath.return_value.resolve.return_value = Path('/fake/candidate')
            result = promote('/fake/candidate')
            promoted_result.append(result)

        # Smoke test: candidate 90% >= current 70% → should promote
        assert promoted_result[0] is True

    def test_golden_veto_true_returns_false(self):
        """golden_veto=True forces promote() to return False."""
        from model_promoter import promote

        candidate_smoke = self._make_smoke_result(pass_rate=0.9)
        current_smoke   = self._make_smoke_result(pass_rate=0.7)

        def _fake_run_eval(adapter_path, with_golden=False):
            if with_golden:
                return candidate_smoke, True   # VETO
            return current_smoke, None

        with patch('model_promoter._run_eval', side_effect=_fake_run_eval), \
             patch('model_promoter._current_production_adapter', return_value='/fake/current'), \
             patch('model_promoter._write_promotion_record'), \
             patch('model_promoter.Path') as MockPath:
            MockPath.return_value.resolve.return_value = Path('/fake/candidate')
            result = promote('/fake/candidate')

        assert result is False

    def test_golden_veto_writes_promotion_record_not_promoted(self):
        """Golden veto calls _write_promotion_record with promoted=False."""
        from model_promoter import promote

        smoke = self._make_smoke_result()

        def _fake_run_eval(adapter_path, with_golden=False):
            return smoke, True if with_golden else None

        mock_write = MagicMock()
        with patch('model_promoter._run_eval', side_effect=_fake_run_eval), \
             patch('model_promoter._current_production_adapter', return_value='/fake/current'), \
             patch('model_promoter._write_promotion_record', mock_write), \
             patch('model_promoter.Path') as MockPath:
            MockPath.return_value.resolve.return_value = Path('/fake/candidate')
            promote('/fake/candidate')

        mock_write.assert_called_once()
        _, kwargs = mock_write.call_args
        assert kwargs['promoted'] is False
