#!/usr/bin/env python3
"""
Health check HTTP server for the AI Trade Agent.

Endpoints
---------
  GET /health    Simple liveness check (existing behaviour).
                 Returns JSON with daemon heartbeat ages and overall status.
                 "status" field: "ok" | "degraded" | "down"

  GET /status    Rich JSON snapshot including:
                   - daemon heartbeats
                   - current allocation tier + position-size %
                   - sector headroom (from portfolio_constraints.json)
                   - recent preflight info (from allocation_config.json)
                   - daily P&L estimate (from heartbeat files)
                   - inference backend / model info

  GET /metrics   Prometheus text-format metrics for scraping by
                 Prometheus / Grafana.  Exposes counters and gauges
                 suitable for dashboarding without a push-based sidecars.

Usage:
  python3 scripts/health_server.py             # default port 8765
  python3 scripts/health_server.py --port 9000

UptimeRobot setup:
  Monitor type : HTTP(s)
  URL          : http://<your-server-ip>:8765/health
  Keyword      : "ok"   (alert if keyword NOT present)

Environment variables:
  HEALTH_PORT              override default port (8765)
  HEALTH_MAX_AGE_SECONDS   max acceptable heartbeat age (default: 300)
"""
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse

_SCRIPTS_DIR = Path(__file__).resolve().parent
_LOGS_DIR    = _SCRIPTS_DIR.parent / 'logs'

sys.path.insert(0, str(_SCRIPTS_DIR))

HEARTBEAT_FILES = {
    'stock':   _LOGS_DIR / 'heartbeat_stock.json',
    'options': _LOGS_DIR / 'heartbeat_options.json',
}

MAX_AGE_SECONDS = int(os.getenv('HEALTH_MAX_AGE_SECONDS', '300'))


# ---------------------------------------------------------------------------
# Heartbeat reader (unchanged from original)
# ---------------------------------------------------------------------------

def _read_heartbeat(name: str, path: Path) -> dict:
    """Return heartbeat status dict for one daemon."""
    if not path.exists():
        return {'daemon': name, 'status': 'missing', 'age_seconds': None, 'healthy': False}
    try:
        data = json.loads(path.read_text())
        ts_str = data.get('ts', '')
        ts = datetime.fromisoformat(ts_str.rstrip('Z')).replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        healthy = age <= MAX_AGE_SECONDS
        return {
            'daemon':          name,
            'status':          data.get('status', 'unknown'),
            'market_open':     data.get('market_open', False),
            'last_heartbeat':  ts_str,
            'age_seconds':     round(age, 1),
            'healthy':         healthy,
            'daily_pnl_pct':   data.get('daily_pnl_pct'),
            'session_trades':  data.get('session_trades'),
        }
    except Exception as e:
        return {'daemon': name, 'status': 'error', 'error': str(e), 'healthy': False}


def build_health_payload() -> tuple[dict, int]:
    """Return (payload_dict, http_status_code)."""
    daemons = {name: _read_heartbeat(name, path) for name, path in HEARTBEAT_FILES.items()}
    alerts_log = _LOGS_DIR / 'alerts.jsonl'
    recent_alerts: list = []

    if alerts_log.exists():
        try:
            lines = alerts_log.read_text().splitlines()
            for line in reversed(lines[-20:]):
                record = json.loads(line)
                if record.get('level') in ('WARNING', 'CRITICAL'):
                    recent_alerts.append(record)
                if len(recent_alerts) >= 5:
                    break
        except Exception:
            pass

    all_healthy = all(d['healthy'] for d in daemons.values())
    any_healthy = any(d['healthy'] for d in daemons.values())

    if all_healthy:
        overall, http_code = 'ok', 200
    elif any_healthy:
        overall, http_code = 'degraded', 200
    else:
        overall, http_code = 'down', 503

    payload = {
        'status':           overall,
        'checked_at':       datetime.now(timezone.utc).isoformat(),
        'max_age_seconds':  MAX_AGE_SECONDS,
        'daemons':          daemons,
        'recent_warnings':  recent_alerts,
    }
    return payload, http_code


# ---------------------------------------------------------------------------
# /status — rich snapshot
# ---------------------------------------------------------------------------

def _read_json_file(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return default


def build_status_payload() -> dict:
    """Aggregate all runtime artefact files into a single status snapshot."""
    daemons = {name: _read_heartbeat(name, p) for name, p in HEARTBEAT_FILES.items()}

    allocation_cfg  = _read_json_file(_LOGS_DIR / 'allocation_config.json',    {})
    constraints_log = _read_json_file(_LOGS_DIR / 'portfolio_constraints.json', [])

    # Most-recent 5 portfolio veto records
    recent_vetoes = (constraints_log or [])[-5:] if isinstance(constraints_log, list) else []

    # Inference backend info (written by trading_daemon heartbeat or read from env)
    inference_backend = os.getenv('INFERENCE_BACKEND', 'ollama')
    inference_model   = (
        os.getenv('VLLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct-AWQ')
        if inference_backend == 'vllm'
        else os.getenv('BASE_MODEL', 'Qwen/Qwen2.5-32B-Instruct') + '+LoRA'
    )

    # Last fine-tune timestamp (trading_daemon writes this)
    finetune_log = _read_json_file(_LOGS_DIR / 'finetune_log.json', {})
    last_finetune = finetune_log.get('completed_at') if finetune_log else None

    # Daily P&L rollup from heartbeats
    pnl_stock   = None
    pnl_options = None
    for name, hb in daemons.items():
        pnl = hb.get('daily_pnl_pct')
        if pnl is not None:
            if name == 'stock':
                pnl_stock = pnl
            else:
                pnl_options = pnl

    return {
        'checked_at':    datetime.now(timezone.utc).isoformat(),
        'daemons':       daemons,
        'inference': {
            'backend': inference_backend,
            'model':   inference_model,
        },
        'allocation': {
            'tier':              allocation_cfg.get('tier'),
            'position_size_pct': allocation_cfg.get('position_size_pct'),
            'computed_at':       allocation_cfg.get('computed_at'),
            'metrics':           allocation_cfg.get('current_metrics'),
        },
        'portfolio': {
            'recent_vetoes': recent_vetoes,
        },
        'performance': {
            'daily_pnl_pct_stock':   pnl_stock,
            'daily_pnl_pct_options': pnl_options,
        },
        'last_finetune': last_finetune,
    }


# ---------------------------------------------------------------------------
# /metrics — Prometheus text format
# ---------------------------------------------------------------------------

def build_prometheus_metrics() -> str:
    """
    Return a Prometheus-compatible text string with key gauges and counters.

    Scraped by `prometheus.yml` → visualised in Grafana.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    lines: list[str] = []

    def gauge(name: str, value, labels: str = '', help_text: str = ''):
        if help_text:
            lines.append(f'# HELP {name} {help_text}')
        lines.append(f'# TYPE {name} gauge')
        label_str = f'{{{labels}}}' if labels else ''
        lines.append(f'{name}{label_str} {value} {now_ms}')

    # ------------------------------------------------------------------
    # Daemon liveness
    # ------------------------------------------------------------------
    for name, path in HEARTBEAT_FILES.items():
        hb = _read_heartbeat(name, path)
        age = hb.get('age_seconds') if hb.get('age_seconds') is not None else -1
        healthy = 1 if hb.get('healthy') else 0
        gauge(
            'aitrade_daemon_heartbeat_age_seconds',
            age,
            labels=f'daemon="{name}"',
            help_text='Seconds since the last daemon heartbeat write.',
        )
        gauge(
            'aitrade_daemon_healthy',
            healthy,
            labels=f'daemon="{name}"',
            help_text='1 if daemon heartbeat is within max_age_seconds.',
        )
        pnl = hb.get('daily_pnl_pct')
        if pnl is not None:
            gauge(
                'aitrade_daily_pnl_pct',
                round(float(pnl), 6),
                labels=f'daemon="{name}"',
                help_text='Today\'s realised P&L as a fraction of starting equity.',
            )
        trades = hb.get('session_trades')
        if trades is not None:
            gauge(
                'aitrade_session_trades_total',
                int(trades),
                labels=f'daemon="{name}"',
                help_text='Number of orders submitted in the current session.',
            )

    # ------------------------------------------------------------------
    # Allocation tier
    # ------------------------------------------------------------------
    alloc = _read_json_file(_LOGS_DIR / 'allocation_config.json', {})
    if alloc:
        tier     = alloc.get('tier', 0)
        size_pct = alloc.get('position_size_pct', 0)
        metrics  = alloc.get('current_metrics', {})
        gauge('aitrade_allocation_tier', tier,
              help_text='Current allocation tier (1=cold-start, 2=promising, 3=proven).')
        gauge('aitrade_allocation_position_size_pct', size_pct,
              help_text='Maximum allowed position size as fraction of equity.')
        if metrics:
            gauge('aitrade_perf_sharpe',   metrics.get('sharpe', 0),
                  help_text='Annualised Sharpe ratio of recent closed trades.')
            gauge('aitrade_perf_max_dd',   metrics.get('max_dd', 0),
                  help_text='Maximum draw-down of recent closed trades.')
            gauge('aitrade_perf_win_rate', metrics.get('win_rate', 0),
                  help_text='Win rate of recent closed trades.')
            gauge('aitrade_perf_total_trades', metrics.get('total_trades', 0),
                  help_text='Number of trades used in allocation calculation.')

    # ------------------------------------------------------------------
    # Portfolio veto count (total entries in constraints log)
    # ------------------------------------------------------------------
    constraints = _read_json_file(_LOGS_DIR / 'portfolio_constraints.json', [])
    if isinstance(constraints, list):
        lines.append('# HELP aitrade_portfolio_vetoes_total Cumulative sector/correlation veto count.')
        lines.append('# TYPE aitrade_portfolio_vetoes_total counter')
        lines.append(f'aitrade_portfolio_vetoes_total {len(constraints)} {now_ms}')

    # ------------------------------------------------------------------
    # Broker-side risk reconciliation (C3 — options positions without a SELL)
    # -1 = API unavailable (unknown), 0 = all protected, N = N unprotected
    # ------------------------------------------------------------------
    reconcile = _read_json_file(_LOGS_DIR / 'reconcile_status.json', {})
    if reconcile:
        gauge(
            'aitrade_unprotected_positions',
            int(reconcile.get('unprotected_positions', -1)),
            help_text=(
                'Number of options positions without any open SELL order '
                '(GTC stop or limit). -1 = status unknown (API error).'
            ),
        )

    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress access log noise
        pass

    def do_GET(self):
        if self.path in ('/health', '/'):
            payload, http_code = build_health_payload()
            body = json.dumps(payload, indent=2).encode()
            self._respond(http_code, 'application/json', body)

        elif self.path == '/status':
            payload = build_status_payload()
            body = json.dumps(payload, indent=2).encode()
            self._respond(200, 'application/json', body)

        elif self.path == '/metrics':
            body = build_prometheus_metrics().encode()
            self._respond(200, 'text/plain; version=0.0.4', body)

        else:
            self.send_response(404)
            self.end_headers()

    def _respond(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Trade Agent health check server')
    parser.add_argument('--port', type=int,
                        default=int(os.getenv('HEALTH_PORT', '8765')),
                        help='Port to listen on (default: 8765)')
    args = parser.parse_args()

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(('0.0.0.0', args.port), HealthHandler)
    print(f"Health server listening on :")
    print(f"  http://0.0.0.0:{args.port}/health   — liveness")
    print(f"  http://0.0.0.0:{args.port}/status   — rich snapshot")
    print(f"  http://0.0.0.0:{args.port}/metrics  — Prometheus text")
    print(f"Daemon heartbeat max age: {MAX_AGE_SECONDS}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nHealth server stopped.")
