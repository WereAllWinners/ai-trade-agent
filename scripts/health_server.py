#!/usr/bin/env python3
"""
Health check HTTP server for the AI Trade Agent.

Serves a single endpoint:
  GET /health  →  JSON with daemon liveness, last heartbeat ages, and
                  a top-level "status" field ("ok" | "degraded" | "down")

Designed to be polled by UptimeRobot (or any HTTP monitor) every 1-5
minutes.  UptimeRobot should alert if the endpoint returns non-200 or
if the response body contains "status": "down".

Usage:
  python3 scripts/health_server.py             # default port 8765
  python3 scripts/health_server.py --port 9000

UptimeRobot setup:
  Monitor type : HTTP(s)
  URL          : http://<your-server-ip>:8765/health
  Keyword      : "ok"   (alert if keyword NOT present → degraded/down)

Systemd service: services/ai-health-server.service

Environment variable (optional):
  HEALTH_PORT   override the default port (8765)
  HEALTH_MAX_AGE_SECONDS  max acceptable heartbeat age (default: 300 = 5 min)
"""
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse

_SCRIPTS_DIR = Path(__file__).resolve().parent
_LOGS_DIR = _SCRIPTS_DIR.parent / 'logs'

HEARTBEAT_FILES = {
    'stock':   _LOGS_DIR / 'heartbeat_stock.json',
    'options': _LOGS_DIR / 'heartbeat_options.json',
}

MAX_AGE_SECONDS = int(os.getenv('HEALTH_MAX_AGE_SECONDS', '300'))


# ---------------------------------------------------------------------------
# Heartbeat reader
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
            'daemon': name,
            'status': data.get('status', 'unknown'),
            'market_open': data.get('market_open', False),
            'last_heartbeat': ts_str,
            'age_seconds': round(age, 1),
            'healthy': healthy,
        }
    except Exception as e:
        return {'daemon': name, 'status': 'error', 'error': str(e), 'healthy': False}


def build_health_payload() -> tuple[dict, int]:
    """Return (payload_dict, http_status_code)."""
    daemons = {name: _read_heartbeat(name, path) for name, path in HEARTBEAT_FILES.items()}
    alerts_log = _LOGS_DIR / 'alerts.jsonl'
    recent_alerts = []

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
        overall = 'ok'
        http_code = 200
    elif any_healthy:
        overall = 'degraded'
        http_code = 200  # still 200 so UptimeRobot uses keyword check
    else:
        overall = 'down'
        http_code = 503

    payload = {
        'status': overall,
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'max_age_seconds': MAX_AGE_SECONDS,
        'daemons': daemons,
        'recent_warnings': recent_alerts,
    }
    return payload, http_code


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # suppress access log noise
        pass

    def do_GET(self):
        if self.path not in ('/health', '/'):
            self.send_response(404)
            self.end_headers()
            return

        payload, http_code = build_health_payload()
        body = json.dumps(payload, indent=2).encode()

        self.send_response(http_code)
        self.send_header('Content-Type', 'application/json')
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

    server = HTTPServer(('0.0.0.0', args.port), HealthHandler)
    print(f"Health server listening on http://0.0.0.0:{args.port}/health")
    print(f"Daemon heartbeat max age: {MAX_AGE_SECONDS}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nHealth server stopped.")
