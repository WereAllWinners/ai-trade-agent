#!/usr/bin/env python3
"""
News Fetcher — Polygon.io free news API (5 calls/min).

Fetches recent headlines per symbol, applies lightweight keyword sentiment,
and formats a compact snippet for LLM prompt injection.

In-memory cache (1-hour TTL) prevents redundant API calls when the same
symbol is analyzed multiple times within a session.
"""
import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_API_KEY      = os.getenv('POLYGON_API_KEY', '')
_BASE_URL     = 'https://api.polygon.io/v2/reference/news'
_CALL_GAP     = 13.0   # seconds between calls: 60s / 5 calls + margin
_CACHE_TTL    = 3600   # 1 hour in seconds

_last_call_ts: float = 0.0
_cache: dict = {}      # symbol -> (fetched_ts, articles)


# ── Rate limiter ──────────────────────────────────────────────────────────────

def _rate_limit() -> None:
    global _last_call_ts
    wait = _CALL_GAP - (time.monotonic() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


# ── Keyword sentiment (no extra API needed) ───────────────────────────────────

_BULLISH = {
    'beat', 'beats', 'exceeds', 'surpasses', 'upgrade', 'upgraded', 'outperform',
    'record', 'rally', 'surge', 'soar', 'growth', 'profit', 'raised', 'raise',
    'bullish', 'strong', 'positive', 'gain', 'buy', 'expansion', 'approved',
}
_BEARISH = {
    'miss', 'misses', 'missed', 'downgrade', 'downgraded', 'underperform',
    'decline', 'drop', 'fall', 'fell', 'loss', 'cut', 'cuts', 'bearish',
    'weak', 'negative', 'warning', 'concern', 'risk', 'probe', 'lawsuit',
    'fine', 'recall', 'investigation', 'fraud', 'breach', 'layoff', 'layoffs',
}

def _sentiment(text: str) -> str:
    words = set(text.lower().split())
    bull  = len(words & _BULLISH)
    bear  = len(words & _BEARISH)
    if bull > bear:   return 'bullish'
    if bear > bull:   return 'bearish'
    return 'neutral'


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_news(symbol: str, limit: int = 5, max_age_hours: int = 24) -> list:
    """
    Return up to `limit` recent articles for `symbol` from Polygon.
    Results are cached for 1 hour. Returns [] if API key absent or call fails.

    Each article dict has: title, description, published_utc, age, sentiment.
    """
    if not _API_KEY:
        log.debug("POLYGON_API_KEY not set — skipping news")
        return []

    # Cache hit
    if symbol in _cache:
        cached_ts, cached_articles = _cache[symbol]
        if time.monotonic() - cached_ts < _CACHE_TTL:
            return cached_articles

    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    ).strftime('%Y-%m-%dT%H:%M:%SZ')

    _rate_limit()
    try:
        resp = requests.get(
            _BASE_URL,
            params={
                'ticker':             symbol,
                'limit':              limit,
                'published_utc.gte':  cutoff,
                'order':              'desc',
                'sort':               'published_utc',
                'apiKey':             _API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get('results', [])
    except Exception as e:
        log.warning(f"News fetch failed for {symbol}: {e}")
        return []

    articles = []
    for a in raw:
        title = a.get('title', '')
        desc  = (a.get('description') or '')[:200]
        pub   = a.get('published_utc', '')
        try:
            pub_dt  = datetime.fromisoformat(pub.replace('Z', '+00:00'))
            age_h   = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
            age_str = f"{age_h:.0f}h ago"
        except Exception:
            age_str = ''

        articles.append({
            'title':       title,
            'description': desc,
            'published_utc': pub,
            'age':         age_str,
            'sentiment':   _sentiment(title + ' ' + desc),
        })

    _cache[symbol] = (time.monotonic(), articles)
    return articles


# ── Prompt formatter ──────────────────────────────────────────────────────────

def format_for_prompt(symbol: str, max_articles: int = 3) -> str:
    """
    Fetch news and return a compact multi-line string ready for prompt injection.
    Returns '' if no recent news or API unavailable.
    """
    articles = fetch_news(symbol)
    if not articles:
        return ''

    lines = [f"\nRecent News ({symbol}, last 24h):"]
    for a in articles[:max_articles]:
        tag = f" [{a['sentiment'].upper()}]" if a['sentiment'] != 'neutral' else ''
        lines.append(f"  • {a['title']}{tag}  ({a['age']})")
    return '\n'.join(lines)
