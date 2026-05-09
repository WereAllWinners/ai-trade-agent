#!/usr/bin/env python3
"""
News Fetcher — Polygon.io free news API (5 calls/min).

Fetches recent headlines per symbol, applies LLM-scored sentiment (with
keyword fallback), and formats a compact snippet for LLM prompt injection.

In-memory cache (1-hour TTL) prevents redundant API calls when the same
symbol is analyzed multiple times within a session.

Env flags:
  NEWS_LLM_SENTIMENT=false  — skip LLM scoring, use keyword fallback only
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

_API_KEY   = os.getenv('POLYGON_API_KEY', '')
_BASE_URL  = 'https://api.polygon.io/v2/reference/news'
_CALL_GAP  = 13.0   # seconds between calls: 60s / 5 calls + margin
_CACHE_TTL = 3600   # 1 hour in seconds

# Gate: set NEWS_LLM_SENTIMENT=false to fall back to keyword matching during
# GPU-contention hours without restarting the daemon.
NEWS_LLM_SENTIMENT: bool = os.getenv('NEWS_LLM_SENTIMENT', 'true').lower() != 'false'

_last_call_ts: float = 0.0
_cache: dict = {}      # symbol -> (fetched_ts, articles)


# ── Rate limiter ──────────────────────────────────────────────────────────────

def _rate_limit() -> None:
    global _last_call_ts
    wait = _CALL_GAP - (time.monotonic() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


# ── Keyword sentiment (fallback, no extra API needed) ─────────────────────────

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


# ── LLM sentiment scoring ─────────────────────────────────────────────────────

def score_sentiment_with_llm(title: str, description: str) -> dict:
    """Score news sentiment via the local Qwen model.

    Returns dict with keys: sentiment, magnitude, sentiment_reason.
    Falls back to keyword _sentiment() on any exception (import error,
    model timeout, parse failure, etc.).

    The LLM call is capped at 60 tokens to keep latency low — it fires
    only on cache miss (once per symbol per hour).
    """
    try:
        from model_inference_lora import get_trading_decision  # lazy import
        prompt = (
            f"Analyze this financial news headline and brief description.\n"
            f"Headline: {title}\n"
            f"Description: {description[:150]}\n\n"
            f"Respond in this exact format:\n"
            f"Sentiment: <bullish|bearish|neutral>\n"
            f"Magnitude: <low|medium|high>\n"
            f"Reason: <one phrase, max 10 words>"
        )
        response = get_trading_decision(prompt, max_new_tokens=60, temperature=0.3)
        sentiment = 'neutral'
        magnitude = 'medium'
        reason    = ''
        for line in response.splitlines():
            lower = line.lower().strip()
            if lower.startswith('sentiment:'):
                val = lower.split(':', 1)[1].strip()
                if val in ('bullish', 'bearish', 'neutral'):
                    sentiment = val
            elif lower.startswith('magnitude:'):
                val = lower.split(':', 1)[1].strip()
                if val in ('low', 'medium', 'high'):
                    magnitude = val
            elif lower.startswith('reason:'):
                reason = line.split(':', 1)[1].strip()[:80]
        return {'sentiment': sentiment, 'magnitude': magnitude, 'sentiment_reason': reason}
    except Exception as e:
        log.debug(f"LLM sentiment failed, using keyword fallback: {e}")
        text = title + ' ' + description
        return {'sentiment': _sentiment(text), 'magnitude': 'medium', 'sentiment_reason': ''}


# ── Core fetch ────────────────────────────────────────────────────────────────

def fetch_news(symbol: str, limit: int = 5, max_age_hours: int = 24) -> list:
    """
    Return up to `limit` recent articles for `symbol` from Polygon.
    Results are cached for 1 hour. Returns [] if API key absent or call fails.

    Each article dict has: title, description, published_utc, age,
    sentiment, magnitude, sentiment_reason.
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

        if NEWS_LLM_SENTIMENT:
            scored = score_sentiment_with_llm(title, desc)
        else:
            scored = {
                'sentiment':        _sentiment(title + ' ' + desc),
                'magnitude':        'medium',
                'sentiment_reason': '',
            }

        articles.append({
            'title':            title,
            'description':      desc,
            'published_utc':    pub,
            'age':              age_str,
            'sentiment':        scored['sentiment'],
            'magnitude':        scored['magnitude'],
            'sentiment_reason': scored['sentiment_reason'],
        })

    _cache[symbol] = (time.monotonic(), articles)
    return articles


# ── Prompt formatter ──────────────────────────────────────────────────────────

def format_for_prompt(symbol: str, max_articles: int = 3) -> str:
    """
    Fetch news and return a compact multi-line string ready for prompt injection.
    Returns '' if no recent news or API unavailable.

    Format example:
      Recent News (NVDA, last 24h):
        • Jensen Huang signals data center demand [BULLISH/HIGH — demand catalyst]  (3h ago)
    """
    articles = fetch_news(symbol)
    if not articles:
        return ''

    lines = [f"\nRecent News ({symbol}, last 24h):"]
    for a in articles[:max_articles]:
        if a['sentiment'] != 'neutral':
            tag = f"[{a['sentiment'].upper()}/{a['magnitude'].upper()}"
            if a.get('sentiment_reason'):
                tag += f" — {a['sentiment_reason']}"
            tag += ']'
            lines.append(f"  • {a['title']} {tag}  ({a['age']})")
        else:
            lines.append(f"  • {a['title']}  ({a['age']})")
    return '\n'.join(lines)
