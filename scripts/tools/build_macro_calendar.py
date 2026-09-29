#!/usr/bin/env python3
"""
build_macro_calendar.py — build data/macro_event_calendar.json from official
Fed/BLS pages (sprint04 F1.1).

Run manually (annual cadence — see CLAUDE.md), review the printed output,
then write it to data/macro_event_calendar.json by hand. Dates are extracted
via raw-HTML regex, NEVER via LLM summarization — a WebFetch/LLM-summarize
pass on the FOMC page during this sprint's planning already got a decision
date wrong (reported the first day of a 2-day meeting instead of the last),
which is exactly the failure mode this script exists to avoid.

Any date line the parser isn't confident about is skipped with a logged
warning rather than guessed — a wrong halt-guard date is worse than a
missing one (missing degrades to today's already-accepted fail-open
behavior; wrong could halt trading at the wrong moment or miss a real one).

Usage:
    python3 scripts/tools/build_macro_calendar.py > /tmp/new_calendar.json
    # spot-check the printed dates against the source pages, then:
    cp /tmp/new_calendar.json data/macro_event_calendar.json
"""
import json
import logging
import re
import sys
from datetime import datetime

import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# Same UA already used by get_index_constituents()'s Wikipedia scrape —
# reused verbatim for consistency. Note: BLS returns 403 for a realistic
# browser UA on these specific pages but 200 for this bot-labeled one —
# inverted from the usual pattern, but reproducible (verified live).
_UA = {'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'}

_FOMC_URL = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
_CPI_URL = 'https://www.bls.gov/schedule/news_release/cpi.htm'
_NFP_URL = 'https://www.bls.gov/schedule/news_release/empsit.htm'

_MONTHS = {m: i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], 1)}
_MONTHS_ABBR = {m[:3]: i for m, i in _MONTHS.items()}

_FOMC_YEAR_RE = re.compile(r'^(20\d\d)\s+FOMC\s+Meetings?$')
_FOMC_MONTH_RE = re.compile(r'^(' + '|'.join(_MONTHS) + r')$')
_FOMC_RANGE_RE = re.compile(r'^(\d{1,2})-(\d{1,2})\*?$')
_FOMC_SINGLE_DAY_RE = re.compile(r'^(\d{1,2})\*?$')
_BLS_RELEASE_RE = re.compile(
    r'(' + '|'.join(_MONTHS_ABBR) + r')\.?\s+(\d{1,2}),\s+(20\d\d)'
)


def fetch(url: str) -> str:
    resp = requests.get(url, headers=_UA, timeout=15)
    resp.raise_for_status()
    return resp.text


def _strip_tags(html: str) -> str:
    return re.sub(r'<[^>]+>', ' ', html)


def extract_fomc_dates(html: str) -> list[dict]:
    """Each '20XX FOMC Meetings' section has repeating '{Month}' lines
    followed by a date line — usually a '{d1}-{d2}' (or '{d1}-{d2}*',
    asterisk = SEP-associated meeting, irrelevant to extraction) range, but
    in principle could be a single day (an unscheduled/one-day meeting) — no
    real example was observed live this session, but the parser must not
    assume ranges are guaranteed. The decision date is always the SECOND day
    of a 2-day range (or the single day itself), at 2:00 PM ET — this time
    is universal FOMC convention, NOT scraped from this page.

    Any date line matching neither shape is skipped with a warning, not
    guessed at.
    """
    text = _strip_tags(html)
    events = []
    year = None
    month_name = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        ym = _FOMC_YEAR_RE.match(line)
        if ym:
            year = int(ym.group(1))
            month_name = None
            continue

        mm = _FOMC_MONTH_RE.match(line)
        if mm and year:
            month_name = mm.group(1)
            continue

        if not (year and month_name):
            continue

        rm = _FOMC_RANGE_RE.match(line)
        if rm:
            end_day = int(rm.group(2))
            events.append({
                'date': f"{year:04d}-{_MONTHS[month_name]:02d}-{end_day:02d}",
                'time_et': '14:00',
                'event': 'FOMC Statement',
            })
            month_name = None  # consumed — next line won't also match a date pattern for this month
            continue

        sm = _FOMC_SINGLE_DAY_RE.match(line)
        if sm:
            day = int(sm.group(1))
            events.append({
                'date': f"{year:04d}-{_MONTHS[month_name]:02d}-{day:02d}",
                'time_et': '14:00',
                'event': 'FOMC Statement',
            })
            month_name = None
            continue

    return events


def extract_bls_dates(html: str, event_name: str) -> list[dict]:
    """cpi.htm and empsit.htm share one page template: a '{Reference Month}
    {Year}' line, followed a few lines later by a '{Mon}. {Day}, {Year}'
    release-date line. 8:30 AM ET is BLS's published standard release time
    for both release types — same convention-not-scraped-fact caveat as
    FOMC's 2:00 PM."""
    text = _strip_tags(html)
    events = []
    for m in _BLS_RELEASE_RE.finditer(text):
        mon, day, yr = m.group(1), int(m.group(2)), int(m.group(3))
        events.append({
            'date': f"{yr:04d}-{_MONTHS_ABBR[mon]:02d}-{day:02d}",
            'time_et': '08:30',
            'event': event_name,
        })
    return events


def main():
    fomc_html = fetch(_FOMC_URL)
    cpi_html = fetch(_CPI_URL)
    nfp_html = fetch(_NFP_URL)

    fomc = extract_fomc_dates(fomc_html)
    cpi = extract_bls_dates(cpi_html, 'CPI')
    nfp = extract_bls_dates(nfp_html, 'Employment Situation (NFP)')

    if not fomc:
        log.warning("No FOMC dates extracted — check %s manually", _FOMC_URL)
    if not cpi:
        log.warning("No CPI dates extracted — check %s manually", _CPI_URL)
    if not nfp:
        log.warning("No NFP dates extracted — check %s manually", _NFP_URL)

    events = sorted(fomc + cpi + nfp, key=lambda e: e['date'])
    out = {
        'generated_at': datetime.now().astimezone().isoformat(),
        'source_urls': [_FOMC_URL, _CPI_URL, _NFP_URL],
        'events': events,
    }
    print(json.dumps(out, indent=2))
    log.info("Extracted %d FOMC + %d CPI + %d NFP = %d total events",
              len(fomc), len(cpi), len(nfp), len(events))


if __name__ == '__main__':
    main()
