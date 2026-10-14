"""`ufe probe`: check every free data source from this machine and write a report."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from ..config import ROOT, sources
from .http import get

CHECKS = [
    ("openfootball leagues", lambda s: s["openfootball_leagues"]["url"].format(season="2026-27", code="en.1"), None),
    ("openfootball cups", lambda s: s["openfootball_cups"]["url"].format(season="2025-26", comp="cl"), None),
    ("FPL history mirror", lambda s: s["fpl_history"]["url"].format(season="2026-27", file="teams.csv"), None),
    ("ClubElo", lambda s: s["clubelo"]["url"].format(date=datetime.now().date().isoformat()), None),
    ("football-data.org", lambda s: s["football_data_org"]["url"] + "/competitions/CL", "UFE_FOOTBALL_DATA_KEY"),
    ("The Odds API", lambda s: s["the_odds_api"]["url"], "UFE_ODDS_API_KEY"),
    ("Understat", lambda s: "https://understat.com/league/EPL/2026", None),
    ("BBC football RSS", lambda s: s["news_rss"]["feeds"][0], None),
    ("FotMob", lambda s: "https://www.fotmob.com/api/leagues?id=42", None),
]


def run() -> str:
    src = sources()
    lines = [f"# Data source probe — {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "| source | status | ms | note |", "|---|---|---|---|"]
    for name, url_fn, key_env in CHECKS:
        url = url_fn(src)
        headers, params, note = {}, {}, ""
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                lines.append(f"| {name} | skipped | – | set {key_env} (free key) |")
                continue
            if "football-data" in url:
                headers["X-Auth-Token"] = key
            else:
                params["apiKey"] = key
        t = time.time()
        try:
            r = get(url, headers=headers, params=params, retries=1, timeout=15)
            ok = r.status_code == 200
            status = "ok" if ok else f"HTTP {r.status_code}"
        except Exception as exc:          # network failures are expected for some unofficial sources
            status, note = "error", type(exc).__name__
        lines.append(f"| {name} | {status} | {int((time.time() - t) * 1000)} | {note} |")
    from . import ucl_fantasy
    url, recs = ucl_fantasy.discover(1)
    lines.append(f"| UCL Fantasy feed | {'ok' if recs else 'not found'} | – | {url or 'set players_url in config/sources.yaml'} |")
    report = "\n".join(lines)
    out = ROOT / "data" / "processed" / "probe_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return report
