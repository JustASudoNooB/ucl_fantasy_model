"""Bookmaker prices -> data/processed/odds_md{N}.csv (used to blend market-implied goal rates).

Source: The Odds API (free tier; set UFE_ODDS_API_KEY). Median decimal price across bookmakers
for 1X2 and the 2.5 goals line. A manual CSV with the same columns works too:
fixture_id, home_odds, draw_odds, away_odds, over25_odds, under25_odds
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, sources
from ..names import ucl_team_id
from .http import get

PROC = ROOT / "data" / "processed"


def fetch(md: int, fixtures: pd.DataFrame) -> Path | None:
    cfg = sources()["the_odds_api"]
    key = os.environ.get(cfg["key_env"])
    if not key:
        return None
    r = get(cfg["url"], params=dict(apiKey=key, regions="uk,eu", markets="h2h,totals", oddsFormat="decimal"))
    if r.status_code != 200:
        return None
    rows = []
    fx = fixtures[fixtures.md == md]
    for ev in r.json():
        h, a = ucl_team_id(ev.get("home_team", "")), ucl_team_id(ev.get("away_team", ""))
        m = fx[(fx.home == h) & (fx.away == a)]
        if m.empty:
            continue
        hp, dp, ap, op, up = [], [], [], [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] == "h2h":
                    o = {x["name"]: x["price"] for x in mk["outcomes"]}
                    if ev["home_team"] in o and ev["away_team"] in o and "Draw" in o:
                        hp.append(o[ev["home_team"]])
                        ap.append(o[ev["away_team"]])
                        dp.append(o["Draw"])
                elif mk["key"] == "totals":
                    for x in mk["outcomes"]:
                        if abs(float(x.get("point", 0)) - 2.5) < 1e-9:
                            (op if x["name"] == "Over" else up).append(x["price"])
        if not hp:
            continue
        rows.append(dict(fixture_id=m.iloc[0].fixture_id, home_odds=np.median(hp), draw_odds=np.median(dp),
                         away_odds=np.median(ap), over25_odds=np.median(op) if op else np.nan,
                         under25_odds=np.median(up) if up else np.nan, n_books=len(hp)))
    if not rows:
        return None
    out = PROC / f"odds_md{md}.csv"
    PROC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
