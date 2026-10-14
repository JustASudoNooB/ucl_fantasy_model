"""football-data.org (free tier includes the Champions League): kick-off times and results.
Set UFE_FOOTBALL_DATA_KEY. Updates data/processed/fixtures.csv in place."""
from __future__ import annotations

import os

import pandas as pd

from ..config import ROOT, sources
from ..names import ucl_team_id
from ..timeutil import UEFA_TZ
from .http import get

PROC = ROOT / "data" / "processed"


def sync_fixtures(fixtures: pd.DataFrame, season: int = 2026) -> pd.DataFrame | None:
    cfg = sources()["football_data_org"]
    key = os.environ.get(cfg["key_env"])
    if not key:
        return None
    r = get(f"{cfg['url']}/competitions/CL/matches", headers={"X-Auth-Token": key}, params={"season": season}, min_gap=6.5)
    if r.status_code != 200:
        return None
    fx = fixtures.drop(columns=[c for c in ["fixture_id"] if c in fixtures.columns]).copy()
    for m in r.json().get("matches", []):
        h = ucl_team_id(m["homeTeam"].get("name", "")) or ucl_team_id(m["homeTeam"].get("shortName", ""))
        a = ucl_team_id(m["awayTeam"].get("name", "")) or ucl_team_id(m["awayTeam"].get("shortName", ""))
        sel = (fx.home == h) & (fx.away == a)
        if not sel.any():
            continue
        ko = pd.Timestamp(m["utcDate"]).tz_convert(UEFA_TZ)
        fx.loc[sel, "date"] = ko.date().isoformat()
        fx.loc[sel, "kickoff_cet"] = ko.strftime("%H:%M")
        ft = (m.get("score") or {}).get("fullTime") or {}
        if m.get("status") == "FINISHED" and ft.get("home") is not None:
            # regularTime holds the 90-minute score when a knockout tie goes to extra time
            reg = (m.get("score") or {}).get("regularTime") or ft
            fx.loc[sel, ["home_goals", "away_goals", "status"]] = [reg["home"], reg["away"], "FT"]
    PROC.mkdir(parents=True, exist_ok=True)
    fx.to_csv(PROC / "fixtures.csv", index=False)
    return fx
