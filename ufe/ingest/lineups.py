"""Confirmed line-ups (~60 minutes before kick-off) -> data/processed/lineups_md{N}.csv.

Primary: FotMob's public match JSON (unofficial; the endpoint and headers change from time to
time, so failures are expected and handled). Fallback: paste the XI into a CSV
(id or name, team, role = start | bench) or use `ufe lineup --md N --team XXX "Name, Name, ..."`.
Whenever a line-up file changes, `ufe live` re-runs the simulation for the remaining fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import ROOT
from ..names import best_player_match
from .http import get

PROC = ROOT / "data" / "processed"


def fotmob_lineups(match_id: int) -> list[dict] | None:
    try:
        r = get(f"https://www.fotmob.com/api/matchDetails?matchId={match_id}", retries=1)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    out = []
    lu = (data.get("content") or {}).get("lineup") or {}
    for side in ("homeTeam", "awayTeam"):
        t = lu.get(side) or {}
        for p in t.get("starters", []) or []:
            out.append(dict(name=p.get("name"), team_name=t.get("name"), role="start"))
        for p in t.get("subs", []) or []:
            out.append(dict(name=p.get("name"), team_name=t.get("name"), role="bench"))
    return out or None


def from_text(players: pd.DataFrame, team: str, starters: str, bench: str = "") -> pd.DataFrame:
    pool = players[players.team == team]
    cands = dict(zip(pool.name, pool.id))
    rows = []
    for role, names in (("start", starters), ("bench", bench)):
        for n in [x.strip() for x in names.split(",") if x.strip()]:
            pid = best_player_match(n, cands, cutoff=75)
            rows.append(dict(id=pid, name=n, team=team, role=role, matched=pid is not None))
    return pd.DataFrame(rows)


def write(md: int, rows: pd.DataFrame) -> Path:
    PROC.mkdir(parents=True, exist_ok=True)
    p = PROC / f"lineups_md{md}.csv"
    rows = rows.dropna(subset=["id"])
    if p.exists():
        prev = pd.read_csv(p)
        rows = pd.concat([prev[~prev.team.isin(rows.team)], rows], ignore_index=True)
    rows.to_csv(p, index=False)
    return p
