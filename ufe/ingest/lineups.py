"""Confirmed line-ups (~60 minutes before kick-off) -> data/processed/lineups_md{N}.csv.

Automatic: API-Football (free key, `ufe sync --lineups MD`, run by the scheduler from 75 minutes
before each kick-off). Also: FotMob's public match JSON (unofficial; the endpoint and headers change from time to
time, so failures are expected and handled). Fallback: paste the XI into a CSV
(id or name, team, role = start | bench) or use `ufe lineup --md N --team XXX "Name, Name, ..."`.
Whenever a line-up file changes, `ufe live` re-runs the simulation for the remaining fixtures.
"""
from __future__ import annotations

from pathlib import Path

import os

import pandas as pd

from ..config import ROOT, sources
from ..names import best_player_match, ucl_team_id
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


def api_football(md: int, players: pd.DataFrame, fixtures: pd.DataFrame, season: int = 2026,
                 only_teams: set[str] | None = None) -> tuple[Path | None, str]:
    """Confirmed XIs for the matchday's games that have them published. Returns (file, note)."""
    cfg = sources()["api_football"]
    key = os.environ.get(cfg["key_env"])
    if not key:
        return None, f"lineups: skipped (no {cfg['key_env']})"
    hdr = {"x-apisports-key": key}
    fx = fixtures[fixtures.md == md]
    rows, n_games = [], 0
    for day in sorted(fx.date.astype(str).unique()):
        r = get(f"{cfg['url']}/fixtures", headers=hdr, params=dict(league=cfg["ucl_league_id"], season=season, date=day))
        if r.status_code != 200:
            return None, f"lineups: fixtures request returned {r.status_code}"
        for g in r.json().get("response", []):
            teams = {ucl_team_id(g["teams"]["home"]["name"]), ucl_team_id(g["teams"]["away"]["name"])}
            if None in teams or (only_teams and not teams & only_teams):
                continue
            lr = get(f"{cfg['url']}/fixtures/lineups", headers=hdr, params=dict(fixture=g["fixture"]["id"]))
            if lr.status_code != 200 or not lr.json().get("response"):
                continue                                     # not published yet
            n_games += 1
            for side in lr.json()["response"]:
                team = ucl_team_id(side["team"]["name"])
                pool = players[players.team == team]
                cands = dict(zip(pool.name, pool.id))
                for role, key_ in (("start", "startXI"), ("bench", "substitutes")):
                    for e in side.get(key_) or []:
                        n = (e.get("player") or {}).get("name") or ""
                        pid = best_player_match(n, cands, cutoff=75)
                        rows.append(dict(id=pid, name=n, team=team, role=role, matched=pid is not None))
    if not rows:
        return None, "lineups: none published yet"
    df = pd.DataFrame(rows)
    miss = df[~df.matched & (df.role == "start")]
    note = f"lineups: {n_games} games, {int(df.matched.sum())}/{len(df)} players matched"
    if len(miss):
        note += f"; unmatched starters: {', '.join(miss.team + ' ' + miss.name)}"
    return write(md, df), note
