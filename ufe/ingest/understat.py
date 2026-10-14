"""Understat player xG / xA for the big five leagues (covers 24 of the 36 league-phase clubs).

League pages embed `var playersData = JSON.parse('...')`; this parses that blob, computes each
player's share of his team's non-penalty xG and xA per 90, and writes
data/processed/understat_players.csv. player_rates uses it the same way as the Premier League
history: observed shares blended with the price prior by minutes played.
"""
from __future__ import annotations

import codecs
import json
import re
from pathlib import Path

import pandas as pd

from ..config import ROOT
from ..names import ucl_team_id
from .http import get

PROC = ROOT / "data" / "processed"
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]


def _blob(html: str, var: str):
    m = re.search(var + r"\s*=\s*JSON\.parse\('(.*?)'\)", html)
    if not m:
        return None
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def league_players(league: str, season: int) -> pd.DataFrame | None:
    r = get(f"https://understat.com/league/{league}/{season}", min_gap=3.0)
    if r.status_code != 200:
        return None
    data = _blob(r.text, "playersData")
    if not data:
        return None
    df = pd.DataFrame(data)
    for c in ["games", "time", "goals", "xG", "assists", "xA", "npg", "npxG", "shots", "key_passes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["team"] = df.team_title.str.split(",").str[-1].map(lambda s: ucl_team_id(s.strip()))
    team_tot = df.groupby("team_title").agg(t_npxg=("npxG", "sum"), t_xa=("xA", "sum"), t_games=("games", "max"))
    df = df.join(team_tot, on="team_title")
    m90 = df.time.clip(lower=1) / 90
    df["xg_share"] = (df.npxG / m90) / (df.t_npxg / df.t_games.clip(lower=1))
    df["xa_share"] = (df.xA / m90) / (df.t_xa / df.t_games.clip(lower=1))
    df["league"] = league
    return df


def sync(season: int = 2026) -> Path | None:
    frames = [f for f in (league_players(lg, season) for lg in LEAGUES) if f is not None]
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    PROC.mkdir(parents=True, exist_ok=True)
    p = PROC / "understat_players.csv"
    out.to_csv(p, index=False)
    return p
