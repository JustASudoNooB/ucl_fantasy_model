"""Premier League player event history (Opta data via the FPL API, mirrored on GitHub).

Used for (1) per-player rates of UCL players at English clubs and (2) population priors
(how minutes, recoveries, cards and attacking shares vary by position and price) that are
applied to every player without club-level data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, sources
from .http import fetch_to_file

RAW = ROOT / "data" / "raw" / "fpl"
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def download(seasons=("2025-26", "2026-27"), force: bool = False) -> None:
    url = sources()["fpl_history"]["url"]
    for s in seasons:
        for f in ("players_raw.csv", "gws/merged_gw.csv", "teams.csv"):
            fetch_to_file(url.format(season=s, file=f), RAW / f"fpl_{s}_{f.replace('/', '_')}", force=force)


@dataclass
class FPLData:
    gw: pd.DataFrame            # per player-match rows (history season)
    players_hist: pd.DataFrame  # players_raw of the history season
    players_now: pd.DataFrame   # players_raw of the current season (squads, prices, set-piece orders)


def load(history: str = "2025-26", current: str = "2026-27") -> FPLData:
    gw = pd.read_csv(RAW / f"fpl_{history}_gws_merged_gw.csv")
    ph = pd.read_csv(RAW / f"fpl_{history}_players_raw.csv")
    pn = pd.read_csv(RAW / f"fpl_{current}_players_raw.csv")
    th = pd.read_csv(RAW / f"fpl_{history}_teams.csv")
    tn = pd.read_csv(RAW / f"fpl_{current}_teams.csv")
    for p, t in ((ph, th), (pn, tn)):
        p["team_name"] = p["team"].map(dict(zip(t["id"], t["name"])))
        p["full_name"] = (p["first_name"].fillna("") + " " + p["second_name"].fillna("")).str.strip()
        p["pos"] = p["element_type"].map(POS)
    gw["pos"] = gw["position"].replace({"GKP": "GK", "AM": "MID"})
    return FPLData(gw=gw, players_hist=ph, players_now=pn)


def player_aggregates(d: FPLData) -> pd.DataFrame:
    """Per player (FPL code) season aggregates, per-90 rates and attacking shares of team xG/xA."""
    gw = d.gw.copy()
    for c in ("recoveries", "saves", "own_goals", "penalties_missed", "starts"):
        if c not in gw:                       # older seasons lack some Opta fields (recoveries: 2025/26+)
            gw[c] = np.nan if c == "recoveries" else 0
    code_of = dict(zip(d.players_hist["id"], d.players_hist["code"]))
    gw["code"] = gw["element"].map(code_of)
    team_fx = gw.groupby(["team", "fixture"]).agg(t_xg=("expected_goals", "sum"), t_xa=("expected_assists", "sum")).reset_index()
    team_rate = team_fx.groupby("team").agg(team_xg_pm=("t_xg", "mean"), team_xa_pm=("t_xa", "mean"), team_matches=("fixture", "nunique"))
    gw["sub_app"] = ((gw["starts"] == 0) & (gw["minutes"] > 0)).astype(int)
    gw["no_start"] = (gw["starts"] == 0).astype(int)
    gw["full90"] = ((gw["starts"] == 1) & (gw["minutes"] >= 88)).astype(int)
    agg = gw.groupby(["code", "team"]).agg(
        name=("name", "last"), pos=("pos", "last"), matches=("fixture", "nunique"), minutes=("minutes", "sum"),
        starts=("starts", "sum"), sub_apps=("sub_app", "sum"), no_starts=("no_start", "sum"), full90=("full90", "sum"),
        goals=("goals_scored", "sum"), assists=("assists", "sum"), xg=("expected_goals", "sum"), xa=("expected_assists", "sum"),
        recoveries=("recoveries", lambda x: x.sum(min_count=1)), saves=("saves", "sum"), yellow=("yellow_cards", "sum"), red=("red_cards", "sum"),
        own_goals=("own_goals", "sum"), pens_missed=("penalties_missed", "sum"), value=("value", "first"),
    ).reset_index()
    agg = agg.sort_values("minutes").drop_duplicates("code", keep="last")      # club with most minutes
    agg = agg.join(team_rate, on="team")
    m90 = agg["minutes"].clip(lower=1) / 90.0
    agg["xg90"] = agg["xg"] / m90
    agg["xa90"] = agg["xa"] / m90
    agg["rec90"] = agg["recoveries"] / m90
    agg["yc90"] = agg["yellow"] / m90
    agg["xg_share"] = agg["xg90"] / agg["team_xg_pm"]
    agg["xa_share"] = agg["xa90"] / agg["team_xa_pm"]
    agg["start_rate"] = agg["starts"] / agg["team_matches"].clip(lower=1)
    agg["p_sub"] = agg["sub_apps"] / agg["no_starts"].clip(lower=1)
    agg["p_full90"] = agg["full90"] / agg["starts"].clip(lower=1)
    agg["start_price"] = agg["value"] / 10.0
    return agg


def population_priors(agg: pd.DataFrame, min_minutes: int = 600) -> dict:
    """Position-level priors and price slopes fitted on the Premier League population."""
    out: dict = {}
    a = agg[agg["minutes"] >= min_minutes].copy()
    for pos, g in a.groupby("pos"):
        x = np.log(g["start_price"].clip(lower=3.8))
        pri = {}
        for col in ("xg_share", "xa_share"):
            y = np.log(g[col].clip(lower=1e-3))
            b, c = np.polyfit(x, y, 1)
            pri[col] = dict(slope=float(b), intercept=float(c), resid_sd=float(np.std(y - (b * x + c))))
        pri["rec90"] = float(g["rec90"].mean())
        pri["yc90"] = float(g["yellow"].sum() / g["minutes"].sum() * 90)
        pri["red90"] = float(g["red"].sum() / g["minutes"].sum() * 90)
        pri["og90"] = float(g["own_goals"].sum() / g["minutes"].sum() * 90)
        pri["p_full90"] = float(g["full90"].sum() / max(g["starts"].sum(), 1))
        out[pos] = pri
    return out
