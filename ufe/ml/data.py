"""Training data for the neural-network challenger: every Premier League player in every match from
2022/23 game week 16 (when FPL began recording starts and xG) to now (FPL archive,
github.com/vaastav/Fantasy-Premier-League), about 105,000 rows.

Each row is one player and one of his club's league matches while he was registered, with
  features   known before kick-off only (every rolling value is shifted by one match), defined so
             the same numbers can be built for a Champions League player (ufe/ml/ucl.py):
               position, price on the FPL scale, price relative to the club's dearest player and
               the price rank inside the club, home/away, the team's and the opponent's expected
               goals for the match, rest days, recent start rate and whether he started the last
               two matches, his xG and xA per 90 as a share of the team's xG per match, and how
               many minutes that history rests on
  targets    started (0/1), minutes, goals, assists

Team expected goals here come from exponentially weighted team xG for and against (half-life 10
matches, home x1.08 / away x0.93); for UCL matches the engine's own goal rates are used instead,
which are on the same scale (expected goals per match).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ROOT, sources
from ..ingest.http import fetch_to_file

RAW = ROOT / "data" / "raw" / "fpl"
OUT = ROOT / "data" / "processed" / "ml_train.parquet"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
CURRENT = "2026-27"
FILES = ["gws/merged_gw.csv", "fixtures.csv"]
POS = ["GK", "DEF", "MID", "FWD"]
FEATURES = ["pos_GK", "pos_DEF", "pos_MID", "pos_FWD", "log_price", "price_rel", "price_rank", "home",
            "lam_for", "lam_against", "rest_days", "start_rate", "has_start_hist", "started_last1",
            "started_last2", "xg_share", "xa_share", "has_att_hist", "log_n90"]
HL_START, HL_ATT, HL_TEAM = 5, 19, 10       # half-lives in matches
HOME, AWAY = 1.08, 0.93


def _path(season: str, f: str):
    return RAW / f"fpl_{season}_{f.replace('/', '_')}"


def download(seasons=SEASONS, refresh_current: bool = True) -> list[str]:
    url = sources()["fpl_history"]["url"]
    got = []
    for s in seasons:
        for f in FILES:
            ok = fetch_to_file(url.format(season=s, file=f), _path(s, f),
                               force=(refresh_current and s == CURRENT), retries=2)
            if ok:
                got.append(f"{s}/{f}")
    return got


def _season_rows(season: str) -> pd.DataFrame:
    g = pd.read_csv(_path(season, "gws/merged_gw.csv"))
    fx = pd.read_csv(_path(season, "fixtures.csv"))
    fx = fx[fx.team_h_score.notna()]
    g = g[g.fixture.isin(fx.id)].copy()
    # FPL began recording starts and xG at 2022/23 game week 16: earlier rows would read as
    # "never started, no xG", so they are dropped
    rec = g.groupby("GW").starts.sum()
    first = int(rec[rec > 0].index.min()) if (rec > 0).any() else 99
    g = g[g.GW >= first]
    g["season"] = season
    g["kickoff"] = pd.to_datetime(g.kickoff_time, utc=True)
    g["home"] = g.was_home.astype(str).str.lower().isin(["true", "1"]).astype(float)
    sc = fx.set_index("id")
    g["gf"] = np.where(g.home == 1, g.fixture.map(sc.team_h_score), g.fixture.map(sc.team_a_score))
    g["ga"] = np.where(g.home == 1, g.fixture.map(sc.team_a_score), g.fixture.map(sc.team_h_score))
    g["pos"] = g.position.replace({"GKP": "GK", "AM": "MID"})
    g = g[g.pos.isin(POS)]
    for c in ("minutes", "starts", "goals_scored", "assists", "expected_goals", "expected_assists", "value"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    return g[["season", "GW", "fixture", "kickoff", "element", "name", "team", "pos", "home", "gf", "ga", "minutes",
              "starts", "goals_scored", "assists", "expected_goals", "expected_assists", "value"]]


def _team_rates(g: pd.DataFrame) -> pd.DataFrame:
    """Per team and fixture: xG for/against before the match (EW over earlier matches, all seasons)
    and the opponent's, combined into expected goals for and against."""
    tm = (g.groupby(["season", "fixture", "team", "kickoff", "home"], as_index=False)
          .agg(xg_for=("expected_goals", "sum")))
    opp = tm[["season", "fixture", "team", "xg_for"]].rename(columns={"team": "opp", "xg_for": "xg_against"})
    tm = tm.merge(opp, on=["season", "fixture"])
    tm = tm[tm.team != tm.opp].sort_values("kickoff")
    league = tm.xg_for.mean()
    for col, out in (("xg_for", "att"), ("xg_against", "dfn")):
        tm[out] = tm.groupby("team")[col].transform(lambda s: s.shift().ewm(halflife=HL_TEAM).mean()).fillna(league)
    tm["rest_days"] = tm.groupby("team").kickoff.diff().dt.total_seconds().div(86400).clip(2, 14).fillna(7.0)
    o = tm[["season", "fixture", "team", "att", "dfn"]].rename(columns={"team": "opp", "att": "o_att", "dfn": "o_dfn"})
    tm = tm.merge(o, on=["season", "fixture", "opp"])
    hf = np.where(tm.home == 1, HOME, AWAY)
    tm["lam_for"] = tm.att * tm.o_dfn / league * hf
    tm["lam_against"] = tm.o_att * tm.dfn / league * (HOME + AWAY - hf)
    tm["team_xg_pm"] = tm.att
    return tm[["season", "fixture", "team", "lam_for", "lam_against", "rest_days", "team_xg_pm"]]


def build(seasons=SEASONS) -> pd.DataFrame:
    g = pd.concat([_season_rows(s) for s in seasons if _path(s, "gws/merged_gw.csv").exists()], ignore_index=True)
    # one id per player across seasons: name is stable in the archive, element ids are not
    g["pid"] = g.name.astype(str)
    g = g.merge(_team_rates(g), on=["season", "fixture", "team"], how="left")
    g = g.sort_values(["pid", "kickoff"]).reset_index(drop=True)
    by = g.groupby("pid")
    # starts: EW rate over earlier matches, the last two matches
    g["start_rate"] = by.starts.transform(lambda s: s.shift().ewm(halflife=HL_START).mean())
    g["n_prev"] = by.cumcount()
    g["started_last1"] = by.starts.shift(1).fillna(0.0)
    g["started_last2"] = by.starts.shift(2).fillna(0.0)
    g["has_start_hist"] = (g.n_prev >= 3).astype(float)
    g["start_rate"] = np.where(g.has_start_hist == 1, g.start_rate, 0.0)
    # attacking history: EW xG and xA per 90 over earlier appearances, relative to team xG per match
    ew = lambda col: by[col].transform(lambda s: s.shift().ewm(halflife=HL_ATT).mean())
    m, xg, xa = ew("minutes"), ew("expected_goals"), ew("expected_assists")
    n90 = by.minutes.transform(lambda s: s.shift().fillna(0).cumsum()) / 90.0
    tx = g.team_xg_pm.clip(lower=0.4)
    g["xg_share"] = np.where(m > 5, xg / m * 90.0 / tx, np.nan)
    g["xa_share"] = np.where(m > 5, xa / m * 90.0 / tx, np.nan)
    g["has_att_hist"] = ((n90 >= 5) & g.xg_share.notna()).astype(float)
    for c in ("xg_share", "xa_share"):
        g[c] = np.where(g.has_att_hist == 1, np.clip(g[c].fillna(0.0), 0, 1.5), 0.0)
    g["log_n90"] = np.log1p(n90.clip(upper=114))                # cap: three seasons of minutes
    # price inside the club at that game week
    g["log_price"] = np.log(g.value.clip(lower=35) / 10.0)
    tmax = g.groupby(["season", "GW", "team"]).value.transform("max")
    g["price_rel"] = g.value / tmax.clip(lower=1)
    g["price_rank"] = g.groupby(["season", "GW", "team"]).value.rank(ascending=False, pct=True)
    for p in POS:
        g[f"pos_{p}"] = (g.pos == p).astype(float)
    g["start"] = (g.starts > 0).astype(float)
    g["played"] = (g.minutes > 0).astype(float)
    keep = ["season", "GW", "kickoff", "pid", "team", "pos", "minutes", "start", "played", "goals_scored",
            "assists"] + FEATURES
    out = g[keep].dropna(subset=["lam_for", "lam_against"]).reset_index(drop=True)
    out[FEATURES] = out[FEATURES].fillna(0.0)
    return out


def save(df: pd.DataFrame) -> str:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p = OUT.with_suffix(".csv.gz")
    df.to_csv(p, index=False, compression="gzip")
    return str(p)


def load() -> pd.DataFrame:
    return pd.read_csv(OUT.with_suffix(".csv.gz"), parse_dates=["kickoff"])
