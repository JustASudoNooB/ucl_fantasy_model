"""Big-five league player seasons (FBref, Opta event data) from the worldfootballR_data archive.

The archive (github.com/JaseZiv/worldfootballR_data, release "fb_big5_advanced_season_stats")
was frozen in September 2025. It holds complete 2023/24 and 2024/25 seasons for the Premier
League, La Liga, Bundesliga, Serie A and Ligue 1, and the first 3-5 rounds of 2025/26. FBref
itself refuses many automated clients; the archive's GitHub release files download normally.

The processed table ships with the engine (data/seed/fbref_big5.csv), so reading it needs no
extra package. Rebuilding it from the .rds files needs `pip install pyreadr`.

One row per player, club and season:
  season (end year), comp, squad, ucl (2026/27 UCL team id or ""), player, born, pos,
  mp, starts, minutes, gls, ast, pk, pkatt, xg, npxg, xag, crdy, crdr, recov and misc_90s (the
  misc table's own minutes/90, which can cover fewer matches than the standard table; NaN when
  the archive lacks the row), and club totals for the same season:
  team_matches, team_npxg, team_xag, team_pkatt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from .http import fetch_to_file

RAW = ROOT / "data" / "raw" / "fbref"
SEED = ROOT / "data" / "seed" / "fbref_big5.csv"
URL = "https://github.com/JaseZiv/worldfootballR_data/releases/download/fb_big5_advanced_season_stats/{name}.rds"
FILES = ("big5_player_standard", "big5_player_playing_time", "big5_player_misc")

# FBref club names of the 2026/27 UCL clubs in the big-five leagues
SQUAD_TO_UCL = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Liverpool": "LIV", "Manchester City": "MCI", "Manchester Utd": "MUN",
    "Bayern Munich": "BAY", "Dortmund": "BVB", "RB Leipzig": "RBL", "Stuttgart": "VFB",
    "Barcelona": "BAR", "Real Madrid": "RMA", "Atlético Madrid": "ATM", "Villarreal": "VIL", "Betis": "BET",
    "Inter": "INT", "Napoli": "NAP", "Roma": "ROM", "Como": "COM",
    "Paris S-G": "PSG", "Lille": "LIL", "Lens": "RCL",
}


def download(force: bool = False) -> list[Path]:
    out = []
    for name in FILES:
        out.append(fetch_to_file(URL.format(name=name), RAW / f"{name}.rds", force=force))
    return out


def build_table(seasons=(2024, 2025, 2026), raw: Path = RAW, out: Path = SEED) -> pd.DataFrame:
    """Read the archive's .rds files and write the processed table."""
    import pyreadr                                   # optional dependency, only for rebuilding

    def rd(name):
        return pyreadr.read_r(str(raw / f"{name}.rds"))[None]

    key = ["Season_End_Year", "Squad", "Comp", "Player", "Born"]
    st = rd("big5_player_standard")
    st = st[st.Season_End_Year.isin(seasons)]
    st = st.rename(columns={"MP_Playing": "mp", "Starts_Playing": "starts", "Min_Playing": "minutes", "Gls": "gls",
                            "Ast": "ast", "PK": "pk", "PKatt": "pkatt", "CrdY": "crdy", "CrdR": "crdr",
                            "xG_Expected": "xg", "npxG_Expected": "npxg", "xAG_Expected": "xag"})
    cols = key + ["Pos", "mp", "starts", "minutes", "gls", "ast", "pk", "pkatt", "crdy", "crdr", "xg", "npxg", "xag"]
    df = st[cols].copy()
    pt = rd("big5_player_playing_time")
    pt = pt[pt.Season_End_Year.isin(seasons)]
    tm = pt.groupby(["Season_End_Year", "Squad"])["MP_Playing.Time"].max().rename("team_matches")
    misc = rd("big5_player_misc")
    misc = misc[misc.Season_End_Year.isin(seasons)][key + ["Recov", "Mins_Per_90"]].drop_duplicates(key)
    df = df.merge(misc, on=key, how="left").rename(columns={"Recov": "recov", "Mins_Per_90": "misc_90s"})
    df = df.join(tm, on=["Season_End_Year", "Squad"])
    for c in ("mp", "starts", "minutes", "gls", "ast", "pk", "pkatt", "crdy", "crdr", "xg", "npxg", "xag"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    tot = df.groupby(["Season_End_Year", "Squad"]).agg(team_npxg=("npxg", "sum"), team_xag=("xag", "sum"),
                                                        team_pkatt=("pkatt", "sum"))
    df = df.join(tot, on=["Season_End_Year", "Squad"])
    df = df.rename(columns={"Season_End_Year": "season", "Squad": "squad", "Comp": "comp", "Player": "player",
                            "Born": "born", "Pos": "pos"})
    df["ucl"] = df.squad.map(SQUAD_TO_UCL).fillna("")
    df["born"] = pd.to_numeric(df.born, errors="coerce").astype("Int64")
    df = df[df.minutes > 0].sort_values(["season", "squad", "player"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.3f")
    return df


def load(path: Path = SEED) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["ucl"] = df["ucl"].fillna("")
    return df


def fbref_pos(pos: str) -> str:
    """FBref position string ('DF,MF') -> the first listed role on the game's scale."""
    first = str(pos).split(",")[0].strip()
    return {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}.get(first, "MID")


def season_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row rates: shares of the club's open-play xG and xA per 90 on the pitch, start rate."""
    d = df.copy()
    m90 = (d.minutes / 90.0).clip(lower=1e-6)
    tm = d.team_matches.clip(lower=1)
    d["team_npxg_pm"] = d.team_npxg / tm
    d["team_xag_pm"] = d.team_xag / tm
    d["xg_share"] = (d.npxg / m90) / d.team_npxg_pm.clip(lower=0.2)
    d["xa_share"] = (d.xag / m90) / d.team_xag_pm.clip(lower=0.15)
    d["start_rate"] = (d.starts / tm).clip(0, 1)
    d["rec90"] = d.recov / d.misc_90s.where(d.misc_90s > 0)
    d["yc90"] = d.crdy / m90
    d["pk_share"] = np.where(d.team_pkatt > 0, d.pkatt / d.team_pkatt.clip(lower=1), np.nan)
    return d
