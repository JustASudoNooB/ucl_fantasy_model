"""This season's Premier League numbers for every player at an English UCL club, from the official
FPL API (bootstrap-static, no key). Writes data/processed/domestic_form.csv in the seed form
schema; history.load_current_form() then uses it in place of the seed rows for those clubs.

Limits of the source: FPL has no appearance count, shots or penalty goals. Appearances are
estimated from starts and minutes, shots are left at 0 and pk at 0 (so xG includes penalties).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..config import ROOT
from ..names import ucl_team_id
from .http import get

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
OUT = ROOT / "data" / "processed" / "domestic_form.csv"
COLS = ["team", "player", "comp", "season", "team_matches", "m", "minutes", "starts", "goals", "assists",
        "pk", "xg", "shots", "sot", "source", "read_at"]


def normalise(data: dict) -> pd.DataFrame:
    played = sum(1 for e in data.get("events", []) if e.get("finished"))
    teams = {t["id"]: ucl_team_id(t["name"]) or ucl_team_id(t.get("short_name", "")) for t in data.get("teams", [])}
    rows = []
    for p in data.get("elements", []):
        team = teams.get(p.get("team"))
        mins = int(p.get("minutes") or 0)
        if team is None or mins == 0:
            continue
        starts = int(p.get("starts") or 0)
        sub_apps = round(max(0, mins - 85 * starts) / 25)              # estimate: FPL has no appearance count
        rows.append(dict(team=team, player=f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
                         comp="Premier League", season="2026-27", team_matches=played,
                         m=min(played, starts + sub_apps) if played else starts + sub_apps,
                         minutes=mins, starts=starts, goals=int(p.get("goals_scored") or 0),
                         assists=int(p.get("assists") or 0), pk=0, xg=float(p.get("expected_goals") or 0.0),
                         shots=0, sot=0, source=URL, read_at=date.today().isoformat()))
    return pd.DataFrame(rows, columns=COLS)


def sync() -> str:
    r = get(URL, retries=2)
    if r.status_code != 200:
        return f"form: FPL API returned {r.status_code}"
    df = normalise(r.json())
    if df.empty:
        return "form: no UCL-club players in the FPL feed"
    keep = df
    if OUT.exists():                                           # keep other clubs' rows from other sources
        old = pd.read_csv(OUT)
        keep = pd.concat([old[~old.team.isin(set(df.team))], df], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keep.to_csv(OUT, index=False)
    return f"form -> {OUT} ({len(df)} players, clubs {sorted(df.team.unique())})"
