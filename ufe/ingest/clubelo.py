"""ClubElo daily ratings: an independent cross-check (and optional prior) for the team model."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from ..config import sources
from ..names import ucl_team_id
from .http import get

CLUBELO_NAMES = {"Paris SG": "PSG", "Bodoe/Glimt": "BOD", "Atletico": "ATM", "Sporting": "SCP", "Slavia Praha": "SLA",
                 "Man City": "MCI", "Man United": "MUN", "Inter": "INT", "Bayern": "BAY", "Dortmund": "BVB",
                 "RB Leipzig": "RBL", "Stuttgart": "VFB", "Lens": "RCL", "Lille": "LIL", "Brugge": "CLB",
                 "Shakhtar": "SHA", "AEK": "AEK", "LASK": "LAS", "Viking": "VIK", "Slovan Bratislava": "SLO",
                 "Sabah": "SAB", "Fenerbahce": "FEN", "Galatasaray": "GAL", "Feyenoord": "FEY", "PSV": "PSV",
                 "Porto": "POR", "Betis": "BET", "Villarreal": "VIL", "Real Madrid": "RMA", "Barcelona": "BAR",
                 "Napoli": "NAP", "Roma": "ROM", "Como": "COM", "Liverpool": "LIV", "Arsenal": "ARS", "Aston Villa": "AVL"}


def ratings(on: date | None = None) -> pd.DataFrame | None:
    on = on or date.today()
    r = get(sources()["clubelo"]["url"].format(date=on.isoformat()))
    if r.status_code != 200 or not r.text.startswith("Rank"):
        return None
    df = pd.read_csv(io.StringIO(r.text))
    df["team"] = [CLUBELO_NAMES.get(c) or ucl_team_id(c) for c in df.Club]
    return df.dropna(subset=["team"])[["team", "Club", "Country", "Elo"]]
