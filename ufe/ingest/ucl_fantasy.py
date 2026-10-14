"""Official UCL Fantasy feed adapter (players, prices, positions, ownership, per-MD stats).

The game is a JavaScript app that loads its data from JSON feeds under
https://gaming.uefa.com/en/uclfantasy/services/feeds/ . The exact path (and a season "tour" id
in it) changes between seasons, so this adapter:
  1. uses `ucl_fantasy_feed.players_url` from config/sources.yaml if you set it, else
  2. tries the candidate templates with several tour ids and matchdays,
  3. parses whatever JSON comes back by searching for the list of player records,
  4. normalises it to the engine's pool schema and writes data/processed/players_feed.csv.

If discovery fails: open the game in Chrome, DevTools > Network, filter "players", copy the
request URL into sources.yaml (players_url) and run `ufe sync --feed` again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import ROOT, sources
from ..names import ucl_team_id
from .http import get

PROC = ROOT / "data" / "processed"
POS_CODES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", "1": "GK", "2": "DEF", "3": "MID", "4": "FWD",
             "G": "GK", "D": "DEF", "M": "MID", "F": "FWD", "GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID",
             "FWD": "FWD", "GOALKEEPER": "GK", "DEFENDER": "DEF", "MIDFIELDER": "MID", "FORWARD": "FWD"}
KEYS = dict(
    id=["id", "playerId", "pId"],
    name=["pFName", "fullName", "name", "pDName", "displayName", "playerName"],
    short=["pDName", "displayName", "shortName"],
    team=["tName", "teamName", "cCode", "tSName", "team"],
    pos=["skill", "position", "pos", "positionId", "role"],
    price=["value", "price", "cost", "playerValue"],
    selected=["selPer", "selectedBy", "selected", "ownership"],
    points=["totPts", "points", "totalPoints"],
    status=["pStatus", "status", "availability"],
)


def _first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _find_player_list(obj) -> list[dict] | None:
    """Depth-first search for the largest list of dicts that look like player records."""
    best = None

    def visit(o):
        nonlocal best
        if isinstance(o, list) and o and isinstance(o[0], dict):
            sample = o[0]
            if _first(sample, KEYS["price"]) is not None and _first(sample, KEYS["name"]) is not None:
                if best is None or len(o) > len(best):
                    best = o
        if isinstance(o, dict):
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o[:5]:
                visit(v)

    visit(obj)
    return best


def normalise(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        team_raw = _first(r, KEYS["team"])
        team = ucl_team_id(str(team_raw)) if team_raw is not None else None
        pos = POS_CODES.get(_first(r, KEYS["pos"]), POS_CODES.get(str(_first(r, KEYS["pos"])).upper()))
        price = _first(r, KEYS["price"])
        if team is None or pos is None or price is None:
            continue
        rows.append(dict(feed_id=_first(r, KEYS["id"]), name=_first(r, KEYS["name"]), short=_first(r, KEYS["short"]),
                         team=team, pos=pos, price=float(price), selected=_first(r, KEYS["selected"]),
                         points=_first(r, KEYS["points"]), status=_first(r, KEYS["status"]), source="ucl_feed",
                         as_of=pd.Timestamp.utcnow().date().isoformat(), note=""))
    return pd.DataFrame(rows)


def discover(md: int = 1) -> tuple[str | None, list[dict] | None]:
    cfg = sources()["ucl_fantasy_feed"]
    urls = [cfg["players_url"]] if cfg.get("players_url") else []
    for tour in cfg["tour_ids"]:
        for tpl in cfg["candidates"]:
            urls.append(tpl.format(base=cfg["base"], tour=tour, md=md))
    urls = list(dict.fromkeys(urls))                 # templates without {tour} appear once
    for u in urls:
        try:
            r = get(u, headers={"Accept": "application/json", "Referer": "https://gaming.uefa.com/en/uclfantasy/"}, retries=1)
        except Exception:
            continue
        if r.status_code != 200:
            continue
        try:
            data = r.json()
        except json.JSONDecodeError:
            continue
        recs = _find_player_list(data)
        if recs:
            return u, recs
    return None, None


def sync(md: int = 1) -> Path | None:
    url, recs = discover(md)
    if not recs:
        return None
    df = normalise(recs)
    if df.empty:
        return None
    PROC.mkdir(parents=True, exist_ok=True)
    out = PROC / "players_feed.csv"
    df.to_csv(out, index=False)
    (PROC / "feed_meta.json").write_text(json.dumps(dict(url=url, n=len(df))))
    return out
