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


# Season-to-date numbers the feed carries for every player (cumulative over the matchdays played).
# Kept in players_feed.csv and in the pre-deadline copy, so the difference between two copies is a
# player's line for the matchday in between. pen_* meanings are UEFA's pS / pC / pE / pM fields.
STATS = dict(mins="minsPlyd", goals="gS", assists="assist", cs="cS", gc="gC", yc="yC", rc="rC", og="oG",
             pen_saved="pS", pen_conceded="pC", pen_earned="pE", pen_missed="pM", saves="saves", recoveries="bR",
             goals_outside="gOB", motm="mOM", last_md_points="lastGdPoints", transfers_in="mTransferIn",
             transfers_out="mTransferOut")
TEXT = dict(trained="trained", q_status="qStatus")       # UEFA's own availability notes


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


# UEFA's own short club names that are ambiguous elsewhere ("Paris FC" is another club in French
# league data), so they are matched only inside this feed.
FEED_CLUBS = {"paris": "PSG", "dortmund": "BVB", "bratislava": "SLO"}


def _feed_team(v) -> str | None:
    t = ucl_team_id(str(v))
    return t or FEED_CLUBS.get(str(v).strip().lower())


def normalise(records: list[dict]) -> pd.DataFrame:
    rows, unmapped = [], set()
    for r in records:
        team = None
        for k in KEYS["team"]:                      # every name field, not just the first present: UEFA's
            if r.get(k) not in (None, ""):           # full name for some clubs is a form ("Paris") we may not know
                team = _feed_team(r[k])
                if team:
                    break
        if team is None:
            unmapped.add(str(_first(r, KEYS["team"])))
        pos = POS_CODES.get(_first(r, KEYS["pos"]), POS_CODES.get(str(_first(r, KEYS["pos"])).upper()))
        price = _first(r, KEYS["price"])
        if team is None or pos is None or price is None:
            continue
        rows.append(dict(feed_id=_first(r, KEYS["id"]), name=_first(r, KEYS["name"]), short=_first(r, KEYS["short"]),
                         team=team, pos=pos, price=float(price), selected=_first(r, KEYS["selected"]),
                         points=_first(r, KEYS["points"]), status=_first(r, KEYS["status"]), source="ucl_feed",
                         as_of=pd.Timestamp.utcnow().date().isoformat(), note="",
                         **{c: r.get(k) for c, k in STATS.items()},
                         **{c: (str(r.get(k) or "").strip() or None) for c, k in TEXT.items()}))
    df = pd.DataFrame(rows)
    df.attrs["unmapped"] = sorted(unmapped)
    return df


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
    meta = dict(url=url, n=len(df), clubs=int(df.team.nunique()), unmapped_clubs=df.attrs.get("unmapped", []))
    try:                                           # what the feed carries, for building on it later
        top = max(recs, key=lambda r: float(_first(r, KEYS["points"]) or 0))
        meta["fields"] = sorted({k for r in recs[:200] for k in r})
        meta["sample"] = {k: (v if not isinstance(v, (list, dict)) else json.loads(json.dumps(v))[:3] if isinstance(v, list) else v)
                          for k, v in top.items()}
    except Exception:
        pass
    (PROC / "feed_meta.json").write_text(json.dumps(meta))
    if meta["unmapped_clubs"] or meta["clubs"] < 36:
        print(f"WARNING feed: {meta['clubs']} of 36 clubs; names not matched: {meta['unmapped_clubs']}", flush=True)
    _save_baseline(df)
    try:
        sync_md_points(url, latest=recs)
    except Exception as e:                           # extra data: never fail the feed sync over it
        print(f"WARNING feed: matchday points not updated ({type(e).__name__}: {str(e)[:120]})", flush=True)
    return out


def played_mds(now=None) -> list[int]:
    """Matchdays whose deadline has passed."""
    from datetime import datetime, timezone
    from ..config import rules
    from ..timeutil import deadline_for
    now = now or datetime.now(timezone.utc)
    return sorted(int(r["md"]) for r in rules()["matchdays"] if deadline_for(r).astimezone(timezone.utc) <= now)


def sync_md_points(url: str, fetch=None, latest: list[dict] | None = None) -> Path | None:
    """Each player's official points in every matchday played: the feed returns one matchday's
    points (curGDPts) per gamedayId. Written to feed_md_points.csv (feed_id, md, points). Earlier
    matchdays are final and fetched once; the latest one is refreshed on every run (UEFA corrects
    points for a day or two)."""
    import re
    if not re.search(r"gamedayId=\d+", url or ""):
        return None
    out = PROC / "feed_md_points.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["feed_id", "md", "points"])
    mds = played_mds()
    if not mds:
        return None
    rows = []
    for g in mds:
        if g != mds[-1] and (old.md == g).any():
            continue
        u = re.sub(r"gamedayId=\d+", f"gamedayId={g}", url)
        try:
            if fetch is not None:
                data = fetch(u)
            else:
                r = get(u, headers={"Accept": "application/json", "Referer": "https://gaming.uefa.com/en/uclfantasy/"},
                        retries=2, timeout=60.0)             # a large file: the default 20 s was not always enough
                data = r.json() if r.status_code == 200 else None
        except Exception:
            data = None
        recs = _find_player_list(data) if data is not None else None
        if not recs or str(recs[0].get("mdId")) != str(g):      # the feed did not serve this matchday
            continue
        rows += [dict(feed_id=_first(r, KEYS["id"]), md=g, points=r.get("curGDPts")) for r in recs]
    got = {r["md"] for r in rows}
    if latest and mds[-1] not in got and not (old.md == mds[-1]).any():
        # the latest matchday could not be fetched on its own: the main feed's "last matchday" points are the same numbers
        rows += [dict(feed_id=_first(r, KEYS["id"]), md=mds[-1], points=r.get("lastGdPoints")) for r in latest
                 if r.get("lastGdPoints") is not None]
    if not rows:
        return out if out.exists() else None
    new = pd.DataFrame(rows)
    keep = pd.concat([old[~old.md.isin(set(new.md))], new], ignore_index=True).sort_values(["md", "feed_id"])
    keep.to_csv(out, index=False)
    return out


def _save_baseline(df: pd.DataFrame) -> Path | None:
    """Before a matchday's deadline, keep a copy of the feed as feed_pre_md{N}.csv: the live call
    reads each player's points for the matchday as the feed's season total minus this copy."""
    from ..report.managers import next_md
    from ..timeutil import deadline_for
    from ..config import rules
    from datetime import datetime, timezone
    md = next_md()
    row = next((r for r in rules()["matchdays"] if int(r["md"]) == md), None)
    if row is None or datetime.now(timezone.utc) >= deadline_for(row).astimezone(timezone.utc):
        return None
    p = PROC / f"feed_pre_md{md}.csv"
    df.to_csv(p, index=False)
    return p
