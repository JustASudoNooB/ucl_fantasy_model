"""This season's league numbers for every player at the 36 Champions League clubs, from
API-Football's official API (free plan: 100 requests a day, about 10 a minute).

The Premier League fantasy feed (fpl_live.py) covers only Premier League players; this fills in
the other leagues: appearances, starts, minutes, goals, assists, penalties taken, shots and shots
on target in each club's domestic league. API-Football has no xG, so xG is estimated from shots:
0.10 per non-penalty shot plus 0.76 per penalty taken, marked in the source column.

Budget: each run refreshes the clubs with the oldest data, up to `max_requests` calls (default
30: about 12 clubs, so every club is refreshed every three days), with a pause between calls for
the per-minute limit. Premier League clubs are skipped (the fantasy feed has real xG for them).
The club-id lookup (one call) is kept in data/processed/api_football_teams.json. A plan error
(for example, a season the free plan does not cover) stops the run and is reported.
Writes into data/processed/domestic_form.csv, replacing only the clubs it fetched.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date

import pandas as pd

from ..config import ROOT, sources
from ..names import ucl_team_id
from .ucl_fantasy import FEED_CLUBS

PROC = ROOT / "data" / "processed"
OUT = PROC / "domestic_form.csv"
TEAMS = PROC / "api_football_teams.json"
PL_CLUBS = {"ARS", "AVL", "LIV", "MCI", "MUN", "CHE", "TOT", "NEW"}      # covered by the fantasy feed
NOT_LEAGUE = ("champions league", "europa", "conference", "cup", "copa", "coppa", "pokal", "coupe", "super",
              "friendl", "taça", "beker", "trophy", "shield", "playoff", "play-off", "qualif")
NP_SHOT_XG, PEN_XG = 0.10, 0.76


class PlanError(RuntimeError):
    pass


def _call(path: str, params: dict, key: str, url: str, pause: float) -> dict:
    from .http import get
    r = get(f"{url}/{path}", headers={"x-apisports-key": key}, params=params, retries=2)
    time.sleep(pause)
    d = r.json() if r.status_code == 200 else {"errors": {"http": r.status_code}}
    err = d.get("errors")
    if err and (not isinstance(err, list) or len(err)):
        raise PlanError(json.dumps(err)[:300])
    return d


def _club(name: str) -> str | None:
    return ucl_team_id(name) or FEED_CLUBS.get(name.strip().lower())


def team_ids(key: str, url: str, league: int, season: int, pause: float) -> tuple[dict[str, int], int]:
    """API-Football's id for each club (cached), and the number of calls it took."""
    if TEAMS.exists():
        d = json.loads(TEAMS.read_text())
        if d.get("season") == season and len(d.get("ids", {})) >= 30:
            return d["ids"], 0
    resp = _call("teams", dict(league=league, season=season), key, url, pause)["response"]
    ids = {}
    for t in resp:
        code = _club(t["team"]["name"])
        if code:
            ids[code] = t["team"]["id"]
    TEAMS.write_text(json.dumps(dict(season=season, ids=ids, unmatched=sorted(
        t["team"]["name"] for t in resp if not _club(t["team"]["name"])))))
    return ids, 1


def league_row(code: str, p: dict, stats: list[dict]) -> dict | None:
    """The player's domestic-league line: the competition with most appearances that is not a cup
    or a European competition."""
    lg = [s for s in stats if not any(w in (s.get("league", {}).get("name") or "").lower() for w in NOT_LEAGUE)]
    if not lg:
        return None
    s = max(lg, key=lambda x: (x.get("games", {}).get("appearences") or 0))
    g, gl, sh, pen = s.get("games", {}), s.get("goals", {}), s.get("shots", {}), s.get("penalty", {})
    num = lambda v: float(v or 0)
    pk = num(pen.get("scored")) + num(pen.get("missed"))
    shots = num(sh.get("total"))
    return dict(team=code, player=p.get("name") or f"{p.get('firstname', '')} {p.get('lastname', '')}".strip(),
                comp=s["league"].get("name"), season=f"{date.today().year if date.today().month >= 7 else date.today().year - 1}-"
                f"{str((date.today().year if date.today().month >= 7 else date.today().year - 1) + 1)[2:]}",
                m=num(g.get("appearences")), minutes=num(g.get("minutes")), starts=num(g.get("lineups")),
                goals=num(gl.get("total")), assists=num(gl.get("assists")), pk=pk,
                xg=round(NP_SHOT_XG * max(shots - pk, 0.0) + PEN_XG * pk, 2), shots=shots, sot=num(sh.get("on")),
                source="api-football (xG estimated from shots)", read_at=date.today().isoformat())


def sync(season: int = 2026, max_requests: int = 30, pause: float = 6.5) -> str:
    cfg = sources()["api_football"]
    key = os.environ.get(cfg["key_env"])
    if not key:
        return f"api-football: skipped (no {cfg['key_env']})"
    used = 0
    try:
        ids, used = team_ids(key, cfg["url"], cfg["ucl_league_id"], season, pause)
        old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["team", "source", "read_at"])
        last = (old[old.source.astype(str).str.startswith("api-football")].groupby("team").read_at.max()
                if len(old) else pd.Series(dtype=str))
        todo = sorted((c for c in ids if c not in PL_CLUBS), key=lambda c: str(last.get(c, "")))
        rows, done = [], []
        for code in todo:
            team_rows, page, pages = [], 1, 1
            while page <= pages and used < max_requests:
                d = _call("players", dict(team=ids[code], season=season, page=page), key, cfg["url"], pause)
                used += 1
                pages = int(d.get("paging", {}).get("total") or 1)
                for it in d.get("response", []):
                    r = league_row(code, it.get("player", {}), it.get("statistics", []))
                    if r and r["m"] > 0:
                        team_rows.append(r)
                page += 1
            if page <= pages:                          # out of budget mid-club: finish it next run
                break
            if team_rows:
                tm = max(r["m"] for r in team_rows)    # matches the club has played (most-used player)
                for r in team_rows:
                    r["team_matches"] = tm
                rows += team_rows
                done.append(code)
    except PlanError as e:
        return f"WARNING api-football: stopped after {used} requests: {e}"
    if not rows:
        return f"api-football: nothing new ({used} requests)"
    new = pd.DataFrame(rows)
    cols = ["team", "player", "comp", "season", "team_matches", "m", "minutes", "starts", "goals", "assists",
            "pk", "xg", "shots", "sot", "source", "read_at"]
    keep = pd.concat([old[~old.team.isin(done)], new[cols]], ignore_index=True) if len(old) else new[cols]
    keep.to_csv(OUT, index=False)
    return f"api-football -> {OUT.name}: {len(new)} players at {len(done)} clubs ({', '.join(done)}); {used} requests"
