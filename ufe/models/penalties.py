"""Penalty takers for every club, from what has happened rather than pre-season articles.

Evidence per player at his current club, scored and summed:
  this season's UCL penalties taken (match events: pen_goal, pen_miss)   3.0 each, + recency
  this season's league penalties taken (starting11.com tables, else `pk`)  2.0 each
  last season's penalty attempts at the same club (FBref)                  0.7 each
  the season before                                                        0.3 each
  the table's current order (FPL penalties_order, notes)                   1.5
  data/seed/news/set_pieces.csv ("penalty taker")                         100 (manual override)
The top player per club becomes the designated taker (pen_order 1) when his score reaches 1.4
(one UCL or domestic penalty this season, or two attempts last season); the runner-up with 1.4 or
more is the backup (pen_order 2). Clubs with no evidence keep the existing order (FPL's
penalties_order or notes), and failing that penalties go by expected-goal share in the simulation.
"""
from __future__ import annotations

import pandas as pd

from ..config import ROOT

SEED = ROOT / "data" / "seed"
PROC = ROOT / "data" / "processed"
MIN_SCORE = 1.4


def _events() -> pd.DataFrame:
    frames = []
    for d in (SEED / "results", PROC / "results"):
        for p in sorted(d.glob("md*_events.csv")) if d.exists() else []:
            try:
                frames.append(pd.read_csv(p))
            except Exception:  # noqa: BLE001
                continue
    if not frames:
        return pd.DataFrame(columns=["md", "team", "player", "event"])
    e = pd.concat(frames, ignore_index=True)
    return e[e.event.isin(["pen_goal", "pen_miss"])]


def evidence() -> pd.DataFrame:
    """(team, name, score, why) rows before matching to the pool."""
    rows = []
    for r in _events().itertuples():
        rows.append((r.team, r.player, 3.0 + 0.05 * float(r.md), f"UCL MD{int(r.md)} penalty"))
    covered = set()
    pt = PROC / "penalty_takers.csv"                  # starting11.com league tables (ingest/setpieces.py)
    if pt.exists():
        d = pd.read_csv(pt)
        covered = set(d.team)
        for r in d.itertuples():
            full = r.full if isinstance(r.full, str) and len(r.full.split()) > len(str(r.name).split()) else r.name
            rows.append((r.team, full, 2.0 * float(r.taken), f"{int(r.scored)}/{int(r.taken)} league penalties 26/27"))
    for p in (SEED / "form" / "domestic_2026_27.csv", PROC / "domestic_form.csv"):
        if p.exists():
            d = pd.read_csv(p)
            if "pk" in d:
                for r in d[(pd.to_numeric(d.pk, errors="coerce").fillna(0) > 0) & ~d.team.isin(covered)].itertuples():
                    rows.append((r.team, r.player, 2.0 * float(r.pk), f"{int(r.pk)} league penalty goal(s) 26/27"))
    fb = SEED / "fbref_big5.csv"
    if fb.exists():
        f = pd.read_csv(fb)
        f = f[f.ucl.notna() & (pd.to_numeric(f.pkatt, errors="coerce").fillna(0) > 0)]
        latest = int(f.season.max()) if len(f) else 0
        for r in f.itertuples():
            w = 0.7 if int(r.season) == latest else (0.3 if int(r.season) == latest - 1 else 0.0)
            if w:
                rows.append((r.ucl, r.player, w * float(r.pkatt), f"{int(r.pkatt)} attempt(s) in {int(r.season) - 1}/{str(r.season)[2:]}"))
    sp = SEED / "news" / "set_pieces.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        for r in s[s.role.astype(str).str.contains("penalt", case=False)].itertuples():
            backup = "backup" in str(r.role).lower() or "second" in str(r.role).lower()
            rows.append((r.team, r.name, 50.0 if backup else 100.0, "set_pieces.csv"))
    return pd.DataFrame(rows, columns=["team", "name", "score", "why"])


def takers(players: pd.DataFrame) -> pd.DataFrame:
    """One row per club with evidence: team, first (id), backup (id or None), scores and reasons."""
    from ..ingest.team_news import best_in_team
    ev = evidence()
    # the order already in the table (FPL's penalties_order, notes) counts as some evidence too
    if "pen_order" in players:
        cur = players[players.pen_order == 1]
        ev = pd.concat([ev, pd.DataFrame(dict(team=cur.team, name=cur.name, score=1.5, why="FPL order / notes"))],
                       ignore_index=True)
    if ev.empty:
        return pd.DataFrame(columns=["team", "first", "backup", "score", "why"])
    by_team = {t: g for t, g in players.groupby("team")}
    ids = []
    for r in ev.itertuples():
        cand = by_team.get(r.team)
        ids.append(best_in_team(str(r.name), cand)[0] if cand is not None else None)
    ev = ev.assign(id=ids).dropna(subset=["id"])
    out = []
    for team, g in ev.groupby("team"):
        tot = g.groupby("id").score.sum().sort_values(ascending=False)
        if tot.iloc[0] < MIN_SCORE:
            continue
        first = tot.index[0]
        backup = tot.index[1] if len(tot) > 1 and tot.iloc[1] >= MIN_SCORE else None
        why = "; ".join(g[g.id == first].why.tolist())
        out.append(dict(team=team, first=first, backup=backup, score=round(float(tot.iloc[0]), 2), why=why))
    return pd.DataFrame(out)


def apply(players: pd.DataFrame) -> pd.DataFrame:
    """Set pen_order 1/2 (pen_src 4) where this season's evidence names a taker."""
    t = takers(players)
    if t.empty:
        return players
    players = players.copy()
    for r in t.itertuples():
        on = players.team == r.team
        players.loc[on & (players.pen_order <= 2), "pen_order"] = 9
        players.loc[players.id == r.first, ["pen_order", "pen_src"]] = [1, 4]
        if r.backup:
            players.loc[players.id == r.backup, ["pen_order", "pen_src"]] = [2, 4]
    return players
