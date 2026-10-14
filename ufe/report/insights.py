"""Dashboard extras: match-by-match grading, the half-time re-plan, day-1 points for the in-browser
half-time planner, domestic form per player, and home/away records per club."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, team_lookup
from ..evaluate import grade
from .export import _r

RUNS = ROOT / "data" / "runs"
PROC = ROOT / "data" / "processed"
CUPS = {"cl", "el", "conf", "ucl_current"}


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None


def grading(engine) -> dict:
    """{md: [graded snapshot, ...]} for every matchday with finished games."""
    out = {}
    fx = engine.fixtures
    for md in sorted(fx[fx.status == "FT"].md.unique()):
        g = grade.grade_md(int(md), fx)
        if g:
            out[str(int(md))] = g
    return out


def replans() -> dict:
    out = {}
    for p in sorted(RUNS.glob("md*/replan_day*.json")):
        d = _load(p)
        if d:
            out.setdefault(str(d["md"]), {})[str(d["after_day"] + 1)] = d
    return out


def live_points(engine) -> dict:
    """{md: {player id: {pts, min, g, a, potm, off, est}}} for finished games (event points plus the
    model's estimate of ball-recovery points). Feeds the half-time planner in the browser."""
    from .replan import realised_points
    out = {}
    fx = engine.fixtures
    for md in sorted(fx[fx.status == "FT"].md.unique()):
        md = int(md)
        try:
            app, _ = realised_points(engine, md, 1)
        except Exception:
            continue
        if app is None or not len(app):
            continue
        rows = {}
        for r in app.dropna(subset=["id"]).itertuples():
            rows[r.id] = dict(pts=_r(r.points, 0), min=int(r.minutes), g=int(r.goal + r.pen_goal), a=int(r.assist),
                              potm=bool(r.potm), off=bool(r.red or r.second_yellow), rec=int(r.rec_pts_est))
        days_done = sorted({int(d) for d in engine.fixture_table(md).query("status == 'FT'").day})
        teams_done = sorted(set(fx[(fx.md == md) & (fx.status == "FT")].home) | set(fx[(fx.md == md) & (fx.status == "FT")].away))
        out[str(md)] = dict(players=rows, days_done=days_done, teams_done=teams_done)
    return out


def player_form(engine, k: int = 400) -> list[dict]:
    """This season's domestic-league numbers next to what the model expected from history."""
    p = engine.players
    if "dom_m" not in p:
        return []
    f = p[p.dom_m.notna()].copy()
    rows = []
    for r in f.itertuples():
        xg90 = getattr(r, "dom_xg90", np.nan)
        prev = getattr(r, "prev_xg90", np.nan)
        rows.append(dict(id=r.id, name=r.name, team=r.team, pos=r.pos, m=int(r.dom_m), starts=int(r.dom_starts),
                         tm=int(r.dom_team_matches), min=int(r.dom_min), xg=_r(r.dom_xg, 2), a=int(r.dom_assists),
                         xg90=_r(xg90, 2), prev_xg90=_r(prev, 2), p_start=_r(r.p_start, 2),
                         start_rate=_r(r.obs_start_rate, 2)))
    rows.sort(key=lambda d: -(d["min"] or 0))
    return rows[:k]


def team_records(engine, days: int = 365) -> dict:
    """Per UCL club: domestic home and away records over the last `days`, this season's UCL games,
    and the last six results in all competitions."""
    m = engine.matches
    as_of = pd.Timestamp(engine.as_of)
    lo = as_of - pd.Timedelta(days=days)
    m = m[(m.date < as_of)]
    out = {}
    for t in team_lookup():
        rows = m[(m.home_id == t) | (m.away_id == t)].sort_values("date")
        if rows.empty:
            continue
        dom = rows[~rows.comp.isin(CUPS) & (rows.date >= lo)]

        def rec(df, home: bool):
            d = df[df.home_id == t] if home else df[df.away_id == t]
            gf = d.hg if home else d.ag
            ga = d.ag if home else d.hg
            w, dr = int((gf > ga).sum()), int((gf == ga).sum())
            n = len(d)
            return dict(n=n, w=w, d=dr, l=n - w - dr, gf=int(gf.sum()), ga=int(ga.sum()),
                        ppg=_r((3 * w + dr) / n, 2) if n else None)

        last = []
        for r in rows.tail(6).itertuples():
            home = r.home_id == t
            gf, ga = (r.hg, r.ag) if home else (r.ag, r.hg)
            opp = r.away_id if home else r.home_id
            res = "W" if gf > ga else ("D" if gf == ga else "L")
            last.append(dict(date=str(r.date.date()), ha="H" if home else "A", opp=str(opp).split(":")[-1].title() if ":" in str(opp) else str(opp),
                             score=f"{int(gf)}-{int(ga)}", res=res, comp="UCL" if r.comp in CUPS else "league"))
        ucl = rows[rows.comp == "ucl_current"]
        out[t] = dict(home=rec(dom, True), away=rec(dom, False), ucl_home=rec(ucl, True), ucl_away=rec(ucl, False),
                      last=last, last_domestic=str(dom.date.max().date()) if len(dom) else None)
    return out


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def backtests() -> dict:
    return _clean(dict(team=_load(PROC / "backtest_form_homeaway.json") or {}, player=_load(PROC / "backtest_player_form.json") or {}))
