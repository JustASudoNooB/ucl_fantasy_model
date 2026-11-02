"""Challenger against the current model on played UCL matchdays (pre-match information only).

For each matchday with actual XIs on file the engine is built as of the matchday's first date,
then run with the network's weight at 0 (current model), 0.5 and 1 (network only). Scored:
  starts      log loss and Brier over every pool player of the clubs that played, and actual
              starters among each club's top 11
  goals       Brier and log loss of P(scores) against the match events (matchdays with events)
  points      correlation between expected points and the points the match events give (ball
              recoveries excluded on both sides)
Writes data/processed/ml_ucl_eval.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import ROOT
from ..evaluate import grade
from ..pipeline import load_fixtures
from .ucl import Challenger

OUT = ROOT / "data" / "processed" / "ml_ucl_eval.json"


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _actual(md: int, eng, players: pd.DataFrame):
    xi, ev = grade.load_xis(md), grade.load_events(md)
    started = {i for i in grade._match_names(xi, players) if i} if xi is not None else set()
    scored, pts = set(), None
    if ev is not None and xi is not None:
        app = grade.appearances(md, eng.fixtures, ev, xi)
        app["id"] = grade._match_names(app, players)
        app = app.dropna(subset=["id"])
        pos = dict(zip(players.id, players.pos))
        fp = grade.fantasy_points(app.set_index("id", drop=False), pos).reset_index(drop=True)
        pts = fp.groupby("id").points_events.sum()
        g = ev[ev.event.isin(["goal", "pen_goal"])]
        scored = {i for i in grade._match_names(g.rename(columns={"player": "name"}), players) if i}
    return started, scored, pts


def run(mds=None, blends=(0.0, 0.5, 1.0), n_sims: int = 6000, verbose: bool = True) -> dict:
    fx = load_fixtures()
    mds = mds or [int(m) for m in sorted(fx.md.unique()) if grade.xis_path(int(m)) is not None]
    out = {}
    for md in mds:
        as_of = str(fx[fx.md == md].date.min())[:10]
        eng = Challenger(as_of=as_of, n_sims=n_sims, blend=0.0)
        teams = set(fx[fx.md == md].home) | set(fx[fx.md == md].away)
        res = {}
        for w in blends:
            eng.blend = float(w)
            eng._xi.clear(); eng._sims.clear(); eng._news.clear()
            pl = eng.players_for(md)
            pl = pl[pl.team.isin(teams)]
            started, scored, pts = _actual(md, eng, pl)
            y = pl.id.isin(started).astype(float)
            r = dict(start_ll=round(_ll(pl.p_start, y), 4), start_brier=round(float(np.mean((pl.p_start - y) ** 2)), 4),
                     top11=int(sum(g.assign(y=y[g.index]).nlargest(11, "p_start").y.sum() for _, g in pl.groupby("team"))),
                     starters=int(y.sum()))
            if scored or pts is not None:
                pr = eng.projections(md).set_index("id")
                pr = pr[pr.index.isin(pl.id)]
                ys = pr.index.isin(scored).astype(float)
                r.update(goal_ll=round(_ll(pr.p_goal, ys), 4), goal_brier=round(float(np.mean((pr.p_goal - ys) ** 2)), 4),
                         goals_exp=round(float(pr.p_goal.sum()), 1), scorers=int(ys.sum()))
                if pts is not None:
                    a = pts.reindex(pr.index).fillna(0.0)
                    xrec_pts = (pr.xrec / 3.0) if "xrec" in pr else 0.0
                    r["points_corr"] = round(float(np.corrcoef(pr.xpts - xrec_pts, a)[0, 1]), 4)
            res[str(w)] = r
            if verbose:
                print(f"MD{md} network weight {w}: {r}", flush=True)
        out[str(md)] = res
    OUT.write_text(json.dumps(out, indent=1))
    return out
