"""Half-matchday re-plan: after one evening's games, refresh the predictions for the games still to
come and decide the captain switch and manual subs for the rest of the matchday.

  1. Refit the team model through the finished day (walk-forward: as_of = the next match date),
     so day-1 results move the ratings used for day 2.
  2. Apply the freshest team news for the clubs still to play (predicted or confirmed XIs).
  3. Score the squad's day-1 players from the match events (goals, assists, clean sheets, cards,
     POTM, minutes; ball recoveries estimated from the model) or from the app's real points.
  4. Compare whole plans (keep auto-subs, move the armband, manual subs, or both).
  5. Freeze the refreshed predictions for the remaining games (snapshot_day{D}.json, write-once
     after that day's first kick-off), so day 2 can be graded twice: as forecast before the
     deadline and as re-forecast at half-time.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, rules
from ..evaluate import grade
from ..ingest import team_news
from ..optimize import policy
from ..timeutil import IST, kickoff_dt
from .export import model_version, _r

RUNS = ROOT / "data" / "runs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def snapshot_day(engine, md: int, day: int, force: bool = False) -> Path:
    """Freeze predictions for one day's games of a matchday (the half-time re-forecast)."""
    p = RUNS / f"md{md}" / f"snapshot_day{day}.json"
    fx = [f for f in engine.match_probabilities(md) if f["day"] == day]
    if not fx:
        raise ValueError(f"MD{md} has no day {day}")
    first = min(kickoff_dt(f["date"], f["kickoff_cet"]) for f in fx).astimezone(timezone.utc)
    if p.exists() and not force and datetime.now(timezone.utc) >= first:
        return p
    sim = engine.simulate(md)
    proj = engine.projections(md)
    teams = {f["home"] for f in fx} | {f["away"] for f in fx}
    idx = sim.index()
    news = team_news.load_for(md, engine.players)
    named = team_news.named_flags(news, engine.players) if news is not None else {}
    lam = {}
    for f in fx:
        lam[f["home"]], lam[f["away"]] = f["xg_home"], f["xg_away"]
    players = []
    for r in proj[proj.team.isin(teams)].itertuples():
        pts = sim.points[idx[r.id]]
        pmf = np.bincount(np.clip(pts + 3, 0, 28).astype(int), minlength=29) / len(pts)
        players.append(dict(id=r.id, name=r.name, team=r.team, pos=r.pos, price=r.price, xpts=_r(r.xpts),
                            p_start_sim=_r(r.p_start_sim), p_goal=_r(r.p_goal), p_assist=_r(r.p_assist), p_cs=_r(r.p_cs),
                            p_potm=_r(r.p_potm), xrec=_r(r.xrec), team_lambda=_r(lam.get(r.team)), named=named.get(r.id),
                            pmf=np.round(pmf, 4).tolist()))
    out = dict(md=md, day=day, created_at=_now(), model_version=model_version(), n_sims=int(sim.meta["n_sims"]),
               as_of=engine.as_of,
               team_model=dict(fitted_through=engine.team_model.fitted_through, n_matches=engine.team_model.n_matches,
                               hyper=engine.team_model.hyper),
               fixtures=fx, players=players)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return p


def realised_points(engine, md: int, day_done: int) -> tuple[pd.DataFrame, set[str]]:
    """Points for players whose games are finished, from the match events.
    Ball recoveries are not in match reports: the model's expected recoveries for the minutes
    actually played are added (floor(x/3)), and the row is flagged 'estimated'."""
    fx = engine.fixtures
    ev, xi = grade.load_events(md), grade.load_xis(md)
    if ev is None or xi is None:
        return pd.DataFrame(), set()
    app = grade.appearances(md, fx, ev, xi)
    players = engine.players
    app["id"] = grade._match_names(app, players)
    pos = {i: p for i, p in zip(players.id, players.pos)}
    app = grade.fantasy_points(app.set_index("id", drop=False), {k: v for k, v in pos.items()})
    app = app.reset_index(drop=True)
    proj = engine.projections(md).set_index("id")
    xrec90 = (proj.xrec / proj.xmins.clip(lower=1) * 90.0) if "xmins" in proj else proj.xrec
    rec = []
    for r in app.itertuples():
        x = float(xrec90.get(r.id, 0.0)) * r.minutes / 90.0 if r.id in xrec90.index else 0.0
        rec.append(int(np.floor(x / rules()["scoring"]["recoveries_step"])))
    app["rec_pts_est"] = rec
    app["points"] = app.points_events + app.rec_pts_est
    sent_off = set(app[(app.red > 0) | (app.second_yellow > 0)].id.dropna())
    return app, sent_off


def run(engine, md: int, day_done: int, squad: list[str], xi: list[str], captain: str,
        points_override: dict[str, float] | None = None, lineups_known: bool | None = None,
        freeze: bool = True, label: str = "dashboard squad", out_name: str | None = None) -> dict:
    players = engine.players_for(md)
    names = players.set_index("id").name
    sim = engine.simulate(md)
    idx = sim.index()
    day_of = {p: int(sim.day[idx[p]]) for p in squad}
    app, sent_off = realised_points(engine, md, day_done)
    real: dict[str, float] = {}
    detail = {}
    if len(app):
        a = app.dropna(subset=["id"]).set_index("id")
        for p in squad:
            if day_of[p] <= day_done:
                if p in a.index:
                    row = a.loc[p]
                    row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                    real[p] = float(row.points)
                    detail[p] = dict(minutes=int(row.minutes), goals=int(row.goal + row.pen_goal), assists=int(row.assist),
                                     potm=bool(row.potm), red=bool(row.red or row.second_yellow), yellow=int(row.yellow),
                                     conceded=int(row.conceded_on), points_events=_r(row.points_events, 0),
                                     rec_pts_est=int(row.rec_pts_est))
                else:
                    real[p] = 0.0
                    detail[p] = dict(minutes=0, note="did not play")
    if points_override:
        real.update(points_override)
        for p in points_override:
            detail.setdefault(p, {})["source"] = "app points"
    bench = [p for p in squad if p not in xi]
    lk = lineups_known if lineups_known is not None else False
    dec = policy.live_decisions(sim, players, squad, xi, captain, bench, md, day_done, real,
                                lineups_known=lk, sent_off=sent_off & set(squad))
    proj = engine.projections(md).set_index("id")
    rows = []
    for p in squad:
        later = day_of[p] > day_done
        rows.append(dict(id=p, name=names[p], team=players.set_index("id").team[p], pos=players.set_index("id").pos[p],
                         day=day_of[p], in_xi=p in xi, captain=p == captain,
                         realised=None if later else _r(real.get(p, 0.0), 0),
                         expected=_r(proj.xpts.get(p), 2) if later else None,
                         p_start=_r(proj.p_start_sim.get(p), 2) if later else None,
                         detail=detail.get(p), sent_off=p in sent_off))
    day_total = sum(real.get(p, 0.0) for p in xi if day_of[p] <= day_done) + (real.get(captain, 0.0) if day_of[captain] <= day_done else 0.0)
    snap = None
    if freeze:
        snap = snapshot_day(engine, md, day_done + 1).name
    out = dict(md=md, after_day=day_done, created_at=_now(), as_of=engine.as_of, model_version=model_version(),
               label=label, squad=rows, captain=captain, xi=xi, bench=bench, day_points=_r(day_total, 0),
               decisions=dec, lineups_known=lk, snapshot=snap,
               note="Day-1 points come from match reports plus the model's estimate of ball-recovery points; "
                    "replace them with the app's numbers (--points) when they differ.")
    p = RUNS / f"md{md}" / (out_name or f"replan_day{day_done + 1}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, default=str))
    return out
