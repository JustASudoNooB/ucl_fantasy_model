"""Snapshots (pre-deadline predictions, used later by the feedback loop) and the dashboard data."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import team_news
from .. import __version__
from ..config import ROOT, rules, settings, team_lookup
from ..optimize import chips as chips_mod
from ..optimize import policy as policy_mod
from ..optimize import price_check
from ..optimize.squad import best_squad, best_xi
from ..rules import budget, chip_allowed, club_cap, formation_label, formations, manual_actions_allowed
from ..timeutil import IST, deadline_for, kickoff_dt

RUNS = ROOT / "data" / "runs"
PROC = ROOT / "data" / "processed"


def model_version() -> str:
    """Code version + hash of the fitted params and settings (any recalibration changes it)."""
    sha = hashlib.sha1()
    for p in (PROC / "params.json", ROOT / "config" / "settings.yaml"):
        if p.exists():
            sha.update(p.read_bytes())
    return f"{__version__}+{sha.hexdigest()[:7]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def snapshot(engine, md: int, force: bool = False) -> Path:
    """Freeze this matchday's predictions before the deadline (honest walk-forward scoring).
    Once the deadline has passed an existing snapshot is never overwritten (unless force=True),
    so later rebuilds of the dashboard cannot leak results into the grading."""
    p = RUNS / f"md{md}" / "snapshot.json"
    if p.exists() and not force:
        r = next((m for m in rules()["matchdays"] if m["md"] == md), None)
        if r is not None and datetime.now(timezone.utc) >= deadline_for(r).astimezone(timezone.utc):
            return p
    out = snapshot_dict(engine, md)
    d = RUNS / f"md{md}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "snapshot.json"
    p.write_text(json.dumps(out))
    return p


def retro_snapshot(engine, md: int, tag: str) -> Path:
    """A re-forecast of a played matchday by today's code, using only pre-match information
    (engine.as_of = the matchday's first date, no confirmed line-ups). Graded alongside the frozen
    snapshot to show whether a model change would have helped; never a substitute for it."""
    out = snapshot_dict(engine, md)
    out["retro"] = dict(tag=tag, as_of=engine.as_of, note="built after the matches with pre-match inputs only")
    d = RUNS / f"md{md}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"retro_{tag}.json"
    p.write_text(json.dumps(out))
    return p


def snapshot_dict(engine, md: int) -> dict:
    sim = engine.simulate(md)
    proj = engine.projections(md)
    fx = engine.match_probabilities(md)
    lam = {}
    for f in fx:
        lam[f["home"]] = f["xg_home"]
        lam[f["away"]] = f["xg_away"]
    idx = sim.index()
    news = team_news.load_for(md, engine.players)
    named = team_news.named_flags(news, engine.players) if news is not None else {}
    players = []
    for r in proj.itertuples():
        pts = sim.points[idx[r.id]]
        pmf = np.bincount(np.clip(pts + 3, 0, 28).astype(int), minlength=29) / len(pts)
        players.append(dict(id=r.id, name=r.name, team=r.team, pos=r.pos, price=r.price, xpts=_r(r.xpts),
                            p_start_sim=_r(r.p_start_sim), p_goal=_r(r.p_goal), p_assist=_r(r.p_assist), p_cs=_r(r.p_cs),
                            p_potm=_r(r.p_potm), xrec=_r(r.xrec), team_lambda=_r(lam.get(r.team)), named=named.get(r.id),
                            pmf=np.round(pmf, 4).tolist()))
    return dict(md=md, created_at=_now(), model_version=model_version(), n_sims=int(sim.meta["n_sims"]),
                as_of=engine.as_of,
                team_model=dict(fitted_through=engine.team_model.fitted_through, n_matches=engine.team_model.n_matches,
                                hyper=engine.team_model.hyper),
                fixtures=fx, players=players)


def _news_by_team(md: int) -> dict:
    p = team_news.news_path(md)
    if p is None:
        return {}
    n = team_news.parse(p)
    out: dict = {}
    for t, g in n.groupby("team"):
        out[t] = {k: g[g.status == k].name.tolist() for k in ("out", "doubt", "susp")}
    return out


def _fixture_detail(engine, md: int, fx: list[dict], proj: pd.DataFrame, sim) -> list[dict]:
    players = engine.players_for(md).set_index("id")
    news_by_team = _news_by_team(md)
    out = []
    for fi, f in enumerate(fx):
        rows = proj[proj.team.isin([f["home"], f["away"]])]
        scorers = rows.sort_values("p_goal", ascending=False).head(8)
        potm = rows.sort_values("p_potm", ascending=False).head(5)
        lineups = {}
        for side in ("home", "away"):
            t = f[side]
            pl = players[players.team == t].sort_values("p_start", ascending=False)
            lineups[side] = [dict(id=i, name=r["name"], pos=r["pos"], p_start=_r(r["p_start"], 2), price=r["price"])
                             for i, r in pl.iterrows()]
        d = dict(f)
        d["scorers"] = [dict(id=r.id, name=r.name, team=r.team, p_goal=_r(r.p_goal), p_assist=_r(r.p_assist)) for r in scorers.itertuples()]
        assisters = rows.sort_values("p_assist", ascending=False).head(6)
        d["assisters"] = [dict(id=r.id, name=r.name, team=r.team, p_assist=_r(r.p_assist), p_goal=_r(r.p_goal)) for r in assisters.itertuples()]
        d["potm"] = [dict(id=r.id, name=r.name, team=r.team, p=_r(r.p_potm)) for r in potm.itertuples()]
        d["potm_unlisted"] = _r(float(sim.potm_other[fi].mean()))
        d["lineups"] = lineups
        d["news"] = {side: news_by_team.get(f[side], {}) for side in ("home", "away")}
        sc = sim.scores[fi]
        d["sim_goals"] = [_r(sc[:, 0].mean(), 2), _r(sc[:, 1].mean(), 2)]
        out.append(d)
    return out


def _news_changes(engine, md: int, k: int = 40) -> list[dict]:
    info = engine._news.get(md) or {}
    log = info.get("log")
    if log is None or len(log) == 0:
        return []
    log = log.assign(delta=(log.p_start_after - log.p_start_before).abs()).sort_values("delta", ascending=False)
    ev_ids = set(engine.projections(md).sort_values("xpts", ascending=False).head(150).id)
    log = log[log.id.isin(ev_ids) | (log.p_start_before > 0.6)]
    return [dict(id=r.id, name=r.name, team=r.team, before=_r(r.p_start_before, 2), after=_r(r.p_start_after, 2),
                 news=r.news) for r in log.head(k).itertuples()]


def matchday_block(engine, md: int, all_formations: bool = True, with_checks: bool = False) -> dict:
    sim = engine.simulate(md)
    proj = engine.projections(md)
    players = engine.players_for(md)
    ev = proj.set_index("id").xpts
    fx = engine.match_probabilities(md)
    md_row = next(r for r in rules()["matchdays"] if r["md"] == md)
    dl = deadline_for(md_row)
    best = best_squad(players, ev, budget(md), club_cap(md))
    # pick the captain by in-matchday policy value among the top candidates
    cap_best, pol_best, cap_scores = policy_mod.best_captain(sim, players, best.squad, best.xi, best.bench, md, ev)
    by_form = {}
    if all_formations:
        for f in formations():
            s = best_squad(players, ev, budget(md), club_cap(md), formation=f)
            by_form[formation_label(f)] = s.to_dict()
    ll = best_squad(players, ev, None, club_cap(md)).to_dict() if chip_allowed("limitless", md) else None
    # the same optimum restricted to published prices, and break-even prices for the rest
    unverified = players[~players.price_status.isin(["verified", "official"])].id.tolist() if "price_status" in players else []
    verified = None
    if any(i in unverified for i in best.squad):
        v = best_squad(players, ev, budget(md), club_cap(md), banned=unverified)
        verified = dict(v.to_dict(), captain=str(ev.reindex(v.xi).idxmax()))
    checks = price_check.run(players, ev, budget(md), club_cap(md), best, per_pos=3) if with_checks else []
    cols = ["id", "name", "team", "pos", "price", "xpts", "p10", "p50", "p90", "p_haul", "p_blank", "p_start_sim",
            "p_play", "xpts_if_start", "p_goal", "p_assist", "xgoals", "xassists", "xrec", "p_cs", "p_potm", "day", "value"]
    src = players.set_index("id").rate_source
    pst = players.set_index("id").price_status if "price_status" in players else pd.Series(dtype=str)
    plist = []
    for r in proj[cols].itertuples(index=False):
        d = {k: (_r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
             for k, v in r._asdict().items()}
        d["src"] = src.get(r.id, "")
        d["ps"] = pst.get(r.id, "verified")
        plist.append(d)
    captains = proj.sort_values("xpts", ascending=False).head(12)
    return dict(
        md=md, stage=md_row["stage"], deadline_cet=dl.strftime("%a %d %b %Y, %H:%M CET"),
        deadline_ist=dl.astimezone(IST).strftime("%a %d %b %Y, %H:%M IST"), deadline_iso=dl.isoformat(),
        manual_actions=manual_actions_allowed(md), days=sorted({f["day"] for f in fx}),
        fixtures=_fixture_detail(engine, md, fx, proj, sim), players=plist,
        best=dict(best.to_dict(), captain=cap_best, captain_policy=cap_scores, policy=pol_best.to_dict()),
        by_formation=by_form, limitless=ll, verified=verified, price_checks=checks,
        news_changes=_news_changes(engine, md),
        captains=[dict(id=r.id, name=r.name, team=r.team, pos=r.pos, xpts=_r(r.xpts), p90=_r(r.p90), p_haul=_r(r.p_haul),
                       day=int(r.day), p_goal=_r(r.p_goal)) for r in captains.itertuples()],
    )


BASE_RATES = (0.505, 0.159, 0.336)     # UCL 2024/25-2025/26 home / draw / away share (378 matches)


def graded_matchdays(engine) -> list[dict]:
    """Score every frozen pre-deadline snapshot against the results played so far."""
    out = []
    fx = engine.fixtures
    for md in sorted(fx.md.unique()):
        p = RUNS / f"md{int(md)}" / "snapshot.json"
        done = fx[(fx.md == md) & (fx.status == "FT")]
        if not p.exists() or done.empty:
            continue
        snap = json.loads(p.read_text())
        res = {(r.home, r.away): (int(r.home_goals), int(r.away_goals)) for r in done.itertuples()}
        rows, ll, llb, rps, rpsb, hit = [], [], [], [], [], []
        for f in snap.get("fixtures", []):
            k = (f["home"], f["away"])
            if k not in res:
                continue
            hg, ag = res[k]
            y = 0 if hg > ag else (1 if hg == ag else 2)
            pr = np.array([f["p_home"], f["p_draw"], f["p_away"]], dtype=float)
            base = np.array(BASE_RATES)
            o = np.cumsum(np.eye(3)[y])[:2]
            ll.append(-np.log(max(pr[y], 1e-9)))
            llb.append(-np.log(base[y]))
            rps.append(float(np.sum((np.cumsum(pr)[:2] - o) ** 2) / 2))
            rpsb.append(float(np.sum((np.cumsum(base)[:2] - o) ** 2) / 2))
            hit.append(int(np.argmax(pr) == y))
            rows.append(dict(home=f["home"], away=f["away"], score=f"{hg}-{ag}", p=[_r(x, 3) for x in pr],
                             p_result=_r(pr[y], 3), xg=[f.get("xg_home"), f.get("xg_away")], hit=bool(np.argmax(pr) == y)))
        if rows:
            out.append(dict(md=int(md), snapshot_created=snap.get("created_at"), model_version=snap.get("model_version"),
                            n=len(rows), log_loss=_r(np.mean(ll), 3), log_loss_base=_r(np.mean(llb), 3),
                            rps=_r(np.mean(rps), 3), rps_base=_r(np.mean(rpsb), 3), correct=int(sum(hit)), rows=rows))
    return out


def dashboard_data(engine, mds: list[int], backtest: dict | None = None, standings: pd.DataFrame | None = None) -> dict:
    engine._mds = mds
    tl = team_lookup()
    tm = engine.team_model
    teams = {}
    for t, info in tl.items():
        a, d, _, _ = tm._ad(t)
        teams[t] = dict(name=info["name"], short=info["short"], country=info["country"], rating=_r(tm.rating(t)),
                        attack=_r(a), defence=_r(d), uncertainty=_r(tm.uncertainty(t)))
    blocks = {str(md): matchday_block(engine, md, with_checks=(i == 0)) for i, md in enumerate(mds)}
    E = engine.ev_matrix(mds)
    cols = ["id", "name", "team", "pos", "price"] + [c for c in ("price_status", "own_pct", "status") if c in engine.players]
    idx_players = engine.players[cols].rename(columns={"price_status": "ps", "own_pct": "own", "status": "st"})
    idx_players = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                   for r in idx_players.to_dict(orient="records")]
    ev = {pid: [_r(E.loc[pid, md], 2) for md in mds] for pid in E.index}
    days = {}
    for md in mds:
        p = engine.projections(md).set_index("id").day
        days[str(md)] = {pid: int(v) for pid, v in p.items()}
    league_mds = [m for m in mds if m <= 8]
    chips = dict(limitless=chips_mod.limitless_targets(engine.players, E, league_mds),
                 wildcard=chips_mod.wildcard_targets(engine.players, E, league_mds, horizon=3))
    sc = rules()["scoring"]
    deadlines = []
    for r in rules()["matchdays"]:
        dl = deadline_for(r)
        deadlines.append(dict(md=r["md"], stage=r["stage"], cet=dl.strftime("%a %d %b, %H:%M"),
                              ist=dl.astimezone(IST).strftime("%a %d %b, %H:%M"), iso=dl.isoformat(),
                              free=rules()["transfers"]["by_md"][r["md"]]["free"],
                              chips_ok=chip_allowed("wildcard", r["md"]), manual=manual_actions_allowed(r["md"])))
    pool_src = engine.pool.get("source", pd.Series(["seed"])).value_counts().to_dict()
    return dict(
        meta=dict(generated_at=_now(), as_of=engine.as_of, model_version=model_version(),
                  n_sims=int(engine.simulate(mds[0]).meta["n_sims"]), team_model_matches=tm.n_matches,
                  team_model_fitted_through=tm.fitted_through, hyper=tm.hyper, pool_size=int(len(engine.players)),
                  price_status=engine.players.get("price_status", pd.Series(["verified"])).value_counts().to_dict(),
                  news_files={str(m): (str(team_news.news_path(m).name) if team_news.news_path(m) else None) for m in mds},
                  pool_sources=pool_src, mds=mds),
        rules=dict(scoring=sc, deadlines=deadlines, budget=rules()["squad"]["budget"], club_cap=rules()["club_cap"],
                   unknowns=rules()["unknowns"]),
        teams=teams, matchdays=blocks, players_index=idx_players, ev=ev, ev_mds=mds, days=days, chips=chips,
        standings=standings.round(4).to_dict(orient="records") if standings is not None else [],
        backtest=backtest or {},
        graded=graded_matchdays(engine),
        verification=_load_json(PROC / "verification.json"),
        changes=_load_json(PROC / "changes.json"),
        **_extras(engine),
    )


def _managers(engine):
    from .managers import all_managers
    return all_managers(engine, list(getattr(engine, "_mds", [])))


def _extras(engine) -> dict:
    from . import insights
    out = {}
    for key, fn in (("grading", lambda: insights.grading(engine)), ("replan", insights.replans),
                    ("live", lambda: insights.live_points(engine)), ("form_players", lambda: insights.player_form(engine)),
                    ("team_records", lambda: insights.team_records(engine)), ("form_tests", insights.backtests),
                    ("managers", lambda: _managers(engine))):
        try:
            out[key] = fn()
        except Exception as e:                       # the dashboard must still build
            out[key] = {} if key != "form_players" else []
            out.setdefault("errors", []).append(f"{key}: {e}")
    return out


def _load_json(p) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return {}
