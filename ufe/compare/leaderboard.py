"""Grade every model on the matchdays already played, with pre-match information only.

For each matchday with actual starting XIs on file, every model is rebuilt as of the matchday's
first date (today's code, the news and odds files saved before that matchday) and scored on:
  starts     log loss and Brier over every listed player of the clubs that played; actual
             starters among each club's 11 most likely starters
  goals      log loss and Brier of P(scores) against the match events
  points     against UEFA's own points for every player in the matchday (feed_md_points.csv):
             correlation, root-mean-square error, and the actual points of each model's 20
             highest-expected players
  matches    result log loss (home/draw/away) and total goals expected vs scored
A matchday without line-ups or events on file is still graded on points and results. The model's
constants (recoveries, assists by position, start cap) are refitted on the matchdays played, so
those matchdays are in-sample for them. A model is not graded on a matchday it cannot predict (the bookmaker model needs prices for at
least half the matches). Saved deadline forecasts (forecast.py) are scored against the points
each manager reports (md{N}.points in the manager file).
Writes data/processed/model_leaderboard.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT
from ..evaluate import grade
from ..ml.evaluate import _actual, _ll
from ..pipeline import Engine, load_fixtures
from . import registry
from .forecast import MANAGERS, RUNS

OUT = ROOT / "data" / "processed" / "model_leaderboard.json"
LOWER_BETTER = {"start_ll", "start_brier", "goal_ll", "goal_brier", "result_ll", "goals_abs_err", "points_rmse"}
METRICS = ["start_ll", "top11_share", "goal_ll", "points_corr", "points_rmse", "top20_points", "result_ll", "goals_abs_err"]


def _match_scores(e, md: int, fx: pd.DataFrame) -> dict:
    done = fx[(fx.md == md) & fx.home_goals.notna()]
    if done.empty:
        return {}
    res = {(r.home, r.away): (int(r.home_goals), int(r.away_goals)) for r in done.itertuples()}
    ll, exp, act = [], 0.0, 0
    for m in e.match_probabilities(md):
        k = (m["home"], m["away"])
        if k not in res:
            continue
        hg, ag = res[k]
        y = 0 if hg > ag else (1 if hg == ag else 2)
        ll.append(-np.log(max([m["p_home"], m["p_draw"], m["p_away"]][y], 1e-9)))
        exp += m["xg_home"] + m["xg_away"]
        act += hg + ag
    return dict(result_ll=round(float(np.mean(ll)), 4), goals_exp=round(exp, 1), goals_actual=act,
                goals_abs_err=round(abs(exp - act), 1), n_matches=len(ll))


def grade_md(md: int, keys: list[str], n_sims: int = 6000, verbose: bool = True) -> dict:
    fx = load_fixtures()
    as_of = str(fx[fx.md == md].date.min())[:10]
    teams = set(fx[fx.md == md].home) | set(fx[fx.md == md].away)
    base = Engine(as_of=as_of, n_sims=n_sims)
    out = {}
    for k in keys:
        try:
            e = registry.build(k, as_of, n_sims=n_sims, base=base)
            if k == "bookmaker":
                cov = e.odds_coverage(md)
                if cov < 0.5:
                    out[k] = dict(skipped=f"no bookmaker prices on file for MD{md}" if cov == 0 else
                                  f"bookmaker prices for only {cov:.0%} of MD{md}")
                    continue
            pl = e.players_for(md)
            pl = pl[pl.team.isin(teams)]
            r = {}
            has_xi = grade.xis_path(md) is not None
            started, scored, pts_ev = _actual(md, e, pl) if has_xi else (set(), set(), None)
            if has_xi:
                y = pl.id.isin(started).astype(float)
                n11 = int(sum(g.assign(y=y[g.index]).nlargest(11, "p_start").y.sum() for _, g in pl.groupby("team")))
                r.update(start_ll=round(_ll(pl.p_start, y), 4), start_brier=round(float(np.mean((pl.p_start - y) ** 2)), 4),
                         top11=n11, starters=int(y.sum()), top11_share=round(n11 / max(int(y.sum()), 1), 4))
            pr = e.projections(md).set_index("id")
            pr = pr[pr.index.isin(pl.id)]
            if scored:
                ys = pr.index.isin(scored).astype(float)
                r.update(goal_ll=round(_ll(pr.p_goal, ys), 4), goal_brier=round(float(np.mean((pr.p_goal - ys) ** 2)), 4),
                         scorers_exp=round(float(pr.p_goal.sum()), 1), scorers=int(ys.sum()))
            # points: UEFA's own matchday points for every player when on file (recoveries and awards
            # included on both sides), else the points the match events give (recoveries left out)
            off = grade.feed_actual_points(md)
            if off is not None:
                q = pr[pr.index.isin(off.index)]
                a = off.reindex(q.index).astype(float)
                x = q.xpts
                r["points_source"] = "UEFA matchday points"
            elif pts_ev is not None:
                a = pts_ev.reindex(pr.index).fillna(0.0)
                x = pr.xpts - ((pr.xrec / 3.0) if "xrec" in pr else 0.0)
                r["points_source"] = "match events (no ball recoveries)"
            else:
                a = x = None
            if a is not None and len(a) > 20:
                r["points_corr"] = round(float(np.corrcoef(x, a)[0, 1]), 4)
                r["points_rmse"] = round(float(np.sqrt(np.mean((x - a) ** 2))), 4)
                r["points_bias"] = round(float(np.mean(x - a)), 4)
                r["top20_points"] = int(a[x.sort_values(ascending=False).index[:20]].sum())
                r["n_players"] = int(len(a))
            r.update(_match_scores(e, md, fx))
            out[k] = r
        except Exception as ex:                       # one model failing must not stop the others
            out[k] = dict(error=f"{type(ex).__name__}: {str(ex)[:200]}")
        if verbose:
            print(f"MD{md} {k}: {out[k]}", flush=True)
    return out


def summarise(by_md: dict, keys: list[str]) -> dict:
    """Per score: each model's average over the matchdays every graded model has that score for
    (a fair head-to-head; the matchdays used are listed), the best model on it, and how many
    (matchday, score) pairs each model won."""
    avg, wins, best, used = {k: {} for k in keys}, {k: 0 for k in keys}, {}, {}
    for m in METRICS:
        have = {k: {md for md, r in by_md.items() if m in (r.get(k) or {})} for k in keys}
        have = {k: v for k, v in have.items() if v}
        if not have:
            continue
        common = sorted(set.intersection(*have.values()), key=int)
        if common:
            used[m] = [int(x) for x in common]
            for k in have:
                avg[k][m] = round(float(np.mean([by_md[md][k][m] for md in common])), 4)
                avg[k][m + "_n"] = len(common)
            if len(have) >= 2:
                got = {k: avg[k][m] for k in have}
                best[m] = (min if m in LOWER_BETTER else max)(got, key=got.get)
        for r in by_md.values():
            got = {k: r[k][m] for k in keys if m in (r.get(k) or {})}
            if len(got) >= 2:
                top = (min if m in LOWER_BETTER else max)(got.values())
                tied = [k for k, v in got.items() if abs(v - top) < 1e-9]
                for k in tied:                        # a tie splits the win (the bookmaker shares the
                    wins[k] += 1.0 / len(tied)        # statistical model's starts, so they often tie)
    return dict(average=avg, best=best, matchdays_used=used, wins={k: round(v, 2) for k, v in wins.items()})


def forecast_scores() -> list[dict]:
    """Saved deadline forecasts against the points each manager reported."""
    rows = []
    for p in sorted(RUNS.glob("md*/forecast_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if "models" not in d:
            continue
        f = MANAGERS / f"{d['manager']}.yaml"
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else {}
        actual = ((cfg or {}).get(f"md{d['md']}") or {}).get("points")
        row = dict(md=d["md"], manager=d["manager"], made_at=d.get("made_at"), pre_deadline=d.get("pre_deadline"),
                   actual=actual, models={})
        for k, v in d["models"].items():
            if "expected" in v:
                row["models"][k] = dict(expected=v["expected"], expected_no_changes=v.get("expected_no_changes"),
                                        error=(round(float(actual) - v["expected"], 1) if actual is not None else None),
                                        p10=v["percentiles"][0], p90=v["percentiles"][-1])
        rows.append(row)
    return rows


def run(mds: list[int] | None = None, n_sims: int = 6000, verbose: bool = True) -> dict:
    import os
    prev = os.environ.get("UFE_FROZEN_INPUTS")
    os.environ["UFE_FROZEN_INPUTS"] = "1"           # news and prices as they were at each deadline
    try:
        return _run(mds, n_sims, verbose)
    finally:
        if prev is None:
            os.environ.pop("UFE_FROZEN_INPUTS", None)
        else:
            os.environ["UFE_FROZEN_INPUTS"] = prev


def _run(mds: list[int] | None, n_sims: int, verbose: bool) -> dict:
    fx = load_fixtures()
    keys = registry.available()
    done = lambda m: grade.xis_path(m) is not None or (fx[fx.md == m].home_goals.notna().all() and grade.feed_actual_points(m) is not None)
    mds = mds or [int(m) for m in sorted(fx.md.unique()) if done(int(m))]
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    by_md = {k: v for k, v in (old.get("mds") or {}).items() if int(k) not in mds}
    for md in mds:
        by_md[str(md)] = grade_md(md, keys, n_sims=n_sims, verbose=verbose)
    by_md = dict(sorted(by_md.items(), key=lambda kv: int(kv[0])))
    out = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), models=keys,
               method="today's code, rebuilt as of each matchday's first date with the team news and prices saved before its deadline",
               mds=by_md, summary=summarise(by_md, keys), forecasts=forecast_scores())
    OUT.write_text(json.dumps(out, indent=1))
    return out
