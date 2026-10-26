"""Walk-forward test of goal-level terms in the team model (MD3 review, 26 Oct 2026).

Through three matchdays the 2026/27 UCL has 4.09 goals a game against 3.44 last season, and the
top domestic leagues score about 11% more than last season. The model's single intercept moves
slowly (500-day half-life), so it under-forecast goals on every matchday (MD2 78 v 59, MD3 74 v 63).

Configurations add log-scale level shifts with N(0, sd^2) priors (see team_ratings.fit):
  season      every match of the current season (July to June)
  cup         every UEFA match (all seasons)
  cup_season  this season's UEFA matches

Scored on (a) UCL 2024/25 + 2025/26, refit before every round, (b) UCL 2026/27 MD1-MD3,
(c) top-flight domestic matches 2025/26 and 2026/27 to date, refit every 14 days.
Metrics: result log loss and RPS, total-goals log score (Poisson), mean goal bias
(forecast minus actual goals per match) and over-2.5 Brier.

  python scripts/backtest_goal_level.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufe.config import settings                      # noqa: E402
from ufe.evaluate import metrics as M                # noqa: E402
from ufe.evaluate.backtest import backtest_team_model  # noqa: E402
from ufe.ingest.openfootball import add_current_ucl, load_matches  # noqa: E402
from ufe.models.scoreline import match_probs         # noqa: E402
from ufe.models.team_ratings import fit              # noqa: E402
from ufe.pipeline import load_fixtures               # noqa: E402

DOMESTIC = ["en.1", "es.1", "de.1", "it.1", "fr.1", "pt.1", "nl.1", "be.1", "tr.1", "at.1", "gr.1", "sco.1"]
DF = None


def _df():
    global DF
    if DF is None:
        DF = add_current_ucl(load_matches(), load_fixtures())
    return DF


def _goal_metrics(pred: pd.DataFrame) -> dict:
    lam = (pred.xg_home + pred.xg_away).to_numpy()
    tot = (pred.hg + pred.ag).to_numpy()
    p_over = 1 - poisson.cdf(2, lam)
    return dict(goal_bias=float(np.mean(lam - tot)), goal_logscore=float(-np.mean(poisson.logpmf(tot, lam))),
                over25_brier=float(np.mean((p_over - (tot > 2.5)) ** 2)), goals_pred=float(lam.sum()),
                goals_act=float(tot.sum()))


def domestic_preds(season: str, step_days: int = 14, start_after_days: int = 45, **kw) -> pd.DataFrame:
    df = _df()
    tgt = df[(df.season == season) & df.comp.isin(DOMESTIC)].sort_values("date")
    if tgt.empty:
        return tgt
    start = tgt.date.min() + pd.Timedelta(days=start_after_days)
    end = tgt.date.max() + pd.Timedelta(days=1)
    rows, t = [], start
    while t < end:
        batch = tgt[(tgt.date >= t) & (tgt.date < t + pd.Timedelta(days=step_days))]
        if len(batch):
            m = fit(df, as_of=t, **kw)
            for r in batch.itertuples():
                grp = m.groups.get(r.home_id, "EUR")
                lh, la = m.lambdas(r.home_id, r.away_id, neutral=bool(r.neutral), comp_group=grp)
                sh, sa = m.log_sd(r.home_id, r.away_id)
                mp = match_probs(lh, la, m.rho, sh=sh, sa=sa)
                rows.append(dict(hg=r.hg, ag=r.ag, xg_home=lh, xg_away=la, p_home=mp.p_home, p_draw=mp.p_draw,
                                 p_away=mp.p_away))
        t += pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)


def _score(pred: pd.DataFrame) -> dict:
    y = M.outcome_index(pred.hg.to_numpy(), pred.ag.to_numpy())
    P = pred[["p_home", "p_draw", "p_away"]].to_numpy()
    return dict(n=int(len(pred)), log_loss=M.log_loss(P, y), rps=M.rps(P, y), **_goal_metrics(pred))


def run_config(cfg: dict) -> dict:
    tm = settings()["team_model"]
    base = dict(half_life_days=tm["half_life_days"], european_weight=tm["european_weight"],
                prior_sd_team=tm["prior_sd_team"], prior_sd_league=tm["prior_sd_league"],
                current_ucl_weight=tm.get("current_ucl_weight", 2.0), form_mix=float(tm.get("form_mix", 0.0)),
                form_half_life=float(tm.get("form_half_life", 60.0)))
    kw = {**base, **cfg}
    out = dict(cfg=json.dumps(cfg))
    hist = pd.concat([backtest_team_model(_df(), s, **kw)[0] for s in ("2024-25", "2025-26")], ignore_index=True)
    cur = backtest_team_model(_df(), "2026-27", comp="ucl_current", **kw)[0]
    dom = pd.concat([domestic_preds(s, **kw) for s in ("2025-26", "2026-27")], ignore_index=True)
    dom_cur = domestic_preds("2026-27", **kw)
    for tag, pred in (("ucl_hist", hist), ("ucl_2627", cur), ("dom", dom), ("dom_2627", dom_cur)):
        for k, v in _score(pred).items():
            out[f"{tag}_{k}"] = v
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
           if any(s in k for s in ("cfg", "log_loss", "rps", "goal_bias", "goal_logscore"))}, flush=True)
    return out


def main():
    grid = [dict(),
            dict(season_level_sd=0.1), dict(season_level_sd=0.2),
            dict(season_level_sd=0.1, cup_season_level_sd=0.1), dict(season_level_sd=0.1, cup_season_level_sd=0.2),
            dict(season_level_sd=0.2, cup_season_level_sd=0.2),
            dict(season_level_sd=0.1, cup_level_sd=0.1, cup_season_level_sd=0.1)]
    with Pool(2) as pool:
        res = pool.map(run_config, grid)
    df = pd.DataFrame(res)
    b = df.iloc[0]
    for tag in ("ucl_hist", "ucl_2627", "dom", "dom_2627"):
        df[f"{tag}_ll_gain"] = b[f"{tag}_log_loss"] - df[f"{tag}_log_loss"]
        df[f"{tag}_rps_gain"] = b[f"{tag}_rps"] - df[f"{tag}_rps"]
        df[f"{tag}_goal_gain"] = b[f"{tag}_goal_logscore"] - df[f"{tag}_goal_logscore"]
    cols = ["cfg"] + [c for c in df.columns if c.endswith(("_gain", "goal_bias", "_n"))]
    print(df[cols].round(4).to_string(index=False))
    out = ROOT / "data" / "processed" / "backtest_goal_level.json"
    out.write_text(json.dumps(dict(results=df.round(6).to_dict(orient="records")), indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
