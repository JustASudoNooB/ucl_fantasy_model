"""Walk-forward backtests: the team model only ever sees matches played before each round."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.scoreline import match_probs
from ..models.team_ratings import fit
from . import metrics as M


def _rounds(dates: pd.Series, gap_days: int = 3) -> list[pd.Timestamp]:
    """Cluster match dates into rounds; return the first date of each round."""
    ds = sorted(pd.to_datetime(dates).unique())
    starts, prev = [], None
    for d in ds:
        if prev is None or (d - prev).days > gap_days:
            starts.append(d)
        prev = d
    return starts


def backtest_team_model(df: pd.DataFrame, season: str, comp: str = "cl", stages: tuple[str, ...] = ("league", "playoffs", "finals"),
                        predictive_sd_scale: float = 1.0, **fit_kw) -> tuple[pd.DataFrame, dict]:
    target = df[(df.comp == comp) & (df.season == season)].copy()
    target = target[target.stage.str.lower().str.startswith(stages)]
    if target.empty:
        return target, {}
    starts = _rounds(target.date)
    rows = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else target.date.max() + pd.Timedelta(days=1)
        batch = target[(target.date >= start) & (target.date < end)]
        model = fit(df, as_of=start, **fit_kw)
        for r in batch.itertuples():
            lh, la = model.lambdas(r.home_id, r.away_id, neutral=bool(r.neutral), comp_group="EUR")
            sh, sa = model.log_sd(r.home_id, r.away_id, predictive_sd_scale)
            mp = match_probs(lh, la, model.rho, sh=sh, sa=sa)
            top = mp.top_scores[0]
            rows.append(dict(date=r.date, home=r.home_id, away=r.away_id, hg=r.hg, ag=r.ag, stage=r.stage,
                             p_home=mp.p_home, p_draw=mp.p_draw, p_away=mp.p_away, xg_home=lh, xg_away=la,
                             cs_home=mp.cs_home, cs_away=mp.cs_away, btts=mp.btts, top_h=top[0], top_a=top[1],
                             p_exact=float(mp.matrix[min(r.hg, 10), min(r.ag, 10)])))
    pred = pd.DataFrame(rows)
    return pred, summarise(pred, df[(df.date < starts[0]) & df.comp.isin(["cl", "el", "conf"])])


def summarise(pred: pd.DataFrame, history: pd.DataFrame | None = None) -> dict:
    y = M.outcome_index(pred.hg.to_numpy(), pred.ag.to_numpy())
    P = pred[["p_home", "p_draw", "p_away"]].to_numpy()
    if history is not None and len(history):
        yh = M.outcome_index(history.hg.to_numpy(), history.ag.to_numpy())
        base = np.bincount(yh, minlength=3) / len(yh)
    else:
        base = np.array([0.45, 0.25, 0.30])
    B = np.tile(base, (len(pred), 1))
    U = np.full_like(P, 1 / 3)
    cs = np.concatenate([pred.cs_home.to_numpy(), pred.cs_away.to_numpy()])
    cs_y = np.concatenate([(pred.ag == 0).to_numpy(), (pred.hg == 0).to_numpy()]).astype(float)
    out = dict(
        n=int(len(pred)),
        log_loss=M.log_loss(P, y), log_loss_base=M.log_loss(B, y), log_loss_uniform=M.log_loss(U, y),
        rps=M.rps(P, y), rps_base=M.rps(B, y), rps_uniform=M.rps(U, y),
        brier=M.brier_multi(P, y), accuracy=M.accuracy(P, y),
        exact_score_hit=float(np.mean((pred.top_h == pred.hg) & (pred.top_a == pred.ag))),
        goals_mae=float(np.mean(np.abs(pred.xg_home - pred.hg)) / 2 + np.mean(np.abs(pred.xg_away - pred.ag)) / 2),
        mean_p_exact=float(pred.p_exact.mean()),
        cs_brier=M.brier_binary(cs, cs_y), cs_ece=M.expected_calibration_error(cs, cs_y, 8),
        calibration_home=M.calibration_table(P[:, 0], (y == 0).astype(float), 8),
        calibration_cs=M.calibration_table(cs, cs_y, 8),
        base_rates=[round(float(x), 4) for x in base],
    )
    out["log_loss_skill_vs_base"] = 1 - out["log_loss"] / out["log_loss_base"]
    out["rps_skill_vs_base"] = 1 - out["rps"] / out["rps_base"]
    return out
