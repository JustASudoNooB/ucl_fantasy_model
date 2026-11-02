"""Train and test the challenger on Premier League seasons the network has not seen.

  train       2022/23 (from GW16), 2023/24, 2024/25
  validate    2025/26                (early stopping picks the epoch here)
  test        2026/27 so far         (touched once, for the report)

The same features go through a linear model (hidden=()), the baseline the network must beat.
Metrics per player-match: start log loss and Brier; for players who played, the Poisson log
loss of goals and assists given minutes, and the Brier score of "scores at least once".
Each model is an average of three networks from different random starts. The production model
is refit on every season with the chosen number of epochs and saved to
data/processed/ml_model.json, with the metrics in ml_metrics.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import ROOT
from .data import FEATURES, load
from .nn import Ensemble, MultiTaskNet

PROC = ROOT / "data" / "processed"
MODEL = PROC / "ml_model.json"
METRICS = PROC / "ml_metrics.json"
TRAIN, VAL, TEST = ["2022-23", "2023-24", "2024-25"], ["2025-26"], ["2026-27"]
HIDDEN = (32, 16)


def tensors(df: pd.DataFrame):
    X = np.nan_to_num(df[FEATURES].to_numpy(dtype=float))
    T = dict(start=df.start.to_numpy(float), played=df.played.to_numpy(float),
             expo=(df.minutes.to_numpy(float) / 90.0), goals=df.goals_scored.to_numpy(float),
             assists=df.assists.to_numpy(float))
    return X, T


def metrics(net, df: pd.DataFrame) -> dict:
    X, T = tensors(df)
    pr = net.predict(X)
    p = np.clip(pr["p_start"], 1e-4, 1 - 1e-4)
    y = T["start"]
    pl = T["played"] > 0
    out = dict(n=int(len(df)), start_ll=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
               start_brier=float(np.mean((p - y) ** 2)), n_played=int(pl.sum()))
    for k, col in (("goal", "goals"), ("assist", "assists")):
        mu = (pr["g90"] if k == "goal" else pr["a90"])[pl] * T["expo"][pl]
        yy = T[col][pl]
        out[f"{k}_poisson_ll"] = float(np.mean(mu - yy * np.log(mu + 1e-12)))
        p1 = 1 - np.exp(-mu)
        out[f"{k}_brier"] = float(np.mean((p1 - (yy > 0)) ** 2))
        out[f"{k}_pred"] = float(mu.sum())
        out[f"{k}_actual"] = float(yy.sum())
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}


def run(verbose: bool = False, hidden=HIDDEN, seeds: int = 3) -> dict:
    df = load()
    tr, va, te = (df[df.season.isin(s)] for s in (TRAIN, VAL, TEST))
    Xtr, Ttr = tensors(tr)
    Xva, Tva = tensors(va)
    res = {}
    nets = {}
    for name, h in (("network", hidden), ("linear", ())):
        k = seeds if h else 1                               # the linear model has one optimum
        ens = Ensemble([MultiTaskNet(len(FEATURES), h, seed=s).fit(Xtr, Ttr, Xva, Tva, verbose=verbose, seed=s)
                        for s in range(k)])
        nets[name] = ens
        res[name] = dict(best_epoch=ens.best_epoch, val=metrics(ens, va), test=metrics(ens, te))
    # production model: every season, the network's epoch count, the same number of random starts
    Xall, Tall = tensors(df)
    ep = int(nets["network"].best_epoch or 20)
    final = Ensemble([MultiTaskNet(len(FEATURES), hidden, seed=s).fit(
        Xall, Tall, Xall[:5000], {c: v[:5000] for c, v in Tall.items()}, epochs=ep, patience=ep + 1, seed=s)
        for s in range(seeds)])
    MODEL.write_text(final.to_json())
    out = dict(trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), features=FEATURES,
               hidden=list(hidden), rows=dict(train=int(len(tr)), val=int(len(va)), test=int(len(te)), all=int(len(df))),
               seasons=dict(train=TRAIN, val=VAL, test=TEST), results=res, production_epochs=ep)
    METRICS.write_text(json.dumps(out, indent=1))
    return out


def load_model() -> Ensemble | None:
    return Ensemble.from_json(MODEL.read_text()) if MODEL.exists() else None
