"""Decision-tree challenger: gradient-boosted trees (LightGBM, free and open source) on the same
Premier League player-matches and features as the neural network.

Three models, each a sum of a few hundred small trees:
  start     binary log loss               P(start)
  goals     Poisson, offset log(minutes/90)   goals per 90
  assists   Poisson, offset log(minutes/90)   assists per 90
Trees split the inputs into ranges ("price above 7.5 and started last match"), so they pick up
thresholds and interactions without the inputs being scaled. The number of trees is chosen on the
2025/26 season (early stopping) and the model is scored once on 2026/27, the same seasons as the
network (train.py). The production model is refit on every season with that many trees and saved
to data/processed/trees_model.json; scores and the inputs each model leans on most go to
trees_metrics.json.

The `predict(X)` interface matches the network's, so the UCL challenger engine (ucl.Challenger)
runs it unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import ROOT
from .data import FEATURES, load
from .train import TEST, TRAIN, VAL, metrics, tensors

PROC = ROOT / "data" / "processed"
MODEL = PROC / "trees_model.json"
METRICS = PROC / "trees_metrics.json"
TARGETS = ("start", "goals", "assists")
PARAMS = dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0, num_threads=2, deterministic=True)


def _lgb():
    import lightgbm as lgb
    return lgb


def _data(X, T, target):
    """(features, label, offset) for one target: starts use every row, goals and assists the rows
    where the player played, with log(minutes/90) as the exposure offset."""
    if target == "start":
        return X, T["start"], None
    m = (T["played"] > 0) & (T["expo"] > 0)
    return X[m], T["goals" if target == "goals" else "assists"][m], np.log(T["expo"][m])


class TreeModel:
    def __init__(self, boosters: dict, best_iter: dict | None = None):
        self.boosters = boosters
        self.best_iter = best_iter or {}

    def predict(self, X) -> dict:
        X = np.asarray(X, dtype=float)
        b = self.boosters
        return dict(p_start=np.clip(b["start"].predict(X), 1e-4, 1 - 1e-4),
                    g90=np.clip(b["goals"].predict(X), 1e-4, 5.0), a90=np.clip(b["assists"].predict(X), 1e-4, 5.0))

    def importance(self, top: int = 6) -> dict:
        out = {}
        for k, b in self.boosters.items():
            g = b.feature_importance(importance_type="gain")
            s = pd.Series(g / max(g.sum(), 1e-12), index=b.feature_name()).sort_values(ascending=False)
            out[k] = {f: round(float(v), 3) for f, v in s.head(top).items()}
        return out

    def to_json(self) -> str:
        return json.dumps(dict(features=self.boosters["start"].feature_name(), best_iter=self.best_iter,
                               models={k: b.model_to_string() for k, b in self.boosters.items()}))

    @classmethod
    def from_json(cls, s: str, check: bool = True) -> "TreeModel":
        lgb = _lgb()
        d = json.loads(s)
        if check and d.get("features") != FEATURES:
            raise ValueError("tree model was trained on a different feature list: retrain with `python -m ufe models train`")
        return cls({k: lgb.Booster(model_str=v) for k, v in d["models"].items()}, d.get("best_iter"))


def fit(X, T, Xv=None, Tv=None, rounds: dict | None = None, max_rounds: int = 1500, params: dict | None = None) -> TreeModel:
    """Early stopping on (Xv, Tv) when given, else exactly `rounds[target]` trees."""
    lgb = _lgb()
    p0 = {**PARAMS, **(params or {})}
    names = FEATURES if X.shape[1] == len(FEATURES) else [f"f{i}" for i in range(X.shape[1])]
    boosters, best = {}, {}
    for t in TARGETS:
        x, y, off = _data(X, T, t)
        p = {**p0, "objective": "binary" if t == "start" else "poisson",
             "min_data_in_leaf": p0["min_data_in_leaf"] if t == "start" else int(p0["min_data_in_leaf"] * 1.5)}
        dtr = lgb.Dataset(x, label=y, init_score=off, feature_name=names, free_raw_data=False)
        if Xv is not None:
            xv, yv, offv = _data(Xv, Tv, t)
            dva = lgb.Dataset(xv, label=yv, init_score=offv, reference=dtr, feature_name=names)
            b = lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            best[t] = int(b.best_iteration or max_rounds)
        else:
            n = int((rounds or {}).get(t, 300))
            b = lgb.train(p, dtr, num_boost_round=n)
            best[t] = n
        boosters[t] = b
    return TreeModel(boosters, best)


def run(verbose: bool = False) -> dict:
    df = load()
    tr, va, te = (df[df.season.isin(s)] for s in (TRAIN, VAL, TEST))
    Xtr, Ttr = tensors(tr)
    Xva, Tva = tensors(va)
    m = fit(Xtr, Ttr, Xva, Tva)
    res = dict(trees=dict(best_iter=m.best_iter, val=metrics(m, va), test=metrics(m, te)))
    nn = json.loads((PROC / "ml_metrics.json").read_text()) if (PROC / "ml_metrics.json").exists() else {}
    for k in ("network", "linear"):                       # same seasons, for the side-by-side table
        if k in (nn.get("results") or {}):
            res[k] = nn["results"][k]
    Xall, Tall = tensors(df)
    final = fit(Xall, Tall, rounds=m.best_iter)
    MODEL.write_text(final.to_json())
    out = dict(trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), library="lightgbm " + _lgb().__version__,
               params=PARAMS, features=FEATURES, rows=dict(train=int(len(tr)), val=int(len(va)), test=int(len(te)), all=int(len(df))),
               seasons=dict(train=TRAIN, val=VAL, test=TEST), results=res, importance=final.importance())
    METRICS.write_text(json.dumps(out, indent=1))
    if verbose:
        print(json.dumps(res, indent=1))
    return out


def load_model() -> TreeModel | None:
    return TreeModel.from_json(MODEL.read_text()) if MODEL.exists() else None
