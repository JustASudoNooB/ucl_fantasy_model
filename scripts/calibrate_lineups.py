"""Calibrate how predicted line-ups (team news) move start probabilities, using real line-ups.

    python scripts/calibrate_lineups.py --md 2

For every club with a known actual XI (data/seed/lineups/md{N}_actual_xis.txt) it compares
  p0     the model's start probability before any news,
  named  whether the predicted XI (the md{N} news file) named the player,
  y      whether he actually started,
and fits  logit P(start) = a + b * logit(p0) + c * named  (players ruled out are excluded).
Prints the fit, the Bayes-rule version it replaces, and log loss / Brier of each.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ufe.ingest import team_news  # noqa: E402
from ufe.pipeline import SEED, Engine  # noqa: E402


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def fit(X, y, ridge=(0.0, 0.0, 0.0), centre=(0.0, 1.0, 1.8)):
    def nll(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        eps = 1e-9
        return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + sum(
            r * (wi - c) ** 2 for r, wi, c in zip(ridge, w, centre))
    w = minimize(nll, x0=np.array(centre), method="BFGS").x
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=int, default=2)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    eng = Engine(as_of="2026-10-13", n_sims=2000)
    base = eng.players
    news = team_news.match(team_news.parse(team_news.news_path(a.md)), base)
    actual = team_news.match(team_news.parse(SEED / "lineups" / f"md{a.md}_actual_xis.txt"), base)
    teams = sorted(set(actual.team))
    rows = []
    for t in teams:
        named = set(news[(news.team == t) & (news.status == "start")].id.dropna())
        out = set(news[(news.team == t) & news.status.isin(["out", "susp"])].id.dropna())
        started = set(actual[actual.team == t].id.dropna())
        miss = actual[(actual.team == t) & actual.id.isna()].name.tolist()
        if miss:
            print(f"{t}: actual starters not in the pool: {miss}")
        for r in base[base.team == t].itertuples():
            if r.id in out:
                continue
            rows.append(dict(team=t, id=r.id, name=r.name, p0=float(r.p_start), named=int(r.id in named), y=int(r.id in started)))
    df = pd.DataFrame(rows)
    print(f"{len(df)} players, {df.y.sum()} starters, named {df.named.sum()}, named & started {((df.named == 1) & (df.y == 1)).sum()}")
    X = np.column_stack([np.ones(len(df)), logit(df.p0.to_numpy()), df.named.to_numpy()])
    y = df.y.to_numpy()
    w = fit(X, y, ridge=(1.0, 1.0, 1.0))
    p_fit = 1 / (1 + np.exp(-(X @ w)))
    cfg = team_news._cfg()
    odds = df.p0.clip(0.02, 0.98) / (1 - df.p0.clip(0.02, 0.98)) * np.where(df.named == 1, cfg["lr_start"], cfg["lr_bench"])
    p_bayes = np.minimum(odds / (1 + odds), cfg["p_max"])
    for nm, p in (("prior only", df.p0.clip(0.01, 0.99)), ("Bayes LR (settings)", p_bayes), ("fitted logistic", p_fit)):
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        br = np.mean((p - y) ** 2)
        print(f"{nm:22s} log loss {ll:.3f}  Brier {br:.3f}  expected starters {p.sum():.1f} (actual {y.sum()})")
    print("fitted: logit P = %.2f + %.2f * logit(p0) + %.2f * named" % tuple(w))
    df["p_bayes"], df["p_fit"] = p_bayes, p_fit
    cross_validate(df, cfg)
    for b in (0.5, 0.75):
        off = b * X[:, 1]
        def nll(v):
            z = v[0] + off + v[1] * X[:, 2]
            p = 1 / (1 + np.exp(-z))
            return -np.sum(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        v = minimize(nll, x0=np.array([-0.5, 1.8]), method="BFGS").x
        print(f"all clubs, slope fixed at {b}: logit P = {v[0]:.2f} + {b} * logit(p0) + {v[1]:.2f} * named")
    if "--show" in sys.argv:
        print(df.sort_values(["team", "y", "p_fit"], ascending=[True, False, False]).to_string(index=False, max_rows=400))



def cross_validate(df: pd.DataFrame, cfg: dict) -> None:
    """Leave-one-club-out log loss for: Bayes LR, free logistic, logistic with b fixed."""
    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = {k: [] for k in ("bayes", "logit_free", "logit_b0.5", "logit_b0.75")}
    for t in sorted(df.team.unique()):
        tr, te = df[df.team != t], df[df.team == t]
        Xtr = np.column_stack([np.ones(len(tr)), logit(tr.p0.to_numpy()), tr.named.to_numpy()])
        Xte = np.column_stack([np.ones(len(te)), logit(te.p0.to_numpy()), te.named.to_numpy()])
        ytr, yte = tr.y.to_numpy(), te.y.to_numpy()
        odds = te.p0.clip(0.02, 0.98) / (1 - te.p0.clip(0.02, 0.98)) * np.where(te.named == 1, cfg["lr_start"], cfg["lr_bench"])
        res["bayes"].append(ll(np.minimum(odds / (1 + odds), cfg["p_max"]), yte))
        w = fit(Xtr, ytr, ridge=(1.0, 1.0, 1.0))
        res["logit_free"].append(ll(1 / (1 + np.exp(-(Xte @ w))), yte))
        for b in (0.5, 0.75):
            # fix the slope on logit(p0), fit intercept and the 'named' effect
            off_tr, off_te = b * Xtr[:, 1], b * Xte[:, 1]
            def nll(v):
                z = v[0] + off_tr + v[1] * Xtr[:, 2]
                p = 1 / (1 + np.exp(-z))
                return -np.sum(ytr * np.log(p + 1e-9) + (1 - ytr) * np.log(1 - p + 1e-9))
            v = minimize(nll, x0=np.array([-0.5, 1.8]), method="BFGS").x
            res[f"logit_b{b}"].append(ll(1 / (1 + np.exp(-(v[0] + off_te + v[1] * Xte[:, 2]))), yte))
    for k, v in res.items():
        print(f"leave-one-club-out log loss  {k:12s} {np.mean(v):.3f}   per club {np.round(v, 3).tolist()}")


if __name__ == "__main__":
    main()
