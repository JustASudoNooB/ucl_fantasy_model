"""Out-of-sample test of the player layer: does last season's data beat price-based priors?

Season-ahead tests (fit on season T-1, score on season T):
  A. Premier League, FPL/Opta data, 2024/25 -> 2025/26. The prior is the production one in
     kind: log(share) linear in the game's start price, by position (fitted on T-1 only).
  B. Big-five leagues, FBref/Opta data, 2023/24 -> 2024/25 (and 2022/23 as a second season
     back, to set the season decay). The prior is the position mean (no prices exist).
  C. Ball recoveries per 90, FBref 2022/23 -> 2023/24 (the last two seasons with full counts).

For each quantity the blend is the production one: log-space weight w = n / (n + k), n in
90-minute units of history. Metric: minutes-weighted mean squared error of log(share), and the
share of that error the blend removes relative to the prior alone.

Run:  python scripts/validate_player_layer.py [--rds data/raw/fbref]
Writes data/processed/player_validation.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufe.ingest import fpl as F          # noqa: E402

FLOOR = {"xg_share": 0.004, "xa_share": 0.004}
K_GRID = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 1e9]


def wmse(y, yhat, w):
    return float(np.sum(w * (y - yhat) ** 2) / np.sum(w))


def blend(prior_log, obs_log, n90, k):
    w = n90 / (n90 + k) if k < 1e8 else np.zeros_like(n90)
    if k == 0:
        w = np.where(n90 > 0, 1.0, 0.0)
    return np.where(np.isfinite(obs_log), w * obs_log + (1 - w) * prior_log, prior_log)


def grid(prior_log, obs_log, n90, y, wt):
    res = {}
    for k in K_GRID:
        res[k] = wmse(y, blend(prior_log, obs_log, n90, k), wt)
    return res


def summarise(name, res, n):
    prior_only = res[1e9]
    data_only = res[0]
    kbest = min(res, key=res.get)
    out = dict(test=name, n_players=int(n), mse_prior=round(prior_only, 4), mse_data_only=round(data_only, 4),
               k_best=kbest, mse_best=round(res[kbest], 4),
               error_removed_vs_prior=round(1 - res[kbest] / prior_only, 3),
               curve={str(int(k) if k < 1e8 else "prior"): round(v, 4) for k, v in res.items()})
    print(f"{name:<44} n={n:<5} prior {prior_only:.4f}  data-only {data_only:.4f}  best k={kbest:<6g} "
          f"{res[kbest]:.4f}  ({out['error_removed_vs_prior']:.0%} of prior error removed)")
    return out


# ------------------------------------------------------------------------------------------ A
def test_fpl() -> list[dict]:
    a24 = F.player_aggregates(F.load("2024-25", "2025-26"))
    a25 = F.player_aggregates(F.load("2025-26", "2026-27"))
    out = []
    t = a25[a25.minutes >= 900].merge(a24, on="code", how="left", suffixes=("", "_prev"))
    t["moved"] = t.team_prev.notna() & (t.team_prev != t.team)
    t["n90"] = (t.minutes_prev.fillna(0) / 90.0)
    fit = a24[a24.minutes >= 900]
    for q in ("xg_share", "xa_share"):
        prior = np.zeros(len(t))
        for pos in ("GK", "DEF", "MID", "FWD"):
            g = fit[fit.pos == pos]
            b, c = np.polyfit(g.start_price, np.log(g[q].clip(lower=FLOOR[q])), 1)
            m = (t.pos == pos).to_numpy()
            prior[m] = c + b * t.start_price.to_numpy()[m]
        y = np.log(t[q].clip(lower=FLOOR[q]).to_numpy())
        obs = np.log(t[f"{q}_prev"].clip(lower=FLOOR[q]).to_numpy())
        obs[(t.minutes_prev.fillna(0) < 90).to_numpy()] = np.nan
        wt = t.minutes.to_numpy()
        nonGK = (t.pos != "GK").to_numpy()
        for label, m in (("all", nonGK), ("same club", nonGK & ~t.moved.to_numpy()), ("moved club", nonGK & t.moved.to_numpy())):
            res = grid(prior[m], obs[m], t.n90.to_numpy()[m], y[m], wt[m])
            out.append(summarise(f"A {q} PL 24/25->25/26, price prior, {label}", res, m.sum()))
    # start rate: same club, every player with a squad season in both (target: share of matches started)
    s = a25.merge(a24, on="code", suffixes=("", "_prev"))
    s = s[(s.team == s.team_prev) & (s.matches >= 5)]
    pr = np.zeros(len(s))
    for pos in ("GK", "DEF", "MID", "FWD"):
        g = a24[(a24.pos == pos) & (a24.matches >= 5)]
        b, c = np.polyfit(g.start_price, g.start_rate, 1)
        m = (s.pos == pos).to_numpy()
        pr[m] = np.clip(c + b * s.start_price.to_numpy()[m], 0.02, 0.95)
    y = s.start_rate.to_numpy()
    o = s.start_rate_prev.to_numpy()
    n = s.matches_prev.to_numpy().astype(float)
    res = {}
    for k in K_GRID:
        w = n / (n + k) if k < 1e8 else np.zeros_like(n)
        res[k] = wmse(y, w * o + (1 - w) * pr, np.ones_like(y))
    out.append(summarise("A start rate PL 24/25->25/26 (matches as n)", res, len(s)))
    return out


# ------------------------------------------------------------------------------------------ B, C
def fb_rows(rds: Path, seasons) -> pd.DataFrame:
    import pyreadr
    st = pyreadr.read_r(str(rds / "big5_player_standard.rds"))[None]
    st = st[st.Season_End_Year.isin(seasons)].copy()
    pt = pyreadr.read_r(str(rds / "big5_player_playing_time.rds"))[None]
    tm = pt[pt.Season_End_Year.isin(seasons)].groupby(["Season_End_Year", "Squad"])["MP_Playing.Time"].max()
    st = st.join(tm.rename("tm"), on=["Season_End_Year", "Squad"])
    for c in ("Min_Playing", "npxG_Expected", "xAG_Expected", "Starts_Playing"):
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0)
    tot = st.groupby(["Season_End_Year", "Squad"])[["npxG_Expected", "xAG_Expected"]].sum()
    st = st.join(tot.rename(columns=lambda c: "team_" + c), on=["Season_End_Year", "Squad"])
    m90 = (st.Min_Playing / 90).clip(lower=1e-6)
    st["xg_share"] = (st.npxG_Expected / m90) / (st.team_npxG_Expected / st.tm)
    st["xa_share"] = (st.xAG_Expected / m90) / (st.team_xAG_Expected / st.tm)
    st["pos"] = st.Pos.astype(str).str.split(",").str[0].map({"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"})
    st["key"] = st.Player.astype(str) + "|" + st.Born.astype(str)
    # one row per player-season: the club with most minutes
    st = st.sort_values("Min_Playing").drop_duplicates(["Season_End_Year", "key"], keep="last")
    return st


def test_fbref(rds: Path) -> list[dict]:
    st = fb_rows(rds, (2023, 2024, 2025))
    T = st[(st.Season_End_Year == 2025) & (st.Min_Playing >= 900) & (st.pos != "GK")]
    P1 = st[st.Season_End_Year == 2024].set_index("key")
    P2 = st[st.Season_End_Year == 2023].set_index("key")
    out = []
    for q in ("xg_share", "xa_share"):
        pri = st[(st.Season_End_Year == 2024) & (st.Min_Playing >= 900)].groupby("pos")[q].apply(
            lambda x: np.log(x.clip(lower=FLOOR[q])).mean())
        prior = T.pos.map(pri).to_numpy()
        y = np.log(T[q].clip(lower=FLOOR[q]).to_numpy())
        wt = T.Min_Playing.to_numpy()
        n1 = T.key.map(P1.Min_Playing).fillna(0).to_numpy() / 90
        n2 = T.key.map(P2.Min_Playing).fillna(0).to_numpy() / 90
        x1 = T.key.map(P1[q]).to_numpy(dtype=float)
        x2 = T.key.map(P2[q]).to_numpy(dtype=float)
        moved = (T.key.map(P1.Squad).fillna(T.Squad) != T.Squad).to_numpy()
        best = None
        for delta in (0.0, 0.25, 0.5, 0.75, 1.0):            # weight of the season two back
            num = np.nan_to_num(x1 * n1) + delta * np.nan_to_num(x2 * n2)
            den = n1 + delta * n2
            obs = np.where(den > 1, np.log(np.clip(num / np.maximum(den, 1e-9), FLOOR[q], None)), np.nan)
            res = grid(prior, obs, den, y, wt)
            kb = min(res, key=res.get)
            print(f"   {q}: season-two-back weight {delta:.2f} -> best k {kb:g}, mse {res[kb]:.4f}")
            if best is None or res[kb] < best[1][best[2]]:
                best = (delta, res, kb, obs, den)
        delta, res, kb, obs, den = best
        r = summarise(f"B {q} big-5 23/24(+22/23)->24/25, position prior", res, len(T))
        r["season_two_back_weight"] = delta
        out.append(r)
        for label, m in (("same club", ~moved), ("moved club", moved)):
            res2 = grid(prior[m], obs[m], den[m], y[m], wt[m])
            out.append(summarise(f"B {q} big-5, {label}", res2, m.sum()))
    return out


def test_recoveries(rds: Path) -> list[dict]:
    import pyreadr
    mi = pyreadr.read_r(str(rds / "big5_player_misc.rds"))[None]
    mi = mi[mi.Season_End_Year.isin((2023, 2024))].copy()
    mi["n90"] = pd.to_numeric(mi.Mins_Per_90, errors="coerce").fillna(0)
    mi["rec90"] = pd.to_numeric(mi.Recov, errors="coerce") / mi.n90.where(mi.n90 > 0)
    mi["pos"] = mi.Pos.astype(str).str.split(",").str[0].map({"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"})
    mi["key"] = mi.Player.astype(str) + "|" + mi.Born.astype(str)
    mi = mi.sort_values("n90").drop_duplicates(["Season_End_Year", "key"], keep="last")
    T = mi[(mi.Season_End_Year == 2024) & (mi.n90 >= 10) & (mi.pos != "GK")]
    P = mi[mi.Season_End_Year == 2023].set_index("key")
    pri = mi[(mi.Season_End_Year == 2023) & (mi.n90 >= 10)].groupby("pos").rec90.median()
    prior = T.pos.map(pri).to_numpy()
    y = T.rec90.to_numpy()
    x = T.key.map(P.rec90).to_numpy(dtype=float)
    n = T.key.map(P.n90).fillna(0).to_numpy()
    res = {}
    for k in K_GRID:
        w = n / (n + k) if k < 1e8 else np.zeros_like(n)
        if k == 0:
            w = (n > 0).astype(float)
        res[k] = wmse(y, np.where(np.isfinite(x), w * x + (1 - w) * prior, prior), T.n90.to_numpy())
    return [summarise("C recoveries/90 big-5 22/23->23/24, position prior", res, len(T))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rds", default=str(ROOT / "data" / "raw" / "fbref"))
    a = ap.parse_args()
    rds = Path(a.rds)
    results = test_fpl()
    if (rds / "big5_player_standard.rds").exists():
        results += test_fbref(rds)
        results += test_recoveries(rds)
    out = ROOT / "data" / "processed" / "player_validation.json"
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
