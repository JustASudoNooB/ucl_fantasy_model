"""Does a player's current-season domestic form improve next-match predictions?

Data: Premier League 2024/25 and 2025/26 per-match Opta numbers from the FPL archive
(minutes, starts, goals, assists, xG, xA). Target season 2025/26, every match from the 4th
game week on; each prediction only uses matches played before it.

Rates (goals and assists per 90) are estimated as
    (k * position prior + W_prev * last-season xG + W_cur * sum_i d^age_i * this-season xG_i)
  / (k            + W_prev * last-season 90s + W_cur * sum_i d^age_i * this-season 90s_i)
with d = 0.5 ** (1 / half_life_matches). W_cur = 0 is what the engine did before (history from
previous seasons only). The prediction for a match uses the minutes actually played, so the test
isolates the rate estimate. Scored by Poisson log loss on goals and assists.

Starts: P(start) for players who appeared in one of the club's last three matches (a proxy for
being fit), from last season's start rate at the club, this season's start rate, or the
recency-weighted rate over the last matches. Scored by log loss.

    python scripts/backtest_player_form.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fpl"


def load(season: str) -> pd.DataFrame:
    g = pd.read_csv(RAW / f"fpl_{season}_gws_merged_gw.csv")
    pr = pd.read_csv(RAW / f"fpl_{season}_players_raw.csv")[["id", "code"]]
    g = g.merge(pr, left_on="element", right_on="id", how="left")
    g["kickoff_time"] = pd.to_datetime(g.kickoff_time)
    g = g.sort_values(["code", "kickoff_time"]).reset_index(drop=True)
    for c in ("expected_goals", "expected_assists"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    return g


def pois_ll(y, lam):
    lam = np.maximum(lam, 1e-9)
    return -(y * np.log(lam) - lam - gammaln(y + 1))


def main():
    prev, cur = load("2024-25"), load("2025-26")
    # last season per player: 90s, xG, xA, start rate at his (last) club
    pa = prev.groupby("code").agg(m=("minutes", "sum"), xg=("expected_goals", "sum"), xa=("expected_assists", "sum"),
                                  starts=("starts", "sum"), n=("minutes", "size"), team=("team", "last"),
                                  pos=("position", "last"))
    pa["m90"] = pa.m / 90.0
    # position priors (per 90) from last season's regulars
    reg = pa[pa.m >= 900]
    prior_g = (reg.groupby("pos").xg.sum() / reg.groupby("pos").m90.sum()).to_dict()
    prior_a = (reg.groupby("pos").xa.sum() / reg.groupby("pos").m90.sum()).to_dict()

    rows = []
    for code, g in cur.groupby("code"):
        g = g.reset_index(drop=True)
        p = pa.loc[code] if code in pa.index else None
        for t in range(len(g)):
            r = g.iloc[t]
            if r.GW < 4:
                continue
            past = g.iloc[:t]
            rows.append(dict(code=code, gw=int(r.GW), pos=r.position, team=r.team, minutes=float(r.minutes),
                             start=int(r.starts), goals=int(r.goals_scored), assists=int(r.assists),
                             prev_m90=float(p.m90) if p is not None else 0.0,
                             prev_xg=float(p.xg) if p is not None else 0.0, prev_xa=float(p.xa) if p is not None else 0.0,
                             prev_start_rate=float(p.starts / max(p.n, 1)) if p is not None and p.team == r.team else np.nan,
                             past_m=past.minutes.to_numpy(float), past_xg=past.expected_goals.to_numpy(float),
                             past_xa=past.expected_assists.to_numpy(float), past_start=past.starts.to_numpy(float)))
    D = pd.DataFrame(rows)
    print("player-matches:", len(D))

    def rate(df, which, k, w_prev, w_cur, hl):
        prior = prior_g if which == "g" else prior_a
        out = np.empty(len(df))
        for i, r in enumerate(df.itertuples()):
            pm, px = r.past_m, (r.past_xg if which == "g" else r.past_xa)
            n = len(pm)
            d = np.ones(n) if not np.isfinite(hl) else 0.5 ** (np.arange(n)[::-1] / hl)
            num = k * prior.get(r.pos, 0.1) + w_prev * (r.prev_xg if which == "g" else r.prev_xa) + w_cur * np.sum(d * px)
            den = k + w_prev * r.prev_m90 + w_cur * np.sum(d * pm) / 90.0
            out[i] = num / den
        return out

    played = D[D.minutes > 0].copy()
    res = []
    for which, col in (("g", "goals"), ("a", "assists")):
        for k in (5.0,):
            for w_cur, hl in ((0.0, np.inf), (0.5, np.inf), (1.0, np.inf), (2.0, np.inf), (1.0, 10.0), (2.0, 10.0),
                              (1.0, 5.0), (2.0, 5.0), (3.0, 10.0)):
                lam = rate(played, which, k, 1.0, w_cur, hl) * played.minutes.to_numpy() / 90.0
                ll = pois_ll(played[col].to_numpy(), lam)
                early = (played.gw <= 12).to_numpy()
                res.append(dict(target=col, k=k, w_cur=w_cur, half_life=hl, ll=float(ll.mean()),
                                ll_early=float(ll[early].mean()), ll_late=float(ll[~early].mean()), n=int(len(ll))))
                print(res[-1], flush=True)
    R = pd.DataFrame(res)
    for col in ("goals", "assists"):
        b = R[(R.target == col) & (R.w_cur == 0)].iloc[0]
        R.loc[R.target == col, "gain"] = b.ll - R.loc[R.target == col, "ll"]
        R.loc[R.target == col, "gain_early"] = b.ll_early - R.loc[R.target == col, "ll_early"]

    # ---- starts
    S = D.copy()
    S["recent_app"] = [bool(np.any(pm[-3:] > 0)) if len(pm) else False for pm in S.past_m]
    S = S[S.recent_app]
    eps = 1e-3

    def ll_bin(p, y):
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    y = S.start.to_numpy(float)
    out_s = []
    prev_r = S.prev_start_rate.to_numpy()
    for name, k0, w_prev, w_cur, hl in (("prev season only", 4, 1.0, 0.0, np.inf), ("this season", 4, 0.0, 1.0, np.inf),
                                        ("both", 4, 1.0, 1.0, np.inf), ("both, prev=3 matches", 2, 0.3, 1.0, np.inf),
                                        ("both, prev=5 matches", 2, 0.5, 1.0, np.inf), ("recent hl=3", 2, 0.0, 1.0, 3.0),
                                        ("recent hl=5", 2, 0.0, 1.0, 5.0), ("both + recent hl=5", 2, 1.0, 1.0, 5.0),
                                        ("both + recent hl=3", 2, 1.0, 1.0, 3.0)):
        p = np.empty(len(S))
        for i, r in enumerate(S.itertuples()):
            ps = r.past_start
            n = len(ps)
            d = np.ones(n) if not np.isfinite(hl) else 0.5 ** (np.arange(n)[::-1] / hl)
            pr = prev_r[i]
            num = k0 * 0.5 + w_cur * np.sum(d * ps)
            den = k0 + w_cur * np.sum(d)
            if np.isfinite(pr) and w_prev > 0:
                num += w_prev * 10 * pr          # last season counts like 10 matches
                den += w_prev * 10
            p[i] = num / den
        ll = ll_bin(p, y)
        early = (S.gw <= 12).to_numpy()
        out_s.append(dict(model=name, ll=float(ll.mean()), ll_early=float(ll[early].mean()), n=int(len(ll))))
        print(out_s[-1], flush=True)
    out = dict(rates=R.round(5).to_dict(orient="records"), starts=out_s,
               note="EPL 2025/26 walk-forward; FPL archive per-match Opta data")
    (ROOT / "data" / "processed" / "backtest_player_form.json").write_text(json.dumps(out, indent=1, default=float))
    print(R.round(5).to_string(index=False))
    print(pd.DataFrame(out_s).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
