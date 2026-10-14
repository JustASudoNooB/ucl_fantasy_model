"""League-phase standings simulation and knockout-tie probabilities.

Why it matters for Fantasy: positions 1-8 skip the knockout play-offs, so their players blank
in MD9 and MD10; 9-24 play both legs; 25-36 are out. Knowing each club's odds of landing in
each band drives transfer and chip timing from MD5 onward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd



def simulate_league_phase(fixtures: pd.DataFrame, lambdas: dict[tuple[str, str], tuple[float, float]], rho: float,
                          n_sims: int = 20000, seed: int = 7, rating_noise: dict[str, float] | None = None) -> pd.DataFrame:
    """fixtures: md, home, away, status, home_goals, away_goals. Played rows are fixed; the rest are simulated.
    rating_noise: per-team sd (log-goal scale) of a per-simulation strength shock (model uncertainty)."""
    rng = np.random.default_rng(seed)
    teams = sorted(set(fixtures.home) | set(fixtures.away))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    pts = np.zeros((n_sims, T))
    gf = np.zeros((n_sims, T))
    ga = np.zeros((n_sims, T))
    agf = np.zeros((n_sims, T))       # away goals scored
    wins = np.zeros((n_sims, T))
    awins = np.zeros((n_sims, T))
    opp = {t: [] for t in teams}
    shock = np.zeros((n_sims, T))
    if rating_noise:
        sd = np.array([rating_noise.get(t, 0.06) for t in teams])
        shock = rng.normal(0, 1, (n_sims, T)) * sd
    for f in fixtures.itertuples():
        h, a = ti[f.home], ti[f.away]
        opp[f.home].append(a)
        opp[f.away].append(h)
        if f.status == "FT":
            hg = np.full(n_sims, int(f.home_goals))
            ag = np.full(n_sims, int(f.away_goals))
        else:
            lh, la = lambdas[(f.home, f.away)]
            # strength shocks move attack up and the opponent's goals down together
            lh_s = lh * np.exp(shock[:, h] - shock[:, a])
            la_s = la * np.exp(shock[:, a] - shock[:, h])
            hg = rng.poisson(lh_s)
            ag = rng.poisson(la_s)
        pts[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        pts[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg
        agf[:, a] += ag
        wins[:, h] += hg > ag
        wins[:, a] += ag > hg
        awins[:, a] += ag > hg
    opp_pts = np.stack([pts[:, opp[t]].sum(axis=1) for t in teams], axis=1)
    noise = rng.random((n_sims, T))
    # UEFA league-phase tiebreakers (points, GD, GF, away GF, wins, away wins, opponents' points, ...)
    keys = (noise, opp_pts, awins, wins, agf, gf, gf - ga, pts)       # lexsort: last key is primary
    rank = np.zeros((n_sims, T), dtype=np.int16)
    for n in range(n_sims):
        o = np.lexsort(tuple(-k[n] for k in keys))
        rank[n, o] = np.arange(1, T + 1)
    out = []
    for t in teams:
        r = rank[:, ti[t]]
        out.append(dict(team=t, exp_points=float(pts[:, ti[t]].mean()), exp_rank=float(r.mean()),
                        p_top8=float((r <= 8).mean()), p_playoff=float(((r >= 9) & (r <= 24)).mean()),
                        p_out=float((r >= 25).mean()), p_first=float((r == 1).mean()),
                        rank_p10=int(np.percentile(r, 10)), rank_p90=int(np.percentile(r, 90))))
    return pd.DataFrame(out).sort_values("exp_rank").reset_index(drop=True)


def tie_advance_probability(l1: tuple[float, float], l2: tuple[float, float], rho: float = -0.05,
                            n_sims: int = 40000, seed: int = 3) -> float:
    """Two-legged tie. l1 = (λ team A at home, λ team B away) leg 1; l2 = (λ A away, λ B home) leg 2.
    Returns P(team A advances). Extra time ~ a third of a match at the second-leg rates;
    shoot-outs are coin flips."""
    rng = np.random.default_rng(seed)
    a1, b1 = rng.poisson(l1[0], n_sims), rng.poisson(l1[1], n_sims)
    a2, b2 = rng.poisson(l2[0], n_sims), rng.poisson(l2[1], n_sims)
    agg_a, agg_b = a1 + a2, b1 + b2
    tied = agg_a == agg_b
    eta, etb = rng.poisson(l2[0] / 3, n_sims), rng.poisson(l2[1] / 3, n_sims)
    agg_a = agg_a + np.where(tied, eta, 0)
    agg_b = agg_b + np.where(tied, etb, 0)
    still = agg_a == agg_b
    pens = rng.random(n_sims) < 0.5
    return float(np.mean((agg_a > agg_b) | (still & pens)))
