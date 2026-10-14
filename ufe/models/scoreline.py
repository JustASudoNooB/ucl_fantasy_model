"""Scoreline distributions, derived match probabilities, and market (odds) handling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import poisson


def dc_matrix(lh: float, la: float, rho: float = -0.06, max_goals: int = 10) -> np.ndarray:
    """P(home=x, away=y) with the Dixon–Coles low-score correction; rows = home goals."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[0, 0] *= max(1 - lh * la * rho, 1e-9)
    m[0, 1] *= max(1 + lh * rho, 1e-9)
    m[1, 0] *= max(1 + la * rho, 1e-9)
    m[1, 1] *= max(1 - rho, 1e-9)
    return m / m.sum()


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(7)     # nodes/weights for a standard normal
_GH_W = _GH_W / _GH_W.sum()


def dc_matrix_predictive(lh: float, la: float, rho: float = -0.06, sh: float = 0.0, sa: float = 0.0,
                         max_goals: int = 10) -> np.ndarray:
    """Posterior-predictive score matrix: averages the Dixon–Coles matrix over uncertainty in the
    log goal rates (sd `sh`, `sa`, independent), by 7x7 Gauss–Hermite quadrature. With zero sd it
    equals dc_matrix. Walk-forward UCL backtests (2024/25-2025/26): log loss 0.8980 -> 0.8970,
    RPS 0.19576 -> 0.19559, clean-sheet ECE 0.0235 -> 0.0214."""
    if sh <= 1e-9 and sa <= 1e-9:
        return dc_matrix(lh, la, rho, max_goals)
    m = np.zeros((max_goals + 1, max_goals + 1))
    for xi, wi in zip(_GH_X, _GH_W):
        for xj, wj in zip(_GH_X, _GH_W):
            m += wi * wj * dc_matrix(lh * np.exp(sh * xi), la * np.exp(sa * xj), rho, max_goals)
    return m / m.sum()


@dataclass
class MatchProbs:
    lh: float
    la: float
    p_home: float
    p_draw: float
    p_away: float
    cs_home: float          # home keeps a clean sheet
    cs_away: float
    btts: float
    over25: float
    top_scores: list        # [(h, a, p), ...]
    matrix: np.ndarray

    def to_dict(self, top: int = 6, grid: int = 5) -> dict:
        return dict(xg_home=round(self.lh, 3), xg_away=round(self.la, 3), p_home=round(self.p_home, 4),
                    p_draw=round(self.p_draw, 4), p_away=round(self.p_away, 4), cs_home=round(self.cs_home, 4),
                    cs_away=round(self.cs_away, 4), btts=round(self.btts, 4), over25=round(self.over25, 4),
                    top_scores=[[int(h), int(a), round(float(p), 4)] for h, a, p in self.top_scores[:top]],
                    grid=np.round(self.matrix[:grid + 1, :grid + 1], 4).tolist())


def match_probs(lh: float, la: float, rho: float = -0.06, max_goals: int = 10, sh: float = 0.0,
                sa: float = 0.0) -> MatchProbs:
    m = dc_matrix_predictive(lh, la, rho, sh, sa, max_goals)
    p_home = float(np.tril(m, -1).sum())
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())
    cs_home = float(m[:, 0].sum())
    cs_away = float(m[0, :].sum())
    btts = float(m[1:, 1:].sum())
    g = np.arange(max_goals + 1)
    over25 = float(m[(g[:, None] + g[None, :]) > 2.5].sum())
    flat = [(h, a, m[h, a]) for h in range(max_goals + 1) for a in range(max_goals + 1)]
    flat.sort(key=lambda x: -x[2])
    return MatchProbs(lh, la, p_home, p_draw, p_away, cs_home, cs_away, btts, over25, flat[:10], m)


# ---------------------------------------------------------------------------- odds
def remove_margin(odds: list[float], method: str = "power") -> np.ndarray:
    """Decimal odds -> fair probabilities. 'power' handles favourite–longshot bias better than
    proportional normalisation; 'shin' models insider trading."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    if method == "proportional":
        return inv / inv.sum()
    if method == "power":
        k = brentq(lambda k: np.sum(inv ** k) - 1.0, 0.5, 3.0)
        return inv ** k
    if method == "shin":
        s = inv.sum()

        def f(z):
            p = (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / s) - z) / (2 * (1 - z))
            return p.sum() - 1.0

        z = brentq(f, 0.0, 0.4)
        return (np.sqrt(z ** 2 + 4 * (1 - z) * inv ** 2 / s) - z) / (2 * (1 - z))
    raise ValueError(method)


def implied_lambdas(p_home: float, p_draw: float, p_away: float, p_over25: float | None = None,
                    rho: float = -0.06, total_prior: float = 2.8) -> tuple[float, float]:
    """Goal rates that reproduce market 1X2 (and optionally over 2.5) probabilities."""
    target = np.array([p_home, p_draw, p_away])

    def loss(x):
        lh, la = np.exp(x)
        mp = match_probs(lh, la, rho, 8)
        err = np.sum((np.array([mp.p_home, mp.p_draw, mp.p_away]) - target) ** 2)
        if p_over25 is not None:
            err += (mp.over25 - p_over25) ** 2
        else:
            err += 1e-3 * (np.log(lh + la) - np.log(total_prior)) ** 2   # weak anchor on total goals
        return err

    best = minimize(loss, x0=np.log([1.4, 1.1]), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10})
    lh, la = np.exp(best.x)
    return float(lh), float(la)


def blend_lambdas(model: tuple[float, float], market: tuple[float, float] | None, w_market: float) -> tuple[float, float]:
    if market is None:
        return model
    lh = np.exp((1 - w_market) * np.log(model[0]) + w_market * np.log(market[0]))
    la = np.exp((1 - w_market) * np.log(model[1]) + w_market * np.log(market[1]))
    return float(lh), float(la)
