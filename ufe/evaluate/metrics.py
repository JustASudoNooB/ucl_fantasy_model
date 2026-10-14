"""Proper scoring rules and calibration diagnostics."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def outcome_index(hg: np.ndarray, ag: np.ndarray) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score for ordered outcomes (home, draw, away). Lower is better."""
    k = p.shape[1]
    cum_p = np.cumsum(p, axis=1)
    onehot = np.eye(k)[y]
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o)[:, :-1] ** 2, axis=1) / (k - 1)))


def brier_multi(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def brier_binary(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss_binary(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability diagram data: mean forecast vs observed frequency per probability bin."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        out.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), forecast=float(p[m].mean()), observed=float(y[m].mean())))
    return out


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    tab = calibration_table(p, y, bins)
    n = sum(t["n"] for t in tab)
    return float(sum(t["n"] * abs(t["forecast"] - t["observed"]) for t in tab) / max(n, 1))


def crps_discrete(pmf: np.ndarray, y: int) -> float:
    """CRPS for a distribution over 0..K (pmf) and an integer outcome y."""
    cdf = np.cumsum(pmf)
    k = np.arange(len(pmf))
    return float(np.sum((cdf - (k >= y)) ** 2))


def crps_samples(samples: np.ndarray, y: float) -> float:
    """CRPS estimated from Monte Carlo samples (energy form)."""
    s = np.asarray(samples, dtype=float)
    return float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))) if len(s) < 3000 else \
        float(np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(np.sort(s)[::7][:, None] - np.sort(s)[::7][None, :])))
