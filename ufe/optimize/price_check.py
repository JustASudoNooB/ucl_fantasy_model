"""Price checks for players whose UCL Fantasy price is estimated or unknown.

For a candidate, the break-even price is the highest price at which forcing him into the
squad still reaches the best achievable objective (within `tol` points). If the price shown in
the app is at or below it, he belongs in the optimal squad; above it, he does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .squad import best_squad


def breakeven(players: pd.DataFrame, ev: pd.Series, budget: float, cap: int, cand: str, base_obj: float,
              lo: float = 4.0, hi: float = 12.0, tol: float = 0.05) -> float | None:
    pl = players.copy()
    m = pl.id == cand
    pl.loc[m, "price_status"] = "verified"
    grid = np.arange(lo, hi + 1e-9, 0.5)

    def ok(p: float) -> bool:
        pl.loc[m, "price"] = p
        try:
            s = best_squad(pl, ev, budget, cap, must_have=[cand], time_limit=20)
        except Exception:
            return False
        return cand in s.squad and s.objective >= base_obj - tol

    if not ok(grid[0]):
        return None
    a, b = 0, len(grid) - 1
    if ok(grid[b]):
        return float(grid[b])
    while b - a > 1:                       # ok(grid[a]) is True, ok(grid[b]) is False
        mid = (a + b) // 2
        if ok(grid[mid]):
            a = mid
        else:
            b = mid
    return float(grid[a])


def candidates(players: pd.DataFrame, ev: pd.Series, best_squad_ids: list[str], per_pos: int = 3) -> list[str]:
    """Estimated-price players in the best squad, plus the top unknown-price players per position
    whose projection beats the weakest pick at that position."""
    df = players.set_index("id")
    out = [i for i in best_squad_ids if df.loc[i, "price_status"] == "est_fpl"]
    for pos in ("GK", "DEF", "MID", "FWD"):
        picks = [i for i in best_squad_ids if df.loc[i, "pos"] == pos]
        floor = min(ev.get(i, 0.0) for i in picks) if picks else 0.0
        pool = df[(df.pos == pos) & (df.price_status != "verified") & (~df.index.isin(best_squad_ids))]
        top = ev.reindex(pool.index).dropna().sort_values(ascending=False)
        out += [i for i in top.index[:per_pos] if top[i] > floor]
    return out


def run(players: pd.DataFrame, ev: pd.Series, budget: float, cap: int, best, per_pos: int = 3) -> list[dict]:
    df = players.set_index("id")
    rows = []
    for c in candidates(players, ev, best.squad, per_pos):
        be = breakeven(players, ev, budget, cap, c, best.objective)
        rows.append(dict(id=c, name=df.loc[c, "name"], team=df.loc[c, "team"], pos=df.loc[c, "pos"],
                         status=df.loc[c, "price_status"], price_used=float(df.loc[c, "price"]),
                         xpts=round(float(ev.get(c, 0.0)), 2), breakeven=be, in_best=c in best.squad))
    return rows
