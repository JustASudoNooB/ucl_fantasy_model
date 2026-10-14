"""Chip planner: when are Wildcard and Limitless worth the most?

For every matchday where a chip is legal:
  Limitless gain  = best unlimited-budget squad's XI EV  -  your squad's best XI EV  (that matchday)
  Wildcard gain   = horizon EV of the best budget-legal squad over the next H matchdays
                    -  horizon EV of your current squad over the same matchdays
Both are computed from the same simulated projections, and the in-matchday option value of
each squad (captain switch / manual subs) can be added from the policy simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings
from ..rules import budget, chip_allowed, club_cap, manual_actions_allowed
from .squad import best_squad, best_xi


def limitless_targets(players: pd.DataFrame, ev: pd.DataFrame, mds: list[int]) -> dict[int, dict]:
    out = {}
    for md in mds:
        if not chip_allowed("limitless", md):
            continue
        sol = best_squad(players, ev[md], budget=None, club_cap=club_cap(md))
        out[md] = sol.to_dict()
    return out


def wildcard_targets(players: pd.DataFrame, ev: pd.DataFrame, mds: list[int], horizon: int = 3) -> dict[int, dict]:
    decay = settings()["optimizer"]["horizon_decay"]
    out = {}
    all_mds = list(ev.columns)
    for md in mds:
        if not chip_allowed("wildcard", md):
            continue
        span = [m for m in all_mds if m >= md][:horizon]
        w = np.array([decay ** h for h in range(len(span))])
        hev = (ev[span] * w).sum(axis=1)
        sol = best_squad(players, hev, budget=budget(md), club_cap=club_cap(md))
        per_md = {int(m): round(best_xi(sol.squad, players, ev[m])["ev_xi"], 2) for m in span}
        out[md] = dict(sol.to_dict(), span=[int(m) for m in span], ev_by_md=per_md,
                       horizon_ev=round(float(sum(per_md[m] * wi for m, wi in zip(span, w))), 2))
    return out


def chip_table(players: pd.DataFrame, ev: pd.DataFrame, squad: list[str], mds: list[int], horizon: int = 3) -> pd.DataFrame:
    decay = settings()["optimizer"]["horizon_decay"]
    ll = limitless_targets(players, ev, mds)
    wc = wildcard_targets(players, ev, mds, horizon)
    rows = []
    for md in mds:
        cur = best_xi(squad, players, ev[md])["ev_xi"]
        row = dict(md=md, current_xi_ev=round(cur, 2), manual_actions=manual_actions_allowed(md))
        if md in ll:
            row["limitless_ev"] = ll[md]["ev_xi"]
            row["limitless_gain"] = round(ll[md]["ev_xi"] - cur, 2)
        if md in wc:
            span = wc[md]["span"]
            w = np.array([decay ** h for h in range(len(span))])
            cur_h = sum(best_xi(squad, players, ev[m])["ev_xi"] * wi for m, wi in zip(span, w))
            row["wildcard_gain"] = round(wc[md]["horizon_ev"] - cur_h, 2)
        rows.append(row)
    return pd.DataFrame(rows)
