"""Multi-matchday transfer planner (mixed-integer programme).

Decides, for the next H matchdays at once: who to sell and buy, the starting XI and captain
each week, and when to take -4 hits, while tracking free-transfer rollover (max one carried in
the league phase, none in the knockouts), club caps by stage, the budget, and optional chips
(Wildcard / Limitless) on given matchdays. Planning jointly avoids the classic mistake of
burning a transfer this week on a player who blanks next week.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import rules, settings
from ..rules import POSITIONS, club_cap, free_transfers, stage_of
from .milp import Model


@dataclass
class TransferOption:
    n_transfers: int
    hits: int
    sell: list[str]
    buy: list[str]
    horizon_ev: float          # weighted, net of hits
    ev_by_md: dict
    plan: list[dict]           # per matchday: squad, xi, captain, in, out, chip
    gain_vs_roll: float = 0.0

    def to_dict(self) -> dict:
        return dict(n_transfers=self.n_transfers, hits=self.hits, sell=self.sell, buy=self.buy,
                    horizon_ev=round(self.horizon_ev, 2), gain_vs_roll=round(self.gain_vs_roll, 2),
                    ev_by_md={int(k): round(v, 2) for k, v in self.ev_by_md.items()}, plan=self.plan)


def _candidates(players: pd.DataFrame, ev: pd.DataFrame, squad: list[str], weights: np.ndarray, per_pos: dict) -> list[str]:
    score = (ev.fillna(0).to_numpy() * weights[None, :]).sum(axis=1)
    sc = pd.Series(score, index=ev.index)
    from .squad import pickable
    df = pickable(players, keep=squad).set_index("id")
    keep = set(squad)
    for p, k in per_pos.items():
        ids = [i for i in sc.index if i in df.index and df.loc[i, "pos"] == p]
        keep |= set(sc.loc[ids].sort_values(ascending=False).index[:k])
    return [i for i in ev.index if i in keep and i in df.index]


def plan(players: pd.DataFrame, ev: pd.DataFrame, squad: list[str], bank: float, free_now: int | None,
         mds: list[int], first_md_transfers: int | None = None, chips: dict[int, str] | None = None,
         per_pos: dict | None = None, time_limit: float = 90.0) -> TransferOption:
    """ev: index = player id, columns = matchdays (same order as `mds`)."""
    cfg = settings()
    bw = cfg["optimizer"]["bench_weight"]
    decay = cfg["optimizer"]["horizon_decay"]
    hit = rules()["transfers"]["hit_cost"]
    chips = chips or {}
    H = len(mds)
    w = np.array([decay ** h for h in range(H)])
    per_pos = per_pos or {"GK": 10, "DEF": 30, "MID": 30, "FWD": 20}
    ids = _candidates(players, ev[mds], squad, w, per_pos)
    df = players.set_index("id").loc[ids]
    P = len(ids)
    E = ev.loc[ids, mds].fillna(0.0).to_numpy()
    pos = df.pos.to_numpy()
    team = df.team.to_numpy()
    price = df.price.to_numpy()
    x0 = np.array([1 if i in squad else 0 for i in ids])
    value0 = float(players.set_index("id").loc[squad].price.sum())
    want = rules()["squad"]["positions"]
    lo, hi = rules()["lineup"]["min"], rules()["lineup"]["max"]
    budget_league = rules()["squad"]["budget"]["league_phase"]
    budget_ko = rules()["squad"]["budget"]["knockout"]

    m = Model()
    X, S, C, TI, TO = [], [], [], [], []
    t_v, e_v, u_v, r_v, f_v = [], [], [], [], []
    for h, md in enumerate(mds):
        chip = chips.get(md)
        x = m.block(f"x{h}", P)
        ti = m.block(f"in{h}", P)
        to = m.block(f"out{h}", P)
        X.append(x)
        TI.append(ti)
        TO.append(to)
        prev = x0 if h == 0 else None
        for i in range(P):
            if h == 0:
                m.add({int(x[i]): 1, int(ti[i]): -1, int(to[i]): 1}, lo=float(prev[i]), hi=float(prev[i]))
            else:
                m.add({int(x[i]): 1, int(X[h - 1][i]): -1, int(ti[i]): -1, int(to[i]): 1}, lo=0, hi=0)
            m.add({int(ti[i]): 1, int(to[i]): 1}, hi=1)
        for p in POSITIONS:
            m.add_sum(x[np.where(pos == p)[0]], lo=want[p], hi=want[p])
        cap = club_cap(md)
        for tm in np.unique(team):
            m.add_sum(x[np.where(team == tm)[0]], hi=cap)
        extra = (budget_ko - budget_league) if stage_of(md) != "league" else 0.0
        m.add_sum(x, price, hi=value0 + bank + extra + 1e-6)
        # line-up on the squad actually used this matchday (temporary squad under Limitless)
        if chip == "limitless":
            y = m.block(f"y{h}", P)
            for p in POSITIONS:
                m.add_sum(y[np.where(pos == p)[0]], lo=want[p], hi=want[p])
            for tm in np.unique(team):
                m.add_sum(y[np.where(team == tm)[0]], hi=cap)
            use = y
            m.add_sum(ti, hi=0)          # permanent squad frozen this matchday
        else:
            use = x
        s = m.block(f"s{h}", P)
        c = m.block(f"c{h}", P)
        S.append(s)
        C.append(c)
        m.add_sum(s, lo=11, hi=11)
        m.add_sum(c, lo=1, hi=1)
        for p in POSITIONS:
            m.add_sum(s[np.where(pos == p)[0]], lo=lo[p], hi=hi[p])
        for i in range(P):
            m.add({int(s[i]): 1, int(use[i]): -1}, hi=0)
            m.add({int(c[i]): 1, int(s[i]): -1}, hi=0)
        m.maximize(s, w[h] * E[:, h] * (1 - bw))
        m.maximize(use, w[h] * E[:, h] * bw)
        m.maximize(c, w[h] * E[:, h])
        # transfer accounting
        t = m.var(f"t{h}", 0, 15)
        e = m.var(f"e{h}", 0, 15)
        u = m.var(f"u{h}", 0, 15)
        r = m.var(f"r{h}", 0, 1)
        f = m.var(f"f{h}", 0, 15)
        t_v.append(t)
        e_v.append(e)
        u_v.append(u)
        r_v.append(r)
        f_v.append(f)
        d = {int(t): 1.0}
        for i in range(P):
            d[int(ti[i])] = -1.0
        m.add(d, lo=0, hi=0)
        base, carry = free_transfers(md)
        unlimited = base is None or chip in ("wildcard",)
        if unlimited:
            m.add({int(e): 1}, lo=0, hi=0)
            m.add({int(r): 1}, lo=0, hi=0)
            m.add({int(f): 1}, lo=15, hi=15)
        else:
            if h == 0:
                fv = free_now if free_now is not None else base
                m.add({int(f): 1}, lo=fv, hi=fv)
            else:
                prev_md = mds[h - 1]
                _, prev_carry = free_transfers(prev_md)
                prev_unlimited = free_transfers(prev_md)[0] is None or chips.get(prev_md) == "wildcard"
                if prev_carry > 0 and not prev_unlimited:
                    m.add({int(f): 1, int(r_v[h - 1]): -1}, lo=base, hi=base)
                else:
                    m.add({int(f): 1}, lo=base, hi=base)
            m.add({int(e): 1, int(t): -1, int(f): 1}, lo=0)           # e >= t - f
            m.add({int(u): 1, int(f): -1, int(t): 1, int(e): -1}, lo=0, hi=0)   # u = f - t + e
            m.add({int(r): 1, int(u): -1}, hi=0)
            if carry == 0:
                m.add({int(r): 1}, lo=0, hi=0)
        m.maximize([e], [-hit * w[h]])
        if h == 0 and first_md_transfers is not None:
            m.add({int(t): 1}, lo=first_md_transfers, hi=first_md_transfers)

    sol, obj = m.solve(time_limit=time_limit)
    plan_rows, ev_by_md = [], {}
    hits_total = 0
    for h, md in enumerate(mds):
        xi = [ids[i] for i in range(P) if sol[S[h][i]] > 0.5]
        cap = [ids[i] for i in range(P) if sol[C[h][i]] > 0.5][0]
        sq = [ids[i] for i in range(P) if sol[X[h][i]] > 0.5]
        bought = [ids[i] for i in range(P) if sol[TI[h][i]] > 0.5]
        sold = [ids[i] for i in range(P) if sol[TO[h][i]] > 0.5]
        e_h = int(round(sol[e_v[h]]))
        hits_total += e_h if h == 0 else 0
        ev_md = float(E[[ids.index(p) for p in xi], h].sum() + E[ids.index(cap), h])
        ev_by_md[md] = ev_md
        plan_rows.append(dict(md=md, chip=chips.get(md), buy=bought, sell=sold, hits=e_h,
                              free=int(round(sol[f_v[h]])) if sol[f_v[h]] < 15 else None,
                              xi=xi, captain=cap, squad=sq, ev=round(ev_md, 2)))
    first = plan_rows[0]
    return TransferOption(n_transfers=len(first["buy"]), hits=first["hits"], sell=first["sell"], buy=first["buy"],
                          horizon_ev=float(obj), ev_by_md=ev_by_md, plan=plan_rows)


def options(players: pd.DataFrame, ev: pd.DataFrame, squad: list[str], bank: float, free_now: int | None,
            mds: list[int], max_k: int = 3, chips: dict[int, str] | None = None) -> list[TransferOption]:
    """Best plan for each number of transfers this matchday (0..max_k) plus the unconstrained best."""
    out = []
    for k in range(0, max_k + 1):
        try:
            out.append(plan(players, ev, squad, bank, free_now, mds, first_md_transfers=k, chips=chips))
        except RuntimeError:
            continue
    if out:
        roll = out[0].horizon_ev
        for o in out:
            o.gain_vs_roll = o.horizon_ev - roll
    return sorted(out, key=lambda o: -o.horizon_ev)
