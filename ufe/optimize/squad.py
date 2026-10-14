"""Squad, starting XI and captain selection (single matchday or Limitless / Wildcard builds)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import rules, settings
from ..rules import POSITIONS, formation_label, formations
from .milp import Model


@dataclass
class SquadSolution:
    squad: list[str]
    xi: list[str]
    captain: str
    bench: list[str]
    formation: str
    ev_xi: float            # starting XI incl. captain double
    cost: float
    objective: float

    def to_dict(self) -> dict:
        return dict(squad=self.squad, xi=self.xi, captain=self.captain, bench=self.bench, formation=self.formation,
                    ev_xi=round(self.ev_xi, 3), cost=round(self.cost, 1), objective=round(self.objective, 3))


PICKABLE = {"official", "verified"}


def pickable(players: pd.DataFrame, keep: list[str] | None = None) -> pd.DataFrame:
    """Players the optimiser may buy: official or published prices. Estimated prices (est_fpl) are
    allowed only while no official list is loaded; unknown prices never (they are only simulated)."""
    if "price_status" not in players:
        return players
    ok = set(PICKABLE)
    if not (players.price_status == "official").any():
        ok.add("est_fpl")
    return players[players.price_status.isin(ok) | players.id.isin(keep or [])]


def best_squad(players: pd.DataFrame, ev: pd.Series, budget: float | None, club_cap: int,
               formation: tuple[int, int, int] | None = None, bench_weight: float | None = None,
               must_have: list[str] | None = None, banned: list[str] | None = None,
               exclude_squads: list[list[str]] | None = None, time_limit: float = 60) -> SquadSolution:
    """players: id, pos, team, price. ev: expected points per player id for the matchday (or horizon)."""
    cfg = settings()
    bw = cfg["optimizer"]["bench_weight"] if bench_weight is None else bench_weight
    players = pickable(players, keep=must_have)
    df = players.set_index("id").loc[ev.index.intersection(players.id)]
    ids = list(df.index)
    P = len(ids)
    e = ev.loc[ids].to_numpy()
    m = Model()
    x = m.block("x", P)
    s = m.block("s", P)
    c = m.block("c", P)
    m.maximize(s, e * (1 - bw))
    m.maximize(x, e * bw)
    m.maximize(c, e)
    pos = df.pos.to_numpy()
    want = rules()["squad"]["positions"]
    lo, hi = rules()["lineup"]["min"], rules()["lineup"]["max"]
    m.add_sum(x, lo=15, hi=15)
    m.add_sum(s, lo=11, hi=11)
    m.add_sum(c, lo=1, hi=1)
    for p in POSITIONS:
        ix = np.where(pos == p)[0]
        m.add_sum(x[ix], lo=want[p], hi=want[p])
        if formation is None:
            m.add_sum(s[ix], lo=lo[p], hi=hi[p])
    if formation is not None:
        d_, m_, f_ = formation
        for p, k in (("GK", 1), ("DEF", d_), ("MID", m_), ("FWD", f_)):
            m.add_sum(s[np.where(pos == p)[0]], lo=k, hi=k)
    if budget is not None:
        m.add_sum(x, df.price.to_numpy(), hi=budget + 1e-6)
    for t in df.team.unique():
        m.add_sum(x[np.where(df.team.to_numpy() == t)[0]], hi=club_cap)
    for i in range(P):
        m.add({int(s[i]): 1, int(x[i]): -1}, hi=0)
        m.add({int(c[i]): 1, int(s[i]): -1}, hi=0)
    for pid in must_have or []:
        if pid in ids:
            m.add({int(x[ids.index(pid)]): 1}, lo=1, hi=1)
    for pid in banned or []:
        if pid in ids:
            m.add({int(x[ids.index(pid)]): 1}, lo=0, hi=0)
    for sq in exclude_squads or []:
        ix = [ids.index(p) for p in sq if p in ids]
        m.add_sum(x[ix], hi=len(ix) - 1)
    sol, obj = m.solve(time_limit=time_limit)
    sq = [ids[i] for i in range(P) if sol[x[i]] > 0.5]
    xi = [ids[i] for i in range(P) if sol[s[i]] > 0.5]
    cap = [ids[i] for i in range(P) if sol[c[i]] > 0.5][0]
    bench = order_bench([p for p in sq if p not in xi], df, ev)
    f = tuple(int((df.loc[xi].pos == p).sum()) for p in ("DEF", "MID", "FWD"))
    ev_xi = float(ev.loc[xi].sum() + ev.loc[cap])
    return SquadSolution(sq, xi, cap, bench, formation_label(f), ev_xi, float(df.loc[sq].price.sum()), float(obj))


def order_bench(bench: list[str], df: pd.DataFrame, ev: pd.Series) -> list[str]:
    """Goalkeeper last-but-not-least convention: outfield bench by EV, keeper fourth."""
    gk = [p for p in bench if df.loc[p, "pos"] == "GK"]
    out = sorted([p for p in bench if df.loc[p, "pos"] != "GK"], key=lambda p: -ev.loc[p])
    return out + gk


def best_xi(squad: list[str], players: pd.DataFrame, ev: pd.Series, formation: tuple[int, int, int] | None = None) -> dict:
    """Best XI for an owned squad: enumerate valid formations (fast, exact)."""
    df = players.set_index("id").loc[squad]
    best = None
    shapes = [formation] if formation else formations()
    for (d_, m_, f_) in shapes:
        pick = []
        ok = True
        for p, k in (("GK", 1), ("DEF", d_), ("MID", m_), ("FWD", f_)):
            cands = df[df.pos == p].index.tolist()
            cands.sort(key=lambda q: -ev.get(q, 0.0))
            if len(cands) < k:
                ok = False
                break
            pick += cands[:k]
        if not ok:
            continue
        cap = max(pick, key=lambda q: ev.get(q, 0.0))
        val = float(sum(ev.get(q, 0.0) for q in pick) + ev.get(cap, 0.0))
        if best is None or val > best["ev_xi"]:
            best = dict(xi=pick, captain=cap, formation=formation_label((d_, m_, f_)), ev_xi=val)
    best["bench"] = order_bench([p for p in squad if p not in best["xi"]], df, ev)
    return best
