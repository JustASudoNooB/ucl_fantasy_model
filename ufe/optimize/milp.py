"""Tiny modelling layer over scipy.optimize.milp (HiGHS). No external solver install needed,
which keeps the engine portable to Windows/macOS laptops."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class Model:
    n: int = 0
    lb: list = field(default_factory=list)
    ub: list = field(default_factory=list)
    integ: list = field(default_factory=list)
    obj: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)      # (dict var->coef, lo, hi)
    names: dict = field(default_factory=dict)

    def var(self, name: str, lb: float = 0.0, ub: float = 1.0, integer: bool = True) -> int:
        i = self.n
        self.n += 1
        self.lb.append(lb)
        self.ub.append(ub)
        self.integ.append(1 if integer else 0)
        self.names[name] = i
        return i

    def block(self, prefix: str, k: int, **kw) -> np.ndarray:
        return np.array([self.var(f"{prefix}[{j}]", **kw) for j in range(k)])

    def add(self, coefs: dict[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        self.rows.append((coefs, lo, hi))

    def add_sum(self, idx, coefs=None, lo=-np.inf, hi=np.inf) -> None:
        idx = np.atleast_1d(idx)
        c = np.ones(len(idx)) if coefs is None else np.asarray(coefs, dtype=float)
        d: dict[int, float] = {}
        for i, v in zip(idx, c):
            d[int(i)] = d.get(int(i), 0.0) + float(v)
        self.add(d, lo, hi)

    def maximize(self, idx, coefs) -> None:
        for i, v in zip(np.atleast_1d(idx), np.atleast_1d(coefs)):
            self.obj[int(i)] = self.obj.get(int(i), 0.0) + float(v)

    def solve(self, time_limit: float = 60.0, mip_gap: float = 1e-4):
        c = np.zeros(self.n)
        for i, v in self.obj.items():
            c[i] = -v                                   # milp minimises
        r, cidx, vals, lo, hi = [], [], [], [], []
        for k, (d, l, h) in enumerate(self.rows):
            for i, v in d.items():
                r.append(k)
                cidx.append(i)
                vals.append(v)
            lo.append(l)
            hi.append(h)
        A = coo_matrix((vals, (r, cidx)), shape=(len(self.rows), self.n)).tocsr()
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.array(self.integ),
                   bounds=Bounds(self.lb, self.ub), options={"time_limit": time_limit, "mip_rel_gap": mip_gap})
        if res.x is None:
            raise RuntimeError(f"MILP failed: {res.message}")
        return np.round(res.x, 6), -res.fun
