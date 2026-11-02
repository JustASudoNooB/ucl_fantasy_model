"""Re-tune the XI-history start update (settings: xi_history) on every matchday with actual XIs.

For each matchday k >= 2 with data/seed/lineups/md{k}_actual_xis.txt, the engine is built as of
the matchday's first date (so only earlier XIs are used) and start probabilities are scored
against who actually started: log loss, Brier and actual starters inside each club's top 11,
for outfield players and keepers separately. Run after each matchday's XIs are on file.

  python scripts/tune_xi_history.py
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufe.evaluate import grade                       # noqa: E402
from ufe.pipeline import Engine, load_fixtures       # noqa: E402

GRID = dict(kappa=[0.5, 1.0, 1.5, 2.5], decay=[0.5, 0.7, 1.0], kappa_gk=[0.25, 0.5, 1.0])


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tables():
    fx = load_fixtures()
    out = {}
    for md in sorted(fx.md.unique()):
        md = int(md)
        if md < 2 or grade.xis_path(md) is None:
            continue
        as_of = str(fx[fx.md == md].date.min())[:10]
        e = Engine(as_of=as_of)
        out[md] = e
    return out


def score(e: Engine, md: int, cfg: dict) -> dict:
    c = copy.deepcopy(e.cfg)
    c["xi_history"] = dict(enabled=True, **cfg)
    e.__dict__["cfg"] = c
    e._xi.clear()
    p = e.players_for(md)
    teams = set(e.fixtures[e.fixtures.md == md].home) | set(e.fixtures[e.fixtures.md == md].away)
    p = p[p.team.isin(teams)].copy()
    started = {i for i in grade._match_names(grade.load_xis(md), p) if i}
    p["y"] = p.id.isin(started).astype(int)
    gk, of = p[p.pos == "GK"], p[p.pos != "GK"]
    hits = int(sum(g.nlargest(11, "p_start").y.sum() for _, g in p.groupby("team")))
    return dict(ll=_ll(p.p_start, p.y), ll_gk=_ll(gk.p_start, gk.y), ll_out=_ll(of.p_start, of.y), hits=hits,
                starters=int(p.y.sum()))


def main():
    engines = tables()
    rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, vals))
        res = {md: score(e, md, cfg) for md, e in engines.items()}
        row = dict(cfg)
        for k in ("ll", "ll_gk", "ll_out"):
            row[k] = round(float(np.mean([r[k] for r in res.values()])), 4)
        row["hits"] = sum(r["hits"] for r in res.values())
        row["per_md"] = {md: round(r["ll"], 4) for md, r in res.items()}
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows).sort_values("ll")
    print(df.head(10).to_string(index=False))
    out = ROOT / "data" / "processed" / "tune_xi_history.json"
    out.write_text(json.dumps(dict(matchdays=list(engines), results=df.to_dict(orient="records")), indent=1, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
