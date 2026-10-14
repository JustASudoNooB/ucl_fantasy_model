"""Your own team: the live call for the current matchday and the plan for the next one.

Reads data/my/team.json (name, squad/XI files, captain, bank, free transfers, the app's points)
and writes data/runs/myteam.json, which the dashboard shows in its "Your team" section.

    python -m ufe --as-of <date> myteam --live-md 2 --next-md 3
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ..config import ROOT
from ..names import best_player_match
from ..optimize import policy, transfers
from ..optimize.squad import best_xi
from .export import _r, model_version

MY = ROOT / "data" / "my"
RUNS = ROOT / "data" / "runs"


def _ids(path, players, n):
    from ..cli import _read_squad
    return _read_squad(str(ROOT / path), players, n=n)


def build(engine, live_md: int | None, next_md: int, after_day: int = 1, horizon: int = 3) -> dict:
    cfg = json.loads((MY / "team.json").read_text(encoding="utf-8"))
    out = dict(name=cfg.get("name", "My team"), updated=cfg.get("updated"), created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               model_version=model_version(), points=cfg.get("points", {}), bank=cfg.get("bank"), bank_known=cfg.get("bank_known", True),
               free_transfers=cfg.get("free_transfers"))
    players = engine.players
    squad = _ids(cfg["squad"], players, 15)
    names = players.set_index("id").name
    price = players.set_index("id").price
    out["squad_cost"] = _r(sum(float(price[p]) for p in squad), 1)
    # ---- live call for the matchday in progress
    if live_md:
        from . import replan
        pl = engine.players_for(live_md)
        xi = _ids(cfg["xi"], pl, 11)
        cap = best_player_match(cfg["captain"], dict(zip(pl.name, pl.id)), cutoff=75)
        override = None
        ap = (cfg.get("app_points") or {}).get(str(live_md))
        if ap:
            override = {}
            for r in pd.read_csv(ROOT / ap).itertuples():
                cand = pl[pl.team == r.team]
                pid = best_player_match(r.name, dict(zip(cand.name, cand.id)), cutoff=75)
                if pid:
                    override[pid] = float(r.points)
        live = replan.run(engine, live_md, after_day, squad, xi, cap, points_override=override, freeze=False,
                          label=out["name"], out_name="replan_myteam.json")
        out["live"] = live
    # ---- next matchday: best XI, captain plan, transfers
    mds = [m for m in range(next_md, next_md + horizon) if m <= 8]
    E = engine.ev_matrix(mds)
    pn = engine.players_for(next_md)
    bx = best_xi(squad, pn, E[next_md])
    cap, rep, table = policy.best_captain(engine.simulate(next_md), pn, squad, bx["xi"], bx["bench"], next_md, E[next_md])
    proj = engine.projections(next_md).set_index("id")
    row = lambda p: dict(id=p, name=names[p], team=pn.set_index("id").team[p], pos=pn.set_index("id").pos[p],
                         xpts=_r(proj.xpts.get(p), 2), p_start=_r(proj.p_start_sim.get(p), 2),
                         horizon=_r(float(E.loc[p].sum()), 1))
    out["next"] = dict(md=next_md, horizon=mds, formation=bx["formation"], xi=[row(p) for p in bx["xi"]],
                       bench=[row(p) for p in bx["bench"]], captain=cap, captain_table=table,
                       ev_passive=_r(rep.ev_passive, 1), ev_policy=_r(rep.ev_active, 1),
                       flags=[dict(name=names[p], team=row(p)["team"], p_start=row(p)["p_start"]) for p in squad
                              if (proj.p_start_sim.get(p, 0) or 0) < 0.5])
    bank = float(cfg.get("bank") or 0.0)
    opts = transfers.options(pn, E, squad, bank, cfg.get("free_transfers"), mds, max_k=2)
    out["next"]["transfers"] = [dict(n=o.n_transfers, hits=o.hits, sell=[names[p] for p in o.sell], buy=[names[p] for p in o.buy],
                                     buy_ids=list(o.buy), gain=_r(o.gain_vs_roll, 1), horizon_ev=_r(o.horizon_ev, 1)) for o in opts]
    (RUNS / "myteam.json").write_text(json.dumps(out, default=str), encoding="utf-8")
    return out
