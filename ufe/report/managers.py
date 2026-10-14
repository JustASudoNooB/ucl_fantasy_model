"""Managers: real squads entered in data/managers/*.yaml, each planned by the engine for the next
matchday (best XI and captain, transfer options over the coming matchdays, chip timing) and
compared with the model's own best squad and with each other."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT, rules
from ..names import best_player_match
from ..optimize import chips as chips_mod
from ..optimize import transfers
from ..optimize.squad import best_squad, best_xi
from ..rules import budget, chip_allowed, club_cap, free_transfers
from ..timeutil import deadline_for
from .export import _r

DIR = ROOT / "data" / "managers"


def next_md() -> int:
    now = datetime.now(timezone.utc)
    for r in rules()["matchdays"]:
        if deadline_for(r).astimezone(timezone.utc) > now:
            return int(r["md"])
    return int(rules()["matchdays"][-1]["md"])


def _match(players, rows) -> tuple[list[str], list[str]]:
    ids, missing = [], []
    for r in rows:
        cand = players[players.team == r.get("team")] if r.get("team") else players
        pid = best_player_match(str(r["name"]), dict(zip(cand.name, cand.id)), cutoff=75)
        (ids.append(pid) if pid else missing.append(str(r["name"])))
    return ids, missing


def plan_manager(engine, cfg: dict, md: int, mds: list[int]) -> dict:
    players = engine.players_for(md)
    by = players.set_index("id")
    squad, missing = _match(players, cfg.get("squad", []))
    out = dict(name=cfg.get("name"), team_name=cfg.get("team_name"), updated=str(cfg.get("updated", "")),
               md=md, missing=missing, squad=squad)
    if missing or len(squad) != 15:
        out["error"] = f"{len(squad)} of 15 players matched; check names: {', '.join(missing)}"
        return out
    horizon = [m for m in mds if m >= md][:3]
    E = engine.ev_matrix(sorted(set(horizon) | {m for m in mds if m >= md}))
    cost = float(by.loc[squad].price.sum())
    bank = cfg.get("bank")
    bank = round(budget(md) - cost, 1) if bank in (None, "") else float(bank)
    ft = cfg.get("free_transfers")
    ft = free_transfers(md)[0] if ft in (None, "") else int(ft)
    bx = best_xi(squad, players, E[md])
    ev = E[md]
    flags = []
    for p in squad:
        r = by.loc[p]
        if float(r.p_start) < 0.3:
            flags.append(dict(id=p, name=r["name"], team=r.team, p_start=_r(r.p_start, 2),
                              why=str(r.get("status") or "unlikely to start")))
    opts = transfers.options(players, E, squad, bank, ft, horizon, max_k=min(3, (ft or 2) + 1))
    model = best_squad(players, ev, budget(md), club_cap(md))
    used = set(cfg.get("chips_used") or [])
    league = [m for m in mds if m >= md and m <= 8]
    tab = chips_mod.chip_table(players, engine.ev_matrix(league), squad, league) if league else None
    chip_rows = tab.round(2).to_dict(orient="records") if tab is not None else []
    advice = []
    for chip, key, thr in (("wildcard", "wildcard_gain", 12.0), ("limitless", "limitless_gain", 25.0)):
        if chip in used or tab is None or key not in tab:
            continue
        t = tab.dropna(subset=[key])
        t = t[[chip_allowed(chip, int(m)) for m in t.md]]
        if t.empty:
            continue
        best = t.loc[t[key].idxmax()]
        now = float(t[t.md == md][key].iloc[0]) if (t.md == md).any() else None
        advice.append(dict(chip=chip, best_md=int(best.md), best_gain=_r(best[key], 1), now_gain=_r(now, 1),
                           use_now=bool(now is not None and now >= thr and now >= 0.9 * float(best[key])), threshold=thr))
    out.update(
        bank=bank, free_transfers=ft, chips_used=sorted(used), squad_value=_r(cost, 1),
        xi=bx["xi"], bench=bx["bench"], captain=bx["captain"], formation=bx["formation"], ev_xi=_r(bx["ev_xi"], 2),
        flags=flags,
        transfers=[dict(n=o.n_transfers, hits=o.hits, sell=o.sell, buy=o.buy, gain=_r(o.gain_vs_roll, 2),
                        horizon_ev=_r(o.horizon_ev, 2)) for o in opts[:4]],
        horizon=horizon, chips=chip_rows, chip_advice=advice,
        model=dict(ev_xi=_r(model.ev_xi, 2), xi=model.xi, squad=model.squad,
                   overlap=len(set(model.squad) & set(squad))),
        md2=cfg.get("md2"),
    )
    return out


def all_managers(engine, mds: list[int]) -> list[dict]:
    if not DIR.exists():
        return []
    md = next_md()
    out = []
    for p in sorted(DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            out.append(plan_manager(engine, cfg, md, mds))
        except Exception as e:                       # one bad file must not break the dashboard
            out.append(dict(name=p.stem, error=str(e)))
    return out
