"""Every model's forecast for each manager's saved team, frozen at the deadline.

For each manager file with a squad, and each available model:
  the saved team (md{N} block: XI, bench order, captain, hits) is simulated with that model's
  numbers, with and without the between-day moves (policy.evaluate), net of hits:
  expected points, percentiles, chances of 80+ / 85+ / 90+ and below 60;
  each player's expected points; and the XI and captain that model would pick from the squad.
Written to data/runs/md{N}/forecast_{manager}.json. Before the deadline the file is rewritten on
every run (the last pre-deadline run wins); after it the file is never touched, unless `force`
(then it is marked pre_deadline: false with the reason). Graded in leaderboard.forecast_scores.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ..config import ROOT, rules
from ..optimize import policy
from ..optimize.squad import best_xi
from ..pipeline import Engine
from ..timeutil import deadline_for
from . import registry

RUNS = ROOT / "data" / "runs"
MANAGERS = ROOT / "data" / "managers"


def path(md: int, name: str) -> Path:
    return RUNS / f"md{md}" / f"forecast_{name.lower()}.json"


def deadline(md: int) -> datetime | None:
    row = next((r for r in rules()["matchdays"] if int(r["md"]) == md), None)
    return deadline_for(row).astimezone(timezone.utc) if row else None


def saved_team(players, cfg: dict, md: int) -> dict | None:
    """The manager's team for matchday md: ids of the squad, XI, bench (in order) and captain."""
    from ..report.managers import _match
    squad, missing = _match(players, cfg.get("squad", []))
    if missing or len(squad) != 15:
        return None
    blk = cfg.get(f"md{md}") or {}
    out = dict(squad=squad, hits=abs(int(blk.get("hits", 0) or 0)) // 4)
    if blk.get("xi") and blk.get("captain"):
        own = players[players.id.isin(squad)]          # names are matched inside the squad only
        xi, _ = _match(own, [{"name": n} for n in blk["xi"]])
        bench, _ = _match(own, [{"name": n} for n in blk.get("bench", [])])
        cap, _ = _match(own, [{"name": blk["captain"]}])
        xi = [p for p in xi if p in squad]
        bench = [p for p in bench if p in squad and p not in xi]
        bench += [p for p in squad if p not in xi and p not in bench]
        if len(xi) == 11 and cap and cap[0] in xi:
            out.update(xi=xi, bench=bench, captain=cap[0], source="saved in the manager file")
    return out


def _own_pick(sim, pl, by, squad, pick, md, hit) -> dict:
    """The XI and captain this model would choose from the squad, scored the same way as the saved team."""
    r = policy.evaluate(sim, pl, squad, pick["xi"], pick["captain"], pick["bench"], md)
    return dict(xi=[by.name[p] for p in pick["xi"]], bench=[by.name[p] for p in pick["bench"]], captain=by.name[pick["captain"]],
                formation=pick.get("formation"), expected=round(r.ev_active - hit, 1), expected_no_changes=round(r.ev_passive - hit, 1))


def team_forecast(e: Engine, md: int, team: dict) -> dict:
    pl = e.players_for(md)
    by = pl.set_index("id")
    sim = e.simulate(md)
    ev = e.projections(md).set_index("id").xpts.reindex(pl.id).fillna(0.0)
    hit = 4 * team.get("hits", 0)
    pick = best_xi(team["squad"], pl, ev)
    xi, bench, cap = team.get("xi") or pick["xi"], team.get("bench") or pick["bench"], team.get("captain") or pick["captain"]
    r = policy.evaluate(sim, pl, team["squad"], xi, cap, bench, md, keep_arrays=True)
    pas, act = (np.asarray(a, float) - hit for a in r.arrays)
    pct = lambda a: [round(float(x), 1) for x in np.percentile(a, [10, 25, 50, 75, 90])]
    return dict(expected=round(float(act.mean()), 1), expected_no_changes=round(float(pas.mean()), 1), hits=-hit,
                percentiles=pct(act), percentiles_no_changes=pct(pas),
                p80=round(float(np.mean(act >= 80)), 3), p85=round(float(np.mean(act >= 85)), 3),
                p90=round(float(np.mean(act >= 90)), 3), p_below_60=round(float(np.mean(act < 60)), 3),
                players={by.name[p]: round(float(ev.get(p, 0.0)), 2) for p in team["squad"]},
                own_pick=_own_pick(sim, pl, by, team["squad"], pick, md, hit),
                rules=[x["text"] for x in r.rules if x.get("kind") in ("captain", "sub")][:6])


def run(md: int | None = None, managers: list[str] | None = None, force: bool = False, n_sims: int = 10000,
        as_of: str | None = None, note: str | None = None, window_hours: float | None = None) -> list[str]:
    from ..report.managers import next_md
    md = md or next_md()
    now = datetime.now(timezone.utc)
    dl = deadline(md)
    files = sorted(MANAGERS.glob("*.yaml"))
    if managers:
        files = [f for f in files if f.stem in managers]
    todo = []
    for f in files:
        p = path(md, f.stem)
        if dl is not None and now >= dl and not force:
            continue                                   # frozen at the deadline
        if window_hours is not None and dl is not None and (dl - now).total_seconds() > window_hours * 3600 and not force:
            continue                                   # too early: keep CI minutes for the last hours
        todo.append((f, p))
    if not todo:
        return []
    as_of = as_of or now.date().isoformat()
    base = Engine(as_of=as_of, n_sims=n_sims)
    engines = {k: registry.build(k, as_of, n_sims=n_sims, base=base) for k in registry.available()}
    players = base.players_for(md)
    written = []
    for f, p in todo:
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        team = saved_team(players, cfg, md)
        if team is None:
            continue
        out = dict(manager=f.stem, md=md, made_at=now.isoformat(timespec="seconds"), deadline=dl.isoformat() if dl else None,
                   pre_deadline=bool(dl is None or now < dl), as_of=as_of, n_sims=n_sims,
                   team_source=team.get("source", "not saved: each model's best XI from the squad"), models={})
        if note:
            out["note"] = note
        if p.exists():
            old = json.loads(p.read_text(encoding="utf-8"))
            if "models" not in old:                    # a hand-written forecast from before this module
                out["saved_before_deadline"] = old
            elif old.get("saved_before_deadline"):
                out["saved_before_deadline"] = old["saved_before_deadline"]
        for k, e in engines.items():
            try:
                out["models"][k] = team_forecast(e, md, team)
            except Exception as ex:                    # one model failing must not lose the others
                out["models"][k] = dict(error=f"{type(ex).__name__}: {str(ex)[:160]}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        written.append(str(p))
    return written
