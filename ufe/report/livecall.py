"""Live captain switch and manual subs for a manager file, after a matchday's first day.

    python -m ufe livecall --manager zero            # the matchday being played, after its last finished day
    python -m ufe livecall --all --auto              # scheduler: only between days, alerts once per change

Points for the players who have played come from the official game feed: the season total now
minus the copy saved before the deadline (data/processed/feed_pre_md{N}.csv). When the feed has
not moved yet, match events (data/seed/results/md{N}_events.csv) are used as before. The XI and
captain are the manager file's `md{N}:` block when given, else the plan frozen at the deadline
(data/runs/md{N}/plan_{name}.json), else the model's best XI. Decisions come from
optimize.policy.live_decisions (whole plans, because any manual change ends auto-subs).
Writes data/runs/md{N}/live_{name}.json and returns a short message for Telegram.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT
from ..timeutil import IST, day_index, kickoff_dt

PROC = ROOT / "data" / "processed"
RUNS = ROOT / "data" / "runs"
MANAGERS = ROOT / "data" / "managers"
GAME_LENGTH = timedelta(hours=2, minutes=10)       # kick-off to final whistle, with stoppages


def days_status(fixtures: pd.DataFrame, md: int, now: datetime | None = None) -> dict:
    """{'finished': last finished day (0 = none), 'next_kickoff': first kick-off of the next day, 'n_days'}."""
    now = now or datetime.now(timezone.utc)
    f = fixtures[fixtures.md == md]
    if f.empty:
        return dict(finished=0, next_kickoff=None, n_days=0)
    di = day_index(f.date.tolist())
    kos = {}
    for r in f.itertuples():
        kos.setdefault(di[r.date], []).append(kickoff_dt(r.date, r.kickoff_cet).astimezone(timezone.utc))
    finished = 0
    for d in sorted(kos):
        if max(kos[d]) + GAME_LENGTH <= now:
            finished = d
        else:
            break
    nxt = min(kos[finished + 1]) if finished + 1 in kos else None
    return dict(finished=finished, next_kickoff=nxt, n_days=len(kos))


def feed_day_points(engine, md: int) -> dict[str, float] | None:
    """This matchday's points so far per player: feed total now minus the pre-deadline copy."""
    from ..pipeline import feed_points
    pre, cur = PROC / f"feed_pre_md{md}.csv", PROC / "players_feed.csv"
    if not pre.exists() or not cur.exists():
        return None
    a, b = feed_points(engine.pool, cur), feed_points(engine.pool, pre)
    if a.empty or b.empty:
        return None
    d = (a - b.reindex(a.index)).dropna()
    if not (d != 0).any():
        return None                      # feed not updated yet (or nobody has played)
    return {k: float(v) for k, v in d.items()}


def _plan_path(md: int, name: str) -> Path:
    return RUNS / f"md{md}" / f"plan_{name.lower()}.json"


def save_plan(plan: dict) -> Path | None:
    """The XI and captain the dashboard recommended, kept as the live call's default once the
    deadline passes (rewritten until then, never after)."""
    from ..config import rules
    from ..timeutil import deadline_for
    md = int(plan["md"])
    row = next((r for r in rules()["matchdays"] if int(r["md"]) == md), None)
    if row is None or "xi" not in plan:
        return None
    p = _plan_path(md, str(plan["name"]))
    if datetime.now(timezone.utc) >= deadline_for(row).astimezone(timezone.utc) and p.exists():
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: plan[k] for k in ("name", "md", "squad", "xi", "bench", "captain", "ev_xi") if k in plan}))
    return p


def run(engine, path: Path, md: int | None = None, after_day: int | None = None,
        now: datetime | None = None, points: dict[str, float] | None = None) -> dict:
    from ..names import best_player_match
    from ..optimize.squad import best_xi
    from . import replan
    from .managers import _match
    now = now or datetime.now(timezone.utc)
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    name = str(cfg.get("name") or Path(path).stem)
    if md is None:
        from ..cli import _live_md
        md = _live_md()
    st = days_status(engine.fixtures, md, now)
    day = after_day if after_day is not None else st["finished"]
    out = dict(manager=name, md=md, after_day=day, created_at=now.isoformat(timespec="minutes"), status="ok")
    if day < 1 or day >= st["n_days"]:
        out["status"] = "no live window" if day < 1 else "matchday finished"
        return out
    players = engine.players_for(md)
    squad, missing = _match(players, cfg.get("squad", []))
    if missing or len(squad) != 15:
        out.update(status="squad not matched", missing=missing)
        return out
    block = cfg.get(f"md{md}") or {}
    plan_p = _plan_path(md, name)
    plan = json.loads(plan_p.read_text()) if plan_p.exists() else {}
    source = "manager file" if block.get("xi") else ("plan frozen at the deadline" if plan.get("xi") else "model's best XI")
    if block.get("xi"):
        xi, _ = _match(players[players.id.isin(squad)], [dict(name=n) if isinstance(n, str) else n for n in block["xi"]])
    elif plan.get("xi") and set(plan["xi"]) <= set(squad):
        xi = plan["xi"]
    else:
        xi = best_xi(squad, players, engine.projections(md).set_index("id").xpts)["xi"]
    cap = block.get("captain")
    captain = best_player_match(cap, dict(zip(players[players.id.isin(squad)].name, players[players.id.isin(squad)].id)),
                                cutoff=75) if cap else plan.get("captain")
    if captain not in xi:
        captain = best_xi(squad, players, engine.projections(md).set_index("id").xpts)["captain"]
    pts_src = "given"
    if points is None:
        points = feed_day_points(engine, md)
        pts_src = "game feed" if points is not None else "match events"
    override = None
    if points is not None:
        sim = engine.simulate(md)
        idx = sim.index()
        override = {p: float(points.get(p, 0.0)) for p in squad if int(sim.day[idx[p]]) <= day}
    res = replan.run(engine, md, day, squad, xi, captain, points_override=override, freeze=False,
                     label=name, out_name=f"live_{name.lower()}.json")
    # chance that each later-day option beats the points already banked by the captain
    sim = engine.simulate(md)
    idx = sim.index()
    real_cap = next((r["realised"] for r in res["squad"] if r["id"] == captain), None)
    for a in res["decisions"].get("actions", []):
        if a.get("kind") == "captain" and real_cap is not None and a.get("to") in idx:
            a["p_beats"] = round(float((sim.points[idx[a["to"]]] > real_cap).mean()), 3)
    names = players.set_index("id").name
    res.update(manager=name, xi_source=source, points_source=pts_src, status="ok",
               next_kickoff_ist=st["next_kickoff"].astimezone(IST).strftime("%a %d %b %H:%M") if st["next_kickoff"] else None,
               captain_name=names.get(captain, captain))
    res["message"] = message(res)
    p = RUNS / f"md{md}" / f"live_{name.lower()}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, default=str))
    return res


def message(res: dict) -> str:
    sq = {r["id"]: r for r in res["squad"]}
    cap = sq.get(res["captain"], {})
    lines = [f"MD{res['md']} after day {res['after_day']} ({res['manager']}): day-{res['after_day']} XI points "
             f"{res['day_points']} ({res['points_source']}); captain {cap.get('name')} {cap.get('realised')}."]
    for a in res["decisions"].get("actions", []):
        extra = f" Chance he beats it: {a['p_beats']:.0%}." if a.get("p_beats") is not None else ""
        lines.append("• " + a["text"] + extra)
    if res.get("next_kickoff_ist"):
        lines.append(f"Changes close at the next kick-off, {res['next_kickoff_ist']} IST. XI from: {res['xi_source']}.")
    return "\n".join(lines)


def _fingerprint(res: dict) -> str:
    acts = [(a.get("kind"), a.get("to"), a.get("into"), a.get("out")) for a in res.get("decisions", {}).get("actions", [])]
    return hashlib.sha1(json.dumps([res.get("md"), res.get("after_day"), acts]).encode()).hexdigest()[:12]


def auto(engine, now: datetime | None = None, send=None) -> list[str]:
    """Scheduler entry: inside a between-days window, run every manager file and send the call
    when it is new or has changed. State: data/processed/livecall_state.json."""
    from ..cli import _live_md
    now = now or datetime.now(timezone.utc)
    md = _live_md()
    st = days_status(engine.fixtures, md, now)
    if st["finished"] < 1 or st["finished"] >= st["n_days"]:
        return []
    state_p = PROC / "livecall_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    notes = []
    for f in sorted(MANAGERS.glob("*.yaml")):
        res = run(engine, f, md=md, now=now)
        if res.get("status") != "ok":
            notes.append(f"{f.stem}: {res.get('status')}")
            continue
        key, fp = f"{f.stem}-md{md}-day{res['after_day']}", _fingerprint(res)
        if state.get(key) != fp:
            if send is not None:
                send(res["message"])
            state[key] = fp
        notes.append(f"{f.stem}: {res['decisions'].get('choice')}")
    state_p.write_text(json.dumps(state, indent=1))
    return notes
