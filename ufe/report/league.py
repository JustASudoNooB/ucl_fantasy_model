"""Mini-league mode: the chance of finishing first, not only expected points.

Every manager file in data/managers/ (yours plus rivals) lists a 15-man squad and, for this mode,
`league_points:` (the total shown in the mini-league table). Rivals are assumed to field their
best expected XI and captain unless their file has an `md{N}:` block.

For the coming matchday the joint simulation gives every manager's points in each simulated
world (shared players move together, so owning the same captain as the leader cancels out).
From it:
  * P(top of the league after the matchday) and P(beat each rival) for your plan, and for
    alternatives: other captains, and the transfer options the manager section lists;
  * differentials: players a rival owns that you do not, and the reverse, with expected points;
  * P(top after MD8), adding the later league-phase matchdays as correlated normal draws
    (means from each squad's best XI per matchday, spread and correlation from this matchday).
Plans are ranked by P(first after MD8), then expected points: when you trail, a differential
captain can raise the chance of finishing first while lowering expected points.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT
from ..optimize.policy import POS_ORDER, auto_subs

MANAGERS = ROOT / "data" / "managers"


def team_sims(sim, players: pd.DataFrame, squad: list[str], xi: list[str], captain: str) -> np.ndarray:
    """Points per simulated matchday for one team (XI + captain + auto-subs in bench order)."""
    idx = sim.index()
    pos_of = players.set_index("id").pos
    pos_arr = np.zeros(len(sim.player_ids), dtype=int)
    for p in squad:
        pos_arr[idx[p]] = POS_ORDER.index(pos_of[p])
    pts = sim.points.astype(float)
    played = sim.minutes > 0
    XI = [idx[p] for p in xi]
    bench = [p for p in squad if p not in xi]
    gk = [idx[p] for p in bench if pos_of[p] == "GK"]
    out = [idx[p] for p in bench if pos_of[p] != "GK"]
    return pts[XI].sum(axis=0) + pts[idx[captain]] + auto_subs(pts, played, pos_arr, XI, gk + out)


def _p_first(totals: np.ndarray, me: int) -> float:
    """totals (M, N): share of worlds where manager `me` is top (ties split)."""
    best = totals.max(axis=0)
    top = totals == best
    return float((top[me] / top.sum(axis=0)).mean())


def run(engine, md: int, mds: list[int]) -> dict | None:
    from ..optimize.squad import best_xi
    from .managers import _match, plan_manager
    files = sorted(MANAGERS.glob("*.yaml"))
    cfgs = []
    for f in files:
        c = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if c.get("league_points") is not None and c.get("squad"):
            cfgs.append((f.stem, c))
    me_i = next((i for i, (stem, c) in enumerate(cfgs) if c.get("me")), None)
    if len(cfgs) < 2 or me_i is None:
        return None
    players = engine.players_for(md)
    sim = engine.simulate(md)
    idx = sim.index()
    E = engine.ev_matrix([m for m in mds if m >= md and m <= 8])
    names = players.set_index("id").name
    teams = []
    for stem, c in cfgs:
        squad, missing = _match(players, c["squad"])
        if missing or len(squad) != 15 or any(p not in idx for p in squad):
            teams.append(dict(stem=stem, name=c.get("name", stem), error=f"squad not matched: {', '.join(missing)}"))
            continue
        block = c.get(f"md{md}") or {}
        bx = best_xi(squad, players, E[md])
        xi, cap = bx["xi"], bx["captain"]
        if block.get("xi"):
            got, _ = _match(players[players.id.isin(squad)], [dict(name=n) for n in block["xi"]])
            if len(got) == 11:
                xi = got
        if block.get("captain"):
            got, _ = _match(players[players.id.isin(xi)], [dict(name=block["captain"])])
            cap = got[0] if got else cap
        teams.append(dict(stem=stem, name=c.get("name", stem), squad=squad, xi=xi, captain=cap,
                          points=float(c["league_points"]), cfg=c))
    ok = [t for t in teams if "error" not in t]
    if len(ok) < 2 or not any(t["stem"] == cfgs[me_i][0] for t in ok):
        return dict(error="need your file and at least one rival with matched squads", teams=[
            {k: t.get(k) for k in ("name", "error")} for t in teams])
    me = next(i for i, t in enumerate(ok) if t["stem"] == cfgs[me_i][0])
    md_pts = np.stack([team_sims(sim, players, t["squad"], t["xi"], t["captain"]) for t in ok])
    now = np.array([t["points"] for t in ok])[:, None]
    after = now + md_pts
    # later league-phase matchdays: correlated normal draws around each squad's best-XI expectation
    later = [m for m in mds if md < m <= 8]
    fut_mean = np.zeros(len(ok))
    for m in later:
        for i, t in enumerate(ok):
            fut_mean[i] += best_xi(t["squad"], engine.players_for(m), E[m])["ev_xi"] if m in E else 0.0
    cov = np.cov(md_pts) * max(len(later), 0)
    rng = np.random.default_rng(int(engine.cfg["simulation"]["seed"]) + 77)
    fut = rng.multivariate_normal(fut_mean, cov, size=md_pts.shape[1]).T if later else np.zeros_like(md_pts)
    final = after + fut

    def summary(i: int) -> dict:
        return dict(name=ok[i]["name"], points=ok[i]["points"], md_exp=round(float(md_pts[i].mean()), 1),
                    md_p10=round(float(np.percentile(md_pts[i], 10)), 0), md_p90=round(float(np.percentile(md_pts[i], 90)), 0),
                    p_top_after_md=round(_p_first(after, i), 3), p_top_md8=round(_p_first(final, i), 3),
                    captain=names.get(ok[i]["captain"], ok[i]["captain"]), me=i == me)

    table = [summary(i) for i in range(len(ok))]
    # your alternatives: captains, then the transfer options of the manager section
    mine = ok[me]
    plans = []

    def score(label: str, squad, xi, cap, extra_ev: float = 0.0) -> dict:
        x = team_sims(sim, players, squad, xi, cap)
        a = after.copy(); a[me] = now[me] + x
        f_ = final.copy(); f_[me] = now[me] + x + fut[me] + extra_ev
        beats = {ok[j]["name"]: round(float((a[me] > a[j]).mean()), 3) for j in range(len(ok)) if j != me}
        return dict(plan=label, md_exp=round(float(x.mean()), 1), p_top_after_md=round(_p_first(a, me), 3),
                    p_top_md8=round(_p_first(f_, me), 3), beats=beats)

    ev_md = E[md]
    for cap in sorted(mine["xi"], key=lambda p: -float(ev_md.get(p, 0)))[:5]:
        plans.append(score(f"captain {names[cap]}" + (" (model pick)" if cap == mine["captain"] else ""),
                           mine["squad"], mine["xi"], cap))
    try:
        mp = plan_manager(engine, mine["cfg"], md, mds)
        for o in (mp.get("transfers") or [])[:4]:
            if not o["n"]:
                continue
            sq = [p for p in mine["squad"] if p not in o["sell"]] + list(o["buy"])
            if any(p not in idx for p in sq):
                continue
            bx = best_xi(sq, players, ev_md)
            later_gain = float(o.get("gain", 0.0)) - (bx["ev_xi"] - best_xi(mine["squad"], players, ev_md)["ev_xi"])
            plans.append(score(f"{' + '.join(names[p] for p in o['buy'])} for {' + '.join(names[p] for p in o['sell'])}"
                               + (f" ({o['hits']} hit)" if o["hits"] else ""), sq, bx["xi"], bx["captain"],
                               extra_ev=max(later_gain, 0.0) - 4 * o["hits"]))
    except Exception:                                     # the transfer search is optional here
        pass
    plans.sort(key=lambda p: (-p["p_top_md8"], -p["md_exp"]))
    # differentials against each rival
    ev_all = E.sum(axis=1)
    diffs = []
    for j, t in enumerate(ok):
        if j == me:
            continue
        theirs = [p for p in t["squad"] if p not in mine["squad"]]
        ours = [p for p in mine["squad"] if p not in t["squad"]]
        row = lambda p: dict(name=names[p], team=str(players.set_index("id").team[p]), md=round(float(ev_md.get(p, 0)), 1),
                             horizon=round(float(ev_all.get(p, 0)), 1))
        diffs.append(dict(rival=t["name"], shared=15 - len(theirs), theirs=[row(p) for p in sorted(theirs, key=lambda p: -float(ev_md.get(p, 0)))],
                          ours=[row(p) for p in sorted(ours, key=lambda p: -float(ev_md.get(p, 0)))],
                          same_captain=t["captain"] == mine["captain"]))
    return dict(md=md, table=sorted(table, key=lambda r: -r["points"]), plans=plans, differentials=diffs,
                later_mds=later, n_sims=int(md_pts.shape[1]),
                note="Rivals assumed to field their best expected XI and captain. Later matchdays use each squad as it "
                     "stands (no transfers), drawn as correlated normals with this matchday's spread.")
