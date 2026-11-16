"""Numbers behind the dashboard's analytics views (player profiles, the lab, the reliability charts).

  season(engine)             each player's Champions League season so far from UEFA's game feed
                             (minutes, goals, assists, clean sheets, recoveries, saves, awards, cards,
                             points per matchday, ownership, transfers), with what the model expected
                             of him before each past deadline (the snapshots saved at the time).
  player_extras(sim, proj)   for one simulated matchday: where each player's expected points come from
                             (appearance, goals, assists, clean sheet, recoveries, award, the rest)
                             and the spread of his points (chance of each total from -2 to 20+).
  reliability()              how the models' probabilities compared with what happened, from the
                             leaderboard's bins (start chance v started, expected v official points).
Everything is keyed by the pool's player id and kept compact: the page embeds it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import ROOT, rules
from ..evaluate import grade

PROC = ROOT / "data" / "processed"
RUNS = ROOT / "data" / "runs"
SEASON_COLS = ["mins", "goals", "assists", "cs", "recoveries", "saves", "motm", "yc", "rc", "points"]
PARTS = ["appearance", "goals", "assists", "clean sheet", "recoveries", "award", "other"]
PMF_LO, PMF_HI = -2, 20


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0
    v = float(v)
    return int(v) if v.is_integer() else round(v, 1)


def season(engine) -> dict:
    f = engine.feed_snapshot
    if f is None or "mins" not in f:
        return {}
    from ..ingest.ucl_fantasy import played_mds
    mds = played_mds()
    pts = {m: grade.official_md_points(m) for m in mds}
    exp = {}
    for m in mds:
        p = RUNS / f"md{m}" / "snapshot.json"
        if p.exists():
            try:
                exp[m] = {r["id"]: r["xpts"] for r in json.loads(p.read_text(encoding="utf-8"))["players"]}
            except Exception:
                pass
    f = f[~f.index.duplicated()]
    players = {}
    for pid, r in f.iterrows():
        if not isinstance(pid, str):
            continue
        md_pts = [None if pts[m] is None or pid not in pts[m].index else _num(pts[m][pid]) for m in mds]
        md_exp = [exp.get(m, {}).get(pid) for m in mds]
        if not _num(r.get("mins")) and not any(md_exp):
            continue                                  # never played and never rated: nothing to show
        d = dict(s=[_num(r.get(c)) for c in SEASON_COLS], p=md_pts)
        if any(v is not None for v in md_exp):
            d["x"] = [None if v is None else round(float(v), 2) for v in md_exp]
        for k, c in (("own", "selected"), ("tin", "transfers_in"), ("tout", "transfers_out")):
            if c in r and not pd.isna(r[c]):
                d[k] = _num(r[c])
        flag = str(r.get("trained") or "")
        if flag.lower().startswith("unlikely"):
            d["flag"] = "Unlikely to start (UEFA)"
        players[pid] = d
    return dict(cols=SEASON_COLS, mds=mds, players=players, as_of=str(f.as_of.dropna().max()) if "as_of" in f else None)


def player_extras(sim, proj: pd.DataFrame, min_parts: float = 0.3, min_pmf: float = 1.0) -> dict:
    sc = rules()["scoring"]
    idx = sim.index()
    out = {}
    for r in proj.itertuples():
        if r.xpts < min_parts or r.id not in idx:
            continue
        i = idx[r.id]
        mins = sim.minutes[i]
        parts = [sc["appearance"] * float((mins > 0).mean()) + sc["sixty_minutes"] * float((mins >= 60).mean()),
                 float(sim.goals[i].mean()) * sc["goal"][r.pos],
                 float(sim.assists[i].mean()) * sc["assist"],
                 float((sim.cs[i] & (mins >= sc["clean_sheet_min_minutes"])).mean()) * sc["clean_sheet"][r.pos],
                 float((sim.recoveries[i] // sc["recoveries_step"]).mean()),
                 float(sim.potm[i].mean()) * sc["player_of_the_match"]]
        parts.append(float(r.xpts) - sum(parts))       # cards, goals conceded, saves, penalties, own goals
        d = dict(c=[round(v, 2) for v in parts])
        if r.xpts >= min_pmf:
            pts = sim.points[i].astype(int)
            pmf = np.bincount(np.clip(pts, PMF_LO, PMF_HI) - PMF_LO, minlength=PMF_HI - PMF_LO + 1) / len(pts)
            d["pmf"] = [int(round(v * 1000)) for v in pmf]
        out[r.id] = d
    return out


def reliability() -> dict:
    """Bins summed over the graded matchdays, per model: [n, sum of predictions, sum of outcomes]."""
    p = PROC / "model_leaderboard.json"
    if not p.exists():
        return {}
    lb = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for md, blk in (lb.get("mds") or {}).items():
        for name, r in blk.items():
            if not isinstance(r, dict):
                continue
            for key in ("cal_start", "cal_points"):
                if key not in r:
                    continue
                acc = out.setdefault(name, {}).setdefault(key, None)
                rows = np.array(r[key], dtype=float)
                out[name][key] = rows if acc is None else acc + rows
            for pos, v in (r.get("pos_points") or {}).items():
                acc = out.setdefault(name, {}).setdefault("pos_points", {})
                acc[pos] = (np.array(v, dtype=float) + acc[pos]) if pos in acc else np.array(v, dtype=float)
            out.setdefault(name, {}).setdefault("mds", []).append(int(md))
    out = {k: v for k, v in out.items() if v.get("cal_start") is not None or v.get("cal_points") is not None}
    for name, d in out.items():
        for key in ("cal_start", "cal_points"):
            if d.get(key) is not None:
                d[key] = [[int(n), round(float(a), 2), round(float(b), 2)] for n, a, b in d[key]]
        if "pos_points" in d:
            d["pos_points"] = {k: [int(v[0]), round(float(v[1]), 1), round(float(v[2]), 1)] for k, v in d["pos_points"].items()}
        d["mds"] = sorted(set(d["mds"]))
    return dict(models=out, start_edges=START_EDGES, points_edges=POINTS_EDGES) if out else {}


START_EDGES = [0, 0.02, 0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0]
POINTS_EDGES = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 20]


def bins(x, y, edges) -> list:
    """[n, sum x, sum y] for each interval of `edges` (the last one closed)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    return [[int((b == i).sum()), round(float(x[b == i].sum()), 3), round(float(y[b == i].sum()), 3)] for i in range(len(edges) - 1)]
