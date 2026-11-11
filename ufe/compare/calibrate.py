"""Refit the model's constants on what UEFA's game feed says happened. Run after each matchday
(weekly workflow), so systematic errors are corrected without anyone editing a file.

  recovery_ratio[pos]    UEFA ball recoveries per unit of the model's Opta-scale rate x minutes
  assist_pos_mult[pos]   the game's assists by position against the model's shares, within each club
  slots[pos]             starters per club by the game's positions (reported; SLOTS in player_rates.py)
  availability           how often players carrying each UEFA flag before a deadline then started
                         (reported; multipliers in settings.yaml)

Each fitted constant moves at most `max_step` (15%) from its current value per run and only with
enough data behind it, so one odd matchday cannot swing the model. Fitted values go to
data/processed/params.json (settings() merges it over settings.yaml); the full report, with sample
sizes and the values before and after, goes to data/processed/calibration.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import ROOT, settings
from ..evaluate import grade
from ..pipeline import Engine, _feed_ids, load_pool

PROC = ROOT / "data" / "processed"
OUT = PROC / "calibration.json"
PARAMS = PROC / "params.json"
MIN_MINUTES = 3000          # player-minutes per position before a recovery ratio is refitted
MIN_ASSISTS = 30            # assists per position before its multiplier is refitted
MIN_CHANGE = 0.03           # smaller differences than this are left alone (noise)


def _step(old: float, target: float, max_step: float) -> float:
    if old <= 0:
        return float(target)
    return float(np.clip(target, old * (1 - max_step), old * (1 + max_step)))


def recovery_ratios(d: pd.DataFrame) -> dict:
    out = {}
    for pos, g in d.groupby("pos"):
        mins = float(g.mins.sum())
        base = float((g.rec90_prior * g.mins / 90.0).sum())
        out[pos] = dict(minutes=round(mins), recoveries=int(g.recoveries.sum()),
                        fitted=round(float(g.recoveries.sum()) / base, 3) if base > 0 else None)
    return out


def assist_split(d: pd.DataFrame) -> dict:
    """Actual assists by position against the model's split of each club's assists (shares x minutes)."""
    act, pred = {}, {}
    for _, g in d.groupby("team"):
        w = g.xa_share * g.mins
        tot = float(g.assists.sum())
        if tot <= 0 or float(w.sum()) <= 0:
            continue
        for pos, gg in g.groupby("pos"):
            act[pos] = act.get(pos, 0.0) + float(gg.assists.sum())
            pred[pos] = pred.get(pos, 0.0) + tot * float(w[gg.index].sum()) / float(w.sum())
    return {p: dict(actual=int(act[p]), expected=round(pred[p], 1), ratio=round(act[p] / pred[p], 3) if pred[p] > 0 else None)
            for p in act}


def slots() -> dict:
    """Starters per club by the game's positions over every line-up on file."""
    pool = load_pool()
    pos = dict(zip(pool.id, pool.pos))
    n, cnt = 0, {}
    for md in range(1, 9):
        if grade.xis_path(md) is None:
            continue
        x = grade.load_xis(md)
        ids = [i for i in grade._match_names(x, pool) if i]
        n += x.team.nunique()
        for i in set(ids):
            cnt[pos.get(i)] = cnt.get(pos.get(i), 0) + 1
    return dict(lineups=n, per_club={p: round(c / n, 2) for p, c in cnt.items() if p}) if n else {}


def availability_check() -> list[dict]:
    """UEFA's flags in the copy saved before each deadline against who then started."""
    pool = load_pool()
    rows = []
    for md in range(1, 9):
        pre = PROC / f"feed_pre_md{md}.csv"
        if not pre.exists() or grade.xis_path(md) is None:
            continue
        f = pd.read_csv(pre)
        if "status" not in f:
            continue
        f = f[f.team.notna() & f.name.notna()]
        ids, _ = _feed_ids(pool, f)
        f = f.assign(id=[ids.get(i) for i in f.index])
        started = {i for i in grade._match_names(grade.load_xis(md), pool) if i}
        flag = f.status.fillna("").astype(str).str.strip().replace("", "none")
        if "trained" in f:
            flag = flag.where(~(f.trained.fillna("").astype(str).str.startswith("Unlikely") & (flag == "none")), "unlikely")
        for k, g in f.assign(flag=flag).groupby("flag"):
            rows.append(dict(md=md, flag=k, players=int(len(g)), started=int(g.id.isin(started).sum())))
    return rows


def run(apply: bool = True, max_step: float = 0.15, as_of: str | None = None) -> dict:
    e = Engine(as_of=as_of or datetime.now(timezone.utc).date().isoformat())
    f = e.feed_snapshot
    rep = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), applied=False, changes={})
    cur = settings()["simulation"]
    if f is not None and "mins" in f:
        d = e.players.set_index("id").join(f[["mins", "goals", "assists", "recoveries"]].apply(pd.to_numeric, errors="coerce"), how="inner")
        d = d[d.mins > 0]
        rep["players"], rep["minutes"] = int(len(d)), int(d.mins.sum())
        rec = recovery_ratios(d)
        ast = assist_split(d)
        new_ratio, new_mult = dict(cur.get("recovery_ratio", {})), dict(cur.get("assist_pos_mult", {}))
        for pos, r in rec.items():
            old = float(new_ratio.get(pos, 1.0))
            r["current"] = old
            if r["fitted"] is not None and r["minutes"] >= MIN_MINUTES:
                r["new"] = round(_step(old, r["fitted"], max_step), 3) if old > 0 else old   # a zero (keepers) stays zero
                if old > 0 and abs(r["new"] / old - 1) >= MIN_CHANGE:
                    rep["changes"][f"recovery_ratio.{pos}"] = [old, r["new"]]
                    new_ratio[pos] = r["new"]
                else:
                    r["new"] = old
        for pos, r in ast.items():
            old = float(new_mult.get(pos, 1.0))
            r["current"] = old
            if r["ratio"] is not None and r["actual"] >= MIN_ASSISTS:
                # the split already includes the current multiplier, so the ratio is what is left to correct;
                # half of it per run (shares are also moved player by player from the same numbers)
                r["new"] = round(_step(old, old * (1 + 0.5 * (r["ratio"] - 1)), max_step), 3)
                if old > 0 and abs(r["new"] / old - 1) >= MIN_CHANGE:
                    rep["changes"][f"assist_pos_mult.{pos}"] = [old, r["new"]]
                    new_mult[pos] = r["new"]
                else:
                    r["new"] = old
        rep["recoveries"], rep["assists"] = rec, ast
        if apply and rep["changes"]:
            tuned = json.loads(PARAMS.read_text(encoding="utf-8")) if PARAMS.exists() else {}
            sim = tuned.setdefault("settings", {}).setdefault("simulation", {})
            sim["recovery_ratio"], sim["assist_pos_mult"] = new_ratio, new_mult
            tuned["calibrated_at"] = rep["generated_at"]
            PARAMS.write_text(json.dumps(tuned, indent=1), encoding="utf-8")
            rep["applied"] = True
    else:
        rep["note"] = "the game feed has no season stats yet"
    rep["slots"] = slots()
    rep["availability"] = availability_check()
    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep
