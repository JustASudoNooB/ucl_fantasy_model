"""Hyperparameter tuning for the team model by walk-forward backtest (no look-ahead).

Objective: mean ranked probability score over the evaluation seasons, with log loss as tie-break.
Tuned values are written to data/processed/params.json and picked up by config.settings().
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import pandas as pd

from ..config import ROOT
from .backtest import backtest_team_model

GRID = dict(
    half_life_days=[180, 300, 500],
    european_weight=[1.0, 1.6, 2.5],
    prior_sd_team=[0.25, 0.35, 0.5],
    prior_sd_league=[0.45],
)


def tune(df: pd.DataFrame, seasons=("2024-25", "2025-26"), grid: dict | None = None, verbose: bool = True) -> dict:
    grid = grid or GRID
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        scores = []
        for s in seasons:
            _, summ = backtest_team_model(df, s, **kw)
            scores.append(summ)
        row = dict(kw, rps=sum(x["rps"] for x in scores) / len(scores),
                   log_loss=sum(x["log_loss"] for x in scores) / len(scores),
                   accuracy=sum(x["accuracy"] for x in scores) / len(scores))
        results.append(row)
        if verbose:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    res = pd.DataFrame(results).sort_values(["rps", "log_loss"]).reset_index(drop=True)
    best = res.iloc[0].to_dict()
    out_path = ROOT / "data" / "processed" / "params.json"
    current = json.loads(out_path.read_text()) if out_path.exists() else {}
    current.setdefault("settings", {}).setdefault("team_model", {})
    for k in keys:
        current["settings"]["team_model"][k] = float(best[k])
    current["team_model_tuning"] = dict(tuned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                        seasons=list(seasons), best=best, grid_size=len(res),
                                        table=res.head(10).round(5).to_dict(orient="records"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return best
