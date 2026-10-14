"""The learning loop: after every matchday, score the stored (pre-deadline) predictions against
what happened, then update the parts of the model that the evidence supports.

Inputs (CSV templates in data/templates/):
  results_md{N}.csv         home, away, home_goals, away_goals
  player_results_md{N}.csv  name or id, team, minutes, started, goals, assists, recoveries, saves,
                            yellow, red, potm, points        (any subset; more columns = more learning)
On your machine `ufe sync` fills both automatically from the official feed when it is reachable.

What gets updated, and how (each update is shrunk so one matchday cannot swing the model):
  - fixtures/results            -> team model refits on the next run (walk-forward)
  - recovery ratio by position  -> multiplicative update, weight grows with sample size
  - start-probability calibration -> Platt scaling once >= 300 player-matches are logged
  - per-player UCL history      -> appended to player_obs.csv, blended into rates next run
  - POTM weights                -> conditional-logit refit once >= 40 matches are logged
Every run appends a scoreboard row and a registry entry (params before/after, metrics).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import team_news
from ..config import ROOT, settings
from ..names import best_player_match, ucl_team_id
from . import metrics as M

PROC = ROOT / "data" / "processed"
RUNS = ROOT / "data" / "runs"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _team(x: str) -> str:
    return x if len(str(x)) == 3 and str(x).isupper() else (ucl_team_id(str(x)) or str(x))


def ingest_results(md: int, results: pd.DataFrame) -> pd.DataFrame:
    """Write final scores into data/processed/fixtures.csv (created from the seed on first use)."""
    from ..pipeline import load_fixtures
    fx = load_fixtures().drop(columns=["fixture_id"])
    for r in results.itertuples():
        h, a = _team(r.home), _team(r.away)
        m = (fx.md == md) & (fx.home == h) & (fx.away == a)
        if not m.any():
            raise ValueError(f"MD{md}: no fixture {h} v {a}")
        fx.loc[m, ["home_goals", "away_goals", "status"]] = [int(r.home_goals), int(r.away_goals), "FT"]
    PROC.mkdir(parents=True, exist_ok=True)
    fx.to_csv(PROC / "fixtures.csv", index=False)
    return fx


def score_matches(md: int, snapshot: dict, results: pd.DataFrame) -> dict:
    preds = {(p["home"], p["away"]): p for p in snapshot["fixtures"]}
    P, y, exact, xg_err, cs_p, cs_y = [], [], [], [], [], []
    for r in results.itertuples():
        h, a = _team(r.home), _team(r.away)
        p = preds.get((h, a))
        if p is None:
            continue
        P.append([p["p_home"], p["p_draw"], p["p_away"]])
        y.append(0 if r.home_goals > r.away_goals else 1 if r.home_goals == r.away_goals else 2)
        top = p["top_scores"][0]
        exact.append(top[0] == r.home_goals and top[1] == r.away_goals)
        xg_err += [abs(p["xg_home"] - r.home_goals), abs(p["xg_away"] - r.away_goals)]
        cs_p += [p["cs_home"], p["cs_away"]]
        cs_y += [r.away_goals == 0, r.home_goals == 0]
    if not P:
        return {}
    P, y = np.array(P), np.array(y)
    return dict(n=len(y), log_loss=M.log_loss(P, y), rps=M.rps(P, y), brier=M.brier_multi(P, y), accuracy=M.accuracy(P, y),
                exact_score_hit=float(np.mean(exact)), goals_mae=float(np.mean(xg_err)),
                cs_brier=M.brier_binary(np.array(cs_p), np.array(cs_y, dtype=float)))


def _match_players(df: pd.DataFrame, snap_players: pd.DataFrame) -> pd.Series:
    if "id" in df.columns:
        return df["id"]
    ids = []
    for r in df.itertuples():
        team = _team(getattr(r, "team", ""))
        cands = snap_players[snap_players.team == team] if team else snap_players
        ids.append(best_player_match(r.name, dict(zip(cands.name, cands.id))))
    return pd.Series(ids, index=df.index)


def score_players(snapshot: dict, actual: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sp = pd.DataFrame(snapshot["players"])
    actual = actual.copy()
    actual["id"] = _match_players(actual, sp)
    j = actual.dropna(subset=["id"]).merge(sp, on="id", suffixes=("", "_pred"))
    out: dict = dict(n_players=int(len(j)))
    if "started" in j:
        out["start_brier"] = M.brier_binary(j.p_start_sim, j.started.astype(float))
        if "named" in j and j.named.notna().any():
            k = j[j.named.notna()]
            st = k.started.astype(int) == 1
            nm = k.named.astype(int) == 1
            out["news_counts"] = dict(started_named=int((st & nm).sum()), started=int(st.sum()),
                                      bench_named=int((~st & nm).sum()), bench=int((~st).sum()))
    if "goals" in j:
        out["goal_brier"] = M.brier_binary(j.p_goal, (j.goals > 0).astype(float))
        out["goal_ece"] = M.expected_calibration_error(j.p_goal, (j.goals > 0).astype(float), 8)
    if "assists" in j:
        out["assist_brier"] = M.brier_binary(j.p_assist, (j.assists > 0).astype(float))
    if "potm" in j:
        out["potm_brier"] = M.brier_binary(j.p_potm, j.potm.astype(float))
    if "points" in j:
        out["points_mae"] = float(np.mean(np.abs(j.xpts - j.points)))
        out["points_bias"] = float(np.mean(j.xpts - j.points))
        if "pmf" in j:
            out["points_crps"] = float(np.mean([M.crps_discrete(np.array(pm), int(pt) + 3) for pm, pt in zip(j.pmf, j.points)]))
    if "recoveries" in j:
        rec = j[j.get("minutes", pd.Series(90, index=j.index)) > 0]
        out["recovery_ratio_obs"] = {p: float(g.recoveries.sum() / max(g.xrec.sum(), 1e-6)) for p, g in rec.groupby("pos")}
        out["recovery_n"] = {p: int(len(g)) for p, g in rec.groupby("pos")}
    return out, j


def update_params(player_metrics: dict) -> dict:
    """Shrunk multiplicative update of the recovery ratio; writes params.json with history."""
    pth = PROC / "params.json"
    params = json.loads(pth.read_text()) if pth.exists() else {}
    cur = settings()["simulation"]["recovery_ratio"]
    new = dict(cur)
    for pos, ratio in player_metrics.get("recovery_ratio_obs", {}).items():
        n = player_metrics["recovery_n"].get(pos, 0)
        w = n / (n + 60.0)                       # ~60 player-matches before trusting the data fully
        new[pos] = round(float(cur[pos] * (ratio ** w)), 4)
    params.setdefault("settings", {}).setdefault("simulation", {})["recovery_ratio"] = new
    params.setdefault("history", []).append(dict(at=_now(), change="recovery_ratio", before=cur, after=new))
    nc = player_metrics.get("news_counts")
    if nc and nc.get("started", 0) + nc.get("bench", 0) > 0:
        # accumulate predicted-XI hit counts across matchdays, then re-derive the likelihood ratios
        acc = params.setdefault("news_counts", dict(started_named=0, started=0, bench_named=0, bench=0))
        for k_, v in nc.items():
            acc[k_] = int(acc.get(k_, 0)) + int(v)
        lr_s, lr_b, sens, fpr = team_news.lr_from_counts(acc["started_named"], acc["started"], acc["bench_named"], acc["bench"])
        before = {k_: settings().raw.get("news", {}).get(k_) for k_ in ("lr_start", "lr_bench")}
        after = dict(lr_start=round(lr_s, 3), lr_bench=round(lr_b, 4))
        params["settings"].setdefault("news", {}).update(after)
        params["history"].append(dict(at=_now(), change="news_likelihood_ratios", before=before, after=after,
                                      sensitivity=round(sens, 4), false_positive_rate=round(fpr, 4), counts=dict(acc)))
    pth.write_text(json.dumps(params, indent=2))
    return new


def append_obs(j: pd.DataFrame, md: int) -> None:
    keep = [c for c in ["id", "name", "team", "pos", "minutes", "started", "goals", "assists", "recoveries", "saves",
                        "yellow", "red", "potm", "points", "team_lambda"] if c in j.columns]
    obs = j[keep].copy()
    obs["md"] = md
    p = PROC / "player_obs.csv"
    if p.exists():
        prev = pd.read_csv(p)
        obs = pd.concat([prev[prev.md != md], obs], ignore_index=True)
    obs.to_csv(p, index=False)


def run(md: int, results_csv: Path, players_csv: Path | None = None) -> dict:
    snap_path = RUNS / f"md{md}" / "snapshot.json"
    if not snap_path.exists():
        raise FileNotFoundError(f"No pre-deadline snapshot for MD{md} at {snap_path}. Run `ufe predict --md {md}` before the deadline.")
    snapshot = json.loads(snap_path.read_text())
    results = pd.read_csv(results_csv)
    ingest_results(md, results)
    entry = dict(md=md, scored_at=_now(), model_version=snapshot.get("model_version"),
                 matches=score_matches(md, snapshot, results))
    if players_csv is not None and Path(players_csv).exists():
        pm, joined = score_players(snapshot, pd.read_csv(players_csv))
        entry["players"] = pm
        append_obs(joined, md)
        if "recovery_ratio_obs" in pm:
            entry["recovery_ratio_after"] = update_params(pm)
    sb_path = PROC / "scoreboard.json"
    board = json.loads(sb_path.read_text()) if sb_path.exists() else []
    board = [b for b in board if b.get("md") != md] + [entry]
    sb_path.write_text(json.dumps(sorted(board, key=lambda b: b["md"]), indent=2))
    reg = PROC / "registry.json"
    registry = json.loads(reg.read_text()) if reg.exists() else []
    registry.append(dict(at=_now(), event=f"feedback MD{md}", model_version=snapshot.get("model_version"),
                         metrics=entry.get("matches"), params_file="data/processed/params.json"))
    reg.write_text(json.dumps(registry, indent=2))
    return entry
