"""Walk-forward test of two team-model additions before adopting them:

  home_sd_team   club-specific home effects (home/away record), prior sd on the log scale
  form_mix       share of match weight given to a short half-life component (recent form)

Each configuration is scored on (a) UCL matches 2024/25 + 2025/26, refit before every round,
and (b) top-flight domestic matches 2024/25 + 2025/26 in the leagues of the 36 UCL clubs,
refit every 14 days. Lower RPS / log loss is better; a change is adopted only if it improves
both on UCL and domestic matches (DECISIONS.md rule).

  python scripts/backtest_form_homeaway.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufe.config import settings                      # noqa: E402
from ufe.evaluate import metrics as M                # noqa: E402
from ufe.evaluate.backtest import backtest_team_model  # noqa: E402
from ufe.ingest.openfootball import load_matches     # noqa: E402
from ufe.models.scoreline import match_probs         # noqa: E402
from ufe.models.team_ratings import fit              # noqa: E402

DOMESTIC = ["en.1", "es.1", "de.1", "it.1", "fr.1", "pt.1", "nl.1", "be.1", "tr.1", "at.1", "gr.1", "sco.1"]
DF = None


def _df():
    global DF
    if DF is None:
        DF = load_matches()
    return DF


def domestic_backtest(season: str, step_days: int = 14, **kw) -> dict:
    df = _df()
    tgt = df[(df.season == season) & df.comp.isin(DOMESTIC)].sort_values("date")
    if tgt.empty:
        return {}
    start = tgt.date.min() + pd.Timedelta(days=45)        # skip the first weeks (tiny current-season sample)
    end = tgt.date.max() + pd.Timedelta(days=1)
    P, y, xg_err = [], [], []
    t = start
    while t < end:
        batch = tgt[(tgt.date >= t) & (tgt.date < t + pd.Timedelta(days=step_days))]
        if len(batch):
            m = fit(df, as_of=t, **kw)
            for r in batch.itertuples():
                grp = m.groups.get(r.home_id, "EUR")
                lh, la = m.lambdas(r.home_id, r.away_id, neutral=bool(r.neutral), comp_group=grp)
                sh, sa = m.log_sd(r.home_id, r.away_id)
                mp = match_probs(lh, la, m.rho, sh=sh, sa=sa)
                P.append([mp.p_home, mp.p_draw, mp.p_away])
                y.append(0 if r.hg > r.ag else (1 if r.hg == r.ag else 2))
                xg_err.append(abs(lh - r.hg) + abs(la - r.ag))
        t += pd.Timedelta(days=step_days)
    P, y = np.array(P), np.array(y)
    return dict(n=len(y), rps=M.rps(P, y), log_loss=M.log_loss(P, y), goals_mae=float(np.mean(xg_err)) / 2)


def run_config(cfg: dict) -> dict:
    tm = settings()["team_model"]
    base = dict(half_life_days=tm["half_life_days"], european_weight=tm["european_weight"],
                prior_sd_team=tm["prior_sd_team"], prior_sd_league=tm["prior_sd_league"])
    kw = {**base, **cfg}
    out = dict(cfg)
    ucl = [backtest_team_model(_df(), s, **kw)[1] for s in ("2024-25", "2025-26")]
    out["ucl_n"] = sum(u["n"] for u in ucl)
    out["ucl_rps"] = float(np.average([u["rps"] for u in ucl], weights=[u["n"] for u in ucl]))
    out["ucl_ll"] = float(np.average([u["log_loss"] for u in ucl], weights=[u["n"] for u in ucl]))
    dom = [domestic_backtest(s, **kw) for s in ("2024-25", "2025-26")]
    dom = [d for d in dom if d]
    out["dom_n"] = sum(d["n"] for d in dom)
    out["dom_rps"] = float(np.average([d["rps"] for d in dom], weights=[d["n"] for d in dom]))
    out["dom_ll"] = float(np.average([d["log_loss"] for d in dom], weights=[d["n"] for d in dom]))
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    grid = [dict()]
    grid += [dict(home_sd_team=s) for s in (0.05, 0.1, 0.2)]
    grid += [dict(form_mix=m, form_half_life=h) for m, h in ((0.3, 30), (0.3, 90), (0.5, 60))]
    if a.quick:
        grid = [dict(), dict(home_sd_team=0.05), dict(form_mix=0.3, form_half_life=30), dict(form_mix=0.3, form_half_life=90)]
    with Pool(2) as pool:
        res = pool.map(run_config, grid)
    df = pd.DataFrame(res)
    base = df.iloc[0]
    df["ucl_rps_gain"] = base.ucl_rps - df.ucl_rps
    df["dom_rps_gain"] = base.dom_rps - df.dom_rps
    df["ucl_ll_gain"] = base.ucl_ll - df.ucl_ll
    df["dom_ll_gain"] = base.dom_ll - df.dom_ll
    print(df.round(5).to_string(index=False))
    out = ROOT / "data" / "processed" / "backtest_form_homeaway.json"
    out.write_text(json.dumps(dict(results=df.round(6).to_dict(orient="records")), indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
