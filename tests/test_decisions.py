"""Decision-layer tests: transfer rollover and hits, live captain/sub calls, feedback scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufe.models.simulate import SimResult


def _pool():
    rows = []
    teams = [f"T{i:02d}" for i in range(12)]
    k = 0
    for pos, n in (("GK", 6), ("DEF", 14), ("MID", 14), ("FWD", 9)):
        for j in range(n):
            rows.append(dict(id=f"{pos}{j}", name=f"{pos} Player {j}", team=teams[k % 12], pos=pos,
                             price=4.0 + (j % 5) * 0.5))
            k += 1
    return pd.DataFrame(rows)


def test_transfer_plan_respects_rollover_and_hits():
    from ufe.optimize import transfers
    pool = _pool()
    rng = np.random.default_rng(0)
    ev = pd.DataFrame({3: rng.uniform(1, 4, len(pool)), 4: rng.uniform(1, 4, len(pool))}, index=pool.id)
    squad = (["GK0", "GK1"] + [f"DEF{j}" for j in range(5)] + [f"MID{j}" for j in range(5)] + [f"FWD{j}" for j in range(3)])
    # make one outsider very attractive only in MD4: best plan saves the transfer in MD3
    ev.loc["MID13", 4] = 30.0
    ev.loc["MID12", 4] = 30.0
    ev.loc["MID11", 4] = 30.0
    opt = transfers.plan(pool, ev, squad, bank=20.0, free_now=2, mds=[3, 4], first_md_transfers=0,
                         per_pos={"GK": 6, "DEF": 14, "MID": 14, "FWD": 9})
    md4 = opt.plan[1]
    assert md4["free"] == 3                     # one unused transfer carried (max one)
    assert {"MID13", "MID12", "MID11"} <= set(md4["squad"])
    assert md4["hits"] == 0


def test_hit_taken_only_when_worth_more_than_four():
    from ufe.optimize import transfers
    pool = _pool()
    ev = pd.DataFrame({3: np.full(len(pool), 2.0)}, index=pool.id)
    squad = (["GK0", "GK1"] + [f"DEF{j}" for j in range(5)] + [f"MID{j}" for j in range(5)] + [f"FWD{j}" for j in range(3)])
    ev.loc["MID0", 3] = 12.0                     # captain is already settled
    ev.loc["FWD8", 3] = 2.0 + 3.0                # +3 over the forward he replaces: not worth -4
    opt = transfers.plan(pool, ev, squad, bank=20.0, free_now=0, mds=[3], per_pos={"GK": 6, "DEF": 14, "MID": 14, "FWD": 9})
    assert opt.hits == 0
    ev.loc["FWD8", 3] = 2.0 + 6.5                # +6.5: worth the hit
    opt = transfers.plan(pool, ev, squad, bank=20.0, free_now=0, mds=[3], per_pos={"GK": 6, "DEF": 14, "MID": 14, "FWD": 9})
    assert "FWD8" in opt.buy and opt.hits == 1


def _two_day_sim():
    ids = (["G1", "G2"] + [f"D{i}" for i in range(5)] + [f"M{i}" for i in range(5)] + [f"F{i}" for i in range(3)])
    pos = ["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = pd.DataFrame(dict(id=ids, name=[f"Name {i}" for i in ids], team="X", pos=pos, price=5.0))
    N = 2000
    rng = np.random.default_rng(1)
    day = np.array([1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2])
    pts = rng.poisson(4, (len(ids), N)).astype(np.int16)
    pts[ids.index("F2")] = 8                     # day-2 forward always scores 8
    mins = np.full((len(ids), N), 90, np.uint8)
    z = np.zeros((len(ids), N))
    sim = SimResult(player_ids=ids, points=pts, minutes=mins, goals=z.astype(np.int8), assists=z.astype(np.int8),
                    cs=z.astype(bool), recoveries=z.astype(np.int16), potm=z.astype(bool), started=np.ones_like(z, bool),
                    day=day, fixture=np.zeros(len(ids), int), scores=np.zeros((1, N, 2)), potm_other=np.zeros((1, N), bool),
                    meta=dict(n_sims=N))
    return players, sim, ids


def test_live_decisions_switch_and_sub():
    from ufe.optimize import policy
    players, sim, ids = _two_day_sim()
    xi = ["G1", "D0", "D1", "D2", "D3", "M0", "M1", "M2", "M3", "F0", "F2"]
    bench = ["M4", "F1", "D4", "G2"]
    out = policy.live_decisions(sim, players, ids, xi, captain="F0", bench=bench, md=3, day_done=1,
                                realised={"F0": 1, "M0": 0, "D0": 6, "G1": 3, "D1": 2, "D2": 2, "M1": 5, "M2": 5})
    kinds = [a["kind"] for a in out["actions"] if a.get("do")]
    assert kinds[0] == "captain"
    assert out["actions"][0]["to"] == "F2"       # the sure 8 on day 2
    assert "sub" in kinds


def test_policy_value_beats_passive_when_days_split():
    from ufe.optimize import policy
    players, sim, ids = _two_day_sim()
    xi = ["G1", "D0", "D1", "D2", "D3", "M0", "M1", "M2", "M3", "F0", "F2"]
    rep = policy.evaluate(sim, players, ids, xi, "F0", ["M4", "F1", "D4", "G2"], md=3)
    assert rep.ev_active >= rep.ev_passive
    assert rep.p_captain_switch > 0.3


def test_feedback_scores_matches():
    from ufe.evaluate.feedback import score_matches
    snap = dict(fixtures=[dict(home="ARS", away="LIL", p_home=0.7, p_draw=0.2, p_away=0.1, xg_home=2.0, xg_away=0.6,
                               cs_home=0.55, cs_away=0.13, top_scores=[[2, 0, 0.15]])])
    res = pd.DataFrame([dict(home="ARS", away="LIL", home_goals=2, away_goals=0)])
    m = score_matches(2, snap, res)
    assert m["n"] == 1 and m["accuracy"] == 1.0 and m["exact_score_hit"] == 1.0
    assert m["log_loss"] == pytest.approx(-np.log(0.7))
