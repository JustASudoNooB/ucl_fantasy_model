"""Core correctness tests: rules, probability models, simulator, optimiser, policy, parsers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufe.models.scoreline import dc_matrix, implied_lambdas, match_probs, remove_margin
from ufe.names import ucl_team_id
from ufe.rules import StatArrays, club_cap, formations, free_transfers, points, validate_squad


def _stats(**kw):
    base = dict(minutes=np.array([90]), goals=np.array([0]), ob_goals=np.array([0]), assists=np.array([0]),
                cs=np.array([False]), conceded=np.array([0]), saves=np.array([0]), pen_saves=np.array([0]),
                recoveries=np.array([0]), potm=np.array([False]), pen_won=np.array([0]), pen_conceded=np.array([0]),
                pen_missed=np.array([0]), yellow=np.array([0]), red=np.array([0]), own_goals=np.array([0]))
    base.update({k: np.array([v]) for k, v in kw.items()})
    return StatArrays(**base)


# ------------------------------------------------------------------ rules
def test_scoring_defender_clean_sheet_goal_recoveries():
    s = _stats(goals=1, cs=True, recoveries=7)
    assert points("DEF", s)[0] == 1 + 1 + 6 + 4 + 2          # app, 60', goal, CS, 7//3 recoveries


def test_scoring_forward_outside_box_potm_card():
    s = _stats(goals=2, ob_goals=1, potm=True, yellow=1)
    assert points("FWD", s)[0] == 1 + 1 + 8 + 1 + 3 - 1


def test_scoring_goalkeeper_saves_conceded_pen_save():
    s = _stats(saves=7, conceded=3, pen_saves=1)
    assert points("GK", s)[0] == 1 + 1 + 2 - 1 + 5


def test_clean_sheet_needs_60_minutes_and_nonplayers_score_zero():
    assert points("DEF", _stats(minutes=59, cs=True))[0] == 1
    assert points("MID", _stats(minutes=0, goals=1))[0] == 0


def test_formations_and_transfer_schedule():
    f = formations()
    assert (3, 4, 3) in f and (5, 4, 1) in f and (4, 2, 4) not in f
    assert all(sum(x) == 10 for x in f)
    assert free_transfers(3) == (2, 1)
    assert free_transfers(9)[0] is None and free_transfers(13) == (5, 0)
    assert club_cap(3) == 3 and club_cap(17) == 8


def test_validate_squad_flags_budget_and_cap():
    pl = ([dict(pos="GK", price=4.5, team=f"T{i}") for i in range(2)] + [dict(pos="DEF", price=4.5, team="A") for _ in range(5)]
          + [dict(pos="MID", price=9.0, team=f"M{i}") for i in range(5)] + [dict(pos="FWD", price=8.0, team=f"F{i}") for i in range(3)])
    errs = validate_squad(pl, md=3)
    assert any("exceeds" in e for e in errs) and any("cap 3" in e for e in errs)


# ------------------------------------------------------------------ probability models
def test_dc_matrix_normalised_and_consistent():
    m = dc_matrix(1.7, 1.1, -0.05)
    assert abs(m.sum() - 1) < 1e-9
    mp = match_probs(1.7, 1.1, -0.05)
    assert abs(mp.p_home + mp.p_draw + mp.p_away - 1) < 1e-9
    assert mp.p_home > mp.p_away


def test_margin_removal_and_implied_lambdas_round_trip():
    fair = remove_margin([1.80, 3.80, 4.60], "power")
    assert abs(fair.sum() - 1) < 1e-9
    mp = match_probs(1.9, 0.9, -0.05)
    lh, la = implied_lambdas(mp.p_home, mp.p_draw, mp.p_away, mp.over25, rho=-0.05)
    assert abs(lh - 1.9) < 0.05 and abs(la - 0.9) < 0.05


def test_team_name_mapping():
    assert ucl_team_id("FC Bayern München") == "BAY"
    assert ucl_team_id("Club Atlético de Madrid") == "ATM"
    assert ucl_team_id("FK Bodø/Glimt") == "BOD"
    assert ucl_team_id("Paris FC") is None


# ------------------------------------------------------------------ team model
def test_team_model_recovers_strength_order():
    from ufe.models.team_ratings import fit
    rng = np.random.default_rng(0)
    strength = {"A": 0.5, "B": 0.0, "C": -0.5}
    rows = []
    d0 = pd.Timestamp("2025-01-01")
    teams = list(strength)
    for k in range(600):
        h, a = rng.choice(teams, 2, replace=False)
        lh = np.exp(0.2 + 0.25 + strength[h] - (-strength[a]) * 0 - strength[a] * 0.5)
        la = np.exp(0.2 + strength[a] - strength[h] * 0.5)
        rows.append(dict(date=d0 + pd.Timedelta(days=k), comp="xx.1", season="s", tier=1, home=h, away=a,
                         home_country="ENG", away_country="ENG", hg=rng.poisson(lh), ag=rng.poisson(la), neutral=False,
                         stage="", extra_time=False, source="t", home_id=f"ENG:{h}", away_id=f"ENG:{a}"))
    m = fit(pd.DataFrame(rows), half_life_days=10000)
    r = {t: m.rating(f"ENG:{t}") for t in teams}
    assert r["A"] > r["B"] > r["C"]


# ------------------------------------------------------------------ simulator
@pytest.fixture(scope="module")
def tiny_world():
    rows = []
    for team in ("HOM", "AWY"):
        for pos, n, price in (("GK", 2, 4.5), ("DEF", 5, 4.5), ("MID", 5, 6.0), ("FWD", 3, 7.0)):
            for k in range(n):
                rows.append(dict(id=f"{team}-{pos}{k}", name=f"{team} {pos}{k}", team=team, pos=pos, price=price,
                                 p_start=0.9 if k < {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}[pos] else 0.1,
                                 p_sub=0.3, p_full90=0.7, xg_share={"GK": 0.0, "DEF": 0.04, "MID": 0.1, "FWD": 0.3}[pos],
                                 xa_share={"GK": 0.0, "DEF": 0.06, "MID": 0.14, "FWD": 0.07}[pos], rec90=4.0, yc90=0.15,
                                 red90=0.004, og90=0.005, pen_order=1 if (pos == "FWD" and k == 0) else 9, ob_share=0.1))
    players = pd.DataFrame(rows)
    others = {t: dict(others_xg=0.05, others_xa=0.05, others_on_pitch=0.2) for t in ("HOM", "AWY")}
    fx = pd.DataFrame([dict(home="HOM", away="AWY", lh=1.8, la=1.1, rho=-0.05, day=1)])
    from ufe.models.simulate import simulate_matchday
    sim = simulate_matchday(fx, players, others, n_sims=6000, seed=3)
    return players, sim


def test_simulated_goals_match_rates_and_allocation(tiny_world):
    players, sim = tiny_world
    assert abs(sim.scores[0][:, 0].mean() - 1.8) < 0.08
    assert abs(sim.scores[0][:, 1].mean() - 1.1) < 0.08
    home = [i for i, p in enumerate(sim.player_ids) if p.startswith("HOM")]
    alloc = sim.goals[home].sum(axis=0).mean()
    assert 1.45 < alloc < 1.8                  # most goals go to listed players; own goals/others take the rest


def test_potm_unique_per_match(tiny_world):
    _, sim = tiny_world
    per_sim = sim.potm.sum(axis=0) + sim.potm_other[0]
    assert np.all(per_sim == 1)


def test_policy_and_optimiser(tiny_world):
    players, sim = tiny_world
    from ufe.optimize import policy
    from ufe.optimize.squad import best_squad
    from ufe.models.simulate import summarise_players
    ev = summarise_players(sim, players).set_index("id").xpts
    sol = best_squad(players, ev, budget=100.0, club_cap=15)
    assert len(sol.squad) == 15 and len(sol.xi) == 11 and sol.captain in sol.xi
    rep = policy.evaluate(sim, players, sol.squad, sol.xi, sol.captain, sol.bench, md=3)
    assert rep.ev_active == pytest.approx(rep.ev_passive)         # single day: nothing to switch
    assert rep.ev_passive >= ev.loc[sol.xi].sum()                   # captain double and auto-subs only add


def test_cup_parser_handles_extra_time(tmp_path):
    from ufe.ingest.openfootball import _parse_cup
    txt = ("= UEFA Champions League 2025/26\n\n▪ Playoffs, Matchday 2\n  Wed Feb 25 2026\n"
           "    21:00  Juventus FC (ITA)       v Galatasaray SK (TUR)     3-2 a.e.t. (3-0, 1-0)\n"
           "           Real Madrid CF (ESP)    v Sport Lisboa e Benfica (POR)  2-1 (1-1)\n"
           "\n▪ Finals, Final\n  Sat May 30\n    18:00  Paris Saint-Germain FC (FRA) v Arsenal FC (ENG)  4-3 pen. 1-1 a.e.t. (1-1, 0-1)\n")
    p = tmp_path / "ucl_2025-26_cl.txt"
    p.write_text(txt, encoding="utf-8")
    rows = _parse_cup(p)
    assert (rows[0]["hg"], rows[0]["ag"], rows[0]["extra_time"]) == (3, 0, True)
    assert (rows[1]["hg"], rows[1]["ag"]) == (2, 1)
    assert rows[2]["neutral"] and (rows[2]["hg"], rows[2]["ag"]) == (1, 1)
    assert rows[2]["date"] == "2026-05-30"
