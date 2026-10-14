"""Official price-list reconciliation, live plan comparison, frozen snapshots."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _importer():
    spec = importlib.util.spec_from_file_location("ios", ROOT / "scripts" / "import_official_squads.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_name_matching_accepts_abbreviations_and_rejects_namesakes():
    m = _importer()
    assert m.name_score("J. Timber", "Jurriën Timber") >= 95
    assert m.name_score("Minjae Kim", "Kim Min-jae") >= 90          # same letters, other order
    assert m.name_score("E. N'Dicka", "Evan Ndicka") >= 90          # apostrophe
    assert m.name_score("Gabriel", "Gabriel Magalhães") >= 95       # one-word game names
    assert m.name_score("Diogo Costa", "Alberto Costa") < 86        # same surname, other player
    assert m.name_score("I. Ba", "Koba Koindredi") < 86             # 'ba' inside another name
    assert m.strict_same_person("K. Adeyemi", "Karim Adeyemi")      # transfer across clubs
    assert not m.strict_same_person("S. Suleiman", "Suleiman Camara")


def _sim_two_days():
    from ufe.models.simulate import SimResult
    ids = ["G1", "G2"] + [f"D{i}" for i in range(5)] + [f"M{i}" for i in range(5)] + [f"F{i}" for i in range(3)]
    pos = ["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = pd.DataFrame(dict(id=ids, name=ids, team="X", pos=pos, price=5.0))
    N = 1000
    day = np.array([1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2])
    pts = np.full((len(ids), N), 4, np.int16)
    mins = np.full((len(ids), N), 90, np.uint8)
    pts[ids.index("M3")] = 0                     # day-2 midfielder ruled out: never plays
    mins[ids.index("M3")] = 0
    z = np.zeros((len(ids), N))
    sim = SimResult(player_ids=ids, points=pts, minutes=mins, goals=z.astype(np.int8), assists=z.astype(np.int8),
                    cs=z.astype(bool), recoveries=z.astype(np.int16), potm=z.astype(bool), started=mins > 0,
                    day=day, fixture=np.zeros(len(ids), int), scores=np.zeros((1, N, 2)),
                    potm_other=np.zeros((1, N), bool), meta=dict(n_sims=N))
    return players, sim


def test_live_plan_keeps_sent_off_player_and_subs_ruled_out_starter():
    from ufe.optimize import policy
    players, sim = _sim_two_days()
    xi = ["G1", "D0", "D1", "D2", "D3", "M0", "M1", "M2", "M3", "F0", "F1"]
    bench = ["D4", "M4", "F2", "G2"]
    realised = {"G1": 3, "D0": 1, "D2": 2, "D3": 6, "M0": -1, "M1": 1, "F0": 2}
    out = policy.live_decisions(sim, players, xi + bench, xi, captain="M0", bench=bench, md=3, day_done=1,
                                realised=realised, sent_off={"M0"})
    subs = [a for a in out["actions"] if a["kind"] == "sub"]
    assert all(a["out"] != "M0" for a in subs)                     # sent off: stays in
    assert any(a["out"] == "M3" for a in subs)                     # ruled out later: subbed by hand
    cap = [a for a in out["actions"] if a["kind"] == "captain"]
    assert cap and cap[0]["to"] in set(xi) | set(bench) and sim.day[sim.index()[cap[0]["to"]]] == 2
    assert out["plans"]["both"] >= out["plans"]["passive"]


def test_snapshot_is_not_overwritten_after_deadline(tmp_path, monkeypatch):
    from ufe.report import export
    monkeypatch.setattr(export, "RUNS", tmp_path)
    d = tmp_path / "md1"
    d.mkdir()
    (d / "snapshot.json").write_text(json.dumps(dict(md=1, frozen=True)))

    class NoEngine:                                                 # any call would raise
        pass
    p = export.snapshot(NoEngine(), 1)
    assert json.loads(p.read_text())["frozen"] is True


def test_logistic_start_update_keeps_prior_order_within_named():
    from ufe.ingest import team_news
    cfg = team_news._cfg()
    hi = team_news.updated_start(0.9, True, cfg)
    lo = team_news.updated_start(0.2, True, cfg)
    out = team_news.updated_start(0.9, False, cfg)
    assert hi > lo > 0.3 and hi <= cfg["p_max"]
    assert out < hi and 0.2 < out < 0.7                            # a benched regular is not ruled out
