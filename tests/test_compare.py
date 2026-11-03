"""Model comparison: decision trees, the bookmaker engine, deadline freezing, leaderboard scoring."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ufe.compare import forecast, leaderboard
from ufe.compare.market import MarketEngine
from ufe.ml import trees
from ufe.models.scoreline import match_probs, remove_margin
from tests.test_ml import _toy


def test_trees_learn_and_round_trip():
    X, T = _toy(3000)
    Xv, Tv = _toy(1500, seed=1)
    m = trees.fit(X, T, Xv, Tv, max_rounds=200, params=dict(min_data_in_leaf=50))
    pr = m.predict(Xv)
    p = pr["p_start"]
    base = Tv["start"].mean()
    ll = -np.mean(Tv["start"] * np.log(p) + (1 - Tv["start"]) * np.log(1 - p))
    ll0 = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    assert ll < ll0 - 0.05                                   # better than the base rate
    assert np.all(pr["g90"] > 0) and np.all(pr["a90"] > 0)
    m2 = trees.TreeModel.from_json(m.to_json(), check=False)
    assert np.allclose(m2.predict(Xv[:20])["p_start"], p[:20])


def _market(odds: pd.DataFrame, base: pd.DataFrame) -> MarketEngine:
    e = MarketEngine(as_of="2026-11-01")
    e.__dict__["cfg"] = {"market": {"margin_method": "proportional"}}
    e.__dict__["team_model"] = type("TM", (), {"rho": -0.06})()
    e.__dict__["ucl_avg_total"] = 3.0
    e._odds = lambda md: odds
    import ufe.compare.market as mk
    mk.Engine._base_fixture_table = lambda self, md: base.copy()
    return e


def test_bookmaker_engine_uses_prices_only(monkeypatch):
    import ufe.compare.market as mk
    orig = mk.Engine._base_fixture_table
    base = pd.DataFrame(dict(fixture_id=["A", "B", "C"], lh=[1.5, 1.2, 1.0], la=[1.0, 1.1, 1.3],
                             lambda_source=["model+market"] * 3, sh=[0.1] * 3, sa=[0.1] * 3))
    odds = pd.DataFrame(dict(home_odds=[1.8, 2.5], draw_odds=[3.8, 3.3], away_odds=[4.5, 2.9],
                             total_line=[2.5, np.nan], over_odds=[1.9, np.nan], under_odds=[1.95, np.nan]),
                        index=pd.Index(["A", "B"], name="fixture_id"))
    try:
        f = _market(odds, base)._base_fixture_table(4).set_index("fixture_id")
    finally:
        mk.Engine._base_fixture_table = orig
    ph = remove_margin([1.8, 3.8, 4.5], "proportional")[0]
    assert abs(match_probs(f.lh["A"], f.la["A"], -0.06).p_home - ph) < 0.02      # reproduces the price
    assert f.lambda_source["A"] == "market" and f.sh["A"] == 0.0
    assert abs(f.lh["B"] + f.la["B"] - 3.0) < 1e-6                              # 1X2 only: average total
    assert f.lambda_source["C"] == "model (no odds)" and f.lh["C"] == 1.0         # no prices: model kept


def test_forecasts_freeze_at_the_deadline(monkeypatch):
    monkeypatch.setattr(forecast, "deadline", lambda md: datetime.now(timezone.utc) - timedelta(minutes=1))
    assert forecast.run(md=4) == []                         # nothing rebuilt, nothing written
    monkeypatch.setattr(forecast, "deadline", lambda md: datetime.now(timezone.utc) + timedelta(hours=30))
    assert forecast.run(md=4, window_hours=3) == []         # too early for the CI window


def test_leaderboard_compares_on_shared_matchdays():
    by_md = {"1": {"a": {"start_ll": 0.6}, "b": {"skipped": "no prices"}},
             "2": {"a": {"start_ll": 0.40}, "b": {"start_ll": 0.38}},
             "3": {"a": {"start_ll": 0.42}, "b": {"start_ll": 0.44}}}
    s = leaderboard.summarise(by_md, ["a", "b"])
    assert s["matchdays_used"]["start_ll"] == [2, 3]
    assert s["average"]["a"]["start_ll"] == 0.41 and s["average"]["b"]["start_ll"] == 0.41
    assert s["wins"] == {"a": 1, "b": 1}
