"""Neural-network challenger: gradients, learning, saving, and no look-ahead in the features."""
import numpy as np
import pandas as pd

from ufe.ml import data as D
from ufe.ml.nn import Ensemble, MultiTaskNet


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 0.5 * X[:, 1] * X[:, 2])))       # an interaction a linear model misses
    expo = rng.uniform(0.2, 1.0, n)
    rate = np.exp(-2 + 0.8 * X[:, 1])
    T = dict(start=(rng.random(n) < p).astype(float), played=np.ones(n), expo=expo,
             goals=rng.poisson(rate * expo).astype(float), assists=rng.poisson(0.5 * rate * expo).astype(float))
    return X, T


def test_backprop_matches_numerical_gradient():
    X, T = _toy(64)
    net = MultiTaskNet(3, (5, 4), l2=1e-3, seed=2)
    _, _, gW, gb = net.loss_and_grads(X, T)
    for li in range(len(net.W)):
        e = 1e-6
        net.W[li][0, 0] += e
        lp = net.loss_and_grads(X, T)[0]
        net.W[li][0, 0] -= 2 * e
        lm = net.loss_and_grads(X, T)[0]
        net.W[li][0, 0] += e
        num = (lp - lm) / (2 * e)
        assert abs(num - gW[li][0, 0]) <= 1e-5 * max(1.0, abs(num))


def test_network_learns_an_interaction_the_linear_model_cannot():
    X, T = _toy(6000)
    Xv, Tv = _toy(3000, seed=1)
    nn = MultiTaskNet(3, (16, 8), seed=0).fit(X, T, Xv, Tv, epochs=40, patience=6)
    lin = MultiTaskNet(3, (), seed=0).fit(X, T, Xv, Tv, epochs=40, patience=6)
    ll = lambda m: m.loss_and_grads((Xv - m.mu) / m.sd, Tv)[1][0]
    assert ll(nn) < ll(lin) - 0.005


def test_ensemble_round_trip():
    X, T = _toy(500)
    e = Ensemble([MultiTaskNet(3, (4,), seed=s).fit(X, T, X, T, epochs=2) for s in range(2)])
    e2 = Ensemble.from_json(e.to_json())
    a, b = e.predict(X[:5]), e2.predict(X[:5])
    assert np.allclose(a["p_start"], b["p_start"]) and np.allclose(a["g90"], b["g90"])


def test_features_use_only_earlier_matches(monkeypatch):
    rows = []
    for i, st in enumerate([1, 1, 0, 1]):
        for team, opp, home in (("AAA", "BBB", True), ("BBB", "AAA", False)):
            rows.append(dict(season="2025-26", GW=i + 1, fixture=i + 1, kickoff=pd.Timestamp(f"2025-09-0{i + 1}", tz="UTC"),
                             element=1 if team == "AAA" else 2, name=f"P {team}", team=team, pos="FWD", home=float(home),
                             gf=1.0, ga=1.0, minutes=90.0 * st if team == "AAA" else 90.0, starts=float(st) if team == "AAA" else 1.0,
                             goals_scored=0.0, assists=0.0, expected_goals=0.5, expected_assists=0.1, value=80.0))
    monkeypatch.setattr(D, "_season_rows", lambda s: pd.DataFrame(rows))
    monkeypatch.setattr(D, "_path", lambda s, f: type("P", (), {"exists": lambda self: True})())
    df = D.build(seasons=["2025-26"])
    a = df[df.pid == "P AAA"].sort_values("kickoff")
    assert list(a.started_last1) == [0.0, 1.0, 1.0, 0.0]            # each row sees only the previous match
    assert a.has_start_hist.iloc[0] == 0.0 and a.start_rate.iloc[0] == 0.0
