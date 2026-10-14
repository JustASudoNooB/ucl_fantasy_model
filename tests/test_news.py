"""Team news layer, unpriced players, and the posterior-predictive score matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _players():
    rows = [("BAY-gk", "Manuel Neuer", "GK", 0.9), ("BAY-gk2", "Jonas Urbig", "GK", 0.1),
            ("BAY-kane", "Harry Kane", "FWD", 0.95), ("BAY-olise", "Michael Olise", "MID", 0.9),
            ("BAY-musiala", "Jamal Musiala", "MID", 0.8), ("BAY-sub", "Bench Guy", "MID", 0.3)]
    return pd.DataFrame([dict(id=i, name=n, team="BAY", pos=p, price=6.0, p_start=s, p_sub=0.3) for i, n, p, s in rows])


def _news(tmp_path):
    f = tmp_path / "md9_test.txt"
    f.write_text("# header\nBAY | Urbig; Laimer; Olise, Luis Díaz; Kane | Musiala | Neuer | \n", encoding="utf-8")
    from ufe.ingest import team_news
    return team_news.parse(f)


def test_parse_positions_and_statuses(tmp_path):
    n = _news(tmp_path)
    st = n[n.status == "start"]
    assert list(st.line_pos) == ["GK", "DEF", "MID", "MID", "FWD"]
    assert set(n[n.status == "out"].name) == {"Musiala"}
    assert set(n[n.status == "doubt"].name) == {"Neuer"}


def test_match_is_position_aware_and_bayes_update(tmp_path):
    from ufe.ingest import team_news
    pl = _players()
    n = team_news.match(_news(tmp_path), pl)
    ids = dict(zip(n.name, n.id))
    assert ids["Urbig"] == "BAY-gk2" and ids["Kane"] == "BAY-kane" and ids["Musiala"] == "BAY-musiala"
    assert pd.isna(ids["Luis Díaz"])                      # not in the pool: a candidate for expansion
    out, log = team_news.apply(pl, n, dict(model="bayes", lr_start=9.3, lr_bench=0.078, doubt_in=0.62,
                                           doubt_out=0.15, p_max=0.985))
    p = out.set_index("id").p_start
    assert p["BAY-musiala"] == 0.0                        # out
    assert p["BAY-gk"] <= 0.15                            # doubtful and not named
    assert p["BAY-gk2"] > 0.5                             # named: 0.1 prior -> ~0.51
    assert p["BAY-kane"] > 0.99 - 1e-9 or p["BAY-kane"] >= 0.98
    assert p["BAY-sub"] < 0.05                            # fit but left out


def test_logistic_news_model_is_calibrated_not_extreme(tmp_path):
    """Calibrated on MD2 line-ups: named regulars ~0.9, named fringe ~0.45, unnamed fringe ~0.2."""
    from ufe.ingest import team_news
    pl = _players()
    n = team_news.match(_news(tmp_path), pl)
    out, _ = team_news.apply(pl, n)                       # settings: logistic model
    p = out.set_index("id").p_start
    assert p["BAY-musiala"] == 0.0
    assert 0.85 < p["BAY-kane"] <= 0.95
    assert 0.35 < p["BAY-gk2"] < 0.6
    assert 0.1 < p["BAY-sub"] < 0.3
    assert p["BAY-gk"] <= 0.15
    old = team_news.softened(team_news._cfg(), 0.5)      # half-strength (carried-forward) news
    out2, _ = team_news.apply(pl, n, old)
    assert pl.set_index("id").p_start["BAY-gk2"] < out2.set_index("id").p_start["BAY-gk2"] < p["BAY-gk2"]


def test_lr_from_counts_shrinks_to_prior():
    from ufe.ingest.team_news import lr_from_counts
    lr_s, lr_b, sens, fpr = lr_from_counts(0, 0, 0, 0)
    assert sens == pytest.approx(0.93) and fpr == pytest.approx(0.10)
    lr_s2, *_ = lr_from_counts(150, 200, 60, 200)        # less reliable predicted XIs -> weaker evidence
    assert lr_s2 < lr_s


def test_optimiser_never_buys_unpriced_players():
    from ufe.optimize.squad import pickable
    pl = pd.DataFrame(dict(id=["a", "b", "c"], price_status=["verified", "unknown", "est_fpl"]))
    assert list(pickable(pl).id) == ["a", "c"]
    assert list(pickable(pl, keep=["b"]).id) == ["a", "b", "c"]   # unless already owned
    # once the official price list is loaded, estimated prices are not bought either
    pl2 = pd.DataFrame(dict(id=["a", "b", "c", "d"], price_status=["verified", "unknown", "est_fpl", "official"]))
    assert list(pickable(pl2).id) == ["a", "d"]


def test_predictive_matrix_reduces_to_point_estimate_and_widens():
    from ufe.models.scoreline import dc_matrix, dc_matrix_predictive
    a = dc_matrix(1.6, 1.1, -0.06)
    b = dc_matrix_predictive(1.6, 1.1, -0.06, 0.0, 0.0)
    assert np.allclose(a, b)
    c = dc_matrix_predictive(1.6, 1.1, -0.06, 0.3, 0.3)
    g = np.arange(a.shape[0])
    var = lambda m: float((m.sum(1) * g ** 2).sum() - (m.sum(1) * g).sum() ** 2)
    assert var(c) > var(a)                                 # rating uncertainty adds spread
