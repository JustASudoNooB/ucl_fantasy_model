"""The dashboard's analytics data and the public edition."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ufe.report import analytics, render


def test_bins_count_every_row_once():
    x = np.array([0.0, 0.01, 0.5, 0.95, 1.0])
    y = np.array([0, 0, 1, 1, 1])
    b = analytics.bins(x, y, analytics.START_EDGES)
    assert len(b) == len(analytics.START_EDGES) - 1
    assert sum(r[0] for r in b) == 5 and abs(sum(r[2] for r in b) - 3) < 1e-9
    assert b[-1][0] == 2                                  # 0.95 and 1.0 share the last interval


def test_player_extras_parts_add_up_to_expected_points():
    n = 400
    rng = np.random.default_rng(0)
    mins = np.where(rng.random(n) < 0.8, 90, 0).astype(np.uint8)
    goals = (rng.random(n) < 0.3).astype(np.int8) * (mins > 0)
    assists = (rng.random(n) < 0.2).astype(np.int8) * (mins > 0)
    cs = rng.random(n) < 0.4
    rec = rng.integers(0, 9, n).astype(np.int16) * (mins > 0)
    potm = (rng.random(n) < 0.05) & (mins > 0)
    pts = (2 * (mins > 0) + 5 * goals + 3 * assists + 1 * (cs & (mins >= 60)) + rec // 3 + 3 * potm - (rng.random(n) < 0.1)).astype(np.int16)
    sim = SimpleNamespace(points=pts[None, :], minutes=mins[None, :], goals=goals[None, :], assists=assists[None, :],
                          cs=cs[None, :], recoveries=rec[None, :], potm=potm[None, :], index=lambda: {"X-mid": 0})
    proj = pd.DataFrame([dict(id="X-mid", pos="MID", xpts=float(pts.mean()))])
    out = analytics.player_extras(sim, proj)["X-mid"]
    assert abs(sum(out["c"]) - pts.mean()) < 0.03          # appearance, goals, assists, clean sheet, recoveries, award, rest
    assert out["c"][6] < 0                                 # the rest here is only the cards
    assert abs(sum(out["pmf"]) - 1000) <= 12 and len(out["pmf"]) == analytics.PMF_HI - analytics.PMF_LO + 1


def test_public_edition_has_no_saved_teams():
    data = dict(meta=dict(model_version="x"), managers=[dict(name="zero")], league=dict(table=[1]), replan={}, live={}, verification={},
                matchdays={"5": {}}, models=dict(forecasts=[dict(manager="zero")], picks_model="combined",
                                                 leaderboard=dict(summary={}, forecasts=[dict(manager="zero")]),
                                                 models=[dict(key="combined", players={}, picks=dict(zero=dict(name="zero")))]))
    pub = render.public_copy(data)
    assert pub["meta"]["public"] is True and "zero" not in json.dumps(pub)
    assert "managers" in data and "forecasts" in data["models"]            # the private data is untouched
    assert pub["models"]["picks_model"] == "combined" and "matchdays" in pub


def test_page_parts_are_placed_and_no_marker_is_left():
    html = render.assemble()
    for mark, _ in render.PART_MARKS:
        assert mark not in html
    assert "function openPlayer" in html and 'id="lab-svg"' in html and "dialog.sheet" in html
    assert html.count(render.MARK) == 1


def test_default_skin_is_set_and_unknown_names_fall_back(tmp_path):
    data = dict(meta=dict(model_version="x", skin="night"), matchdays={})
    html = render.render(data, tmp_path / "a.html", standalone=False).read_text(encoding="utf-8")
    assert 'var k = "night"' in html and "__UFE_SKIN__" not in html
    data["meta"]["skin"] = "nonsense"
    html = render.render(data, tmp_path / "b.html", standalone=False).read_text(encoding="utf-8")
    assert 'var k = "cards"' in html
    for t in render.SKINS[:-1]:                                    # every template has its fonts and its hero
        assert f'"{t}":' in html and f':root[data-skin="{t}"]' in html
