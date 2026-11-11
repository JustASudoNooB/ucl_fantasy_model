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
    tie = leaderboard.summarise({"1": {"a": {"start_ll": 0.5}, "b": {"start_ll": 0.5}}}, ["a", "b"])
    assert tie["wins"] == {"a": 0.5, "b": 0.5}                 # ties split the win


def test_pre_deadline_copies_freeze_at_the_deadline(tmp_path, monkeypatch):
    from ufe.ingest import frozen
    p = tmp_path / "news_md4.txt"
    p.write_text("before")
    before = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)          # MD4 deadline 17:45 UTC
    after = datetime(2026, 11, 3, 20, 0, tzinfo=timezone.utc)
    assert frozen.keep_pre_deadline_copy(p, 4, now=before) == tmp_path / "news_md4_pre.txt"
    p.write_text("after the deadline")
    assert frozen.keep_pre_deadline_copy(p, 4, now=after) is None
    assert (tmp_path / "news_md4_pre.txt").read_text() == "before"


def test_frozen_mode_reads_pre_deadline_news(tmp_path, monkeypatch):
    from ufe.ingest import team_news
    monkeypatch.setattr(team_news, "PROC", tmp_path)
    monkeypatch.setattr(team_news, "NEWS_DIR", tmp_path / "seed")
    (tmp_path / "seed").mkdir()
    (tmp_path / "news_md2.txt").write_text("updated after the deadline")
    (tmp_path / "seed" / "md2_uefa_predicted.txt").write_text("pre-match")
    monkeypatch.delenv("UFE_FROZEN_INPUTS", raising=False)
    assert team_news.news_path(2).name == "news_md2.txt"               # live use: the latest news
    monkeypatch.setenv("UFE_FROZEN_INPUTS", "1")
    assert team_news.news_path(2).name == "md2_uefa_predicted.txt"     # backtests: no post-deadline file
    (tmp_path / "news_md2_pre.txt").write_text("frozen at the deadline")
    assert team_news.news_path(2).name == "news_md2_pre.txt"


def test_feed_maps_clubs_by_any_name_field():
    from ufe.ingest.ucl_fantasy import normalise
    recs = [dict(id=1, pFName="Désiré Doué", tName="Paris", cCode="PSG", skill=3, value=8.5, totPts=30),
            dict(id=2, pFName="Unknown", tName="Atlantis FC", skill=3, value=5.0, totPts=0)]
    df = normalise(recs)
    assert list(df.team) == ["PSG"] and df.attrs["unmapped"] == ["Atlantis FC"]


def test_api_football_league_row_picks_the_domestic_league(monkeypatch):
    from ufe.ingest import api_football as af
    stats = [dict(league=dict(name="UEFA Champions League"), games=dict(appearences=4, lineups=4, minutes=360), goals=dict(total=2), shots=dict(total=9, on=5), penalty={}),
             dict(league=dict(name="La Liga"), games=dict(appearences=11, lineups=10, minutes=880), goals=dict(total=5, assists=3),
                  shots=dict(total=30, on=14), penalty=dict(scored=1, missed=1)),
             dict(league=dict(name="Copa del Rey"), games=dict(appearences=1), goals={}, shots={}, penalty={})]
    r = af.league_row("BAR", dict(name="L. Yamal"), stats)
    assert r["comp"] == "La Liga" and r["starts"] == 10 and r["pk"] == 2
    assert abs(r["xg"] - (0.10 * 28 + 0.76 * 2)) < 1e-9
    monkeypatch.delenv("UFE_API_FOOTBALL_KEY", raising=False)
    assert "skipped" in af.sync()


def test_feed_keeps_stats_and_matchday_points(tmp_path, monkeypatch):
    from ufe.ingest import ucl_fantasy as uf
    rec = dict(id=7, pFName="Nicolás Paz", pDName="N. Paz", tName="Como", cCode="COM", skill=3, value=7.0, totPts=43,
               minsPlyd=318, gS=3, assist=5, bR=15, mOM=1, trained="In contention to start next game", mdId="1", curGDPts=8)
    df = uf.normalise([rec])
    assert df.mins[0] == 318 and df.recoveries[0] == 15 and df.trained[0].startswith("In contention")
    monkeypatch.setattr(uf, "PROC", tmp_path)
    monkeypatch.setattr(uf, "played_mds", lambda now=None: [1, 2])
    served = {1: [dict(rec, mdId="1", curGDPts=8)], 2: [dict(rec, mdId="1", curGDPts=8)]}     # gameday 2 not honoured
    fetch = lambda u: {"data": {"value": {"playerList": served[int(u.split("gamedayId=")[1][0])]}}}
    out = uf.sync_md_points("https://x/players?gamedayId=1&language=en", fetch=fetch)
    got = __import__("pandas").read_csv(out)
    assert got[["md", "points"]].values.tolist() == [[1, 8]]            # the mismatched matchday is not stored
    # with the main feed's records: the finished latest matchday comes from "last matchday" points,
    # and a single matchday still missing is the season total minus the rest
    monkeypatch.setattr(uf, "played_mds", lambda now=None: [1, 2, 3])
    monkeypatch.setattr(uf, "_finished", lambda md: True)
    served = {1: [dict(rec, mdId="1", curGDPts=8)], 2: [dict(rec, mdId="1")], 3: [dict(rec, mdId="1")]}
    out = uf.sync_md_points("https://x/players?gamedayId=1&language=en", fetch=fetch, latest=[dict(rec, lastGdPoints=13)])
    got = __import__("pandas").read_csv(out).sort_values("md")
    assert got[["md", "points"]].values.tolist() == [[1, 8], [2, 22], [3, 13]]     # 43 - 8 - 13 = 22


def test_calibration_refits_ratios_and_split_within_bounds():
    from ufe.compare import calibrate as c
    d = pd.DataFrame(dict(team=["A"] * 4 + ["B"] * 2, pos=["DEF", "MID", "MID", "FWD", "MID", "FWD"],
                          mins=[360, 360, 180, 360, 360, 360], rec90_prior=[4.0, 5.0, 5.0, 2.0, 5.0, 2.0],
                          recoveries=[24, 14, 7, 4, 16, 6], xa_share=[0.1, 0.3, 0.3, 0.1, 0.4, 0.1],
                          assists=[0, 2, 1, 3, 1, 1]))
    r = c.recovery_ratios(d)
    assert r["DEF"]["fitted"] == 1.5                      # 24 recoveries / (4.0 x 4 matches)
    assert r["MID"]["fitted"] == round(37 / (5.0 * 10), 3)
    a = c.assist_split(d)
    assert a["FWD"]["actual"] == 4 and a["FWD"]["ratio"] > 1.5      # forwards out-assist their share
    assert c._step(1.0, 2.0, 0.15) == 1.15 and c._step(1.0, 0.5, 0.15) == 0.85 and c._step(1.0, 1.05, 0.15) == 1.05


def test_start_normalisation_gives_eleven_and_caps_outfield():
    from ufe.pipeline import Engine
    e = Engine(as_of="2026-11-01")
    e.__dict__["cfg"] = {"start": {"normalise": True, "team_total": 10.8, "cap": 0.85, "cap_gk": 0.97}}
    pl = pd.DataFrame(dict(team=["A"] * 16, pos=["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 6 + ["FWD"] * 3,
                           p_start=[0.95, 0.05] + [0.9, 0.9, 0.8, 0.6, 0.0] + [0.95, 0.9, 0.7, 0.5, 0.3, 0.1] + [0.9, 0.4, 0.2]))
    out = e._normalise_starts(pl)
    assert abs(out.p_start.sum() - 10.8) < 0.01
    assert out.p_start[out.pos != "GK"].max() <= 0.85 + 1e-9 and out.p_start[0] > 0.9      # keeper above the outfield cap
    assert out.p_start[6] == 0.0                                                           # a player ruled out stays out
    mids = out.p_start.to_numpy()[7:13]
    assert (np.diff(mids) <= 1e-12).all()                                                  # the order inside the club is kept


def test_availability_from_uefa_flags_fades():
    from ufe.pipeline import Engine
    e = Engine(as_of="2026-11-10")
    e.__dict__["cfg"] = {"availability": {"injured": 0.03, "doubtful": 0.65, "suspended": 0.0, "not_in_squad": 0.0, "unlikely": 0.4, "fade": 0.5}}
    e.__dict__["md_at_as_of"] = 5
    e.__dict__["feed_snapshot"] = pd.DataFrame(dict(status=["I", "D", "S", "NIS", None, None],
                                                    trained=["Unlikely to start next game"] * 4 + ["Unlikely to start next game", "In contention to start next game"]),
                                               index=["inj", "dbt", "sus", "nis", "unl", "fit"])
    a5, a6 = e.availability(5), e.availability(6)
    want = {"inj": 0.03, "dbt": 0.65, "sus": 0.0, "nis": 0.0, "unl": 0.4, "fit": 1.0}
    assert all(abs(a5[k] - v) < 1e-9 for k, v in want.items())
    assert abs(a6["inj"] - 0.515) < 1e-9 and a6["sus"] == 1.0 and a6["nis"] == 0.0 and abs(a6["unl"] - 0.7) < 1e-9


def test_normalising_starts_alone_does_not_move_goal_rates():
    """Without team news the line-up adjustment must leave every club's goal rate untouched."""
    from ufe.pipeline import Engine
    e = Engine(as_of="2026-11-01")
    e.__dict__["cfg"] = {"start": {"normalise": True, "team_total": 10.8, "cap": 0.85, "cap_gk": 0.97}, "news": {}}
    rng = np.random.default_rng(0)
    rows = []
    for t in ("A", "B"):
        for i, pos in enumerate(["GK", "GK"] + ["DEF"] * 6 + ["MID"] * 7 + ["FWD"] * 3):
            rows.append(dict(id=f"{t}{i}", team=t, pos=pos, p_start=float(rng.uniform(0.05, 0.97)), xg_share=float(rng.uniform(0, 0.3)),
                             xa_share=float(rng.uniform(0, 0.2))))
    pl = pd.DataFrame(rows)
    e.xi_players = lambda md: pl
    e.players_for = lambda md: e._normalise_starts(pl)
    e._others_given = lambda now: {}
    e.others_for = lambda md: {}
    f = pd.DataFrame(dict(home=["A"], away=["B"], lh=[1.7], la=[1.1]))
    out = e._lineup_adjust(5, f)
    assert abs(out.lh[0] - 1.7) < 1e-9 and abs(out.la[0] - 1.1) < 1e-9


def test_unconfirmed_absences_from_a_weaker_file_are_partial():
    """A third-party file's 'out' that UEFA's feed does not flag halves the chance at most; UEFA's
    own file, or a flag in the feed, still means out."""
    from ufe.ingest import team_news as tn
    cfg = tn.softened(tn._cfg(), 0.7)
    assert tn.absence_mult({"out"}, tn._cfg(), 1.0) == 0.0                 # full-strength file
    assert tn.absence_mult({"out"}, cfg, 0.03) == 0.0                      # the feed flags him too
    assert tn.absence_mult({"out"}, cfg, 1.0) == cfg["out_unconfirmed"]
    assert tn.absence_mult({"susp"}, cfg, 1.0) == cfg["susp_unconfirmed"]
    assert abs(tn.absence_mult({"out"}, cfg, None) - 0.3) < 1e-9           # no feed: the file's strength
    players = pd.DataFrame(dict(id=["a", "b", "c"], name=["A", "B", "C"], team="XXX", p_start=[0.8, 0.8, 0.8], p_sub=[0.1, 0.1, 0.1]))
    news = pd.DataFrame(dict(id=["a", "b"], team="XXX", name=["A", "B"], status=["out", "out"], conf=1.0))
    out, _ = tn.apply(players, news, cfg, uefa={"a": 1.0, "b": 0.03, "c": 1.0})
    p = out.set_index("id").p_start
    assert abs(p["a"] - 0.8 * cfg["out_unconfirmed"]) < 1e-9 and p["b"] == 0.0
