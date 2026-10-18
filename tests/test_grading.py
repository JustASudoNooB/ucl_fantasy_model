"""Match grading, event points, the half-time plan comparison and the new model terms."""
import numpy as np
import pandas as pd

from ufe.evaluate import grade
from ufe.models import history as H
from ufe.models.team_ratings import fit, match_weights


def _fixtures():
    return pd.DataFrame(dict(md=[2, 2], home=["AAA", "CCC"], away=["BBB", "DDD"], home_goals=[2, 0], away_goals=[1, 0],
                             status=["FT", "NS"], date=["2026-10-13", "2026-10-14"], kickoff_cet=["21:00", "21:00"]))


def _snapshot():
    grid = np.zeros((7, 7))
    grid[1, 1], grid[2, 1], grid[1, 0] = 0.12, 0.10, 0.08
    players = [dict(id="AAA-x", name="Xavier Striker", team="AAA", pos="FWD", p_goal=0.5, p_assist=0.1, p_potm=0.3, p_start_sim=0.9),
               dict(id="AAA-y", name="Yann Mid", team="AAA", pos="MID", p_goal=0.2, p_assist=0.4, p_potm=0.1, p_start_sim=0.8),
               dict(id="BBB-z", name="Zed Back", team="BBB", pos="DEF", p_goal=0.05, p_assist=0.05, p_potm=0.05, p_start_sim=0.95)]
    fx = [dict(fixture_id="MD2-AAA-BBB", home="AAA", away="BBB", day=1, p_home=0.5, p_draw=0.25, p_away=0.25, xg_home=1.6,
               xg_away=1.0, top_scores=[[1, 1, 0.12], [2, 1, 0.10]], grid=grid.tolist(), over25=0.5, btts=0.5, cs_home=0.3, cs_away=0.2)]
    return dict(md=2, created_at="t", model_version="test", fixtures=fx, players=players)


def _events():
    return pd.DataFrame([
        dict(md=2, home="AAA", away="BBB", team="AAA", player="Xavier Striker", event="goal", minute=10, note="", source=""),
        dict(md=2, home="AAA", away="BBB", team="AAA", player="Yann Mid", event="assist", minute=10, note="", source=""),
        dict(md=2, home="AAA", away="BBB", team="BBB", player="Zed Back", event="goal", minute=50, note="", source=""),
        dict(md=2, home="AAA", away="BBB", team="AAA", player="Xavier Striker", event="pen_goal", minute=80, note="", source=""),
        dict(md=2, home="AAA", away="BBB", team="AAA", player="Xavier Striker", event="potm", minute=np.nan, note="", source=""),
        dict(md=2, home="AAA", away="BBB", team="AAA", player="Yann Mid", event="sub_off", minute=55, note="", source=""),
        dict(md=2, home="AAA", away="BBB", team="AAA", player="Sub Guy", event="sub_on", minute=55, note="", source=""),
    ])


def _xis():
    return pd.DataFrame([dict(team="AAA", name="Xavier Striker", status="start", line_pos="FWD", order=0),
                         dict(team="AAA", name="Yann Mid", status="start", line_pos="MID", order=1),
                         dict(team="BBB", name="Zed Back", status="start", line_pos="DEF", order=0)])


def test_grade_match_by_match():
    g = grade.grade(_snapshot(), _fixtures(), _events(), _xis())
    assert len(g["matches"]) == 1                              # the unplayed match is not graded
    m = g["matches"][0]
    assert m["outcome"] == "H" and m["correct"] and m["p_outcome"] == 0.5
    assert m["score_rank"] == 2 and not m["exact"]             # 2-1 was the second likeliest score
    names = [s["name"] for s in m["scorers"]]
    assert names.count("Xavier Striker") == 2 and m["scorers"][0]["rank"] == 1
    assert m["potm"]["name"] == "Xavier Striker" and m["potm"]["rank"] == 1
    assert m["predicted_scorers"][0]["hit"]
    assert g["summary"]["1"]["scorer_top3_team"] == 3          # all three goals by top-3 players


def test_event_points_follow_the_rules():
    app = grade.appearances(2, _fixtures(), _events(), _xis())
    pts = grade.fantasy_points(app.set_index(app.team + "|" + app.name, drop=False),
                               {"AAA|Xavier Striker": "FWD", "AAA|Yann Mid": "MID", "BBB|Zed Back": "DEF", "AAA|Sub Guy": "MID"})
    by = dict(zip(pts.name, pts.points_events))
    assert by["Xavier Striker"] == 1 + 1 + 2 * 4 + 3           # 90 min, two goals (FWD), POTM
    assert by["Yann Mid"] == 1 + 3                             # 55 min (no 60-min point), assist, no clean sheet
    assert by["Zed Back"] == 1 + 1 + 6 - 1                     # goal (DEF), conceded 2 -> -1
    assert by["Sub Guy"] == 1


def test_form_weights_and_home_effects():
    df = pd.DataFrame(dict(date=pd.to_datetime(["2026-01-01", "2026-09-01"]), comp=["en.1", "en.1"], tier=[1, 1]))
    w0 = match_weights(df, pd.Timestamp("2026-10-01"), 500, 1.0, 2.0)
    w1 = match_weights(df, pd.Timestamp("2026-10-01"), 500, 1.0, 2.0, form_mix=0.3, form_half_life=90)
    assert w1[1] / w1[0] > w0[1] / w0[0]                       # recent matches gain weight with the form term


def test_current_season_leads_start_rate():
    r = type("R", (), dict(id="T-p"))()
    prev = (2026, True, 30.0, 3.0, 1.5, 2.0, 1.2, 30.0, 38.0, np.nan, 0.0, 2.0, "FPL 25/26", 34.0)
    cur = (H.CURRENT_SEASON, True, 0.3, 0.0, 1.5, 0.0, 1.2, 0.0, 6.0, np.nan, 0.0, 0.0, "PL 26/27", 1.0)
    out = H._summarise(r, [prev, cur], None)
    assert out["h_start_cur"] and out["h_start_rate"] < 0.5    # no starts in 6 this season outweighs 30 of 38 last season
    only_prev = H._summarise(r, [prev], None)
    assert only_prev["h_start_rate"] > 0.75


def test_fpl_form_normalise():
    from ufe.ingest import fpl_live
    data = dict(events=[dict(finished=True)] * 8 + [dict(finished=False)],
                teams=[dict(id=1, name="Arsenal", short_name="ARS"), dict(id=2, name="Burnley", short_name="BUR")],
                elements=[dict(team=1, first_name="Bukayo", second_name="Saka", minutes=600, starts=7, goals_scored=3,
                               assists=2, expected_goals="2.95"),
                          dict(team=2, first_name="A", second_name="B", minutes=500, starts=5),
                          dict(team=1, first_name="C", second_name="D", minutes=0, starts=0)])
    df = fpl_live.normalise(data)
    assert list(df.team) == ["ARS"] and df.team_matches.iloc[0] == 8 and df.xg.iloc[0] == 2.95
    assert list(df.columns) == fpl_live.COLS


def test_football_data_names_map_to_openfootball():
    from ufe.ingest.football_data_uk import name_map
    known = ["Manchester City FC", "Nottingham Forest FC", "Club Atlético de Madrid", "Athletic Club", "Real Sociedad de Fútbol"]
    m = name_map(["Man City", "Nott'm Forest", "Ath Madrid", "Sociedad", "Brand New FC"], known)
    assert m["Man City"] == "Manchester City FC" and m["Nott'm Forest"] == "Nottingham Forest FC"
    assert m["Ath Madrid"] == "Club Atlético de Madrid" and m["Sociedad"] == "Real Sociedad de Fútbol"
    assert m["Brand New FC"] == "Brand New"


def test_api_football_lineups(monkeypatch, tmp_path):
    import types
    from ufe.ingest import lineups
    monkeypatch.setenv("UFE_API_FOOTBALL_KEY", "x")
    monkeypatch.setattr(lineups, "PROC", tmp_path)
    fixtures = pd.DataFrame(dict(md=[3], date=["2026-10-20"], home=["ARS"], away=["BAY"], status=["NS"]))
    players = pd.DataFrame(dict(id=["ARS-bukayo-saka", "ARS-declan-rice", "BAY-harry-kane"],
                                name=["Bukayo Saka", "Declan Rice", "Harry Kane"], team=["ARS", "ARS", "BAY"]))

    def get(url, headers=None, params=None, **k):
        if url.endswith("/fixtures"):
            body = {"response": [{"fixture": {"id": 7}, "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Bayern München"}}}]}
        else:
            body = {"response": [
                {"team": {"name": "Arsenal"}, "startXI": [{"player": {"name": "B. Saka"}}], "substitutes": [{"player": {"name": "D. Rice"}}]},
                {"team": {"name": "Bayern München"}, "startXI": [{"player": {"name": "H. Kane"}}], "substitutes": []}]}
        return types.SimpleNamespace(status_code=200, json=lambda: body)

    monkeypatch.setattr(lineups, "get", get)
    p, note = lineups.api_football(3, players, fixtures)
    df = pd.read_csv(p)
    assert dict(zip(df.id, df.role)) == {"ARS-bukayo-saka": "start", "ARS-declan-rice": "bench", "BAY-harry-kane": "start"}
    assert "1 games" in note


def test_football_data_fills_unplayed_fixtures(monkeypatch, tmp_path):
    import json
    from ufe.ingest import football_data_uk as fd
    monkeypatch.setattr(fd, "RAW_DIR", tmp_path)
    (tmp_path / "2026-27_en.1.json").write_text(json.dumps({"matches": [
        {"date": "2026-10-04", "team1": "Arsenal FC", "team2": "Chelsea FC"},
        {"date": "2026-09-20", "team1": "Chelsea FC", "team2": "Arsenal FC", "score": {"ft": [0, 0]}}]}))
    res = pd.DataFrame(dict(date=["2026-10-04", "2026-09-20", "2026-10-11"], h=["Arsenal", "Chelsea", "Arsenal"],
                            a=["Chelsea", "Arsenal", "Brand New"], hg=[2, 0, 1], ag=[1, 0, 1]))
    assert fd.merge("2026-27", "en.1", res) == 2               # one filled in place, one new, one already there
    ms = json.loads((tmp_path / "2026-27_en.1.json").read_text())["matches"]
    assert ms[0]["score"]["ft"] == [2, 1] and len(ms) == 3
