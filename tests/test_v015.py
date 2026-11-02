"""v0.15: free team news (starting11.com), free odds (rg.org) with goal lines, penalty takers,
live captain/sub calls from the game feed, mini-league probabilities."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ufe.ingest import rg_odds, setpieces, starting11, team_news
from ufe.models import penalties
from ufe.models.scoreline import implied_lambdas, match_probs, p_over_line, remove_margin
from ufe.report import league, livecall

S11 = """<html><time dateTime="2026-11-02T10:30:00Z" class="x">Updated Mon 2 Nov</time>
<p><strong>Barcelona predicted lineup, 4-3-3 shape:</strong> <span>GK <span>García (<span class="hp-b hp-bL">probable</span>)</span></span><span><span class="hp-sep"> · </span>DC <span>Cubarsí (<span class="hp-b hp-bS">toss-up</span>)</span><span>, Martín (<span class="hp-b hp-bS">toss-up</span>)</span></span><span><span class="hp-sep"> · </span>FW <span>Raphinha (<span class="hp-b hp-bD">doubt: knee</span>)</span></span></p>
<div class="hp-dbl"><div>Out</div><p class="hp-out">Barcelona: Olmo (groin injury), Koundé (suspended) and 2 more. Aston Villa: Cash (suspended).</p></div>
<div class="hp-dbl"><div>Doubts</div><ul><li>Pedri (Barcelona): knock.</li></ul></div>
<p><strong>Aston Villa predicted lineup, 4-2-3-1 shape:</strong> <span>GK <span>Suzuki (<span class="hp-b hp-bS">toss-up</span>)</span></span></p>
self.__next_f.push([1,"ignored"])</html>"""

RG = """<h2>Barcelona Vs Aston Villa Odds</h2><table><thead><tr><th>Market</th><th>BookA</th><th>BookB</th></tr></thead><tbody>
<tr><td>Moneyline: Barcelona</td><td>-400</td><td>1.25</td></tr><tr><td>Moneyline: Draw</td><td>+600</td><td>7.0</td></tr>
<tr><td>Moneyline: Aston Villa</td><td>+900</td><td>10.0</td></tr><tr><td>Over 4.5 goals</td><td>+120</td><td>-</td></tr>
<tr><td>Under 4.5 goals</td><td>-150</td><td>-</td></tr></tbody></table>"""

SP = """<article id="club-atletico-madrid" class="x"><a href="/atletico-madrid"><h3 class="t">Atletico Madrid</h3></a><ul>
<li class="g"><a class="n" href="/player/alex-grimaldo">Grimaldo</a><span class="font-mono tracking-[0.14em] q">DEF</span><span class="r">2/2<span>100%</span></span></li>
<li class="g"><a class="n" href="/player/julian-alvarez-77">Álvarez</a><span class="font-mono tracking-[0.14em] q">FWD</span><span class="r">0/1<span>0%</span></span></li></ul></article>"""


def test_starting11_parse_roles_labels_and_absences():
    p = starting11.parse(S11)
    bar = p["teams"]["BAR"]
    assert [x[1] for x in bar["xi"]] == ["García", "Cubarsí", "Martín", "Raphinha"]
    assert bar["xi"][2][0] == "DC" and bar["xi"][3][2] == "doubt"            # role carried, "doubt: knee" -> doubt
    assert bar["out"] == ["Olmo"] and bar["susp"] == ["Koundé"] and bar["doubt"] == ["Pedri"]
    assert p["teams"]["AVL"]["susp"] == ["Cash"] and p["updated"].startswith("2026-11-02")
    txt = starting11.to_news(dict(p, teams={"BAR": dict(bar, xi=bar["xi"] * 3)}), 4)
    assert "BAR | García [probable];" in txt and "# strength: 0.7" in txt


def test_news_labels_scale_the_update(tmp_path):
    f = tmp_path / "n.txt"
    f.write_text("AAA | Keeper; Sure Thing [nailed on], Maybe Man [toss-up] | | |\n", encoding="utf-8")
    news = team_news.parse(f)
    assert dict(zip(news.name, news.conf)) == {"Keeper": 1.0, "Sure Thing": 1.0, "Maybe Man": 0.5}
    players = pd.DataFrame(dict(id=["a", "b", "c"], name=["Keeper", "Sure Thing", "Maybe Man"], team="AAA",
                                pos=["GK", "MID", "MID"], p_start=[0.4, 0.4, 0.4], p_sub=0.2))
    out, _ = team_news.apply(players, team_news.match(news, players))
    q = out.set_index("id").p_start
    assert q["b"] > q["c"] > 0.4                                             # toss-up gets half the boost


def test_lone_surname_shared_by_two_is_not_guessed():
    cand = pd.DataFrame(dict(id=["j", "e"], name=["Joan García", "Eric García"]))
    assert team_news.best_in_team("García", cand)[0] is None
    assert team_news.best_in_team("Eric García", cand)[0] == "e"


def test_rg_odds_parse_and_goal_line():
    assert abs(rg_odds.to_decimal("-400") - 1.25) < 1e-9 and abs(rg_odds.to_decimal("+600") - 7.0) < 1e-9
    g = rg_odds.parse(RG)
    assert g[0]["home"] == "BAR" and g[0]["away"] == "AVL"
    fx = pd.DataFrame(dict(md=[4], home=["BAR"], away=["AVL"], fixture_id=["MD4-BAR-AVL"]))
    df = rg_odds.combine(g, fx, 4, "test")
    r = df.iloc[0]
    assert r.n_books == 2 and abs(r.home_odds - 1.25) < 1e-9 and r.total_line == 4.5 and abs(r.over_odds - 2.2) < 1e-9


def test_implied_lambdas_fit_a_3_5_line_and_asian_push():
    ph, pdr, pa = remove_margin([2.2, 3.9, 2.9], "power")
    po = remove_margin([2.05, 1.79], "proportional")[0]
    lh, la = implied_lambdas(ph, pdr, pa, po, rho=-0.03, line=3.5)
    m = match_probs(lh, la, -0.03, 10).matrix
    assert abs(p_over_line(m, 3.5) - po) < 0.02
    tot = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    assert abs(p_over_line(m, 3) - m[tot > 3].sum() / (1 - m[tot == 3].sum())) < 1e-12


def test_setpieces_parse_and_penalty_ranking(monkeypatch):
    d = setpieces.parse(SP, "la-liga")
    assert list(d.full) == ["alex grimaldo", "julian alvarez"] and list(d.taken) == [2, 1]
    ev = pd.DataFrame(dict(team=["ATM", "ATM", "ATM"], name=["alex grimaldo", "julian alvarez", "Koke"],
                           score=[4.0, 2.0, 0.3], why=["x", "y", "z"]))
    monkeypatch.setattr(penalties, "evidence", lambda: ev)
    players = pd.DataFrame(dict(id=["g", "a", "k"], name=["Grimaldo", "Julián Álvarez", "Koke"], team="ATM",
                                pos=["DEF", "FWD", "MID"], pen_order=[9, 1, 9], pen_src=[0, 2, 0]))
    out = penalties.apply(players).set_index("id")
    assert out.pen_order["g"] == 1 and out.pen_order["a"] == 2 and out.pen_order["k"] == 9


def test_days_status_between_days():
    fx = pd.DataFrame(dict(md=[4, 4, 4], date=["2026-11-03", "2026-11-03", "2026-11-04"],
                           kickoff_cet=["18:45", "21:00", "21:00"]))
    st = livecall.days_status(fx, 4, datetime(2026, 11, 4, 6, 0, tzinfo=timezone.utc))
    assert st["finished"] == 1 and st["n_days"] == 2 and st["next_kickoff"].hour == 20
    assert livecall.days_status(fx, 4, datetime(2026, 11, 3, 21, 0, tzinfo=timezone.utc))["finished"] == 0


def test_p_first_splits_ties():
    t = np.array([[10, 5, 7], [8, 5, 9]])
    assert abs(league._p_first(t, 0) - (1 + 0.5 + 0) / 3) < 1e-12
