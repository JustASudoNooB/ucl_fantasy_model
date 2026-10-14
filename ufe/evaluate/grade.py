"""Match-by-match grading of frozen predictions against what happened.

Inputs
  data/runs/md{N}/snapshot.json          pre-deadline predictions (write-once after the deadline)
  data/runs/md{N}/snapshot_day{D}.json   re-plan predictions frozen before day D's first kick-off
  data/seed/results/md{N}_events.csv     goals, assists, cards, subs, POTM (or data/processed/results/)
  data/seed/lineups/md{N}_actual_xis.txt actual starting XIs (TEAM | XI)
  fixtures (status FT)                   final scores

What gets graded, per match and per matchday day (day 1 = first evening, day 2 = second)
  result      probability given to what happened, log loss and RPS vs the UCL base rates
  scoreline   rank and probability of the actual score; goals vs expected goals
  scorers     each actual scorer's P(goal) and rank in the match; how many of the model's top-3 scored
  assisters   the same for P(assist)
  POTM        probability and rank of the actual winner
  line-ups    actual starters inside the predicted XI (top 11 by start probability)
  points      approximate fantasy points from the events (ball recoveries and most saves are not
              in match reports, so they are left out and flagged)
Player probabilities are also scored across every pool player of the clubs that played (Brier
skill vs a constant base rate), which is the fairest single number for the player layer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, rules
from ..ingest import team_news

SEED = ROOT / "data" / "seed"
PROC = ROOT / "data" / "processed"
RUNS = ROOT / "data" / "runs"
BASE_RATES = (0.505, 0.159, 0.336)          # UCL 2024/25-2025/26 home / draw / away share


def _r(x, k=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ------------------------------------------------------------------------------------- inputs
def events_path(md: int) -> Path | None:
    for d in (PROC / "results", SEED / "results"):
        p = d / f"md{md}_events.csv"
        if p.exists():
            return p
    return None


def load_events(md: int) -> pd.DataFrame | None:
    p = events_path(md)
    if p is None:
        return None
    e = pd.read_csv(p)
    e["minute"] = pd.to_numeric(e["minute"], errors="coerce")
    e["note"] = e["note"].fillna("")
    return e


def xis_path(md: int) -> Path | None:
    for d in (PROC / "lineups", SEED / "lineups"):
        p = d / f"md{md}_actual_xis.txt"
        if p.exists():
            return p
    return None


def load_xis(md: int) -> pd.DataFrame | None:
    p = xis_path(md)
    if p is None:
        return None
    x = team_news.parse(p)
    return x[x.status == "start"].reset_index(drop=True)


def _match_names(rows: pd.DataFrame, pool: pd.DataFrame, cutoff: int = 86) -> list:
    """Pool id for each (team, name) row; None when the player was not in the pool."""
    out = []
    by_team = {t: g for t, g in pool.groupby("team")}
    for r in rows.itertuples():
        cand = by_team.get(r.team)
        best, best_s = None, 0
        if cand is not None:
            for c in cand.itertuples():
                s = team_news._score(r.name, c.name)
                if s > best_s:
                    best, best_s = c.id, s
        out.append(best if best_s >= cutoff else None)
    return out


# ------------------------------------------------------------------------------------- minutes + points
def appearances(md: int, fixtures: pd.DataFrame, events: pd.DataFrame, xis: pd.DataFrame) -> pd.DataFrame:
    """One row per player who played: team, name, on, off, minutes, started, plus event counts."""
    rows = {}
    ft = fixtures[(fixtures.md == md) & (fixtures.status == "FT")]
    teams_done = set(ft.home) | set(ft.away)
    for r in xis[xis.team.isin(teams_done)].itertuples():
        rows[(r.team, r.name)] = dict(team=r.team, name=r.name, on=0.0, off=90.0, started=1, pos_hint=r.line_pos)

    def key(team, name):
        for (t, n) in rows:
            if t == team and team_news._score(name, n) >= 86:
                return (t, n)
        return None

    ev = events[events.md == md] if "md" in events else events
    for r in ev.sort_values("minute", na_position="last").itertuples():
        k = key(r.team, r.player)
        if r.event == "sub_on":
            if k is None:
                rows[(r.team, r.player)] = dict(team=r.team, name=r.player, on=float(r.minute), off=90.0, started=0, pos_hint="")
        elif r.event in ("sub_off", "red", "second_yellow") and k is not None and pd.notna(r.minute):
            rows[k]["off"] = min(rows[k]["off"], float(min(r.minute, 90)))
    df = pd.DataFrame(rows.values())
    df["minutes"] = (df.off - df.on).clip(lower=0)
    # events per player
    for col in ("goal", "pen_goal", "assist", "yellow", "red", "second_yellow", "potm", "pen_won", "pen_conceded",
                "pen_miss", "pen_save", "own_goal"):
        df[col] = 0
    df["saves"] = np.nan
    for r in ev.itertuples():
        k = key(r.team, r.player)
        if k is None:
            continue
        i = df.index[(df.team == k[0]) & (df.name == k[1])][0]
        if r.event == "saves":
            df.at[i, "saves"] = float(r.minute)
        elif r.event in df.columns:
            df.at[i, r.event] += 1
    # goals conceded while on the pitch
    goals = ev[ev.event.isin(["goal", "pen_goal", "own_goal"])]
    opp = {}
    for f in ft.itertuples():
        opp[f.home], opp[f.away] = f.away, f.home
    conc = []
    for r in df.itertuples():
        g = goals[(goals.team == opp.get(r.team)) & (goals.event != "own_goal")]
        og = goals[(goals.team == r.team) & (goals.event == "own_goal")]
        allg = pd.concat([g, og])
        conc.append(int(((allg.minute.fillna(45) >= r.on) & (allg.minute.fillna(45) <= max(r.off, r.on))).sum()))
    df["conceded_on"] = conc
    return df


def fantasy_points(app: pd.DataFrame, pos: dict[str, str]) -> pd.DataFrame:
    """Event points by the 2026/27 scoring rules. Ball recoveries, the outside-the-box bonus and
    unreported saves are not in match reports and are left out (column 'partial' = True)."""
    sc = rules()["scoring"]
    out = []
    for r in app.itertuples():
        p = pos.get(r.Index) or pos.get((r.team, r.name)) or ("GK" if r.pos_hint == "GK" else None)
        if p is None:
            out.append(np.nan)
            continue
        pts = 0
        if r.minutes > 0:
            pts += sc["appearance"]
        if r.minutes >= 60:
            pts += sc["sixty_minutes"]
        pts += (r.goal + r.pen_goal) * sc["goal"][p]
        pts += r.assist * sc["assist"]
        if r.minutes >= sc["clean_sheet_min_minutes"] and r.conceded_on == 0:
            pts += sc["clean_sheet"][p]
        pts += (r.conceded_on // sc["goals_conceded_step"]) * sc["goals_conceded_penalty"][p]
        if p == "GK" and pd.notna(r.saves):
            pts += int(r.saves // sc["saves_step"])
        pts += r.pen_save * sc["penalty_save"] + r.potm * sc["player_of_the_match"]
        pts += r.pen_won * sc["penalty_won"] + r.pen_conceded * sc["penalty_conceded"] + r.pen_miss * sc["penalty_missed"]
        pts += r.own_goal * sc["own_goal"]
        if r.red or r.second_yellow:
            pts += sc["red_card"]
        else:
            pts += r.yellow * sc["yellow_card"]
        out.append(pts)
    app = app.copy()
    app["points_events"] = out
    app["partial"] = True
    return app


# ------------------------------------------------------------------------------------- grading
def _outcome(hg: int, ag: int) -> int:
    return 0 if hg > ag else (1 if hg == ag else 2)


def _rps(p: np.ndarray, y: int) -> float:
    o = np.cumsum(np.eye(3)[y])[:2]
    return float(np.sum((np.cumsum(p)[:2] - o) ** 2) / 2)


def _score_rank(grid: list[list[float]], hg: int, ag: int) -> tuple[float | None, int | None]:
    g = np.asarray(grid, dtype=float)
    if hg >= g.shape[0] or ag >= g.shape[1]:
        return None, None
    p = float(g[hg, ag])
    return p, int((g > p).sum()) + 1


def grade(snapshot: dict, fixtures: pd.DataFrame, events: pd.DataFrame | None, xis: pd.DataFrame | None,
          label: str = "pre-deadline") -> dict:
    md = int(snapshot["md"])
    ft = fixtures[(fixtures.md == md) & (fixtures.status == "FT")]
    res = {(r.home, r.away): (int(r.home_goals), int(r.away_goals)) for r in ft.itertuples()}
    pl = pd.DataFrame(snapshot["players"])
    if pl.empty:
        return {}
    if "p_start_sim" not in pl:
        pl["p_start_sim"] = np.nan
    ev = events[events.md == md] if events is not None else pd.DataFrame(columns=["team", "player", "event", "minute", "note", "home", "away"])
    if len(ev):
        ev = ev.assign(id=_match_names(ev.rename(columns={"player": "name"}), pl))
    xi, xi_ids, xi_teams = None, set(), set()
    if xis is not None and len(xis):
        xi = xis.assign(id=_match_names(xis, pl))
        xi_ids, xi_teams = set(xi.id.dropna()), set(xi.team)
    by_id = pl.set_index("id")
    matches, all_rows = [], []
    for f in snapshot["fixtures"]:
        k = (f["home"], f["away"])
        if k not in res:
            continue
        hg, ag = res[k]
        y = _outcome(hg, ag)
        pr = np.array([f["p_home"], f["p_draw"], f["p_away"]], dtype=float)
        base = np.array(BASE_RATES)
        p_sc, rank_sc = _score_rank(f.get("grid") or [[0]], hg, ag)
        top = f["top_scores"][0]
        teams = [f["home"], f["away"]]
        mpl = pl[pl.team.isin(teams)]
        mev = ev[(ev.home == f["home"]) & (ev.away == f["away"])] if len(ev) else ev

        def ranked(col):
            return mpl.sort_values(col, ascending=False).reset_index(drop=True)

        def who(event_types, col):
            out = []
            rk = ranked(col)
            for r in mev[mev.event.isin(event_types)].itertuples():
                if r.event == "goal" and "own goal" in str(r.note).lower():
                    continue
                pid = r.id
                if pid is not None and pid in set(rk.id):
                    i = int(rk.index[rk.id == pid][0])
                    out.append(dict(name=r.player, team=r.team, minute=_r(r.minute, 0), id=pid,
                                    p=_r(float(rk.loc[i, col])), rank=i + 1, in_pool=True,
                                    pen=r.event == "pen_goal"))
                else:
                    out.append(dict(name=r.player, team=r.team, minute=_r(r.minute, 0), id=None, p=None, rank=None,
                                    in_pool=False, pen=r.event == "pen_goal"))
            return out

        scorers = who(["goal", "pen_goal"], "p_goal")
        assisters = who(["assist"], "p_assist")
        potm = who(["potm"], "p_potm")
        scored_ids = {s["id"] for s in scorers if s["id"]}
        assist_ids = {s["id"] for s in assisters if s["id"]}
        top_sc = ranked("p_goal").head(5)
        top_as = ranked("p_assist").head(5)
        top_potm = ranked("p_potm").head(3)
        # line-ups: predicted XI = top 11 by simulated start probability, per team
        xi_hits = {}
        if xi is not None:
            for t in teams:
                act = xi[xi.team == t]
                if act.empty:
                    continue
                pred = pl[pl.team == t].sort_values("p_start_sim", ascending=False).head(11)
                hits = int(act.id.dropna().isin(pred.id).sum())
                xi_hits[t] = dict(hits=hits, of=int(len(act)), missing_from_pool=int(act.id.isna().sum()),
                                  wrong=[r.name for r in pred.itertuples() if r.id not in set(act.id.dropna())],
                                  missed=[r.name for r in act.itertuples() if r.id is None or r.id not in set(pred.id)])
        m = dict(
            fixture_id=f.get("fixture_id"), home=f["home"], away=f["away"], day=int(f.get("day", 1)),
            kickoff_ist=f.get("kickoff_ist"), score=[hg, ag],
            p=[_r(x) for x in pr], outcome=["H", "D", "A"][y], p_outcome=_r(pr[y]), pick=["H", "D", "A"][int(np.argmax(pr))],
            correct=bool(np.argmax(pr) == y), log_loss=_r(-np.log(max(pr[y], 1e-9))), log_loss_base=_r(-np.log(base[y])),
            rps=_r(_rps(pr, y)), rps_base=_r(_rps(base, y)),
            xg=[f.get("xg_home"), f.get("xg_away")], likeliest=[top[0], top[1], _r(top[2])],
            exact=bool(top[0] == hg and top[1] == ag), p_score=_r(p_sc), score_rank=rank_sc,
            top_scores=[[a, b, _r(c)] for a, b, c in f["top_scores"][:5]],
            over25=_r(f.get("over25")), btts=_r(f.get("btts")), actual_over25=hg + ag > 2, actual_btts=hg > 0 and ag > 0,
            cs=[_r(f.get("cs_home")), _r(f.get("cs_away"))], actual_cs=[ag == 0, hg == 0],
            scorers=scorers, assisters=assisters, potm=potm[0] if potm else None,
            predicted_scorers=[dict(id=r.id, name=r.name, team=r.team, p=_r(r.p_goal), hit=r.id in scored_ids) for r in top_sc.itertuples()],
            predicted_assisters=[dict(id=r.id, name=r.name, team=r.team, p=_r(r.p_assist), hit=r.id in assist_ids) for r in top_as.itertuples()],
            predicted_potm=[dict(id=r.id, name=r.name, team=r.team, p=_r(r.p_potm), hit=bool(potm and r.id == potm[0]["id"])) for r in top_potm.itertuples()],
            top3_scorers_hit=int(sum(1 for r in top_sc.head(3).itertuples() if r.id in scored_ids)),
            xi=xi_hits,
        )
        matches.append(m)
        # player-level rows for calibration (every pool player of both clubs)
        for r in mpl.itertuples():
            all_rows.append(dict(day=m["day"], id=r.id, team=r.team, p_goal=r.p_goal, p_assist=r.p_assist, p_potm=r.p_potm,
                                 p_start=r.p_start_sim, scored=int(r.id in scored_ids), assisted=int(r.id in assist_ids),
                                 potm=int(bool(potm) and r.id == potm[0]["id"]),
                                 started=(int(r.id in xi_ids) if xi is not None and r.team in xi_teams else None)))
    prow = pd.DataFrame(all_rows)
    summary = {}
    for day in sorted({m["day"] for m in matches}) + ["all"]:
        ms = [m for m in matches if day == "all" or m["day"] == day]
        pr_ = prow if day == "all" else prow[prow.day == day]
        summary[str(day)] = _summarise(ms, pr_)
    return dict(md=md, label=label, created_at=snapshot.get("created_at"), model_version=snapshot.get("model_version"),
                matches=matches, summary=summary)


def _brier_skill(p: pd.Series, y: pd.Series) -> dict:
    p, y = p.astype(float).clip(0, 1), y.astype(float)
    if len(p) == 0:
        return {}
    base = float(y.mean())
    b = float(np.mean((p - y) ** 2))
    b0 = float(np.mean((base - y) ** 2))
    eps = 1e-6
    ll = float(-np.mean(y * np.log(p.clip(eps, 1)) + (1 - y) * np.log((1 - p).clip(eps, 1))))
    ll0 = float(-np.mean(y * np.log(max(base, eps)) + (1 - y) * np.log(max(1 - base, eps))))
    return dict(n=int(len(p)), events=int(y.sum()), predicted=_r(p.sum(), 2), brier=_r(b, 4), brier_base=_r(b0, 4),
                skill=_r(1 - b / b0 if b0 > 0 else np.nan, 3), log_loss=_r(ll, 4), log_loss_base=_r(ll0, 4))


def _summarise(ms: list[dict], prow: pd.DataFrame) -> dict:
    if not ms:
        return {}
    sc = [s for m in ms for s in m["scorers"]]
    asis = [s for m in ms for s in m["assisters"]]
    potm = [m["potm"] for m in ms if m["potm"]]
    xi_h = [v for m in ms for v in m["xi"].values()]
    out = dict(
        n=len(ms), correct=sum(m["correct"] for m in ms),
        log_loss=_r(np.mean([m["log_loss"] for m in ms])), log_loss_base=_r(np.mean([m["log_loss_base"] for m in ms])),
        rps=_r(np.mean([m["rps"] for m in ms])), rps_base=_r(np.mean([m["rps_base"] for m in ms])),
        exact=sum(m["exact"] for m in ms), mean_p_score=_r(np.mean([m["p_score"] or 0 for m in ms])),
        goals=sum(sum(m["score"]) for m in ms), xg=_r(sum(sum(x or 0 for x in m["xg"]) for m in ms), 2),
        scorers=len(sc), scorers_in_pool=sum(1 for s in sc if s["in_pool"]),
        scorer_mean_p=_r(np.mean([s["p"] for s in sc if s["p"] is not None])) if any(s["p"] is not None for s in sc) else None,
        scorer_top3_team=sum(1 for s in sc if s["rank"] is not None and s["rank"] <= 3),
        top3_scorer_hits=sum(m["top3_scorers_hit"] for m in ms),
        assists=len(asis), assisters_in_pool=sum(1 for s in asis if s["in_pool"]),
        potm_in_top3=sum(1 for p in potm if p["rank"] is not None and p["rank"] <= 3), potm_n=len(potm),
        potm_mean_p=_r(np.mean([p["p"] for p in potm if p["p"] is not None])) if any(p["p"] is not None for p in potm) else None,
        xi_hits=sum(v["hits"] for v in xi_h), xi_of=sum(v["of"] for v in xi_h),
        xi_missing_from_pool=sum(v["missing_from_pool"] for v in xi_h),
    )
    if len(prow):
        out["p_goal"] = _brier_skill(prow.p_goal, prow.scored)
        out["p_assist"] = _brier_skill(prow.p_assist, prow.assisted)
        st = prow.dropna(subset=["started"])
        if len(st):
            out["p_start"] = _brier_skill(st.p_start, st.started)
    return out


# ------------------------------------------------------------------------------------- convenience
def snapshot_paths(md: int) -> dict[str, Path]:
    d = RUNS / f"md{md}"
    out = {}
    if (d / "snapshot.json").exists():
        out["pre-deadline"] = d / "snapshot.json"
    for p in sorted(d.glob("snapshot_day*.json")):
        out[f"re-plan before day {p.stem.split('day')[-1]}"] = p
    for p in sorted(d.glob("retro_*.json")):
        out[f"retro {p.stem[6:]}"] = p
    return out


def grade_md(md: int, fixtures: pd.DataFrame) -> list[dict]:
    ev, xi = load_events(md), load_xis(md)
    out = []
    for label, p in snapshot_paths(md).items():
        snap = json.loads(p.read_text())
        g = grade(snap, fixtures, ev, xi, label=label)
        if g and g.get("matches"):
            g["file"] = p.name
            out.append(g)
    return out
