"""Command line: `python -m ufe <command>` (or `ufe <command>` after `pip install -e .`).

Weekly rhythm (all times IST; the scheduler does this for you):
  T-72h  ufe sync --all            fixtures, results, feed, odds, Understat, news
  T-48h  ufe predict --md N        projections + pre-deadline snapshot (for honest scoring later)
  T-24h  ufe advise --md N ...     best XI, captain policy, transfer options, chip table
  T-1h   ufe lineup / ufe sync --lineups   confirmed XIs for the first games -> re-run
  after day 1   ufe --as-of <day-2 date> replan --md N --squad s.csv   refit, captain switch + subs, freeze day 2
  after day 1   ufe live --md N --points day1.csv     captain switch + manual sub calls (no refit)
  after MD      ufe grade --md N                      match-by-match grading of every snapshot
  after MD      ufe results --md N --results r.csv --players p.csv   learning loop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import ROOT, rules, settings

PROC = ROOT / "data" / "processed"


def _engine(args):
    from .pipeline import Engine
    return Engine(as_of=getattr(args, "as_of", None) or pd.Timestamp.today().date().isoformat(),
                  n_sims=getattr(args, "sims", None))


def _mds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _read_squad(path: str, players: pd.DataFrame, n: int = 15) -> list[str]:
    from .names import best_player_match
    df = pd.read_csv(path)
    ids = []
    for r in df.itertuples():
        if hasattr(r, "id") and isinstance(r.id, str) and r.id in set(players.id):
            ids.append(r.id)
            continue
        cands = players if not hasattr(r, "team") else players[players.team == r.team]
        pid = best_player_match(r.name, dict(zip(cands.name, cands.id)), cutoff=75)
        if pid is None:
            sys.exit(f"Could not match '{r.name}'. Use the id column (see `ufe players --search`).")
        ids.append(pid)
    if len(ids) != n:
        sys.exit(f"File {path} has {len(ids)} players; needs {n}.")
    return ids


# ------------------------------------------------------------------------------------ commands
def cmd_probe(args):
    from .ingest import probe
    print(probe.run())


def cmd_sync(args):
    from .ingest import fpl, openfootball
    from .pipeline import load_fixtures
    done = []
    if args.all or args.history:
        openfootball.download(settings()["team_model"]["history_seasons"])
        fpl.download()
        done.append("history")
    if args.all or args.fixtures:
        from .ingest.football_data_org import sync_fixtures
        done.append("fixtures" if sync_fixtures(load_fixtures()) is not None else "fixtures: skipped (no UFE_FOOTBALL_DATA_KEY or unreachable)")
    if args.all or args.feed:
        from .ingest import ucl_fantasy
        p = ucl_fantasy.sync(args.md or 1)
        done.append(f"feed -> {p}" if p else "feed: not found (set players_url in config/sources.yaml)")
    if args.all or args.understat:
        from .ingest import understat
        done.append(f"understat -> {understat.sync()}")
    if args.odds:
        from .ingest import odds
        p = odds.fetch(args.odds, load_fixtures())
        done.append(f"odds -> {p}" if p else "odds: skipped (no UFE_ODDS_API_KEY or no prices yet)")
    print("\n".join(done) or "nothing selected (use --all)")


def cmd_tune(args):
    from .evaluate.tune import tune
    from .ingest.openfootball import load_matches
    print(json.dumps(tune(load_matches()), indent=2))


def cmd_backtest(args):
    from .evaluate.backtest import backtest_team_model
    from .ingest.openfootball import load_matches
    tm = settings()["team_model"]
    df = load_matches()
    for s in args.seasons.split(","):
        _, summ = backtest_team_model(df, s, half_life_days=tm["half_life_days"], european_weight=tm["european_weight"],
                                      prior_sd_team=tm["prior_sd_team"], prior_sd_league=tm["prior_sd_league"],
                                      form_mix=float(tm.get("form_mix", 0.0)), form_half_life=float(tm.get("form_half_life", 60.0)))
        keep = {k: round(v, 4) if isinstance(v, float) else v for k, v in summ.items() if not k.startswith("calibration")}
        print(s, json.dumps(keep, indent=2))


def cmd_predict(args):
    from .report.export import snapshot
    eng = _engine(args)
    md = args.md
    print(f"\nMD{md} fixtures (model {eng.team_model.fitted_through}, {eng.team_model.n_matches} matches)\n")
    for f in eng.match_probabilities(md):
        top = f["top_scores"][0]
        print(f"  day {f['day']}  {f['home']:>3} v {f['away']:<3}  H {f['p_home']:.0%}  D {f['p_draw']:.0%}  A {f['p_away']:.0%}"
              f"   xG {f['xg_home']:.2f}-{f['xg_away']:.2f}   likeliest {top[0]}-{top[1]} ({top[2]:.0%})   {f['kickoff_ist']} IST")
    proj = eng.projections(md)
    cols = ["name", "team", "pos", "price", "xpts", "p90", "p_start_sim", "p_goal", "p_assist", "p_potm", "day"]
    print("\nTop projected players\n", proj[cols].head(args.top).round(2).to_string(index=False))
    p = snapshot(eng, md)
    print(f"\nPre-deadline snapshot saved: {p}")


def cmd_advise(args):
    from .optimize import chips, policy, transfers
    from .optimize.squad import best_xi
    from .rules import free_transfers
    eng = _engine(args)
    md = args.md
    players = eng.players_for(md)
    squad = _read_squad(args.squad, players)
    horizon = [m for m in range(md, md + args.horizon) if m <= 8 or m <= md]
    E = eng.ev_matrix(horizon)
    names = players.set_index("id").name
    form = tuple(int(x) for x in args.formation.split("-")) if args.formation else None
    bx = best_xi(squad, players, E[md], form)
    cap, rep, table = policy.best_captain(eng.simulate(md), players, squad, bx["xi"], bx["bench"], md, E[md])
    print(f"\nBest XI ({bx['formation']}): {', '.join(names[p] for p in bx['xi'])}")
    print(f"Captain: {names[cap]} (chosen by switch-plan value)   Bench: {', '.join(names[p] for p in bx['bench'])}")
    for row in table:
        print(f"    captain {names[row['id']]:<24} set-and-forget {row['ev_passive']:.1f}   with plan {row['ev_policy']:.1f}")
    print(f"Expected: {rep.ev_passive:.1f} passive, {rep.ev_active:.1f} with the in-matchday policy")
    for r in rep.rules:
        print("  -", r["text"])
    ft = args.ft if args.ft is not None else free_transfers(md)[0]
    print("\nTransfer options (this matchday, planned over MD" + ",".join(map(str, horizon)) + ")")
    for o in transfers.options(players, E, squad, args.bank, ft, horizon, max_k=args.max_transfers):
        sell = ", ".join(names[p] for p in o.sell) or "-"
        buy = ", ".join(names[p] for p in o.buy) or "-"
        print(f"  {o.n_transfers} transfer(s), hits {o.hits}: OUT {sell} | IN {buy}   gain vs roll {o.gain_vs_roll:+.2f}")
    if md <= 8:
        tab = chips.chip_table(players, eng.ev_matrix(list(range(md, 9))), squad, list(range(md, 9)))
        print("\nChip values by matchday\n", tab.to_string(index=False))


def cmd_live(args):
    """After a day's games: captain switch and manual subs from realised points."""
    from .names import best_player_match
    from .optimize import policy
    from .optimize.squad import best_xi
    eng = _engine(args)
    md = args.md
    players = eng.players_for(md)
    squad = _read_squad(args.squad, players)
    E = eng.projections(md).set_index("id").xpts
    bx = best_xi(squad, players, E)
    xi = bx["xi"] if not args.xi else _read_squad(args.xi, players, n=11)
    captain = bx["captain"]
    if args.captain:
        captain = best_player_match(args.captain, dict(zip(players.name, players.id)), cutoff=75) or captain
    bench = [p for p in squad if p not in xi]
    pts = pd.read_csv(args.points)
    realised = {}
    for r in pts.itertuples():
        pid = getattr(r, "id", None) or best_player_match(r.name, dict(zip(players.name, players.id)), cutoff=75)
        if pid:
            realised[pid] = float(r.points)
    sent_off = set()
    for nm in (args.sent_off or "").split(","):
        if nm.strip():
            pid = best_player_match(nm.strip(), dict(zip(players.name, players.id)), cutoff=75)
            if pid:
                sent_off.add(pid)
    out = policy.live_decisions(eng.simulate(md), players, squad, xi, captain, bench, md, args.day, realised,
                                lineups_known=args.lineups_known, sent_off=sent_off)
    print(f"\nAfter day {args.day} (auto-sub cover still worth {out['cover']:.2f} pts):")
    for k, v in out.get("plans", {}).items():
        print(f"  plan {k:13s} {v:6.2f}{'  <- best' if k == out.get('choice') else ''}")
    for a in out["actions"]:
        print("  -", a["text"])


def cmd_replan(args):
    """Half-matchday re-plan: after day D, refresh the remaining games and decide captain/subs."""
    from .names import best_player_match
    from .optimize.squad import best_xi
    from .report import replan
    eng = _engine(args)
    md = args.md
    players = eng.players_for(md)
    squad = _read_squad(args.squad, players)
    E = eng.projections(md).set_index("id").xpts
    bx = best_xi(squad, players, E)
    xi = bx["xi"] if not args.xi else _read_squad(args.xi, players, n=11)
    captain = bx["captain"]
    if args.captain:
        captain = best_player_match(args.captain, dict(zip(players.name, players.id)), cutoff=75) or captain
    override = None
    if args.points:
        override = {}
        for r in pd.read_csv(args.points).itertuples():
            pid = getattr(r, "id", None) or best_player_match(r.name, dict(zip(players.name, players.id)), cutoff=75)
            if pid:
                override[pid] = float(r.points)
    out = replan.run(eng, md, args.after_day, squad, xi, captain, points_override=override,
                     lineups_known=args.lineups_known, freeze=not args.no_freeze, label=args.label)
    names = players.set_index("id").name
    print(f"\nMD{md} after day {args.after_day}: your XI scored {out['day_points']} on day {args.after_day} "
          f"(model refit through {eng.team_model.fitted_through})")
    for r in out["squad"]:
        tag = "C " if r["captain"] else ("  " if r["in_xi"] else "b ")
        val = f"{r['realised']:>3} pts" if r["realised"] is not None else f"exp {r['expected']:.1f} (start {r['p_start']:.0%})"
        print(f"  {tag}{r['name']:<24} {r['team']} {r['pos']} day {r['day']}  {val}{'  SENT OFF' if r['sent_off'] else ''}")
    d = out["decisions"]
    print(f"\nPlans (auto-sub cover worth {d['cover']:.2f}):")
    for k, v in d.get("plans", {}).items():
        print(f"  {k:13s} {v:6.2f}{'  <- best' if k == d.get('choice') else ''}")
    for a in d["actions"]:
        print("  -", a["text"])
    if out.get("snapshot"):
        print(f"\nDay-{args.after_day + 1} predictions frozen: data/runs/md{md}/{out['snapshot']}")


def cmd_grade(args):
    """Match-by-match grading of every frozen snapshot of a matchday (and retro re-forecasts)."""
    from .evaluate import grade
    eng = _engine(args)
    out = grade.grade_md(args.md, eng.fixtures)
    for g in out:
        print(f"\n== MD{g['md']} {g['label']} ({g['model_version']}, {g['created_at']})")
        for day, s in g["summary"].items():
            if not s:
                continue
            print(f"  day {day}: {s['correct']}/{s['n']} results, log loss {s['log_loss']} (base {s['log_loss_base']}), "
                  f"RPS {s['rps']} (base {s['rps_base']}), exact scores {s['exact']}, goals {s['goals']} vs xG {s['xg']}")
            print(f"         scorers in model's top 3 of their match: {s['scorer_top3_team']}/{s['scorers']} goals; "
                  f"POTM in top 3: {s['potm_in_top3']}/{s['potm_n']}; XI {s['xi_hits']}/{s['xi_of']}")
            for k in ("p_goal", "p_assist", "p_start"):
                if s.get(k):
                    print(f"         {k}: Brier skill {s[k]['skill']} (n={s[k]['n']}, events {s[k]['events']}, predicted {s[k]['predicted']})")
    (PROC / f"grading_md{args.md}.json").write_text(json.dumps(out, default=str))


def cmd_retro(args):
    from .report.export import retro_snapshot
    eng = _engine(args)
    print(retro_snapshot(eng, args.md, args.tag))


def cmd_lineup(args):
    from .ingest import lineups
    eng = _engine(args)
    rows = lineups.from_text(eng.players, args.team, args.start, args.bench or "")
    print(rows.to_string(index=False))
    print("saved:", lineups.write(args.md, rows))


def cmd_news(args):
    from .ingest import news
    eng = _engine(args)
    teams = sorted(set(eng.fixtures[eng.fixtures.md == args.md].home) | set(eng.fixtures[eng.fixtures.md == args.md].away))
    out = news.run(eng.players, teams, write=args.write)
    print(out[["name", "team", "status", "p_available", "confidence", "title"]].to_string(index=False) if len(out) else "no signals")


def cmd_results(args):
    from .evaluate import feedback
    print(json.dumps(feedback.run(args.md, Path(args.results), Path(args.players) if args.players else None), indent=2))


def cmd_dashboard(args):
    from .evaluate.backtest import backtest_team_model
    from .report.export import dashboard_data, snapshot
    from .report.render import render
    eng = _engine(args)
    mds = _mds(args.mds)
    tm = settings()["team_model"]
    bt = {}
    for s in ("2024-25", "2025-26"):
        _, summ = backtest_team_model(eng.matches, s, half_life_days=tm["half_life_days"], european_weight=tm["european_weight"],
                                      prior_sd_team=tm["prior_sd_team"], prior_sd_league=tm["prior_sd_league"],
                                      form_mix=float(tm.get("form_mix", 0.0)), form_half_life=float(tm.get("form_half_life", 60.0)))
        bt[s] = summ
    data = dashboard_data(eng, mds, backtest=bt, standings=eng.league_table_projection())
    for md in mds:
        snapshot(eng, md)
    out = Path(args.out) if args.out else ROOT / "data" / "runs" / "dashboard.html"
    render(data, out)
    (PROC / "dashboard_data.json").write_text(json.dumps(data))
    print("dashboard:", out)


def cmd_teamnews(args):
    """Register a predicted line-ups / injuries file for a matchday and widen the pool from it."""
    import shutil
    from .ingest import team_news
    src = Path(args.file)
    dst = PROC / f"news_md{args.md}.txt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    news = team_news.parse(dst)
    print(f"MD{args.md}: {news.team.nunique()} clubs, {int((news.status == 'start').sum())} named starters, "
          f"{int((news.status == 'out').sum())} out, {int((news.status == 'doubt').sum())} doubtful, "
          f"{int((news.status == 'susp').sum())} suspended -> {dst}")
    import subprocess, sys
    subprocess.run([sys.executable, str(ROOT / "scripts" / "expand_pool.py"), "--md", str(args.md)], check=False)
    eng = _engine(args)
    n = team_news.load_for(args.md, eng.players)
    eng.players_for(args.md)
    log = eng._news.get(args.md, {}).get("log")
    if log is not None and len(log):
        log = log.assign(d=(log.p_start_after - log.p_start_before).abs()).sort_values("d", ascending=False)
        print(log.drop(columns="d").head(args.top).to_string(index=False))
    miss = n[(n.status == "start") & n.id.isna()]
    if len(miss):
        print(f"{len(miss)} named starters still unmatched:", ", ".join(f"{r.name} ({r.team})" for r in miss.itertuples()))


def cmd_players(args):
    eng = _engine(args)
    df = eng.players
    if args.search:
        df = df[df.name.str.contains(args.search, case=False) | (df.team == args.search.upper())]
    print(df[["id", "name", "team", "pos", "price", "p_start", "rate_source"]].to_string(index=False))


def cmd_schedule(args):
    from .scheduler import loop
    loop(once=args.once)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ufe", description="UCL Fantasy Engine")
    ap.add_argument("--as-of", dest="as_of", help="model date (default: today)")
    ap.add_argument("--sims", type=int, help="Monte Carlo draws per fixture")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    s = sub.add_parser("sync")
    for flag in ("all", "history", "fixtures", "feed", "understat"):
        s.add_argument(f"--{flag}", action="store_true")
    s.add_argument("--odds", type=int, metavar="MD")
    s.add_argument("--md", type=int)
    s.set_defaults(fn=cmd_sync)
    sub.add_parser("tune").set_defaults(fn=cmd_tune)
    b = sub.add_parser("backtest")
    b.add_argument("--seasons", default="2024-25,2025-26")
    b.set_defaults(fn=cmd_backtest)
    p = sub.add_parser("predict")
    p.add_argument("--md", type=int, required=True)
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(fn=cmd_predict)
    a = sub.add_parser("advise")
    a.add_argument("--md", type=int, required=True)
    a.add_argument("--squad", required=True, help="CSV with name[,team] or id per row (15 rows)")
    a.add_argument("--bank", type=float, default=0.0)
    a.add_argument("--ft", type=int)
    a.add_argument("--formation")
    a.add_argument("--horizon", type=int, default=3)
    a.add_argument("--max-transfers", dest="max_transfers", type=int, default=3)
    a.set_defaults(fn=cmd_advise)
    lv = sub.add_parser("live")
    lv.add_argument("--md", type=int, required=True)
    lv.add_argument("--day", type=int, default=1)
    lv.add_argument("--squad", required=True)
    lv.add_argument("--xi")
    lv.add_argument("--captain")
    lv.add_argument("--points", required=True, help="CSV name|id, points for players who have played")
    lv.add_argument("--lineups-known", dest="lineups_known", action="store_true")
    lv.add_argument("--sent-off", dest="sent_off", help="comma-separated names of players sent off (they cannot be subbed)")
    lv.set_defaults(fn=cmd_live)
    rp = sub.add_parser("replan", help="after a matchday's first day: refresh the rest, captain switch + subs")
    rp.add_argument("--md", type=int, required=True)
    rp.add_argument("--after-day", dest="after_day", type=int, default=1)
    rp.add_argument("--squad", required=True)
    rp.add_argument("--xi")
    rp.add_argument("--captain")
    rp.add_argument("--points", help="optional CSV name|id, points (the app's numbers override the event estimate)")
    rp.add_argument("--lineups-known", dest="lineups_known", action="store_true")
    rp.add_argument("--no-freeze", dest="no_freeze", action="store_true")
    rp.add_argument("--label", default="my squad")
    rp.set_defaults(fn=cmd_replan)
    gr = sub.add_parser("grade", help="grade frozen predictions match by match")
    gr.add_argument("--md", type=int, required=True)
    gr.set_defaults(fn=cmd_grade)
    rt = sub.add_parser("retro", help="re-forecast a played matchday with today's code (pre-match inputs only)")
    rt.add_argument("--md", type=int, required=True)
    rt.add_argument("--tag", required=True)
    rt.set_defaults(fn=cmd_retro)
    lu = sub.add_parser("lineup")
    lu.add_argument("--md", type=int, required=True)
    lu.add_argument("--team", required=True)
    lu.add_argument("--start", required=True)
    lu.add_argument("--bench")
    lu.set_defaults(fn=cmd_lineup)
    n = sub.add_parser("news")
    n.add_argument("--md", type=int, required=True)
    n.add_argument("--write", action="store_true")
    n.set_defaults(fn=cmd_news)
    tn = sub.add_parser("teamnews", help="load a predicted line-ups / injuries file (see data/seed/news/)")
    tn.add_argument("--md", type=int, required=True)
    tn.add_argument("--file", required=True, help="TEAM | XI (GK first, lines split by ;) | out | doubtful | suspended")
    tn.add_argument("--top", type=int, default=25)
    tn.set_defaults(fn=cmd_teamnews)
    r = sub.add_parser("results")
    r.add_argument("--md", type=int, required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--players")
    r.set_defaults(fn=cmd_results)
    d = sub.add_parser("dashboard")
    d.add_argument("--mds", default="2-8")
    d.add_argument("--out")
    d.set_defaults(fn=cmd_dashboard)
    pl = sub.add_parser("players")
    pl.add_argument("--search")
    pl.set_defaults(fn=cmd_players)
    sc = sub.add_parser("schedule")
    sc.add_argument("--once", action="store_true")
    sc.set_defaults(fn=cmd_schedule)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
