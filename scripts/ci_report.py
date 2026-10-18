"""Prints what the data syncs actually produced, as GitHub Actions notices (visible on the run
page and through the API without opening the log)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ufe.config import ROOT, team_lookup  # noqa: E402
from ufe.ingest.openfootball import load_matches  # noqa: E402

PROC = ROOT / "data" / "processed"


def note(msg: str) -> None:
    print(f"::notice::{msg}")


m = load_matches()
cur = m[(m.season == "2026-27") & (m.tier > 0)]
by = cur.groupby("comp").agg(n=("date", "size"), last=("date", "max"))
fd = cur[cur.source.astype(str).str.contains("football-data", na=False)] if "source" in cur else cur.iloc[:0]
note("domestic 2026-27: " + ", ".join(f"{c} {r.n} to {r.last:%d %b}" for c, r in by.iterrows()))
ids = set(team_lookup())
last = pd.concat([m[["date", "home_id"]].rename(columns={"home_id": "t"}), m[["date", "away_id"]].rename(columns={"away_id": "t"})])
last = last[last.t.isin(ids) & (last.date.dt.year >= 2026)].groupby("t").date.max()
stale = sorted(t for t in ids if t not in last.index or (pd.Timestamp.today() - last[t]).days > 21)
note(f"UCL clubs with a domestic league match in the last 21 days: {len(ids) - len(stale)}/36; stale: {', '.join(stale) or 'none'}")
f = PROC / "domestic_form.csv"
if f.exists():
    d = pd.read_csv(f)
    note(f"domestic_form.csv: {len(d)} players, clubs {', '.join(sorted(d.team.unique()))}, team matches {int(d.team_matches.max())}")
else:
    note("domestic_form.csv: not written")
for name in ("players_feed.csv", "understat_players.csv", "fixtures.csv"):
    p = PROC / name
    note(f"{name}: {len(pd.read_csv(p)) if p.exists() else 'absent'}")
for p in sorted(PROC.glob("odds_md*.csv")) + sorted(PROC.glob("lineups_md*.csv")):
    note(f"{p.name}: {len(pd.read_csv(p))} rows")
# a club playing twice within two days in one league means a name failed to map (double-counted result)
both = pd.concat([cur[["comp", "date", "home_id", "home"]].rename(columns={"home_id": "t", "home": "n"}),
                  cur[["comp", "date", "away_id", "away"]].rename(columns={"away_id": "t", "away": "n"})]).sort_values("date")
gap = both.groupby(["comp", "t"]).date.diff().dt.days
bad = both[(gap <= 1)]
names = bad.groupby("comp").n.apply(lambda s: "; ".join(sorted(set(s)))[:300])
note(f"double-counted results (club twice within 2 days): {len(bad)} " + (" | ".join(f"{c}: {v}" for c, v in names.items()) or ""))
if (PROC / "players_feed.csv").exists():
    from ufe.pipeline import _official_pool, _overlay_feed, _read_feed
    base, feed = _official_pool(), _read_feed()
    if feed is None:
        note("game feed: present but rejected as incomplete")
    else:
        out = _overlay_feed(base)
        j = base.merge(out, on="id", suffixes=("", "_f"))
        moved = j[(j.price - j.price_f).abs() > 1e-9]
        note(f"game feed: {len(feed)} players, {int((out.source == 'ucl_feed').sum())} matched or added, "
             f"{len(out) - len(base)} added as new, {len(moved)} prices changed"
             + (f" (e.g. {', '.join(moved.name.head(8) + ' ' + moved.price.astype(str) + '->' + moved.price_f.astype(str))})" if len(moved) else ""))
