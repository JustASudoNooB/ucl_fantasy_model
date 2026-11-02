"""How often each team-news label actually started (starting11.com: nailed on / probable / toss-up /
punt / doubt), so team_news.LABEL_CONF can be set from data instead of priors.

Needs the matchday's news file (data/processed/news_md{N}.txt) and its actual XIs
(data/seed/lineups/md{N}_actual_xis.txt). Prints the start rate per label and for players of the
same clubs who were not named.

  python scripts/calibrate_news_labels.py 4
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufe.evaluate import grade             # noqa: E402
from ufe.ingest import team_news           # noqa: E402
from ufe.pipeline import load_pool         # noqa: E402


def main(md: int) -> None:
    news_p, xi = team_news.news_path(md), grade.load_xis(md)
    if news_p is None or xi is None:
        print(f"MD{md}: need both the news file and the actual XIs")
        return
    pool = load_pool()
    news = team_news.match(team_news.parse(news_p), pool)
    started = {i for i in grade._match_names(xi, pool) if i}
    named = news[(news.status == "start") & news.id.notna()].copy()
    named["started"] = named.id.isin(started)
    print(f"MD{md}: {len(named)} named starters matched, {len(started)} actual starters matched")
    for lab, g in named.groupby(named.label.replace("", "(none)")):
        print(f"  {lab:<10} n={len(g):>3}  started {g.started.mean():.0%}")
    teams = set(named.team)
    others = pool[pool.team.isin(teams) & ~pool.id.isin(set(named.id))]
    print(f"  not named  n={len(others):>3}  started {others.id.isin(started).mean():.0%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
