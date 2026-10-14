"""Widen the player pool from team news: predicted starters missing from the price lists.

    python scripts/expand_pool.py --md 2
writes data/seed/players_added_2026_27.csv (verified / est_fpl / unknown prices) and prints the
FPL->UCL price model diagnostics. Run again whenever a new news file is added.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ufe.ingest import team_news  # noqa: E402
from ufe.pipeline import SEED, base_pool  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=int, nargs="+", default=[2])
    a = ap.parse_args()
    pool = base_pool()
    extra = pd.read_csv(SEED / "prices_extra_2026_27.csv")
    out_p = SEED / "players_added_2026_27.csv"
    prev = pd.read_csv(out_p) if out_p.exists() else pd.DataFrame()
    added = []
    for md in a.md:
        news = team_news.load_for(md, pool)
        if news is None:
            print(f"no news file for MD{md}")
            continue
        rows, model = team_news.expand_pool(pool, news, extra, as_of=str(pd.Timestamp.now().date()), source=f"news_md{md}")
        print("FPL->UCL price model:", model)
        added.append(rows)
    allrows = pd.concat([prev] + added, ignore_index=True) if added else prev
    if not allrows.empty:
        allrows = allrows.drop_duplicates(subset=["team", "name"], keep="last")
        allrows.to_csv(out_p, index=False)
        print(allrows.price_status.value_counts().to_string())
        print(allrows.sort_values(["price_status", "team"]).to_string(max_rows=200))


if __name__ == "__main__":
    main()
