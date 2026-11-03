"""Pre-deadline copies of a matchday's inputs, so backtests only see what was known at the deadline.

Team news (news_md{N}.txt) and prices (odds_md{N}.csv) keep updating after the deadline: confirmed
line-ups and late prices help the between-days live call. Every write before the deadline also
writes <name>_pre<suffix>; after the deadline the copy is left alone, so it holds the last version
the deadline forecasts were made from. Backtests (compare/leaderboard.py) run with
UFE_FROZEN_INPUTS=1, which makes the engine read the _pre copies (and, for news, fall back to the
hand-saved pre-match files in data/seed/news rather than a file updated after the deadline).
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def frozen_mode() -> bool:
    return os.environ.get("UFE_FROZEN_INPUTS") == "1"


def pre_path(path: Path) -> Path:
    return path.with_name(path.stem + "_pre" + path.suffix)


def keep_pre_deadline_copy(path: Path, md: int, now: datetime | None = None) -> Path | None:
    from ..config import rules
    from ..timeutil import deadline_for
    row = next((r for r in rules()["matchdays"] if int(r["md"]) == int(md)), None)
    now = now or datetime.now(timezone.utc)
    if row is None or not Path(path).exists() or now >= deadline_for(row).astimezone(timezone.utc):
        return None
    dst = pre_path(Path(path))
    shutil.copyfile(path, dst)
    return dst
