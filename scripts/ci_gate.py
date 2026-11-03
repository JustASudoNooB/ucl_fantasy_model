"""Gate for the scheduled matchday workflow: is this run worth its minutes?

GitHub's free plan gives a private repository 2,000 Actions minutes a month. Scheduled runs only go
ahead from 72 hours before a matchday's first kick-off to 3 hours after its last one (team news,
prices, the frozen forecast, the between-days live call); outside that window the workflow stops
after this check, which costs about a minute. Manual runs always go ahead. The Wednesday
live-call slots run "light": data sync, live call and save only (no dashboard, no scheduled jobs).

Standard library only, so it runs before any install. Writes run=/light= to $GITHUB_OUTPUT.
Test with NOW_UTC=2026-11-04T07:00 SCHEDULE="23 1,4,7,10,13 * * 3" EVENT=schedule.
"""
from __future__ import annotations

import csv
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
UEFA = ZoneInfo("Europe/Paris")
LIVE_SLOTS = "23 1,4,7,10,13 * * 3"          # must match the cron line in matchday.yml
BEFORE, AFTER = timedelta(hours=72), timedelta(hours=3)


def windows() -> list[tuple[int, datetime, datetime]]:
    synced = ROOT / "data" / "processed" / "fixtures.csv"
    path = synced if synced.exists() else ROOT / "data" / "seed" / "fixtures_2026_27.csv"
    kos: dict[int, list[datetime]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"][:10])
            h, m = (int(x) for x in ((r.get("kickoff_cet") or "21:00").strip() or "21:00").split(":"))
            kos.setdefault(int(r["md"]), []).append(datetime(d.year, d.month, d.day, h, m, tzinfo=UEFA).astimezone(timezone.utc))
    return [(md, min(v), max(v)) for md, v in sorted(kos.items())]


def decide(event: str, schedule: str, now: datetime) -> tuple[bool, bool, str]:
    if event != "schedule":
        return True, False, f"{event}: always runs"
    for md, first, last in windows():
        if first - BEFORE <= now <= last + AFTER:
            light = schedule.strip() == LIVE_SLOTS
            return True, light, f"MD{md} window ({first:%d %b %H:%M} to {last:%d %b %H:%M} UTC){' - light run' if light else ''}"
    return False, False, "no matchday within 72 hours: skipped to save Actions minutes"


if __name__ == "__main__":
    now = datetime.fromisoformat(os.environ["NOW_UTC"]).replace(tzinfo=timezone.utc) if os.environ.get("NOW_UTC") \
        else datetime.now(timezone.utc)
    run, light, why = decide(os.environ.get("EVENT", "schedule"), os.environ.get("SCHEDULE", ""), now)
    print(f"::notice::gate: {'run' if run else 'skip'} ({why})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"run={'true' if run else 'false'}\nlight={'true' if light else 'false'}\n")
