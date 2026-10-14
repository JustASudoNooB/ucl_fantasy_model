"""Time handling. UEFA publishes kick-offs in CET/CEST; the user lives in IST."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

UEFA_TZ = ZoneInfo("Europe/Paris")
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


def kickoff_dt(day: str | date, hhmm: str | None) -> datetime:
    """Kick-off as an aware datetime in UEFA local time (defaults to 21:00 when unknown)."""
    if isinstance(day, str):
        day = date.fromisoformat(day)
    hhmm = hhmm or "21:00"
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=UEFA_TZ)


def to_ist(dt: datetime) -> datetime:
    return dt.astimezone(IST)


def fmt_ist(dt: datetime) -> str:
    return to_ist(dt).strftime("%a %d %b, %H:%M IST")


def deadline_for(md_row: dict) -> datetime:
    day, hhmm = md_row["first_kickoff"].split(" ")
    return kickoff_dt(day, hhmm)


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def hours_until(dt: datetime, ref: datetime | None = None) -> float:
    ref = ref or now_utc()
    return (dt - ref).total_seconds() / 3600.0


def day_index(dates: list[str]) -> dict[str, int]:
    """Map each distinct match date of a matchday to 1, 2, 3 ... in order."""
    ordered = sorted(set(dates))
    return {d: i + 1 for i, d in enumerate(ordered)}


def add_days(d: str, n: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()
