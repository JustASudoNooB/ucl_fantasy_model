"""Self-running loop: `ufe schedule` (leave it running) or `ufe schedule --once` from cron /
Windows Task Scheduler every 15 minutes. Jobs fire once per matchday at offsets from the
deadline (first kick-off) and send Telegram alerts. State lives in data/processed/scheduler_state.json.

Put your squad in data/my/squad.csv (name,team per row) to get personalised advice.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from .alerts.telegram import send
from .config import ROOT, rules
from .timeutil import IST, deadline_for

STATE = ROOT / "data" / "processed" / "scheduler_state.json"
SQUAD = ROOT / "data" / "my" / "squad.csv"

JOBS = [  # (hours before deadline, job name, command)
    (72, "sync", ["sync", "--all", "--md", "{md}"]),
    (48, "predict", ["predict", "--md", "{md}"]),
    (30, "news", ["news", "--md", "{md}", "--write"]),
    (24, "advise", ["advise", "--md", "{md}", "--squad", str(SQUAD)]),
    (6, "odds", ["sync", "--odds", "{md}"]),
    (5.5, "predict-final", ["predict", "--md", "{md}"]),
    (5, "dashboard", ["dashboard", "--mds", "{md}-8"]),
]


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def _run(cmd: list[str]) -> int:
    return subprocess.call([sys.executable, "-m", "ufe", *cmd], cwd=ROOT)


def tick(now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    state = _state()
    fired = []
    for row in rules()["matchdays"]:
        md = row["md"]
        dl = deadline_for(row)
        hours = (dl - now).total_seconds() / 3600
        if hours < -60 or hours > 80:
            continue
        for h in rules().get("alerts_hours", [48, 24, 6, 2]):
            key = f"md{md}-alert-{h}"
            if 0 < hours <= h and key not in state:
                send(f"⏰ UCL Fantasy MD{md} deadline in {hours:.0f}h — {dl.astimezone(IST):%a %d %b, %H:%M} IST")
                state[key] = now.isoformat()
                fired.append(key)
        for before, name, cmd in JOBS:
            key = f"md{md}-{name}"
            if hours <= before and hours > -1 and key not in state:
                if name == "advise" and not SQUAD.exists():
                    state[key] = "skipped: no squad file"
                    continue
                rc = _run([c.format(md=md) for c in cmd])
                state[key] = f"{now.isoformat()} rc={rc}"
                fired.append(key)
                if name == "predict-final" and rc == 0:
                    send(f"📊 MD{md} projections refreshed with the latest odds and news. Open the dashboard before "
                         f"{dl.astimezone(IST):%H:%M} IST.")
    fired += _lineup_jobs(now, state)
    _save(state)
    return fired


def _lineup_jobs(now: datetime, state: dict) -> list[str]:
    """From 75 minutes before each kick-off slot: fetch confirmed XIs, re-run the projections and
    rebuild the dashboard, once per slot (a second try 30 minutes later if none were out)."""
    import pandas as pd
    from .pipeline import load_fixtures
    from .timeutil import kickoff_dt
    fired = []
    fx = load_fixtures()
    fx = fx[fx.status != "FT"]
    for (md, day, ko), _ in fx.groupby(["md", "date", "kickoff_cet"]):
        t = kickoff_dt(str(day), str(ko)).astimezone(timezone.utc)
        mins = (t - now).total_seconds() / 60
        if not 0 < mins <= 75:
            continue
        key = f"md{md}-lineups-{day}-{ko}"
        tries = [k for k in state if k.startswith(key)]
        if any(state[k].endswith("rc=0") for k in tries) or len(tries) >= 2 or (tries and mins > 45):
            continue
        rc = _run(["sync", "--lineups", str(md)])
        state[f"{key}#{len(tries) + 1}"] = f"{now.isoformat()} rc={rc}"
        if rc == 0:
            _run(["dashboard", "--mds", f"{md}-8"])
            send(f"📋 MD{md}: confirmed line-ups for the {ko} CET games are in the model. Check the dashboard "
                 f"before {t.astimezone(IST):%H:%M} IST.")
        fired.append(key)
    return fired


def loop(once: bool = False, every_min: int = 15) -> None:
    while True:
        fired = tick()
        if fired:
            print(datetime.now().isoformat(timespec="seconds"), "fired:", ", ".join(fired))
        if once:
            return
        time.sleep(every_min * 60)
