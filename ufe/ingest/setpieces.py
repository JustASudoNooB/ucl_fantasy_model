"""This season's league penalty takers for the UCL clubs, from starting11.com (free, no key).

https://starting11.com/set-piece-takers/{league} lists, per club, every player who has taken a
league penalty this season with scored/taken ("2/2 100%"). Ten league pages cover 28 of the 36
UCL clubs (not Norway, Austria, Czechia, Slovakia, Azerbaijan, Ukraine or Greece). Writes
data/processed/penalty_takers.csv, which models/penalties.py adds to the UCL match events.
"""
from __future__ import annotations

import html as H
import re
from datetime import date

import pandas as pd

from ..config import ROOT
from ..names import ucl_team_id
from .http import get

BASE = "https://starting11.com/set-piece-takers/"
LEAGUES = ["premier-league", "la-liga", "bundesliga", "serie-a", "ligue-1", "eredivisie", "super-lig",
           "primeira-liga", "belgian-pro-league", "scottish-premiership"]
OUT = ROOT / "data" / "processed" / "penalty_takers.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0 Safari/537.36"}


def parse(html: str, league: str = "") -> pd.DataFrame:
    cut = html.find("self.__next_f")
    s = re.sub(r"<!--.*?-->", "", html[:cut] if cut > 0 else html)
    rows = []
    for art in re.finditer(r'<article id="club-([^"]+)"(.*?)</article>', s, flags=re.S):
        body = art.group(2)
        h = re.search(r"<h3[^>]*>([^<]+)</h3>", body)
        team = ucl_team_id(H.unescape(h.group(1)).strip()) if h else None
        if team is None:
            continue
        for li in re.finditer(r"<li[^>]*>(.*?)</li>", body, flags=re.S):
            x = li.group(1)
            a = re.search(r'href="/player/([^"]+)">([^<]+)</a>', x)
            sc = re.search(r">(\d+)/(\d+)<", x)
            pos = re.search(r"tracking-\[0\.14em\][^>]*>([A-Z]{2,3})</span>", x)
            if a and sc:
                full = re.sub(r"(-\d+)+$", "", a.group(1)).replace("-", " ")     # "dembele-153" -> "dembele"
                rows.append(dict(team=team, name=H.unescape(a.group(2)).strip(), slug=a.group(1),
                                 full=full, pos=pos.group(1) if pos else "",
                                 scored=int(sc.group(1)), taken=int(sc.group(2)), league=league))
    return pd.DataFrame(rows, columns=["team", "name", "slug", "full", "pos", "scored", "taken", "league"])


def sync(pages: dict[str, str] | None = None) -> str:
    frames, failed = [], []
    for lg in LEAGUES:
        try:
            html = pages[lg] if pages is not None else None
            if html is None and pages is None:
                r = get(BASE + lg, headers=UA, retries=1)
                if r.status_code != 200:
                    failed.append(f"{lg}:{r.status_code}")
                    continue
                html = r.text
            if html:
                frames.append(parse(html, lg))
        except Exception as e:  # noqa: BLE001
            failed.append(f"{lg}:{type(e).__name__}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return "penalty takers: nothing parsed" + (f" (failed {', '.join(failed)})" if failed else "")
    df["read_at"] = date.today().isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    return (f"penalty takers -> {OUT.name}: {df.team.nunique()} UCL clubs, {len(df)} players"
            + (f"; failed {', '.join(failed)}" if failed else ""))
