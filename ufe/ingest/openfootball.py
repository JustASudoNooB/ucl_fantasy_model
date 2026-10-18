"""openfootball results: domestic leagues (football.json) and UEFA competitions (Football.TXT).

These files are the team model's training data. They are fetched from GitHub raw URLs and
cached under data/raw/openfootball/. Parsing keeps 90-minute scores (extra time and
shoot-outs are tracked separately) because the model predicts 90-minute goals.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import ROOT, sources
from ..names import club_key, norm, ucl_team_id
from .http import fetch_to_file

RAW_DIR = ROOT / "data" / "raw" / "openfootball"

LEAGUE_COUNTRY = {
    "en": "ENG", "es": "ESP", "de": "GER", "it": "ITA", "fr": "FRA", "nl": "NED", "pt": "POR",
    "be": "BEL", "tr": "TUR", "at": "AUT", "gr": "GRE", "sco": "SCO", "ch": "SUI", "dk": "DEN", "no": "NOR",
}

MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

MATCH_RE = re.compile(
    r"^\s*(?:(\d{1,2}:\d{2})\s+)?(?P<t1>.+?)\s+\((?P<c1>[A-Z]{3})\)\s+v\s+(?P<t2>.+?)\s+\((?P<c2>[A-Z]{3})\)\s*(?P<rest>.*)$")
DATE_RE = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
SCORE_RE = re.compile(r"(\d+)-(\d+)")

# Non-UCL clubs that appear under different names in league vs cup files.
MANUAL_ALIASES = {
    ("POR", "sporting braga"): "sporting braga",
    ("POR", "sporting clube braga"): "sporting braga",
    ("POR", "sc braga"): "sporting braga",
    ("POR", "braga"): "sporting braga",
    ("GRE", "olympiakos piraeus"): "olympiakos",
    ("GRE", "olympiakos"): "olympiakos",
    ("GRE", "paok saloniki"): "paok",
    ("UKR", "dinamo kiev"): "dynamo kyiv",
    ("UKR", "dynamo kyiv"): "dynamo kyiv",
    ("AZE", "qarabag agdam"): "qarabag",
    ("AZE", "qarabag"): "qarabag",
    ("ITA", "lazio roma"): "lazio",
    ("BEL", "royale union saint gilloise"): "union saint gilloise",
    ("TUR", "besiktas jk"): "besiktas",
    ("NED", "az alkmaar"): "az",
    ("POR", "sport lisboa e benfica"): "benfica",
    ("POR", "sl benfica"): "benfica",
    ("POR", "benfica"): "benfica",
}


def download(seasons: list[str], leagues: list[str] | None = None, comps: tuple[str, ...] = ("cl", "el", "conf"),
             force: bool = False) -> list[Path]:
    """Fetch league JSON and cup TXT files for the given seasons into the cache."""
    src = sources()
    leagues = leagues or src["openfootball_leagues"]["codes"]
    got = []
    for season in seasons:
        for code in leagues:
            url = src["openfootball_leagues"]["url"].format(season=season, code=code)
            dest = RAW_DIR / f"{season}_{code}.json"
            if fetch_to_file(url, dest, force=force, ok_missing=True):
                got.append(dest)
        for comp in comps:
            url = src["openfootball_cups"]["url"].format(season=season, comp=comp)
            dest = RAW_DIR / f"ucl_{season}_{comp}.txt"
            if fetch_to_file(url, dest, force=force, ok_missing=True):
                got.append(dest)
    return got


def _league_rows(path: Path) -> list[dict]:
    season, code = path.stem.split("_", 1)
    country = LEAGUE_COUNTRY.get(code.split(".")[0], code.upper())
    tier = int(code.split(".")[1]) if "." in code else 1
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for m in data.get("matches", []):
        sc = m.get("score")
        ft = sc if isinstance(sc, list) else (sc or {}).get("ft")
        if not ft or len(ft) != 2 or m.get("date") is None:
            continue
        rows.append(dict(date=m["date"], comp=code, season=season, tier=tier, home=m["team1"], away=m["team2"],
                         home_country=country, away_country=country, hg=int(ft[0]), ag=int(ft[1]),
                         neutral=False, stage=m.get("round", ""), extra_time=False, source=f"openfootball:{code}"))
    return rows


def _parse_cup(path: Path) -> list[dict]:
    _, season, comp = path.stem.split("_", 2)
    text = path.read_text(encoding="utf-8").splitlines()
    start_year = int(season[:4])
    year, last_month = start_year, None
    current_date = None
    stage = ""
    rows = []
    for line in text:
        if line.startswith("▪"):
            stage = line.strip("▪ ").strip()
            continue
        dm = DATE_RE.match(line)
        if dm:
            mon, dd, yy = MONTHS[dm.group(2)], int(dm.group(3)), dm.group(4)
            if yy:
                year = int(yy)
            elif last_month is not None and mon < last_month - 3:
                year += 1
            last_month = mon
            current_date = date(year, mon, dd).isoformat()
            continue
        mm = MATCH_RE.match(line)
        if not mm or current_date is None:
            continue
        rest = mm.group("rest")
        scores = SCORE_RE.findall(rest)
        if not scores:
            continue
        et = "a.e.t." in rest
        if et:
            # "3-2 a.e.t. (3-0, 1-0)" or "4-3 pen. 1-1 a.e.t. (1-1, 0-1)": 90-min score is first in parentheses
            paren = re.search(r"\(([^)]*)\)", rest)
            ft = SCORE_RE.findall(paren.group(1))[0] if paren else scores[-1]
        else:
            ft = scores[0]
        rows.append(dict(date=current_date, comp=comp, season=season, tier=0,
                         home=mm.group("t1").strip(), away=mm.group("t2").strip(),
                         home_country=mm.group("c1"), away_country=mm.group("c2"),
                         hg=int(ft[0]), ag=int(ft[1]), neutral=stage.lower().endswith("final") and "semi" not in stage.lower()
                         and "quarter" not in stage.lower(),
                         stage=stage, extra_time=et, source=f"openfootball:{comp}"))
    return rows


class TeamResolver:
    """Gives every club a stable id across sources: UCL ids for the 36, 'CCC:key' otherwise."""

    def __init__(self, domestic: pd.DataFrame):
        self.authoritative: dict[str, set[str]] = {}
        for col, ccol in (("home", "home_country"), ("away", "away_country")):
            for n, c in zip(domestic[col], domestic[ccol]):
                self.authoritative.setdefault(c, set()).add(self._key(n, c))
        self.cache: dict[tuple[str, str], str] = {}

    @staticmethod
    def _key(name: str, country: str) -> str:
        k = club_key(name).replace(" clube ", " ").replace("clube ", "")
        return MANUAL_ALIASES.get((country, k), k)

    def resolve(self, name: str, country: str) -> str:
        ck = (name, country)
        if ck in self.cache:
            return self.cache[ck]
        uid = ucl_team_id(name)
        if uid is not None:
            from ..config import team_lookup
            if team_lookup()[uid]["country"] != country:
                uid = None
        if uid is None:
            key = self._key(name, country)
            pool = self.authoritative.get(country)
            if pool is None:            # cup-only country: merge variants among cup names seen so far
                pool = self.authoritative.setdefault(f"_cup_{country}", set())
                toks = set(key.split())
                supers = [p for p in pool if toks and toks < set(p.split())]
                subs = [p for p in pool if set(p.split()) and set(p.split()) < toks]
                if len(supers) == 1:
                    key = supers[0]
                elif len(subs) == 1:
                    key = subs[0]
                pool.add(key)
                pool = set()
            if key not in pool and pool:
                toks = set(key.split())
                supers = [p for p in pool if toks and toks <= set(p.split())]
                subs = [p for p in pool if set(p.split()) and set(p.split()) <= toks]
                if len(supers) == 1:
                    key = supers[0]
                elif len(subs) == 1:
                    key = subs[0]
            # a domestic name may itself belong to a UCL club (e.g. "Racing Club de Lens")
            uid = None
            for alias_name in (name, key):
                cand = ucl_team_id(alias_name)
                if cand is not None:
                    from ..config import team_lookup
                    if team_lookup()[cand]["country"] == country:
                        uid = cand
                        break
            out = uid or f"{country}:{key}"
        else:
            out = uid
        self.cache[ck] = out
        return out


def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All cached results as one frame with resolved team ids."""
    league_rows, cup_rows = [], []
    for p in sorted(raw_dir.glob("*.json")):
        league_rows.extend(_league_rows(p))
    for p in sorted(raw_dir.glob("ucl_*.txt")):
        cup_rows.extend(_parse_cup(p))
    dom = pd.DataFrame(league_rows)
    cup = pd.DataFrame(cup_rows)
    res = TeamResolver(dom[dom.tier == 1] if len(dom) else dom)
    df = pd.concat([dom, cup], ignore_index=True)
    df["home_id"] = [res.resolve(n, c) for n, c in zip(df.home, df.home_country)]
    df["away_id"] = [res.resolve(n, c) for n, c in zip(df.away, df.away_country)]
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "home_id", "away_id"]).sort_values("date").reset_index(drop=True)
    return df


def add_current_ucl(df: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Append this season's played UCL matches (team ids already canonical)."""
    played = fixtures[fixtures.status == "FT"].copy()
    if played.empty:
        return df
    from ..config import team_lookup
    tl = team_lookup()
    rows = pd.DataFrame(dict(
        date=pd.to_datetime(played["date"]), comp="ucl_current", season="2026-27", tier=0,
        home=played.home.map(lambda t: tl[t]["name"]), away=played.away.map(lambda t: tl[t]["name"]),
        home_country=played.home.map(lambda t: tl[t]["country"]), away_country=played.away.map(lambda t: tl[t]["country"]),
        hg=played.home_goals.astype(int), ag=played.away_goals.astype(int), neutral=False,
        stage=played.md.map(lambda m: f"League, Matchday {m}"), extra_time=False, source="seed:fixtures",
        home_id=played.home.values, away_id=played.away.values))
    out = pd.concat([df, rows], ignore_index=True)
    return out.drop_duplicates(subset=["date", "home_id", "away_id"], keep="last").sort_values("date").reset_index(drop=True)
