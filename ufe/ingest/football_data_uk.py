"""Current-season domestic results from football-data.co.uk (free CSVs, no key, updated a day or
two after each round). openfootball stops being updated during the season and has no 2026/27
file at all for Belgium, Turkey, Austria, Greece or Scotland, and Norway was never covered, so
the team model was rating several UCL clubs on results from last season.

For each league the CSV is converted to openfootball's JSON layout and merged into
data/raw/openfootball/{season}_{code}.json: matches already in the openfootball file are kept,
new ones are added. Club names are mapped to the names openfootball uses for the same country
(abbreviations expanded, then fuzzy-matched), so a club keeps one id across seasons.
"""
from __future__ import annotations

import io
import json
import re

import pandas as pd
from rapidfuzz import fuzz, process

from ..names import club_key
from .http import get
from .openfootball import LEAGUE_COUNTRY, RAW_DIR

BASE = "https://www.football-data.co.uk"
MAIN = {  # football-data.co.uk division -> openfootball code
    "E0": "en.1", "E1": "en.2", "SP1": "es.1", "SP2": "es.2", "D1": "de.1", "D2": "de.2", "I1": "it.1",
    "I2": "it.2", "F1": "fr.1", "F2": "fr.2", "N1": "nl.1", "B1": "be.1", "P1": "pt.1", "T1": "tr.1",
    "G1": "gr.1", "SC0": "sco.1",
}
EXTRA = {"AUT": "at.1", "NOR": "no.1"}  # the "new leagues" files (all seasons in one CSV)

ABBR = [  # applied to the club key (lower case, accents stripped) before fuzzy matching
    (r"^man ", "manchester "), (r"^nott m ", "nottingham "), (r"^ath madrid$", "atletico madrid"),
    (r"^ath bilbao$", "athletic bilbao"), (r"^sp lisbon$", "sporting clube de portugal"), (r"^sp gijon$", "sporting gijon"),
    (r"^ein frankfurt$", "eintracht frankfurt"), (r"^m gladbach$", "borussia monchengladbach"),
    (r"^paris sg$", "paris saint germain"), (r"^wolves$", "wolverhampton wanderers"),
    (r"^sheffield utd$", "sheffield united"), (r"^sheffield weds$", "sheffield wednesday"),
    (r"^qpr$", "queens park rangers"), (r"^fc koln$", "koln"), (r"^st pauli$", "fc st pauli"),
    (r"^vallecano$", "rayo vallecano"), (r"^espanol$", "espanyol"), (r"^sociedad$", "real sociedad"),
    (r"^celta$", "celta vigo"), (r"^st etienne$", "saint etienne"), (r"^st truiden$", "sint truidense"),
    (r"^rennes$", "stade rennais"), (r"^sp braga$", "sc braga"), (r"^west brom$", "west bromwich albion"),
    (r"^st gilloise$", "union saint gilloise"), (r"^hearts$", "heart of midlothian"), (r"^for sittard$", "fortuna sittard"),
    (r"^nijmegen$", "nec"), (r"^buyuksehyr$", "istanbul basaksehir"), (r"^ofi crete$", "ofi"),
    (r"^ad alcorcon$", "alcorcon"), (r"^austria vienna$", "austria wien"), (r"^rapid vienna$", "rapid wien"), (r"^cambuur$", "sc cambuur"), (r"^mgladbach$", "borussia monchengladbach"),
]
RESERVE = re.compile(r"\b(b|ii|u19|u21|u23|reserves?)\b")


def _expand(name: str) -> str:
    k = club_key(name)
    for pat, rep in ABBR:
        k = re.sub(pat, rep, k)
    return k


def _is_appended(m: dict) -> bool:
    """A match this module added (not an openfootball fixture it only filled in)."""
    return m.get("source") == "football-data.co.uk" and not m.get("round")


def _known_names(code: str) -> list[str]:
    """Club names openfootball itself uses in this league (any season). Names this module added
    are left out, so an earlier mapping miss cannot become its own target."""
    names = set()
    for p in RAW_DIR.glob(f"*_{code}.json"):
        for m in json.loads(p.read_text(encoding="utf-8")).get("matches", []):
            if not _is_appended(m):
                names.update([m["team1"], m["team2"]])
    return sorted(names)


def _country_names(code: str) -> list[str]:
    """The other tiers of the same country (promoted and relegated clubs), reserve sides excluded."""
    country = code.split(".")[0]
    names = set()
    for p in RAW_DIR.glob(f"*_{country}.*.json"):
        if p.stem.split("_", 1)[1] == code:
            continue
        for m in json.loads(p.read_text(encoding="utf-8")).get("matches", []):
            if not _is_appended(m):
                names.update(n for n in (m["team1"], m["team2"]) if not RESERVE.search(club_key(n)))
    return sorted(names)


def _prefix_score(e: str, cand: str) -> int:
    """92 when every word of the short name starts a different word of the candidate
    ("lyon" -> "olympique lyonnais", "west brom" -> "west bromwich albion"), else 0."""
    a, b = e.split(), cand.split()
    if not a or any(len(t) < 3 for t in a):
        return 0
    used = set()
    for t in a:
        j = next((j for j, u in enumerate(b) if j not in used and u.startswith(t)), None)
        if j is None:
            return 0
        used.add(j)
    return 92


def _best(e: str, keys: dict[str, str]) -> tuple[str | None, int]:
    if not keys:
        return None, 0
    scored = [(max(fuzz.token_set_ratio(e, k), _prefix_score(e, k)), k) for k in keys]
    scored.sort(reverse=True)
    top, k = scored[0]
    if len(scored) > 1 and scored[1][0] == top and top < 100:
        return None, 0                                   # two equally good candidates: do not guess
    return keys[k], top


UNMAPPED: list[str] = []


def name_map(fd_names: list[str], known: list[str], wider: list[str] | None = None) -> dict[str, str]:
    """football-data name -> openfootball name: exact or fuzzy within the league first, then the
    country's other tiers; the expanded name itself when nothing is close (recorded in UNMAPPED)."""
    keys = {club_key(n): n for n in known}
    more = {club_key(n): n for n in (wider or []) if club_key(n) not in keys and not RESERVE.search(club_key(n))}
    out = {}
    for n in fd_names:
        e = _expand(n)
        if e in keys:
            out[n] = keys[e]
            continue
        hit, sc = _best(e, keys)
        if hit is None or sc < 85:
            hit, sc = _best(e, more)
        if hit is not None and sc >= 85:
            out[n] = hit
        else:
            out[n] = e.title()
            if keys:
                UNMAPPED.append(n)
    return out


def _csv(content: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):          # the site has used all three over the years
        try:
            return pd.read_csv(io.StringIO(content.decode(enc)))
        except UnicodeDecodeError:
            continue
    raise ValueError("undecodable CSV")


def _rows(df: pd.DataFrame, home: str, away: str, hg: str, ag: str) -> pd.DataFrame:
    df = df.dropna(subset=[home, away, hg, ag]).copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date.astype(str)
    return df.rename(columns={home: "h", away: "a", hg: "hg", ag: "ag"})[["date", "h", "a", "hg", "ag"]]


def _near(matches: list[dict], day: str, t1: str, t2: str, days: int = 3) -> dict | None:
    """The same fixture listed under a date a day or two off (kick-off after midnight, local vs
    UK date, or a short postponement)."""
    d = pd.Timestamp(day)
    for m in matches:
        if m.get("team1") == t1 and m.get("team2") == t2 and m.get("date") and abs((pd.Timestamp(m["date"]) - d).days) <= days:
            return m
    return None


def merge(season: str, code: str, res: pd.DataFrame) -> int:
    """Add the results to the openfootball JSON for that league and season; returns matches added.
    Rows added by an earlier run are rebuilt from scratch, so an improved name mapping never leaves
    a match in the file twice under two spellings."""
    nm = name_map(sorted(set(res.h) | set(res.a)), _known_names(code), _country_names(code))
    path = RAW_DIR / f"{season}_{code}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"name": f"{code} {season}", "matches": []}
    data["matches"] = [m for m in data["matches"] if not _is_appended(m)]
    # openfootball season files also list fixtures not yet played (no score): fill those in place
    by_key = {(m.get("date"), m.get("team1"), m.get("team2")): m for m in data["matches"]}
    added = 0
    for r in res.itertuples():
        t1, t2 = nm[r.h], nm[r.a]
        if r.date == "NaT":
            continue
        ft = [int(r.hg), int(r.ag)]
        old = by_key.get((r.date, t1, t2)) or _near(data["matches"], r.date, t1, t2)
        if old is not None:
            sc = old.get("score")
            has = sc if isinstance(sc, list) else (sc or {}).get("ft")
            if not has:
                old["date"] = r.date
                old["score"] = {"ft": ft}
                old["source"] = "football-data.co.uk"
                added += 1
            continue
        m = dict(round="", date=r.date, team1=t1, team2=t2, score={"ft": ft}, source="football-data.co.uk")
        data["matches"].append(m)
        by_key[(r.date, t1, t2)] = m
        added += 1
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return added


def sync(season: str = "2026-27") -> str:
    yy = season[2:4] + season[5:7]                              # "2026-27" -> "2627"
    done, failed = [], []
    for div, code in MAIN.items():
        try:
            r = get(f"{BASE}/mmz4281/{yy}/{div}.csv", retries=1)
            if r.status_code != 200:
                failed.append(f"{div}:{r.status_code}")
                continue
            df = _csv(r.content)
            rows = _rows(df, 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG')
            done.append(f"{code}+{merge(season, code, rows)} ({len(df)} rows, {len(rows)} played, last {rows.date.max()})")
        except Exception as e:  # noqa: BLE001
            failed.append(f"{div}:{type(e).__name__}")
    for f, code in EXTRA.items():
        try:
            r = get(f"{BASE}/new/{f}.csv", retries=1)
            if r.status_code != 200:
                failed.append(f"{f}:{r.status_code}")
                continue
            df = _csv(r.content)
            start = int(season[:4])
            # autumn-spring leagues label seasons "2026/2027"; Norway plays the calendar year "2026"
            df = df[df.Season.astype(str).str.startswith(str(start))]
            rows = _rows(df, 'Home', 'Away', 'HG', 'AG')
            done.append(f"{code}+{merge(season, code, rows)} ({len(rows)} played, last {rows.date.max()})")
        except Exception as e:  # noqa: BLE001
            failed.append(f"{f}:{type(e).__name__}")
    out = f"results (football-data.co.uk): {', '.join(done) or 'none'}" + (f"; failed {', '.join(failed)}" if failed else "")
    if UNMAPPED:
        out += f"; names kept as new clubs: {', '.join(sorted(set(UNMAPPED)))}"
    return out
