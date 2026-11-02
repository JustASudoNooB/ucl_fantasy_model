"""Predicted line-ups and team news from starting11.com (free, no key) -> data/processed/news_md{N}.txt.

https://starting11.com/champions-league publishes an "Oracle XI" for every club of the coming
matchday, each player tagged nailed on / probable / toss-up / punt (or doubt), plus each match's
absences ("Out": injured or suspended) and doubts. The page is server-rendered: per club

    <p><strong>{Club} predicted lineup, {shape} shape:</strong> <span>GK <span>Name (<span ..>label</span>)</span></span> ...

followed by `<p class="hp-out">ClubA: Name (reason), ... ClubB: ...</p>` and a list of
`<li>Name (Club): reason.</li>` doubts. Fuller names come from the pitch graphic's titles.

The labels go into the news file as "[label]" after each name; team_news.apply scales the update
for that player by LABEL_CONF, and the whole file carries `# strength: 0.7`, since the source is
an algorithm whose hit rate is not yet measured (recalibrate from MD4: scripts/calibrate_news_labels.py).
"""
from __future__ import annotations

import html as H
import re
from datetime import datetime, timezone

import pandas as pd

from ..config import ROOT
from ..names import norm, ucl_team_id
from .http import get

URL = "https://starting11.com/champions-league"
PROC = ROOT / "data" / "processed"
STRENGTH = 0.7
LABEL_CONF = {"nailed on": 1.0, "probable": 0.8, "toss-up": 0.5, "punt": 0.3, "doubt": 0.5}
GK_ROLES = {"GK"}


def _team(name: str) -> str | None:
    name = H.unescape(name).strip()
    return ucl_team_id(name) or ucl_team_id(re.sub(r"\s+(FA|FC|SK|CF|AC)$", "", name))


def _static(html: str) -> str:
    cut = html.find("self.__next_f")
    return re.sub(r"<!--.*?-->", "", html[:cut] if cut > 0 else html)


def _full_names(s: str) -> dict[tuple[str, str], str]:
    """(club, short name) -> the pitch graphic's title name, when it is longer."""
    out = {}
    for m in re.finditer(r'aria-label="([^"]+?) predicted lineup[^"]*"(.*?)(?=aria-label="|<p><strong>|$)', s, flags=re.S):
        club = _team(m.group(1))
        for t, short in re.findall(r'title="([^"]+?) · [^"]+"[^>]*>.*?data-n="([^"]+)"', m.group(2), flags=re.S):
            t, short = H.unescape(t).strip(), H.unescape(short).strip()
            if club and len(t.split()) > len(short.split()):
                out[(club, short)] = t
    return out


def _split_people(text: str) -> list[tuple[str, str]]:
    """"A (groin injury), B (suspended), C and 2 more." -> [(A, groin injury), (B, suspended), (C, '')]."""
    text = re.sub(r"\s+and \d+ more\.?$", "", text.strip().rstrip("."))
    parts = re.split(r",\s*|\s+and\s+", text)
    out = []
    for p in parts:
        m = re.match(r"\s*(.+?)\s*(?:\(([^)]*)\))?\s*$", p)
        if m and m.group(1):
            out.append((m.group(1).strip(), (m.group(2) or "").strip()))
    return out


def parse(html: str) -> dict:
    """{'updated': iso or None, 'teams': {code: {'shape', 'xi': [(role, name, label)], 'out', 'doubt', 'susp'}}}."""
    s = _static(html)
    upd = re.search(r'<time dateTime="([^"]+)"[^>]*>\s*Updated', s)
    full = _full_names(s)
    teams: dict[str, dict] = {}
    for m in re.finditer(r"<p><strong>(.+?) predicted lineup, ([^<]+?) shape:</strong>(.*?)</p>", s, flags=re.S):
        code = _team(m.group(1))
        if code is None:
            continue
        xi, role = [], ""
        # "DC <span>Cubarsí (<span ..>toss-up</span>)</span><span>, Martín (...)</span>": one role, two players
        for r_, name, label in re.findall(r'([A-Z]{1,4})?\s*<span>(?:,\s*)?([^<(]+?)\s*\(<span class="hp-b[^"]*">([^<]+)</span>\)',
                                          m.group(3)):
            role = r_ or role
            name = H.unescape(name).strip()
            label = H.unescape(label).strip().lower().split(":")[0].strip()      # "doubt: ribs injury" -> doubt
            xi.append((role, full.get((code, name), name), label))
        teams[code] = dict(shape=m.group(2).strip(), xi=xi, out=[], doubt=[], susp=[])
    for m in re.finditer(r'<p class="hp-out">(.*?)</p>', s, flags=re.S):
        text = H.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
        # "Barcelona: Olmo (groin injury), ... . Aston Villa: Wan-Bissaka (muscle injury), ..."
        for club, body in re.findall(r"(?:^|\.\s+)([^:.]+?):\s+(.*?)(?=\.\s+[^:.]+?:\s|\.?$)", text.strip()):
            code = _team(club)
            if code not in teams:
                continue
            for name, why in _split_people(body):
                (teams[code]["susp"] if "suspend" in why.lower() else teams[code]["out"]).append(name)
    for m in re.finditer(r"<li>([^<(]+?)\s*\(([^)]+)\):\s*([^<]*)</li>", s):
        code = _team(m.group(2))
        if code in teams:
            teams[code]["doubt"].append(H.unescape(m.group(1)).strip())
    return dict(updated=upd.group(1) if upd else None, teams=teams)


ROLE_POS = {"GK": "GK"}


def _role_pos(role: str) -> str:
    if role in ROLE_POS:
        return ROLE_POS[role]
    if role.startswith("D") and not role.startswith("DM"):
        return "DEF"
    if role in ("FW", "ST", "CF", "SS"):
        return "FWD"
    return "MID"


def _aliases() -> dict[tuple[str, str], str]:
    p = ROOT / "data" / "seed" / "news" / "name_aliases.csv"
    if not p.exists():
        return {}
    a = pd.read_csv(p)
    return {(r.team, str(r.alias)): str(r.name) for r in a.itertuples()}


def resolve(parsed: dict, players: pd.DataFrame) -> dict:
    """Replace the page's short names with the pool's full names where one player clearly fits:
    keepers among keepers, and a surname shared by two players settled by the slot's position."""
    from .team_news import best_in_team
    alias = _aliases()
    by_team = {t: g for t, g in players.groupby("team")}

    def one(code: str, name: str, role: str | None) -> str:
        name = alias.get((code, name), name)
        cand = by_team.get(code)
        if cand is None:
            return name
        if role is not None:
            cand = cand[cand.pos == "GK"] if role == "GK" else cand[cand.pos != "GK"]
        pid, _ = best_in_team(name, cand)
        if pid is None and role is not None:
            pos = _role_pos(role)
            sub = cand[cand.pos == pos]
            if len(sub):
                pid, _ = best_in_team(name, sub)
        if pid is None and role is not None and "p_start" in cand:
            # two players share the surname in that slot (Porto's keepers Diogo and João Costa):
            # take the regular when the model rates him far likelier to start
            tok = norm(name).split()
            same = cand[[all(t in norm(str(n)).split() for t in tok) for n in cand.name]].sort_values("p_start", ascending=False)
            if len(same) >= 2 and same.p_start.iloc[0] >= 0.3 and same.p_start.iloc[0] >= 2 * same.p_start.iloc[1]:
                pid = same.id.iloc[0]
        return str(cand.set_index("id").name[pid]) if pid is not None else name

    out = {}
    for code, t in parsed["teams"].items():
        out[code] = dict(t, xi=[(r, one(code, n, r), lab) for r, n, lab in t["xi"]],
                         out=[one(code, n, None) for n in t["out"]], doubt=[one(code, n, None) for n in t["doubt"]],
                         susp=[one(code, n, None) for n in t["susp"]])
    return dict(parsed, teams=out)


def to_news(parsed: dict, md: int) -> str:
    lines = [f"# MD{md} predicted line-ups and team news: starting11.com Oracle XI ({URL}),",
             f"# page updated {parsed.get('updated') or 'unknown'}, read {datetime.now(timezone.utc).isoformat(timespec='minutes')}.",
             "# Written by ufe/ingest/starting11.py. Labels in [brackets] scale each named player's update.",
             f"# strength: {STRENGTH}",
             "# Format: TEAM | XI (GK first) | out | doubtful | suspended"]
    for code, t in sorted(parsed["teams"].items()):
        if len(t["xi"]) < 9:
            continue
        gk = [x for x in t["xi"] if x[0] in GK_ROLES][:1]
        rest = [x for x in t["xi"] if x not in gk]
        cell = lambda xs: ", ".join(f"{n} [{lab}]" for _, n, lab in xs)
        xi = "; ".join(p for p in (cell(gk), cell(rest)) if p)
        doubt = list(dict.fromkeys(t["doubt"] + [n for _, n, lab in t["xi"] if lab == "doubt"]))
        lines.append(f"{code} | {xi} | {', '.join(t['out'])} | {', '.join(doubt)} | {', '.join(t['susp'])}")
    return "\n".join(lines) + "\n"


def covers(parsed: dict, fixtures: pd.DataFrame, md: int, min_share: float = 0.8) -> bool:
    """The page is for this matchday: most of its fixtures' clubs have a predicted XI."""
    clubs = set(fixtures[fixtures.md == md].home) | set(fixtures[fixtures.md == md].away)
    have = {c for c, t in parsed["teams"].items() if len(t["xi"]) >= 9}
    return bool(clubs) and len(have & clubs) >= min_share * len(clubs) and len(have - clubs) <= 2


def sync(md: int, fixtures: pd.DataFrame, players: pd.DataFrame | None = None, html: str | None = None) -> tuple[str, str]:
    """Fetch, check it is this matchday's page, write data/processed/news_md{md}.txt."""
    if html is None:
        r = get(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0 Safari/537.36"}, retries=2)
        if r.status_code != 200:
            return "", f"team news (starting11): HTTP {r.status_code}"
        html = r.text
    parsed = parse(html)
    if not covers(parsed, fixtures, md):
        return "", f"team news (starting11): page does not cover MD{md} yet ({len(parsed['teams'])} clubs)"
    if players is not None:
        parsed = resolve(parsed, players)
    text = to_news(parsed, md)
    PROC.mkdir(parents=True, exist_ok=True)
    p = PROC / f"news_md{md}.txt"
    p.write_text(text, encoding="utf-8")
    n = sum(1 for ln in text.splitlines() if ln and not ln.startswith("#"))
    return str(p), f"team news (starting11) -> {p.name}: {n} clubs, page updated {parsed.get('updated')}"
