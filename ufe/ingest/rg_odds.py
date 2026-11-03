"""Bookmaker prices without a key: rg.org's Champions League matchday odds articles -> odds_md{N}.csv.

rg.org publishes one article per matchday with a price table for every game (1X2 and a goals
line, several bookmakers): a US edition in American odds (DraftKings, FanDuel, BetRivers, ...) and
a Canadian one in decimal odds (1xBet, Tonybet, 22Bet, ...). The article is found from the soccer
news index (a link containing "champions-league" and "odds"), every table is matched to a fixture
of the matchday by its heading ("Galatasaray Vs Stuttgart Odds"), and for each game the median
price across all bookmakers of both editions is kept. The goals line is whichever line most
bookmakers quote (3.5 is common this season; integer lines are Asian, a push refunds).
"""
from __future__ import annotations

import html as H
import re
from statistics import median

import pandas as pd

from ..config import ROOT
from ..names import ucl_team_id
from .http import get

BASE = "https://rg.org"
INDEXES = ["/news/soccer", "/en-ca/news/soccer"]
PROC = ROOT / "data" / "processed"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/129.0 Safari/537.36"}


def to_decimal(x: str) -> float | None:
    x = x.strip().replace("−", "-")
    if not x or x == "-":
        return None
    try:
        if x[0] in "+-" and abs(float(x)) >= 100:            # American
            v = float(x)
            return 1 + v / 100 if v > 0 else 1 + 100 / abs(v)
        v = float(x)
        return v if v > 1 else None
    except ValueError:
        return None


def _club(name: str) -> str | None:
    name = H.unescape(name).strip()
    return ucl_team_id(name) or ucl_team_id(re.sub(r"\s+(FA|FC|SK|CF)$", "", name))


def _summary_table(tbl: str) -> list[dict]:
    """A table with a "Match (Home | Draw | Away)" column and one "home | draw | away" cell per book."""
    txt = lambda c: H.unescape(re.sub(r"<[^>]+>", "", c)).strip()
    rows = re.findall(r"<tr>(.*?)</tr>", tbl, flags=re.S)
    if not rows:
        return []
    books = [txt(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", rows[0], flags=re.S)][1:]
    out = []
    for r in rows[1:]:
        cells = [txt(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S)]
        if not cells:
            continue
        m = re.match(r"(.+?)\s+[Vv][Ss]\.?\s+(.+?)\s*$", cells[0])
        home, away = (_club(m.group(1)), _club(m.group(2))) if m else (None, None)
        if not home or not away:
            continue
        quotes = []
        for b, v in zip(books, cells[1:]):
            parts = [to_decimal(x.strip()) for x in v.split("|")]
            if len(parts) == 3 and all(x is not None for x in parts):
                quotes.append(dict(book=b, home=parts[0], draw=parts[1], away=parts[2]))
        if quotes:
            out.append(dict(home=home, away=away, quotes=quotes))
    return out


def parse(html: str) -> list[dict]:
    """[{home, away, quotes: [{book, home, draw, away, line, over, under}]}] from one article."""
    cut = html.find("self.__next_f")
    s = html[:cut] if cut > 0 else html
    out = []
    for t in re.finditer(r"<table>(.*?)</table>", s, flags=re.S):
        th0 = re.search(r"<th[^>]*>(.*?)</th>", t.group(1), flags=re.S)
        if th0 and re.sub(r"<[^>]+>", "", th0.group(1)).strip().lower().startswith("match (home"):
            out += _summary_table(t.group(1))           # MD2 style: one row per match, "home | draw | away" per book
            continue
        heads = re.findall(r"<h[23][^>]*>(.*?)</h[23]>", s[max(0, t.start() - 4000):t.start()], flags=re.S)
        if not heads:
            continue
        title = H.unescape(re.sub(r"<[^>]+>", "", heads[-1])).strip()
        m = re.match(r"(.+?)\s+[Vv][Ss]\.?\s+(.+?)\s+Odds", title)
        if not m:
            # older articles (MD1-2): the heading is just "Home vs Away", and the table's first header
            # cell repeats it with the kick-off time in brackets
            th0 = re.search(r"<th[^>]*>(.*?)</th>", t.group(1), flags=re.S)
            for cand in (title, H.unescape(re.sub(r"<[^>]+>", "", th0.group(1))).strip() if th0 else ""):
                m = re.match(r"(.+?)\s+[Vv][Ss]\.?\s+(.+?)\s*(?:\(.*\))?\s*$", cand)
                if m and _club(m.group(1)) and _club(m.group(2)):
                    break
                m = None
        if not m:
            continue
        home, away = _club(m.group(1)), _club(m.group(2))
        if not home or not away:
            continue
        rows = re.findall(r"<tr>(.*?)</tr>", t.group(1), flags=re.S)
        books = [H.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<th[^>]*>(.*?)</th>", rows[0], flags=re.S)][1:]
        q = {b: dict(book=b) for b in books}
        lines: dict[str, dict] = {b: {} for b in books}
        for r in rows[1:]:
            cells = [H.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S)]
            if not cells:
                continue
            label, vals = cells[0], cells[1:]
            lm = re.match(r"(Over|Under)\s+([0-9.]+)\s+goals", label, flags=re.I)
            for b, v in zip(books, vals):
                d = to_decimal(v)
                if d is None:
                    continue
                if lm:
                    lines[b].setdefault(float(lm.group(2)), {})[lm.group(1).lower()] = d
                elif label.endswith(": Draw") or label.strip().lower() == "draw":
                    q[b]["draw"] = d
                elif _club(label.split(":", 1)[-1]) == home:
                    q[b]["home"] = d
                elif _club(label.split(":", 1)[-1]) == away:
                    q[b]["away"] = d
        for b in books:
            for ln, ou in lines[b].items():
                if "over" in ou and "under" in ou:
                    q[b].setdefault("totals", {})[ln] = (ou["over"], ou["under"])
        out.append(dict(home=home, away=away, quotes=list(q.values())))
    return out


def combine(games: list[dict], fixtures: pd.DataFrame, md: int, source: str) -> pd.DataFrame:
    fx = fixtures[fixtures.md == md]
    key = {(r.home, r.away): r.fixture_id for r in fx.itertuples()}
    by_fix: dict[str, list] = {}
    for g in games:
        fid = key.get((g["home"], g["away"]))
        if fid:
            by_fix.setdefault(fid, []).extend(g["quotes"])
    rows = []
    for fid, qs in by_fix.items():
        full = [q for q in qs if all(k in q for k in ("home", "draw", "away"))]
        if not full:
            continue
        row = dict(fixture_id=fid, home_odds=round(median(q["home"] for q in full), 3),
                   draw_odds=round(median(q["draw"] for q in full), 3), away_odds=round(median(q["away"] for q in full), 3),
                   n_books=len(full))
        tot: dict[float, list] = {}
        for q in qs:
            for ln, ou in (q.get("totals") or {}).items():
                tot.setdefault(ln, []).append(ou)
        if tot:
            # most-quoted line; a half line wins a tie (no push)
            ln = max(tot, key=lambda x: (len(tot[x]), x % 1 == 0.5))
            row.update(total_line=ln, over_odds=round(median(o for o, _ in tot[ln]), 3),
                       under_odds=round(median(u for _, u in tot[ln]), 3), n_books_total=len(tot[ln]))
            if ln == 2.5:
                row.update(over25_odds=row["over_odds"], under25_odds=row["under_odds"])
        row["source"] = source
        rows.append(row)
    cols = ["fixture_id", "home_odds", "draw_odds", "away_odds", "over25_odds", "under25_odds", "total_line",
            "over_odds", "under_odds", "n_books", "n_books_total", "source"]
    return pd.DataFrame(rows).reindex(columns=cols)


def discover() -> list[str]:
    urls = []
    for ix in INDEXES:
        try:
            r = get(BASE + ix, headers=UA, retries=1)
        except Exception:  # noqa: BLE001
            continue
        if r.status_code == 200:
            urls += [BASE + u if u.startswith("/") else u
                     for u in re.findall(r'href="([^"]*champions-league[^"]*odds[^"]*)"', r.text)]
    return list(dict.fromkeys(urls))


def sync(md: int, fixtures: pd.DataFrame, pages: dict[str, str] | None = None, min_games: int = 9) -> tuple[str, str]:
    """Write data/processed/odds_md{md}.csv when the articles cover at least `min_games` of the matchday."""
    if pages is None:
        pages = {}
        for u in discover()[:6]:
            try:
                r = get(u, headers=UA, retries=1)
                if r.status_code == 200:
                    pages[u] = r.text
            except Exception:  # noqa: BLE001
                continue
    games, used = [], []
    for u, html in pages.items():
        g = parse(html)
        fx = fixtures[fixtures.md == md]
        hits = sum((x["home"], x["away"]) in set(zip(fx.home, fx.away)) for x in g)
        if hits >= 3:
            games += g
            used.append(u)
    df = combine(games, fixtures, md, source="rg.org: " + ", ".join(u.replace(BASE, "") for u in used))
    if len(df) < min_games:
        return "", f"odds (rg.org): {len(df)} MD{md} games found, not written"
    PROC.mkdir(parents=True, exist_ok=True)
    p = PROC / f"odds_md{md}.csv"
    df.to_csv(p, index=False)
    return str(p), (f"odds (rg.org) -> {p.name}: {len(df)} games, goals line for {int(df.total_line.notna().sum())}, "
                    f"median {int(df.n_books.median())} bookmakers")
