"""Team-news layer: RSS / Google News -> structured availability signals.

Each headline or summary is scanned for squad names; matched players get a status from
keyword rules (OUT / DOUBT / FIT / ROTATION) with a confidence that depends on the phrase and
the source. Signals are merged by recency and confidence into data/processed/availability.csv,
which the engine multiplies into start and sub probabilities. Nothing here is applied blindly:
`ufe news --review` prints the proposed changes first, and every row keeps its URL.

Optional: if an Ollama model is running locally (http://localhost:11434), `extract_llm` asks it
for {player, status, expected_return} JSON per article as a second opinion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from ..config import ROOT, sources, team_lookup
from ..names import norm, strip_accents
from .http import get

PROC = ROOT / "data" / "processed"

RULES = [  # (pattern, status, p_available, confidence)
    (r"\b(ruled out|out for|will miss|misses|sidelined|surgery|torn|fracture|ACL|suspended|banned|red card ban)\b", "OUT", 0.0, 0.85),
    (r"\b(doubt(ful)?|fitness test|late test|50-50|race against time|knock|questionable|not certain)\b", "DOUBT", 0.5, 0.6),
    (r"\b(back in training|returns?|available|passed fit|fit again|in contention|recovered)\b", "FIT", 0.95, 0.55),
    (r"\b(rested|rotation|rotate|start(s)? on the bench|benched|managed minutes)\b", "ROTATION", 0.6, 0.45),
]
SOURCE_WEIGHT = {"uefa.com": 1.0, "bbc": 0.95, "theguardian": 0.9, "espn": 0.85, "news.google": 0.75}


@dataclass
class Signal:
    player_id: str
    name: str
    team: str
    status: str
    p_available: float
    confidence: float
    title: str
    url: str
    published: str


def feeds_for(teams: list[str]) -> list[str]:
    cfg = sources()["news_rss"]["feeds"]
    out = [f for f in cfg if "{club}" not in f]
    tpl = next((f for f in cfg if "{club}" in f), None)
    if tpl:
        tl = team_lookup()
        out += [tpl.replace("{club}", quote_plus(tl[t]["name"])) for t in teams]
    return out


def _parse(url: str) -> list[dict]:
    import feedparser
    try:
        r = get(url, retries=1, min_gap=1.5)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    fp = feedparser.parse(r.content)
    items = []
    for e in fp.entries:
        items.append(dict(title=e.get("title", ""), summary=re.sub("<[^>]+>", " ", e.get("summary", "")),
                          url=e.get("link", url), published=e.get("published", "")))
    return items


def scan(players: pd.DataFrame, items: list[dict]) -> list[Signal]:
    surname = {}
    for r in players.itertuples():
        toks = norm(r.name).split()
        if not toks:
            continue
        key = toks[-1] if len(toks[-1]) > 3 else " ".join(toks[-2:])
        surname.setdefault(key, []).append(r)
    sigs = []
    for it in items:
        text = strip_accents(f"{it['title']}. {it['summary']}")
        low = norm(text)
        for key, cands in surname.items():
            if f" {key} " not in f" {low} ":
                continue
            for c in cands:
                tl = team_lookup().get(c.team, {})
                club_hint = any(norm(x) in low for x in [tl.get("name", ""), tl.get("short", "")] if x)
                for pat, status, pav, conf in RULES:
                    if re.search(pat, text, flags=re.I):
                        src_w = next((w for k, w in SOURCE_WEIGHT.items() if k in it["url"]), 0.7)
                        sigs.append(Signal(c.id, c.name, c.team, status, pav, conf * src_w * (1.0 if club_hint else 0.6),
                                           it["title"][:160], it["url"], it["published"]))
                        break
    return sigs


def merge(signals: list[Signal]) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame(columns=["id", "name", "team", "status", "p_available", "confidence", "title", "url", "published"])
    df = pd.DataFrame([s.__dict__ for s in signals]).rename(columns={"player_id": "id"})
    df = df.sort_values("confidence", ascending=False).drop_duplicates("id")
    # shrink toward available according to confidence
    df["p_available"] = df.confidence * df.p_available + (1 - df.confidence) * 0.95
    df["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df


def run(players: pd.DataFrame, teams: list[str], write: bool = False) -> pd.DataFrame:
    items = []
    for f in feeds_for(teams):
        items += _parse(f)
    out = merge(scan(players, items))
    if write and not out.empty:
        PROC.mkdir(parents=True, exist_ok=True)
        p = PROC / "availability.csv"
        if p.exists():
            prev = pd.read_csv(p)
            out = pd.concat([prev[~prev.id.isin(out.id)], out], ignore_index=True)
        out.to_csv(p, index=False)
    return out


def extract_llm(text: str, model: str = "llama3.1:8b") -> list[dict]:
    """Second-opinion extraction with a local Ollama model (free, offline). Returns [] if unavailable."""
    import httpx
    prompt = ("Extract football team news as JSON list of objects with keys player, club, status "
              "(OUT|DOUBT|FIT|ROTATION), expected_return. Only facts stated in the text.\n\nTEXT:\n" + text[:4000])
    try:
        r = httpx.post("http://localhost:11434/api/generate", json=dict(model=model, prompt=prompt, stream=False,
                                                                       format="json"), timeout=60)
        data = json.loads(r.json().get("response", "[]"))
        return data if isinstance(data, list) else data.get("items", [])
    except Exception:
        return []
