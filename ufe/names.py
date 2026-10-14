"""Name normalisation and matching across data sources (teams and players)."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from rapidfuzz import fuzz, process

from .config import teams

_CLUB_TOKENS = {
    "fc", "cf", "ac", "sc", "sk", "fk", "afc", "as", "ssc", "kv", "cp", "bc", "sl", "pae", "sfp",
    "rsc", "ogc", "rc", "club", "de", "the", "calcio", "sad", "cd", "ud", "sv", "ss", "osc", "losc",
    "balompie", "1907", "1899", "1909", "04", "05", "vfl", "vfb", "tsg", "bsc", "if", "bk", "ik",
}


def strip_accents(s: str) -> str:
    s = s.replace("ø", "o").replace("Ø", "O").replace("ß", "ss").replace("ł", "l").replace("Ł", "L")
    s = s.replace("ı", "i").replace("İ", "I").replace("đ", "d").replace("Đ", "D").replace("æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s: str) -> str:
    s = strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def club_key(name: str) -> str:
    toks = [t for t in norm(name).split() if t not in _CLUB_TOKENS]
    return " ".join(toks) or norm(name)


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for t in teams():
        for n in [t["name"], t["short"], *t.get("aliases", [])]:
            idx[club_key(n)] = t["id"]
            idx[norm(n)] = t["id"]
    return idx


def ucl_team_id(name: str) -> str | None:
    """Map any source's club name to one of the 36 league-phase team ids (or None)."""
    idx = _alias_index()
    for key in (norm(name), club_key(name)):
        if key in idx:
            return idx[key]
    return None


def player_key(name: str) -> str:
    return norm(name)


def best_player_match(name: str, candidates: dict[str, str], cutoff: int = 86) -> str | None:
    """candidates: {display_name: id}. Token-set fuzzy match on accent-stripped names."""
    if not candidates:
        return None
    choices = {norm(k): v for k, v in candidates.items()}
    hit = process.extractOne(norm(name), list(choices), scorer=fuzz.token_set_ratio, score_cutoff=cutoff)
    if hit is None:
        # surname-only fallback (e.g. "Raya" vs "David Raya Martín")
        last = norm(name).split()[-1] if norm(name) else ""
        hits = [k for k in choices if last and last in k.split()]
        return choices[hits[0]] if len(hits) == 1 else None
    return choices[hit[0]]
