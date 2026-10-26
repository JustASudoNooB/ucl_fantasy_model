"""Team news and predicted line-ups -> start probabilities, and a wider player pool.

Input is one text file per matchday (data/seed/news/md{N}_*.txt, or data/processed/news_md{N}.txt
which takes precedence), one line per club:

    TEAM | predicted XI (keeper first, lines separated by ';') | out | doubtful | suspended

How it is used
1. Every listed player's start probability is updated by Bayes' rule. A predicted starter
   multiplies the prior odds by `lr_start` (default 9.3 = 0.93 / 0.10: journalists' predicted
   XIs contain ~93% of actual starters, and ~10% of non-starters get wrongly named); a fit
   player left out multiplies them by `lr_bench` (0.078). Out or suspended -> 0. Doubtful
   players are capped (0.62 if still predicted to start, 0.15 if not).
2. Predicted starters who are missing from the price list are added to the pool, with
   price_status: 'verified' (a published price exists in data/seed/players_extra_2026_27.csv),
   'est_fpl' (Premier League club: UCL price estimated from the FPL price), or 'unknown'
   (simulated so that team-mates' shares stay right, but never picked by the optimiser).
The likelihood ratios are re-estimated from uploaded line-ups by the feedback loop over time.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ..config import ROOT, settings
from ..names import norm

SEED = ROOT / "data" / "seed"
PROC = ROOT / "data" / "processed"
NEWS_DIR = SEED / "news"
PL_CLUBS = {"ARS", "AVL", "LIV", "MCI", "MUN"}

TOP = {"BAY", "ARS", "LIV", "MCI", "PSG", "RMA", "BAR", "INT"}
THIRD = {"SHA", "AEK", "SLA", "SLO", "SAB", "VIK", "BOD", "LAS"}
TIER_PRICE = {0: {"GK": 5.0, "DEF": 5.0, "MID": 6.5, "FWD": 7.0},
              1: {"GK": 4.5, "DEF": 4.5, "MID": 5.5, "FWD": 6.0},
              2: {"GK": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 5.5}}

DEFAULTS = dict(lr_start=6.0, lr_bench=0.17, doubt_in=0.62, doubt_out=0.15, p_max=0.95, carry_forward=0.5,
                model="logistic", logit_a=-0.94, logit_b=0.5, logit_c=1.86)


def _logit(p: float) -> float:
    p = float(np.clip(p, 0.01, 0.99))
    return float(np.log(p / (1 - p)))


def updated_start(p0: float, named: bool, cfg: dict) -> float:
    """Start probability after a predicted (or previous) XI: calibrated logistic or Bayes rule.
    cfg['_k'] < 1 blends toward the prior in logit space (older, carried-forward news)."""
    if cfg.get("model", "logistic") == "logistic":
        z0 = _logit(p0)
        z1 = cfg["logit_a"] + cfg["logit_b"] * z0 + cfg["logit_c"] * (1.0 if named else 0.0)
        k = float(cfg.get("_k", 1.0))
        z = (1 - k) * z0 + k * z1
        return min(1 / (1 + np.exp(-z)), cfg["p_max"])
    p = float(np.clip(p0, 0.02, 0.98))
    odds = p / (1 - p) * (cfg["lr_start"] if named else cfg["lr_bench"])
    return min(odds / (1 + odds), cfg["p_max"])


def _cfg() -> dict:
    try:
        return {**DEFAULTS, **(settings().raw.get("news") or {})}
    except Exception:
        return dict(DEFAULTS)


# ------------------------------------------------------------------------------------- parsing
def news_path(md: int) -> Path | None:
    p = PROC / f"news_md{md}.txt"
    if p.exists():
        return p
    hits = sorted(NEWS_DIR.glob(f"md{md}_*.txt"))
    return hits[-1] if hits else None


def latest_news_before(md: int) -> tuple[int, Path] | None:
    """The most recent earlier matchday that has a news file (used, softened, for later matchdays)."""
    for m in range(md - 1, 0, -1):
        p = news_path(m)
        if p is not None:
            return m, p
    return None


def _names(cell: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,;]", cell or "") if x.strip()]


def _line_pos(i: int, n: int) -> str:
    if i == 0:
        return "GK"
    if i == 1:
        return "DEF"
    if i == n - 1:
        return "FWD"
    return "MID"


def parse(path: Path) -> pd.DataFrame:
    """One row per named player. A header line '# strength: 0.5' marks weaker evidence (e.g. the
    previous match's actual XI rather than a predicted XI for this match)."""
    rows = []
    strength = 1.0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*#\s*strength\s*:\s*([0-9.]+)", raw)
        if m:
            strength = float(m.group(1))
            continue
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in raw.split("|")]
        parts += [""] * (5 - len(parts))
        team, xi, out, doubt, susp = parts[:5]
        lines = [ln for ln in xi.split(";") if ln.strip()]
        k = 0
        for li, ln in enumerate(lines):
            for nm in _names(ln):
                rows.append(dict(team=team, name=nm, status="start", line_pos=_line_pos(li, len(lines)), order=k))
                k += 1
        for status, cell in (("out", out), ("doubt", doubt), ("susp", susp)):
            for nm in _names(cell):
                rows.append(dict(team=team, name=nm, status=status, line_pos="", order=-1))
    out = pd.DataFrame(rows)
    out.attrs["strength"] = strength
    return out


# ------------------------------------------------------------------------------------- matching
def _score(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    s = fuzz.token_set_ratio(a, b)
    if len(a.split()) == 1 and a not in b.split():      # a lone surname must appear as a token
        s = min(s, 70)
    return s


def best_in_team(name: str, cand: pd.DataFrame, cutoff: int = 86):
    """Best pool id for a name among one club's players. When no spelling is close enough, a
    surname that exactly one candidate shares still counts ("Tasos Douvikas" and "Anastasios
    Douvikas"); two players with that surname (the Haugens at Viking) means no guess."""
    best, best_s = None, 0
    for c in cand.itertuples():
        sc = _score(name, c.name)
        if sc > best_s:
            best, best_s = c.id, sc
    if best_s >= cutoff:
        return best, best_s
    last = norm(name).split()[-1:] or [""]
    if len(last[0]) >= 4:
        same = [c.id for c in cand.itertuples() if norm(str(c.name)).split()[-1:] == last]
        if len(same) == 1:
            return same[0], 88
    return None, best_s


def match(news: pd.DataFrame, pool: pd.DataFrame, cutoff: int = 86) -> pd.DataFrame:
    """Attach pool ids (keepers only to keepers; one pool player per news line)."""
    news = news.copy()
    ids = []
    for r in news.itertuples():
        cand = pool[pool.team == r.team]
        if r.status == "start":
            cand = cand[(cand.pos == "GK")] if r.line_pos == "GK" else cand[cand.pos != "GK"]
        best, _ = best_in_team(r.name, cand, cutoff)
        ids.append(best)
    news["id"] = ids
    return news


# ------------------------------------------------------------------------------------- pool expansion
def _fpl_now():
    from . import fpl as fpl_mod
    d = fpl_mod.load()
    now = d.players_now.copy()
    from ..names import ucl_team_id
    now["ucl_team"] = now["team_name"].map(ucl_team_id)
    now["fpl_price"] = now["now_cost"] / 10.0
    now["fpl_pos"] = now["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
    return now[now.ucl_team.isin(PL_CLUBS)]


def _fpl_lookup(now: pd.DataFrame, team: str, name: str):
    best, best_s = None, 0
    for c in now[now.ucl_team == team].itertuples():
        s = max(_score(name, c.full_name), _score(name, str(c.web_name)) - 2)
        if s > best_s:
            best, best_s = c, s
    return best if best_s >= 86 else None


def fpl_price_model(pool: pd.DataFrame, now: pd.DataFrame) -> dict:
    """UCL price ~ a + b * FPL price, per position group, fitted on players listed in both."""
    rows = []
    for r in pool[pool.team.isin(PL_CLUBS)].itertuples():
        c = _fpl_lookup(now, r.team, r.name)
        if c is not None:
            rows.append(dict(pos=r.pos, ucl=float(r.price), fpl=float(c.fpl_price)))
    df = pd.DataFrame(rows)
    out = {}
    for grp, poss in (("back", ("GK", "DEF")), ("front", ("MID", "FWD"))):
        g = df[df.pos.isin(poss)]
        b, a = np.polyfit(g.fpl, g.ucl, 1) if len(g) >= 4 else (0.75, 0.6)
        pred = a + b * g.fpl
        out[grp] = dict(a=float(a), b=float(b), n=int(len(g)), mae=float(np.mean(np.abs(pred - g.ucl))) if len(g) else np.nan)
    return out


def _round_half(x: float, lo: float, hi: float) -> float:
    return float(np.clip(np.round(x * 2) / 2, lo, hi))


def expand_pool(pool: pd.DataFrame, news: pd.DataFrame, extra: pd.DataFrame | None = None,
                as_of: str = "", source: str = "news") -> tuple[pd.DataFrame, dict]:
    """Rows for predicted starters missing from the pool (verified, est_fpl or unknown prices)."""
    extra = extra if extra is not None else pd.DataFrame(columns=["name", "team", "pos", "price"])
    now = _fpl_now()
    model = fpl_price_model(pool, now)
    lo = {"GK": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}
    starters = news[(news.status == "start") & news.id.isna()]
    rows = []
    for r in starters.itertuples():
        ex = extra[(extra.team == r.team)]
        hit = None
        for c in ex.itertuples():
            if _score(r.name, c.name) >= 86:
                hit = c
                break
        if hit is not None:
            rows.append(dict(name=hit.name, team=r.team, pos=hit.pos, price=float(hit.price), source=hit.source,
                             as_of=hit.as_of, note="", price_status="verified"))
            continue
        if r.team in PL_CLUBS:
            c = _fpl_lookup(now, r.team, r.name)
            if c is not None:
                pos = c.fpl_pos if r.line_pos != "GK" else "GK"
                m = model["back" if pos in ("GK", "DEF") else "front"]
                price = _round_half(m["a"] + m["b"] * c.fpl_price, lo[pos], 11.0)
                first = str(c.first_name).split()[0] if str(c.first_name).strip() else ""
                disp = r.name if len(r.name.split()) >= 2 or not first else f"{first} {r.name}"
                rows.append(dict(name=disp, team=r.team, pos=pos, price=price,
                                 source=f"FPL £{c.fpl_price:.1f}m mapped", as_of=as_of,
                                 note=f"UCL price estimated from FPL price (±{m['mae']:.1f}); check in the app",
                                 price_status="est_fpl"))
                continue
        rows.append(dict(name=r.name, team=r.team, pos=r.line_pos, price=np.nan, source=source, as_of=as_of,
                         note="not in any price list; projected for team-mates' shares only",
                         price_status="unknown"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["team", "name"])
        # unknown prices only shape the share prior (these players are never picked): a written
        # guess where one exists (data/seed/news/price_guesses.csv), else a starter's typical
        # price for the club's tier
        gp = NEWS_DIR / "price_guesses.csv"
        guesses = pd.read_csv(gp) if gp.exists() else pd.DataFrame(columns=["team", "name", "pos", "guess"])
        for i, r in out[out.price_status == "unknown"].iterrows():
            g = guesses[(guesses.team == r.team)]
            hit = next((c for c in g.itertuples() if _score(r["name"], c.name) >= 86), None)
            if hit is not None:
                out.at[i, "pos"] = hit.pos
                out.at[i, "price"] = float(hit.guess)
                out.at[i, "note"] = f"price unknown (guess €{hit.guess:.1f}m shapes the share prior only)"
                continue
            tier = 0 if r.team in TOP else (2 if r.team in THIRD else 1)
            out.at[i, "price"] = TIER_PRICE[tier][r.pos]
        sp = NEWS_DIR / "set_pieces.csv"
        if sp.exists():
            for c in pd.read_csv(sp).itertuples():
                for i, r in out[out.team == c.team].iterrows():
                    if _score(r["name"], c.name) >= 86 and c.role not in str(r["note"]):
                        out.at[i, "note"] = f"{c.role}; {r['note']}".strip("; ")
    return out, model


# ------------------------------------------------------------------------------------- applying
def apply(players: pd.DataFrame, news: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bayesian start-probability update. Returns (players, log of changes)."""
    cfg = {**_cfg(), **(cfg or {})}
    players = players.copy()
    teams = set(news.team)
    status: dict[str, set] = {}
    for r in news.dropna(subset=["id"]).itertuples():
        status.setdefault(r.id, set()).add(r.status)
    log = []
    for i, r in players.iterrows():
        if r.team not in teams:
            continue
        s = status.get(r.id, set())
        p0 = float(r.p_start)
        if ("out" in s or "susp" in s) and cfg.get("_carried"):
            q, sub = min(p0, 0.25), float(r.p_sub) * 0.5          # out last time: likely still out
        elif "out" in s or "susp" in s:
            q, sub = 0.0, 0.0
        else:
            q = updated_start(p0, "start" in s, cfg)
            sub = float(r.p_sub)
            if "doubt" in s:
                q = min(q, cfg["doubt_in"] if "start" in s else cfg["doubt_out"])
                sub *= 0.5
        players.at[i, "p_start"] = q
        players.at[i, "p_sub"] = sub
        if abs(q - p0) > 0.05:
            log.append(dict(id=r.id, name=r["name"], team=r.team, p_start_before=round(p0, 3), p_start_after=round(q, 3),
                            news="/".join(sorted(s)) or "not in predicted XI"))
    return players, pd.DataFrame(log)


def named_flags(news: pd.DataFrame, players: pd.DataFrame) -> dict[str, int]:
    """1 = named in the predicted XI, 0 = club has news but player not named (or out)."""
    teams = set(news.team)
    named = set(news[(news.status == "start")].dropna(subset=["id"]).id)
    return {r.id: int(r.id in named) for r in players[players.team.isin(teams)].itertuples()}


def lr_from_counts(started_named: int, started: int, bench_named: int, bench: int,
                   prior=(93.0, 7.0, 10.0, 90.0)) -> tuple[float, float, float, float]:
    """Beta-shrunk sensitivity and false-positive rate of predicted XIs -> likelihood ratios."""
    a1, b1, a0, b0 = prior
    sens = (a1 + started_named) / (a1 + b1 + started)
    fpr = (a0 + bench_named) / (a0 + b0 + bench)
    return sens / fpr, (1 - sens) / (1 - fpr), sens, fpr


def load_for(md: int, pool: pd.DataFrame) -> pd.DataFrame | None:
    p = news_path(md)
    if p is None:
        return None
    raw = parse(p)
    out = match(raw, pool)
    out.attrs["strength"] = raw.attrs.get("strength", 1.0)
    return out


def softened(cfg: dict, k: float) -> dict:
    """Older news counts for less: likelihood ratios raised to the power k (Bayes mode) or a
    logit-space blend toward the prior with weight k (logistic mode); doubts fade too."""
    c = dict(cfg)
    c["_k"] = k
    c["lr_start"] = cfg["lr_start"] ** k
    c["lr_bench"] = cfg["lr_bench"] ** k
    c["doubt_in"] = 1 - (1 - cfg["doubt_in"]) * k       # doubts fade too
    c["doubt_out"] = cfg["doubt_out"] + (1 - cfg["doubt_out"]) * (1 - k) * 0.5
    return c
