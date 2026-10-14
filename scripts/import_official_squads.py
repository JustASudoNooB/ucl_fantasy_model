"""Import the full UCL Fantasy price list (every club's squad page) and reconcile it with the pool.

    python scripts/import_official_squads.py

Reads data/raw/uclfantasy/teams_*.txt (club pages: name | position | status | price),
data/raw/uclfantasy/rankings_p1_*.txt (ownership, MD points, per-90 rates for the top 100 by price)
and data/raw/uclfantasy/ownership_*.csv (ownership for further players), matches every row to the
existing pool by club + name, and writes data/seed/official_squads_2026_27.csv.

It also prints a price audit: every pool price that disagrees with the official list.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ufe.names import norm  # noqa: E402
from ufe.pipeline import SEED, legacy_pool  # noqa: E402

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "uclfantasy"
OUT = SEED / "official_squads_2026_27.csv"

SITE_TEAM = {  # club names used on uclfantasy.com rankings -> engine codes
    "AEK Athens": "AEK", "Arsenal": "ARS", "Aston Villa": "AVL", "Atleti": "ATM", "B. Dortmund": "BVB",
    "Barcelona": "BAR", "Bayern München": "BAY", "Club Brugge": "CLB", "Como": "COM", "Bodø/Glimt": "BOD",
    "Fenerbahçe": "FEN", "Feyenoord": "FEY", "Galatasaray": "GAL", "Inter": "INT", "LASK": "LAS",
    "Leipzig": "RBL", "Lens": "RCL", "Lille": "LIL", "Liverpool": "LIV", "Man City": "MCI", "Man Utd": "MUN",
    "Napoli": "NAP", "Paris": "PSG", "Porto": "POR", "PSV": "PSV", "Real Betis": "BET", "Real Madrid": "RMA",
    "Roma": "ROM", "Shakhtar": "SHA", "S. Bratislava": "SLO", "Sabah": "SAB", "Slavia Praha": "SLA",
    "Sporting CP": "SCP", "Stuttgart": "VFB", "Viking": "VIK", "Villarreal": "VIL",
}


def parse_team_pages() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob("teams_*.txt")):
        team = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("## "):
                team = line[3:].split()[0]
                continue
            if not line or line.startswith("#") or team is None:
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 4:
                continue
            name, pos, status, price = parts[:4]
            m = re.search(r"(\d+(?:\.\d+)?)", price)
            rows.append(dict(team=team, name_official=name, pos=pos, status=status,
                             price=float(m.group(1)) if m else float("nan"), source=f"uclfantasy.com/teams ({path.stem[6:16]})"))
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["team", "name_official"])


def parse_rankings() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob("rankings_p*.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.startswith("#"):
                continue
            p = [x.strip() for x in raw.split("|")]
            if len(p) < 14:
                continue
            f = lambda x: float(x) if x not in ("", "—", "-") else float("nan")  # noqa: E731
            rows.append(dict(team=SITE_TEAM.get(p[2], p[2]), name_official=p[1], pos=p[3], price_rank=f(p[4]),
                             own_pct=f(p[5]), mins_sample=f(p[6]), start_pct=f(p[7]), pts_to_date=f(p[8]),
                             sot90=f(p[9]), kp90=f(p[10]), g90=f(p[11]), a90=f(p[12]), gi90=f(p[13])))
    return pd.DataFrame(rows)


def parse_ownership() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(RAW.glob("ownership_*.csv"))]
    if not frames:
        return pd.DataFrame(columns=["team", "name_official", "own_pct"])
    o = pd.concat(frames, ignore_index=True)
    o["team"] = o.team.map(lambda t: SITE_TEAM.get(t, t))
    return o.drop_duplicates(subset=["team", "name_official"], keep="last")


def _tokens(s: str) -> list[str]:
    return [t for t in norm(s).split() if t]


def name_score(official: str, full: str) -> float:
    """Similarity between an abbreviated game name ('J. Timber', 'Gabriel') and a full name."""
    o, f = _tokens(official), _tokens(full)
    if not o or not f:
        return 0.0
    # compact forms catch N'Dicka / Ndicka, Min-jae / Minjae, St. Juste / Juste
    oc = "".join(o[1:] if len(o[0]) == 1 and len(o) > 1 else o)
    ini_ok = len(o[0]) != 1 or f[0].startswith(o[0]) or len(f) == 1
    if len(oc) >= 4 and ini_ok and (oc == "".join(f) or oc == "".join(f[1:]) or "".join(f).endswith(oc)):
        return 95.0
    if len(oc) >= 6 and len(o[0]) > 1 and sorted("".join(o)) == sorted("".join(f)):
        return 96.0                      # same letters, different order: 'Minjae Kim' / 'Kim Min-jae'
    if len(oc) >= 6 and (sorted(o[1:] if len(o[0]) == 1 else o) == sorted(f)
                         or fuzz.ratio(oc, "".join(f[1:]) or "".join(f)) >= 90):
        return 92.0
    initial = o[0] if len(o[0]) == 1 else None
    core = o[1:] if initial else o
    core_s, full_s = " ".join(core), " ".join(f)
    if initial:
        # surname must match the end (or any part) of the full name, initial the first name
        if core == f[-len(core):] or set(core) <= set(f):
            return 100.0 if f[0].startswith(initial) else 90.0
        return fuzz.token_set_ratio(core_s, full_s) * (1.0 if f[0].startswith(initial) else 0.85)
    if len(core) == 1:
        return 98.0 if core[0] in f else fuzz.partial_ratio(core_s, full_s) * 0.9
    return float(fuzz.token_set_ratio(core_s, full_s))


def strict_same_person(official: str, full: str) -> bool:
    """Cross-club test: initial + exact surname (last tokens), or identical full names."""
    o, f = _tokens(official), _tokens(full)
    if len(o) < 2 or len(f) < 2:
        return False
    if len(o[0]) == 1:
        core = o[1:]
        return f[-len(core):] == core and f[0].startswith(o[0])
    return o == f


def reconcile(official: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Match each official row to a pool player: same club first, then (for transfers the article
    lists missed) any club, which moves the player to his new club."""
    out = []
    used: set[str] = set()
    pending = []
    for team, g in official.groupby("team", sort=False):
        cand = pool[pool.team == team]
        for r in g.itertuples():
            best, best_s = None, 0.0
            for c in cand.itertuples():
                if c.id in used:
                    continue
                s = name_score(r.name_official, c.name) + (3 if c.pos == r.pos else 0)
                if s > best_s:
                    best, best_s = c, s
            row = r._asdict()
            row.pop("Index", None)
            if best is not None and best_s >= 86:
                used.add(best.id)
                row.update(id=best.id, name=best.name, pool_price=best.price, pool_status=best.price_status,
                           pool_pos=best.pos, pool_team=best.team, match_score=round(best_s, 1))
                out.append(row)
            else:
                pending.append(row)
    # second pass: players who changed club since the articles were written
    priced_teams = set(official.loc[official.price.notna(), "team"])
    rest = pool[~pool.id.isin(used) & pool.team.isin(priced_teams)]
    for row in pending:
        best, best_s = None, 0.0
        for c in rest.itertuples():
            if c.id in used or c.team == row["team"] or not strict_same_person(row["name_official"], c.name):
                continue
            s = name_score(row["name_official"], c.name)
            if s > best_s:
                best, best_s = c, s
        if best is not None:
            used.add(best.id)
            from ufe.models.player_rates import slug
            row.update(id=f"{row['team']}-{slug(best.name)}", name=best.name, pool_price=best.price,
                       pool_status=best.price_status, pool_pos=best.pos, pool_team=best.team,
                       match_score=round(best_s, 1), moved_from=best.id)
        else:
            row.update(id=None, name=row["name_official"], pool_price=float("nan"), pool_status="", pool_pos="",
                       pool_team="", match_score=round(best_s, 1))
        out.append(row)
    return pd.DataFrame(out)


def expand_names(rec: pd.DataFrame) -> list[str]:
    """'J. Musiala' -> 'Jamal Musiala' when FBref or FPL has exactly one such player at that club."""
    full: dict[str, list[str]] = {}
    fbp = SEED / "fbref_big5.csv"
    if fbp.exists():
        fb = pd.read_csv(fbp, usecols=["player", "ucl", "season"])
        for r in fb[fb.ucl.notna() & (fb.season >= 2025)].drop_duplicates(["player", "ucl"]).itertuples():
            full.setdefault(r.ucl, []).append(r.player)
    fplp = Path(__file__).resolve().parents[1] / "data" / "raw" / "fpl" / "fpl_2026-27_players_raw.csv"
    if fplp.exists():
        from ufe.names import ucl_team_id
        fp = pd.read_csv(fplp, usecols=["first_name", "second_name", "team"])
        tm = pd.read_csv(fplp.with_name("fpl_2026-27_teams.csv"), usecols=["id", "name"])
        tid = {r.id: ucl_team_id(r.name) for r in tm.itertuples()}
        for r in fp.itertuples():
            t = tid.get(r.team)
            if t:
                full.setdefault(t, []).append(f"{r.first_name} {r.second_name}")
    out = []
    for r in rec.itertuples():
        nm = r.name
        if r.pool_status == "" and isinstance(nm, str) and re.match(r"^[A-ZÀ-Ž]\. ", nm):
            hits = {c for c in full.get(r.team, []) if strict_same_person(nm, c)}
            if len(hits) == 1:
                nm = hits.pop()
        out.append(nm)
    return out


def main() -> None:
    off = parse_team_pages()
    rk = parse_rankings()
    own = parse_ownership()
    if not rk.empty:   # players on the rankings page whose club page was cut short
        k_off = set(off.team + "|" + off.name_official.map(norm))
        add = rk[~(rk.team + "|" + rk.name_official.map(norm)).isin(k_off)]
        add = add.assign(price=add.price_rank, status="Available", source="uclfantasy.com/rankings")
        off = pd.concat([off, add[["team", "name_official", "pos", "status", "price", "source"]]], ignore_index=True)
    if not own.empty:
        k_off = set(off.team + "|" + off.name_official.map(norm))
        add = own[~(own.team + "|" + own.name_official.map(norm)).isin(k_off)]
        add = add.assign(status="Available", source="uclfantasy.com/transfers")
        off = pd.concat([off, add[["team", "name_official", "pos", "status", "price", "source"]]], ignore_index=True)
    pool = legacy_pool()
    rec = reconcile(off, pool)
    # attach ranking stats and ownership by (team, abbreviated name)
    key = lambda d: d.team + "|" + d.name_official.map(norm)  # noqa: E731
    for extra, cols in ((rk, ["own_pct", "mins_sample", "start_pct", "pts_to_date", "sot90", "kp90", "g90", "a90"]),
                        (own, ["own_pct"])):
        if extra.empty:
            continue
        m = dict(zip(key(extra), extra[cols].to_dict("records")))
        for c in cols:
            vals = [m.get(k, {}).get(c) for k in key(rec)]
            if c in rec:
                rec[c] = [v if v is not None and pd.notna(v) else old for v, old in zip(vals, rec[c])]
            else:
                rec[c] = vals
    rec["name"] = expand_names(rec)
    from ufe.models.player_rates import slug
    rec["id"] = [i if isinstance(i, str) else f"{t}-{slug(n)}" for i, t, n in zip(rec.id, rec.team, rec.name)]
    rec["as_of"] = "2026-10-14"
    cols = ["id", "name", "name_official", "team", "pos", "price", "status", "own_pct", "pts_to_date", "mins_sample",
            "start_pct", "sot90", "kp90", "g90", "a90", "pool_price", "pool_status", "pool_pos", "pool_team", "moved_from",
            "match_score", "source", "as_of"]
    rec = rec[[c for c in cols if c in rec]]
    rec.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(rec)} players, {rec.price.notna().sum()} with prices, "
          f"{rec.pool_status.ne('').sum()} matched to the existing pool")
    # ---- audit
    m = rec[rec.pool_status.ne("") & rec.price.notna()]
    diff = m[(m.price - m.pool_price).abs() > 0.01]
    print("\nPrice audit (matched players):")
    print(m.groupby("pool_status").apply(lambda g: f"{len(g)} matched, {(abs(g.price - g.pool_price) > .01).sum()} wrong").to_string())
    print(diff.sort_values("price", ascending=False)[["team", "name", "pos", "pool_pos", "pool_price", "price", "pool_status"]].to_string(index=False))
    pos_diff = m[m.pos != m.pool_pos]
    print(f"\nPosition differences: {len(pos_diff)}")
    print(pos_diff[["team", "name", "pool_pos", "pos"]].to_string(index=False))
    mv = rec[rec.get("moved_from").notna()] if "moved_from" in rec else rec.iloc[:0]
    print(f"\nMoved club since the articles: {len(mv)}")
    print(mv[["name", "pool_team", "team", "pool_price", "price", "match_score"]].to_string(index=False))
    low = rec[rec.pool_status.ne("") & (rec.match_score < 97)]
    print(f"\nLower-confidence matches: {len(low)}")
    print(low[["team", "name_official", "name", "pos", "pool_pos", "match_score"]].to_string(index=False))
    un = rec[rec.pool_status.eq("") & (rec.price >= 5.5)]
    print(f"\nNew to the pool (price >= 5.5): {len(un)}")
    print(un[["team", "name_official", "pos", "price", "match_score"]].to_string(index=False))
    left = pool[~pool.id.isin(rec.id) & (pool.team != "SLO")]
    print(f"\nPool players not on any club page: {len(left)}")
    print(left[["team", "name", "pos", "price", "price_status"]].to_string(index=False, max_rows=300))


if __name__ == "__main__":
    main()
