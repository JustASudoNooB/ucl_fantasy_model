"""Per-player history from Opta event data: what each player actually did, season by season.

Sources
  FPL/Opta   Premier League 2025/26 (every match; xG includes penalties, so designated takers
             have an estimated penalty xG removed).
  FBref/Opta big-five leagues 2023/24, 2024/25 and the first rounds of 2025/26
             (ufe/ingest/fbref.py; non-penalty xG, xAG, starts, recoveries, cards, penalties).

How seasons combine (all constants come from scripts/validate_player_layer.py, which fits on
one season and scores the next):
  * a season k years older than the most recent counts DECAY**k (0.5: the season two back
    carried half the weight of the season one back in the big-five test);
  * shares of club xG / xA are ratio estimates: sum(w * player xG) / sum(w * club xG per match
    * player 90s), so a player's share is measured against the club he played for that season;
  * the blend with the price prior is w = n / (n + k) in log space, n = decayed 90s played.
    k is 12 (xG) and 20 (xA) when most of the history is at the current club, 40 for players
    who changed club (their old share says much less: 0-15% of the error removed vs 20-30%).
  * start rates use only seasons at the current club, weight n / (n + 20) in team matches;
  * recoveries are converted to the FPL/Opta scale (FBref counts more: x0.77 defenders,
    x0.85 midfielders, x0.59 forwards; keepers are not comparable and keep the prior), k = 12.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ..ingest import fbref as fb_mod
from ..names import best_player_match, norm

DECAY = 0.5
# current-season domestic form (scripts/backtest_player_form.py, Premier League 2025/26 walk-forward):
#   per 90, this season's xG counts as much as last season's (W_CUR_XG = 1: goals log loss -1.1%),
#   actual assists count half (W_CUR_XA = 0.5; only assists, not xA, are available for 2026/27);
#   for start rates this season dominates (log loss 0.668 -> 0.566): last season's rate at the club
#   counts like W_PREV_START matches and the blend with the price prior uses K_START_CUR, not K_START.
CURRENT_SEASON = 2027
W_CUR_XG = 1.0
W_CUR_XA = 0.5
W_PREV_START = 3.0
K_START_CUR = 4.0
K_XG = {"same": 12.0, "moved": 40.0}
K_XA = {"same": 20.0, "moved": 40.0}
K_START = 20.0
FIT_UNUSED = 0.35   # share of the matches a player missed entirely that he was fit for (unused sub);
                    # the rest were injuries/suspensions. Start rates are per match he was fit for.
K_REC = 12.0
K_YC = 12.0
REC_FB_TO_FPL = {"GK": None, "DEF": 0.77, "MID": 0.85, "FWD": 0.59}
PEN_XG = 0.76                 # xG of a penalty
PENS_PER_TEAM_MATCH = 0.11
LATEST = {"fpl": 2026, "fbref": 2026}          # season end year that counts as "one season back"
SEASON_LABEL = {2024: "23/24", 2025: "24/25", 2026: "25/26"}
POS_OK = {"GK": {"GK"}, "DEF": {"DEF", "MID"}, "MID": {"MID", "DEF", "FWD"}, "FWD": {"FWD", "MID"}}


def _key(df: pd.DataFrame) -> pd.Series:
    return df.player.astype(str) + "|" + df.born.astype(str)


def match_fbref(pool: pd.DataFrame, fb: pd.DataFrame) -> dict[str, str]:
    """pool id -> FBref player key. Same club: fuzzy (surname fallback). Other clubs: an exact
    full name, or a surname that is unique across the whole archive, with a compatible role."""
    fb = fb.copy()
    fb["key"] = _key(fb)
    fb["p"] = fb.pos.map(fb_mod.fbref_pos)
    recent = fb[fb.season >= 2025]
    by_club = {t: g.drop_duplicates("key") for t, g in fb[fb.ucl != ""].groupby("ucl")}
    full = {}
    for r in recent.drop_duplicates("key").itertuples():
        full.setdefault(norm(r.player), set()).add(r.key)
    last_tok = {}
    for r in fb.drop_duplicates("key").itertuples():
        toks = norm(r.player).split()
        if toks:
            last_tok.setdefault(toks[-1], set()).add(r.key)
    pos_of = fb.drop_duplicates("key", keep="last").set_index("key").p
    out: dict[str, str] = {}
    for r in pool.itertuples():
        nm = norm(r.name)
        if not nm:
            continue
        g = by_club.get(r.team)
        if g is not None and len(g):
            hit = best_player_match(r.name, dict(zip(g.player, g.key)), cutoff=88)
            if hit is not None and pos_of.get(hit, r.pos) in POS_OK[r.pos]:
                out[r.id] = hit
                continue
        toks = nm.split()
        cand = set()
        if len(toks) >= 2:
            cand = full.get(nm, set())
            if not cand:
                best, score = None, 0
                for name_n, keys in full.items():
                    s = fuzz.token_sort_ratio(nm, name_n)
                    if s > score:
                        best, score = keys, s
                if score >= 95:
                    cand = best
        else:
            cand = last_tok.get(toks[0], set())
        if len(cand) == 1:
            k = next(iter(cand))
            if pos_of.get(k, r.pos) in POS_OK[r.pos]:
                out[r.id] = k
    return out


def load_current_form() -> pd.DataFrame:
    """This season's domestic-league numbers per player (minutes, starts, goals, assists, xG).
    data/processed/domestic_form.csv (synced on your laptop) replaces the bundled seed file
    club by club."""
    from ..config import ROOT
    parts = []
    seed = ROOT / "data" / "seed" / "form" / "domestic_2026_27.csv"
    proc = ROOT / "data" / "processed" / "domestic_form.csv"
    if seed.exists():
        parts.append(pd.read_csv(seed))
    if proc.exists():
        p = pd.read_csv(proc)
        if parts:
            parts[0] = parts[0][~parts[0].team.isin(set(p.team))]
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    f = pd.concat(parts, ignore_index=True)
    for c in ("team_matches", "m", "minutes", "starts", "goals", "assists", "pk", "xg"):
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
    f["xg_np"] = (f.xg - PEN_XG * f.pk).clip(lower=0.0)
    tot = f.groupby("team").agg(team_xg_np=("xg_np", "sum"), team_a=("assists", "sum"), tm=("team_matches", "max"))
    f = f.join(tot, on="team")
    f["team_xg_pm"] = f.team_xg_np / f.tm.clip(lower=1)
    f["team_a_pm"] = f.team_a / f.tm.clip(lower=1)
    return f


def match_current(pool: pd.DataFrame, form: pd.DataFrame) -> dict[str, int]:
    """pool id -> row index in the current-form table (same club, fuzzy name)."""
    out: dict[str, int] = {}
    if form is None or form.empty:
        return out
    by_team = {t: g for t, g in form.groupby("team")}
    for r in pool.itertuples():
        g = by_team.get(r.team)
        if g is None:
            continue
        hit = best_player_match(r.name, {n: str(i) for i, n in zip(g.index, g.player)}, cutoff=86)
        if hit is not None:
            out[r.id] = int(hit)
    return out


def observations(pool: pd.DataFrame, fpl_codes: dict, fpl_agg: pd.DataFrame | None,
                 fpl_hist: pd.DataFrame | None, fb: pd.DataFrame | None, use_current: bool = True) -> pd.DataFrame:
    """One row per pool player with decayed history summaries (NaN where nothing is known)."""
    rows = []
    form = load_current_form() if use_current else pd.DataFrame()
    cur = match_current(pool, form)
    fbm = match_fbref(pool, fb) if fb is not None else {}
    fbs = fb_mod.season_rows(fb) if fb is not None else None
    if fbs is not None:
        fbs["key"] = _key(fbs)
        fb_by_key = {k: g for k, g in fbs.groupby("key")}
    else:
        fb_by_key = {}
    agg = fpl_agg.set_index("code") if fpl_agg is not None else None
    pen_codes = set()
    if fpl_hist is not None and "penalties_order" in fpl_hist:
        pen_codes = set(fpl_hist.loc[pd.to_numeric(fpl_hist.penalties_order, errors="coerce") == 1, "code"])
    for r in pool.itertuples():
        obs = []                          # (season_end, same_club, m90, xg_np, team_xg_pm, xa, team_xa_pm, starts, team_matches, rec90_fpl, rec_n90, yc, src)
        code = fpl_codes.get(r.id)
        used_fpl_2026 = False
        if agg is not None and code is not None and code in agg.index:
            a = agg.loc[code]
            m90 = float(a.minutes) / 90.0
            if m90 > 0:
                pen_adj = PEN_XG * PENS_PER_TEAM_MATCH if code in pen_codes else 0.0
                xg_np = max(float(a.xg) - pen_adj * m90, 0.5 * float(a.xg))
                t_np = max(float(a.team_xg_pm) - PEN_XG * PENS_PER_TEAM_MATCH, 0.3)
                same = fpl_team_matches(a, r.team)
                rec = float(a.rec90) if np.isfinite(a.rec90) else np.nan
                obs.append((2026, same, m90, xg_np, t_np, float(a.xa), float(a.team_xa_pm), float(a.starts),
                            float(a.team_matches), rec, m90, float(a.yellow), "FPL 25/26",
                            float(a.starts) + float(a.get("sub_apps", 0.0))))
                used_fpl_2026 = True
        k = fbm.get(r.id)
        pen_taker = None
        if k is not None and k in fb_by_key:
            for s in fb_by_key[k].itertuples():
                if s.season == 2026 and used_fpl_2026:          # FPL already covers 2025/26 in full
                    continue
                m90 = float(s.minutes) / 90.0
                if m90 <= 0:
                    continue
                same = (s.ucl == r.team)
                conv = REC_FB_TO_FPL.get(r.pos)
                rec = float(s.rec90) * conv if (conv and np.isfinite(s.rec90)) else np.nan
                rec_n = float(s.misc_90s) if (conv and np.isfinite(s.misc_90s)) else 0.0
                obs.append((int(s.season), same, m90, float(s.npxg), float(s.team_npxg_pm), float(s.xag),
                            float(s.team_xag_pm), float(s.starts), float(s.team_matches), rec, rec_n, float(s.crdy),
                            f"FBref {SEASON_LABEL.get(int(s.season), s.season)}", float(getattr(s, "mp", s.starts))))
            latest = fb_by_key[k][fb_by_key[k].ucl == r.team].sort_values("season").tail(1)
            if len(latest) and latest.season.iloc[0] >= 2025:
                l = latest.iloc[0]
                if l.pkatt >= 2 and np.isfinite(l.pk_share) and l.pk_share >= 0.5:
                    pen_taker = True
        if code is not None and code in pen_codes and fpl_team_matches(agg.loc[code], r.team):
            pen_taker = True
        if r.id in cur:
            c = form.loc[cur[r.id]]
            m90 = float(c.minutes) / 90.0
            obs.append((CURRENT_SEASON, True, m90, float(c.xg_np), max(float(c.team_xg_pm), 0.3), float(c.assists),
                        max(float(c.team_a_pm), 0.2), float(c.starts), float(c.team_matches), np.nan, 0.0, 0.0,
                        f"{c.comp} 26/27", float(c.m)))
            if float(c.pk) >= 1 and float(c.goals) >= 1:
                pen_taker = True if pen_taker is None else pen_taker
        rows.append(_summarise(r, obs, pen_taker))
    return pd.DataFrame(rows)


def fpl_team_matches(a, team: str) -> bool:
    from ..names import ucl_team_id
    return ucl_team_id(str(a.team)) == team if isinstance(a.team, str) else False


def _summarise(r, obs: list, pen_taker) -> dict:
    out = dict(id=r.id, h_xg_share=np.nan, h_xa_share=np.nan, h_n90=0.0, h_k_xg=K_XG["same"], h_k_xa=K_XA["same"],
               h_start_rate=np.nan, h_start_n=0.0, h_rec90=np.nan, h_rec_n=0.0, h_yc90=np.nan, h_yc_n=0.0,
               h_pen=bool(pen_taker), h_text="", h_start_cur=False, cur_m=np.nan, cur_min=np.nan, cur_starts=np.nan,
               cur_team_matches=np.nan, cur_goals=np.nan, cur_assists=np.nan, cur_xg=np.nan, cur_xg90=np.nan,
               prev_xg90=np.nan, prev_xa90=np.nan, prev_n90=0.0)
    if not obs:
        return out
    is_cur = np.array([o[0] == CURRENT_SEASON for o in obs], dtype=bool)
    prior_seasons = [o[0] for o in obs if o[0] != CURRENT_SEASON]
    newest = max(prior_seasons) if prior_seasons else CURRENT_SEASON - 1
    w = np.array([W_CUR_XG if c else DECAY ** (newest - o[0]) for o, c in zip(obs, is_cur)])
    w_xa = np.where(is_cur, W_CUR_XA, w)
    if is_cur.any():
        o = obs[int(np.argmax(is_cur))]
        out.update(cur_m=o[13], cur_min=o[2] * 90, cur_starts=o[7], cur_team_matches=o[8], cur_assists=o[5],
                   cur_xg=o[3], cur_xg90=o[3] / o[2] if o[2] > 0.5 else np.nan)
    same = np.array([o[1] for o in obs], dtype=bool)
    m90 = np.array([o[2] for o in obs])
    xg = np.array([o[3] for o in obs]); txg = np.array([o[4] for o in obs])
    xa = np.array([o[5] for o in obs]); txa = np.array([o[6] for o in obs])
    st = np.array([o[7] for o in obs]); tm = np.array([o[8] for o in obs])
    rec = np.array([o[9] for o in obs]); recn = np.array([o[10] for o in obs])
    yc = np.array([o[11] for o in obs])
    pm = ~is_cur
    if pm.any() and float((w[pm] * m90[pm]).sum()) > 0.5:
        out["prev_n90"] = float((w[pm] * m90[pm]).sum())
        out["prev_xg90"] = float((w[pm] * xg[pm]).sum() / out["prev_n90"])
        out["prev_xa90"] = float((w[pm] * xa[pm]).sum() / out["prev_n90"])
    n = float((w * m90).sum())
    if n > 0.5:
        out["h_xg_share"] = float((w * xg).sum() / max((w * txg * m90).sum(), 1e-6))
        out["h_xa_share"] = float((w_xa * xa).sum() / max((w_xa * txa * m90).sum(), 1e-6))
        out["h_n90"] = n
        frac_same = float((w * m90 * same).sum() / n)
        if frac_same < 0.5:
            out["h_k_xg"], out["h_k_xa"] = K_XG["moved"], K_XA["moved"]
        out["h_yc90"] = float((w * yc).sum() / n)
        out["h_yc_n"] = n
    if same.any():
        mp = np.array([o[13] if len(o) > 13 else o[8] for o in obs], dtype=float)
        mp = np.minimum(np.maximum(mp, st), tm)
        fit_m = mp + FIT_UNUSED * (tm - mp)          # matches he was fit for (estimate)
        prev = same & ~is_cur
        ws = np.where(prev, w, 0.0)
        r_prev, n_prev = np.nan, 0.0
        if (ws * fit_m).sum() > 0:
            r_prev = float(np.clip((ws * st).sum() / (ws * fit_m).sum(), 0, 1))
            n_prev = float((ws * fit_m).sum())
            out["h_start_rate"], out["h_start_n"] = r_prev, n_prev
        if is_cur.any():
            i = int(np.argmax(is_cur))
            s_cur, n_cur = float(st[i]), float(fit_m[i])
            c = min(n_prev, W_PREV_START) if np.isfinite(r_prev) else 0.0
            if n_cur + c > 0:
                out["h_start_rate"] = float(np.clip((s_cur + c * (r_prev if np.isfinite(r_prev) else 0.0)) / (n_cur + c), 0, 1))
                out["h_start_n"] = float(n_cur + c)
                out["h_start_cur"] = True
    okr = np.isfinite(rec) & (recn > 0)
    if okr.any():
        out["h_rec90"] = float((w[okr] * rec[okr] * recn[okr]).sum() / (w[okr] * recn[okr]).sum())
        out["h_rec_n"] = float((w[okr] * recn[okr]).sum())
    labels = {}
    for o in obs:
        labels[o[12]] = labels.get(o[12], 0) + o[2] * 90
    out["h_text"] = " + ".join(f"{k} {int(round(v)):,} min" for k, v in sorted(labels.items(), key=lambda kv: kv[0], reverse=True))
    return out
