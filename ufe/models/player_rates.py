"""Per-player rates for the UCL Fantasy pool.

Each player gets attacking shares (the fraction of his team's xG / xA he accounts for per 90
on the pitch), recovery and card rates, a start probability, minutes behaviour and set-piece
roles. Sources, in order of preference:

1. Player-specific history (Premier League Opta data in the bundled build; Understat and the
   official UCL feed on your machine), shrunk toward the prior by an empirical-Bayes weight.
2. Within-club priors indexed by position and price. A club's attacking output is shared
   among its players; the game's prices tell us who takes the bigger share. The price slopes
   were fitted on Premier League 2025/26 with team fixed effects.

Start probabilities come from a "slots" model: each club fields ~1 GK, ~4.1 DEF, ~3.6 MID and
~2.3 FWD per match, shared among the club's players by price (and by observed start rate when
known). Players missing from a partial pool are represented by "phantoms" so the visible
players are not over-credited; the phantoms' share is the simulator's "others" bucket.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from ..config import settings
from ..ingest import fbref as fb_mod
from ..ingest import fpl as fpl_mod
from . import history as hist_mod
from ..names import norm, ucl_team_id

PL_CLUBS = {"ARS", "AVL", "LIV", "MCI", "MUN"}
SQUAD_SIZE = {"GK": 3, "DEF": 8, "MID": 8, "FWD": 5}
SLOTS = {"GK": 1.0, "DEF": 4.1, "MID": 3.6, "FWD": 2.3}
BETA = {"GK": 3.0, "DEF": 2.2, "MID": 1.5, "FWD": 1.5}
# listed players come from articles that mostly discuss regular starters; unlisted ones are weighted down
PHANTOM_W = {"GK": 0.25, "DEF": 1.0, "MID": 0.6, "FWD": 0.5}
MIN_PRICE = {"GK": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 5.0}
TYPICAL_PRICE = {"GK": 4.5, "DEF": 4.5, "MID": 5.5, "FWD": 6.5}   # unlisted starters at a fully unlisted position
# within-team price slopes (per £m, FPL scale) and base shares of a regular starter (PL 2025/26)
XG_BASE = {"GK": 0.0005, "DEF": 0.045, "MID": 0.105, "FWD": 0.30}
XA_BASE = {"GK": 0.003, "DEF": 0.066, "MID": 0.144, "FWD": 0.069}
XG_SLOPE = {"GK": 0.0, "DEF": 0.251, "MID": 0.312, "FWD": 0.073}
XA_SLOPE = {"GK": 0.0, "DEF": 0.34, "MID": 0.208, "FWD": 0.0}
REF_FEQ = {"GK": 5.0, "DEF": 5.0, "MID": 6.5, "FWD": 7.5}
# caps stop the exponential price curve extrapolating past what elite players actually take
XG_CAP = {"GK": 0.01, "DEF": 0.12, "MID": 0.38, "FWD": 0.50}
XA_CAP = {"GK": 0.02, "DEF": 0.25, "MID": 0.40, "FWD": 0.30}
REC90 = {"GK": 8.17, "DEF": 3.60, "MID": 4.65, "FWD": 2.31}
# role-adjusted recoveries prior: rec90 = a + b * (xg_share + xa_share)   (PL 2025/26, >=900 min)
REC_ROLE = {"GK": (8.17, 0.0), "DEF": (3.42, 2.96), "MID": (4.95, -2.26), "FWD": (2.61, -1.00)}
# unlisted players at stronger clubs are pricier (rotation stars, not academy fillers)
PHANTOM_SPREAD = {"GK": 0.5, "DEF": 0.5, "MID": 2.0, "FWD": 2.0}
TEAM_SHARE_TOTAL = 0.96        # remaining ~4% of goals are own goals
SHRINK_90S = 10.0              # prior weight in 90-minute equivalents
# P(sub appearance | not starting) by start-probability band (Premier League 2025/26)
P_SUB_BANDS = [(0.0, {"GK": 0.0, "DEF": 0.04, "MID": 0.08, "FWD": 0.13}),
               (0.2, {"GK": 0.01, "DEF": 0.27, "MID": 0.44, "FWD": 0.56}),
               (0.5, {"GK": 0.02, "DEF": 0.33, "MID": 0.48, "FWD": 0.73}),
               (0.8, {"GK": 0.0, "DEF": 0.26, "MID": 0.47, "FWD": 0.56})]


def fpl_equivalent_price(ucl_price) -> np.ndarray:
    """UCL Fantasy price on the FPL price scale (anchors: €4.5m≈£4.5m, €7.0m≈£7.2m,
    €8.5m≈£9.6m, €11.0m≈£15.5m). Only differences within a club are used."""
    return 4.5 * np.exp(0.19 * (np.asarray(ucl_price, dtype=float) - 4.5))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm(name)).strip("-")


def _note_flags(note) -> dict:
    n = note.lower() if isinstance(note, str) else ""
    return dict(pen=("penalt" in n and "not " not in n), injured=("injur" in n and "fully fit" not in n),
                set_pieces=("set piece" in n or "set-piece" in n or "corner" in n or "set play" in n))


def match_fpl(pool: pd.DataFrame, d: fpl_mod.FPLData) -> dict[str, int]:
    """pool player id -> FPL code. Same-club matches are fuzzy; cross-club matches need near-exact names."""
    now = d.players_now.copy()
    now["ucl_team"] = now["team_name"].map(ucl_team_id)
    hist = d.players_hist
    out: dict[str, int] = {}
    for r in pool.itertuples():
        best, best_score = None, 0
        nm = norm(r.name)
        if r.team in PL_CLUBS:
            cands = now[now["ucl_team"] == r.team]
            for c in cands.itertuples():
                s = fuzz.token_set_ratio(nm, norm(c.full_name))
                wn = norm(str(c.web_name))
                if wn and wn == nm.split()[-1] and wn in norm(c.full_name).split():
                    s = max(s, 95 if (cands.web_name == c.web_name).sum() == 1 else 0)
                if s > best_score:
                    best, best_score = c.code, s
            if best_score >= 85:
                out[r.id] = int(best)
                continue
        if len(nm.split()) >= 2:
            best, best_score = None, 0
            for c in hist.itertuples():
                s = fuzz.token_sort_ratio(nm, norm(c.full_name))
                if s > best_score:
                    best, best_score = c.code, s
            if best_score >= 93:
                out[r.id] = int(best)
    return out


SITE_K_G = 25.0    # 90s of prior weight for actual goals per 90 (xG history uses 12: goals are noisier)
SITE_K_A = 40.0    # same for assists (xA history uses 20)
SITE_N90_CAP = 40.0
ASSISTS_PER_GOAL = 0.75


def build(pool: pd.DataFrame, d: fpl_mod.FPLData | None = None, use_fpl_availability: bool = False,
          team_strength: dict[str, float] | None = None,
          team_gpm: dict[str, float] | None = None, use_current_form: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (players, others): others[team] = phantom ('unlisted players') shares and minutes."""
    cfg = settings()
    sim = cfg["simulation"]
    pool = pool.copy()
    pool["note"] = pool.get("note", pd.Series([""] * len(pool))).fillna("")
    if "id" not in pool:
        pool["id"] = [f"{t}-{slug(n)}" for t, n in zip(pool.team, pool.name)]
    d = d or fpl_mod.load()
    agg = fpl_mod.player_aggregates(d)
    pri = fpl_mod.population_priors(agg)
    codes = match_fpl(pool, d)
    agg_by_code = agg.set_index("code")
    now_by_code = d.players_now.set_index("code")
    fb = fb_mod.load()
    H = hist_mod.observations(pool, codes, agg, d.players_hist, fb, use_current=use_current_form).set_index("id")

    rows = []
    for r in pool.itertuples():
        pos = r.pos
        flags = _note_flags(r.note)
        row = dict(id=r.id, name=r.name, team=r.team, pos=pos, price=float(r.price), source=r.source, note=r.note,
                   price_status=str(getattr(r, "price_status", "verified") or "verified"),
                   fpl_code=None, rate_source="prior: club + position + price", obs_xg_share=np.nan,
                   obs_xa_share=np.nan, w_obs=0.0, w_obs_xa=0.0, rec90=REC90[pos], rec90_obs=np.nan, rec_n=0.0,
                   yc90=pri[pos]["yc90"], red90=pri[pos]["red90"],
                   og90=pri[pos]["og90"], p_full90=pri[pos]["p_full90"], obs_start_rate=np.nan, obs_start_n=0.0,
                   minutes_hist=0.0, pen_order=1 if flags["pen"] else 9, pen_src=3 if flags["pen"] else 0,
                   set_pieces=bool(flags["set_pieces"]), available=0.0 if flags["injured"] else 1.0)
        code = codes.get(r.id)
        if code is not None and code in agg_by_code.index:
            a = agg_by_code.loc[code]
            row.update(fpl_code=int(code))
            if a["starts"] > 0:
                ws = a["starts"] / (a["starts"] + 8.0)
                row["p_full90"] = ws * float(a["p_full90"]) + (1 - ws) * pri[pos]["p_full90"]
        # season-by-season Opta history (FPL + FBref), weights from scripts/validate_player_layer.py
        h = H.loc[r.id] if r.id in H.index else None
        if h is not None and h.h_n90 > 0.5 and np.isfinite(h.h_xg_share):
            row.update(obs_xg_share=float(h.h_xg_share), obs_xa_share=float(h.h_xa_share),
                       w_obs=float(h.h_n90 / (h.h_n90 + h.h_k_xg)), w_obs_xa=float(h.h_n90 / (h.h_n90 + h.h_k_xa)),
                       minutes_hist=float(h.h_n90 * 90), rate_source=f"Opta: {h.h_text}")
            wy = h.h_yc_n / (h.h_yc_n + hist_mod.K_YC)
            row["yc90"] = wy * float(h.h_yc90) + (1 - wy) * pri[pos]["yc90"]
        elif team_gpm and r.team in team_gpm and np.isfinite(float(getattr(r, "g90", np.nan) or np.nan)) \
                and float(getattr(r, "mins_sample", 0) or 0) >= 450 and pos != "GK":
            # no Opta history (mostly clubs outside the big five): goals and assists per 90 from the
            # game-stats mirror, as shares of the club's goals per domestic match (team model)
            n90 = min(float(r.mins_sample) / 90.0, SITE_N90_CAP)
            gpm = float(team_gpm[r.team])
            g_sh = max(float(r.g90), 0.01) / gpm
            a_sh = max(float(getattr(r, "a90", 0.0) or 0.0), 0.01) / (ASSISTS_PER_GOAL * gpm)
            row.update(obs_xg_share=float(min(g_sh, 0.6)), obs_xa_share=float(min(a_sh, 0.5)),
                       w_obs=float(n90 / (n90 + SITE_K_G)), w_obs_xa=float(n90 / (n90 + SITE_K_A)),
                       minutes_hist=float(r.mins_sample),
                       rate_source=f"per-90 goals/assists, all comps ({int(r.mins_sample):,} min; uclfantasy.com)")
        if h is not None and np.isfinite(h.h_start_rate):
            row.update(obs_start_rate=float(h.h_start_rate), obs_start_n=float(h.h_start_n),
                       start_k=hist_mod.K_START_CUR if bool(h.h_start_cur) else hist_mod.K_START)
        if h is not None and np.isfinite(h.cur_m):
            row.update(dom_m=float(h.cur_m), dom_min=float(h.cur_min), dom_starts=float(h.cur_starts),
                       dom_team_matches=float(h.cur_team_matches), dom_xg=float(h.cur_xg),
                       dom_assists=float(h.cur_assists), dom_xg90=float(h.cur_xg90) if np.isfinite(h.cur_xg90) else np.nan,
                       prev_xg90=float(h.prev_xg90) if np.isfinite(h.prev_xg90) else np.nan,
                       prev_n90=float(h.prev_n90))
        if h is not None and np.isfinite(h.h_rec90):
            row.update(rec90_obs=float(h.h_rec90), rec_n=float(h.h_rec_n))
        if h is not None and bool(h.h_pen) and row["pen_src"] < 2:
            row.update(pen_order=1, pen_src=1)
        if code is not None and code in now_by_code.index:
            n = now_by_code.loc[code]
            if pd.notna(n.get("penalties_order")) and int(n["penalties_order"]) == 1 and row["pen_src"] < 3:
                row.update(pen_order=1, pen_src=2)
            cop = n.get("chance_of_playing_next_round")
            # only trust FPL injury flags from a live API pull, never from an old snapshot
            if use_fpl_availability and pd.notna(cop) and r.team in PL_CLUBS:
                row["available"] = min(row["available"], float(cop) / 100.0)
        rows.append(row)
    players = pd.DataFrame(rows)
    # one designated taker per club: team-news notes > current FPL order > FBref history
    for team, g in players[players.pen_order == 1].groupby("team"):
        if len(g) > 1:
            keep = g.sort_values(["pen_src", "minutes_hist"], ascending=False).index[0]
            players.loc[g.index.difference([keep]), "pen_order"] = 9
    players = _apply_understat(players)
    strength = _strength_scale(team_strength, players.team.unique())
    players, others = _start_probabilities(players, strength)
    players, others = _attacking_shares(players, others)
    # role-adjusted recoveries prior for players without their own history
    att = players.xg_share + players.xa_share
    prior_rec = [REC_ROLE[p][0] + REC_ROLE[p][1] * a for p, a in zip(players.pos, att)]
    wr = (players.rec_n / (players.rec_n + hist_mod.K_REC)).to_numpy()
    obs_rec = players.rec90_obs.to_numpy()
    prior_rec = np.maximum(np.array(prior_rec), 0.8)
    players["rec90"] = np.where(np.isfinite(obs_rec), wr * np.nan_to_num(obs_rec) + (1 - wr) * prior_rec, prior_rec)
    players = apply_ucl_observations(players)
    players["p_sub"] = [_p_sub(ps, pos) for ps, pos in zip(players.p_start, players.pos)]
    players["ob_share"] = players.pos.map(sim["outside_box_share"]).astype(float)
    return players, others


def _p_sub(p_start: float, pos: str) -> float:
    val = P_SUB_BANDS[0][1][pos]
    for lo, tab in P_SUB_BANDS:
        if p_start >= lo:
            val = tab[pos]
    return float(val)


def _cap_redistribute(raw: np.ndarray, slots: float, cap: float = 0.95) -> np.ndarray:
    w = raw / raw.sum()
    p = slots * w
    for _ in range(25):
        over = p > cap
        if not over.any():
            break
        excess = (p[over] - cap).sum()
        p[over] = cap
        free = p < cap
        if not free.any():
            break
        p[free] += excess * w[free] / w[free].sum()
    return np.minimum(p, cap)


def _strength_scale(team_strength: dict | None, teams) -> dict:
    if not team_strength:
        return {t: 0.5 for t in teams}
    vals = np.array([team_strength.get(t, np.nan) for t in teams], dtype=float)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    return {t: float(np.clip((team_strength.get(t, (lo + hi) / 2) - lo) / max(hi - lo, 1e-9), 0, 1)) for t in teams}


def _start_probabilities(players: pd.DataFrame, strength: dict | None = None) -> tuple[pd.DataFrame, dict]:
    strength = strength or {}
    players = players.copy()
    players["p_start"] = 0.0
    others: dict[str, dict] = {}
    for team in sorted(players.team.unique()):
        for pos in ("GK", "DEF", "MID", "FWD"):
            g = players[(players.team == team) & (players.pos == pos)]
            if g.empty:                      # nobody listed: the whole position is phantom
                others.setdefault(team, {})[pos] = dict(
                    n=int(SQUAD_SIZE[pos]), p_start=float(SLOTS[pos]) * 0.92,
                    price=float(TYPICAL_PRICE[pos] + strength.get(team, 0.5) * PHANTOM_SPREAD[pos] * 0.5))
                continue
            _fill_position(players, others, team, pos, g, strength.get(team, 0.5))
    return players, others


def _fill_position(players: pd.DataFrame, others: dict, team: str, pos: str, g: pd.DataFrame, s: float = 0.5) -> None:
    n_ph = max(SQUAD_SIZE[pos] - len(g), 0)
    min_p = min(float(g.price.min()), MIN_PRICE[pos] + s * PHANTOM_SPREAD[pos])
    prices = np.concatenate([g.price.to_numpy(), np.full(n_ph, min_p)])
    w = np.exp(BETA[pos] * (prices - prices.min()))
    w[len(g):] *= PHANTOM_W[pos]
    avail = np.concatenate([g.available.to_numpy(), np.full(n_ph, 0.92)])
    obs = np.concatenate([g.obs_start_rate.to_numpy(), np.full(n_ph, np.nan)])
    nobs = np.concatenate([g.obs_start_n.to_numpy() if "obs_start_n" in g else np.zeros(len(g)), np.zeros(n_ph)])
    p = _cap_redistribute(w * np.where(avail > 0, 1.0, 1e-6), SLOTS[pos])
    known = np.isfinite(obs)
    kst = np.concatenate([g.start_k.fillna(hist_mod.K_START).to_numpy() if "start_k" in g else np.full(len(g), hist_mod.K_START),
                          np.full(n_ph, hist_mod.K_START)])
    # 20 team matches of prior weight for past seasons (validated season-ahead); 4 when this
    # season's domestic starts are known (in-season walk-forward, scripts/backtest_player_form.py)
    wk = nobs[known] / (nobs[known] + kst[known])
    p[known] = (1 - wk) * p[known] + wk * np.clip(obs[known], 0.02, 0.97)
    p = np.minimum(p * avail, 0.95)
    players.loc[g.index, "p_start"] = p[: len(g)]
    others.setdefault(team, {})[pos] = dict(n=int(n_ph), p_start=float(p[len(g):].sum()), price=float(min_p))


def _attacking_shares(players: pd.DataFrame, others: dict) -> tuple[pd.DataFrame, dict]:
    """Within-club allocation of team xG and xA, normalised so each club's expected on-pitch
    shares sum to TEAM_SHARE_TOTAL; observed shares (when known) are blended in log space."""
    players = players.copy()
    feq = fpl_equivalent_price(players.price.to_numpy())
    pos = players.pos.to_numpy()
    players["xg_share"] = [min(XG_CAP[p], XG_BASE[p] * np.exp(XG_SLOPE[p] * (f - REF_FEQ[p]))) for p, f in zip(pos, feq)]
    players["xa_share"] = [min(XA_CAP[p], XA_BASE[p] * np.exp(XA_SLOPE[p] * (f - REF_FEQ[p]))) for p, f in zip(pos, feq)]
    for team, g in players.groupby("team"):
        ph = others.get(team, {})
        ph_xg = sum(v["p_start"] * XG_BASE[p] * np.exp(XG_SLOPE[p] * (float(fpl_equivalent_price(v["price"])) - REF_FEQ[p]))
                    for p, v in ph.items())
        ph_xa = sum(v["p_start"] * XA_BASE[p] * np.exp(XA_SLOPE[p] * (float(fpl_equivalent_price(v["price"])) - REF_FEQ[p]))
                    for p, v in ph.items())
        for col, ph_mass, obs_col in (("xg_share", ph_xg, "obs_xg_share"), ("xa_share", ph_xa, "obs_xa_share")):
            vals = g[col].to_numpy().copy()
            tot = float((g.p_start * vals).sum() + ph_mass)
            scale = TEAM_SHARE_TOTAL / tot if tot > 0 else 1.0
            vals *= scale
            obs = g[obs_col].to_numpy()
            wv = (g.w_obs if col == "xg_share" else g.get("w_obs_xa", g.w_obs)).to_numpy()
            have = np.isfinite(obs) & (obs > 0)
            vals[have] = np.exp(wv[have] * np.log(obs[have]) + (1 - wv[have]) * np.log(vals[have]))
            # renormalise keeping the phantom mass at its scaled level
            tot2 = float((g.p_start * vals).sum() + ph_mass * scale)
            scale2 = TEAM_SHARE_TOTAL / tot2 if tot2 > 0 else 1.0
            players.loc[g.index, col] = vals * scale2
            key = "others_xg" if col == "xg_share" else "others_xa"
            others.setdefault(team, {})[key] = float(ph_mass * scale * scale2)
        others[team]["others_on_pitch"] = float(sum(v["p_start"] for p, v in ph.items() if isinstance(v, dict) and "p_start" in v))
    return players, others


def apply_ucl_observations(players: pd.DataFrame, path=None) -> pd.DataFrame:
    """Blend this season's UCL results (logged by the feedback loop) into the rates.

    Goals and assists update the attacking shares through a Gamma-Poisson ratio:
        multiplier = (observed + k) / (expected + k),  k = 2 pseudo-events
    where expected = share * team expected goals * minutes/90 summed over logged matches.
    Recoveries update rec90 the same way (converted back to the Opta scale), and observed
    starts move the start probability. Small samples move little; a run of matches moves a lot.
    """
    from ..config import ROOT
    path = path or ROOT / "data" / "processed" / "player_obs.csv"
    if not path.exists():
        return players
    obs = pd.read_csv(path)
    if obs.empty or "id" not in obs:
        return players
    ratio = settings()["simulation"].get("recovery_ratio", {})
    players = players.copy().set_index("id")
    for pid, g in obs.groupby("id"):
        if pid not in players.index:
            continue
        r = players.loc[pid]
        m90 = float(g.get("minutes", pd.Series([0])).sum()) / 90.0
        lam = float(g.get("team_lambda", pd.Series([1.4] * len(g))).fillna(1.4).mean())
        if m90 > 0 and "goals" in g:
            exp_g = r.xg_share * lam * m90
            players.at[pid, "xg_share"] = r.xg_share * (g.goals.sum() + 2.0) / (exp_g + 2.0)
        if m90 > 0 and "assists" in g:
            exp_a = r.xa_share * lam * m90 * 0.86
            players.at[pid, "xa_share"] = r.xa_share * (g.assists.sum() + 2.0) / (exp_a + 2.0)
        if m90 > 0 and "recoveries" in g:
            k = 4.0                                   # four matches of prior weight
            obs_rate = g.recoveries.sum() / m90 / max(ratio.get(r.pos, 1.0), 1e-6)
            players.at[pid, "rec90"] = (obs_rate * m90 + r.rec90 * k) / (m90 + k)
        if "started" in g and len(g):
            n = len(g)
            players.at[pid, "p_start"] = float(np.clip((g.started.sum() + 3 * r.p_start) / (n + 3), 0, 0.97))
        players.at[pid, "rate_source"] = f"{r.rate_source} + UCL 26/27 ({len(g)} matches)"
    return players.reset_index()


def _apply_understat(players: pd.DataFrame) -> pd.DataFrame:
    """Use Understat shares (big-five leagues, current season) for players without other history."""
    from ..config import ROOT
    from ..names import best_player_match
    p = ROOT / "data" / "processed" / "understat_players.csv"
    if not p.exists():
        return players
    us = pd.read_csv(p)
    players = players.copy()
    for i, r in players.iterrows():
        if np.isfinite(r.obs_xg_share):
            continue
        cands = us[us.team == r.team]
        if cands.empty:
            continue
        hit = best_player_match(r["name"], dict(zip(cands.player_name, cands.index)), cutoff=88)
        if hit is None:
            continue
        u = us.loc[hit]
        m90 = float(u.time) / 90.0
        w = m90 / (m90 + SHRINK_90S)
        players.at[i, "obs_xg_share"] = float(u.xg_share)
        players.at[i, "obs_xa_share"] = float(u.xa_share)
        players.at[i, "w_obs"] = w
        players.at[i, "minutes_hist"] = float(u.time)
        players.at[i, "rate_source"] = f"Understat {u.league} ({int(u.time)} min)"
    return players
