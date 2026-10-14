"""Joint Monte Carlo simulation of a matchday, event by event.

For every fixture and simulation draw:
  1. a 90-minute scoreline from the Dixon–Coles matrix;
  2. every player's minutes (start / bench / sub-on minute / sub-off minute);
  3. each goal's minute, type (open play, penalty, own goal), scorer and assister, drawn
     only from players on the pitch at that minute;
  4. goals conceded while each defender/keeper was on the pitch (clean sheets, -1 per 2);
  5. missed penalties and saves, penalties won and conceded, own goals;
  6. ball recoveries (negative binomial), saves, cards;
  7. UEFA Player of the Match as the arg-max of a performance score with Gumbel noise.
Points then follow config/rules_2026_27.yaml exactly. Because everything inside a fixture is
drawn jointly, team-mates' points are correlated the way real ones are (a 3-0 win lifts the
keeper, the back four and the scorer together), which the optimiser and the captaincy
policy rely on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import settings
from ..rules import StatArrays, points
from .scoreline import dc_matrix, dc_matrix_predictive

OFF_Q = ([0, .10, .25, .50, .75, .90, 1.0], [20, 45, 61, 71, 79, 84, 89])      # sub-off minute (PL 2025/26)
SUBMIN_Q = ([0, .10, .25, .50, .75, .90, 1.0], [1, 2, 7, 16, 26, 38, 60])     # minutes played as a sub
OG_POS_W = {"GK": 0.0056, "DEF": 0.0089, "MID": 0.0018, "FWD": 0.0026}
PEN_CONC_W = {"GK": 0.05, "DEF": 0.6, "MID": 0.3, "FWD": 0.05}
REC_RATIO_DEFAULT = {"GK": 0.35, "DEF": 1.5, "MID": 1.3, "FWD": 1.1}          # UEFA vs Opta recoveries
NB_SIZE = {"GK": 30.0, "DEF": 12.0, "MID": 12.0, "FWD": 7.5}


@dataclass
class SimResult:
    player_ids: list
    points: np.ndarray            # (P, N) int16
    minutes: np.ndarray           # (P, N) uint8
    goals: np.ndarray             # (P, N) int8
    assists: np.ndarray
    cs: np.ndarray                # (P, N) bool  (clean-sheet points earned)
    recoveries: np.ndarray        # (P, N) int16
    potm: np.ndarray              # (P, N) bool
    started: np.ndarray           # (P, N) bool
    day: np.ndarray               # (P,) day index within the matchday
    fixture: np.ndarray           # (P,) fixture index
    scores: np.ndarray            # (F, N, 2)
    potm_other: np.ndarray        # (F, N) bool -> POTM went to an unlisted player
    meta: dict = field(default_factory=dict)

    def index(self) -> dict:
        return {p: i for i, p in enumerate(self.player_ids)}


def _interp_q(u: np.ndarray, q) -> np.ndarray:
    return np.interp(u, q[0], q[1])


def _goal_times(rng, shape, slope: float) -> np.ndarray:
    """Minute of each goal; density f(t) ∝ 1 + slope * t/90 on (0, 90]."""
    u = rng.random(shape)
    a = slope / 2.0
    # inverse of F(x) = (x + a x^2)/(1 + a), x = t/90
    x = (-1 + np.sqrt(1 + 4 * a * u * (1 + a))) / (2 * a) if a > 0 else u
    return np.clip(x * 90.0, 0.5, 90.0)


def _weighted_pick(rng, weights: np.ndarray, other_w: np.ndarray | float) -> np.ndarray:
    """weights (..., P) >= 0, other_w broadcastable to (...). Returns index in [0, P] (P = others)."""
    cum = np.cumsum(weights, axis=-1)
    total = cum[..., -1] + other_w
    u = rng.random(total.shape) * total
    idx = (cum < u[..., None]).sum(axis=-1)
    return idx


def _count(idx: np.ndarray, mask: np.ndarray, n: int, p: int) -> np.ndarray:
    """Count occurrences of player index per sim. idx (N, G) in [0, p]; returns (N, p)."""
    rows = np.broadcast_to(np.arange(n)[:, None], idx.shape)
    sel = mask & (idx < p)
    flat = rows[sel] * p + idx[sel]
    return np.bincount(flat, minlength=n * p).reshape(n, p)


def _minutes(rng, n, p_start, p_sub, p_full):
    P = len(p_start)
    start = rng.random((n, P)) < p_start
    full = rng.random((n, P)) < p_full
    off_min = _interp_q(rng.random((n, P)), OFF_Q)
    sub = (~start) & (rng.random((n, P)) < p_sub)
    sub_mins = _interp_q(rng.random((n, P)), SUBMIN_Q)
    on = np.where(start, 0.0, np.where(sub, 90.0 - sub_mins, 999.0))
    off = np.where(start, np.where(full, 90.0, off_min), np.where(sub, 90.0, -1.0))
    minutes = np.clip(off - on, 0, 90)
    minutes = np.where(start | sub, np.maximum(minutes, 1), 0)
    return start, on, off, minutes


def simulate_fixture(rng, n: int, lh: float, la: float, rho: float, home: pd.DataFrame, away: pd.DataFrame,
                     others: dict, cfg: dict, rec_ratio: dict | None = None, sh: float = 0.0, sa: float = 0.0) -> dict:
    """Simulate one fixture. home/away: player rows (with rate columns) of each side.
    sh/sa: posterior sd of the log goal rates (rating uncertainty), integrated into the score matrix."""
    sim = cfg["simulation"]
    pot = cfg["potm"]
    rec_ratio = rec_ratio or sim.get("recovery_ratio") or REC_RATIO_DEFAULT
    m = dc_matrix_predictive(lh, la, rho, sh, sa, max_goals=10)
    flat = rng.choice(m.size, size=n, p=m.ravel())
    goals_side = [flat // m.shape[1], flat % m.shape[1]]           # [home goals, away goals]
    lam = [lh, la]
    sides = [home.reset_index(drop=True), away.reset_index(drop=True)]
    oth = [others.get(home.team.iloc[0] if len(home) else "", {}), others.get(away.team.iloc[0] if len(away) else "", {})]

    # ---------------------------------------------------------------- minutes
    mins, ons, offs, starts = [], [], [], []
    for s in range(2):
        df = sides[s]
        st, on, off, mi = _minutes(rng, n, df.p_start.to_numpy(), df.p_sub.to_numpy(), df.p_full90.to_numpy())
        starts.append(st)
        ons.append(on)
        offs.append(off)
        mins.append(mi)

    # ---------------------------------------------------------------- goal events per scoring side
    ev = []
    for s in range(2):
        G = goals_side[s]
        gmax = int(min(max(G.max(), 1), 9))
        valid = np.arange(gmax)[None, :] < np.minimum(G, gmax)[:, None]
        t = _goal_times(rng, (n, gmax), sim["goal_time_slope"])
        pen_rate = sim["penalties_per_team_match"] * (lam[s] / 1.4) ** 0.7
        q_pen = min(0.35, pen_rate * sim["penalty_conversion"] / max(lam[s], 0.05))
        u = rng.random((n, gmax))
        og = valid & (u < sim["own_goal_share"])
        pen = valid & ~og & (u < sim["own_goal_share"] + q_pen)
        opn = valid & ~og & ~pen
        ev.append(dict(G=G, gmax=gmax, valid=valid, t=t, og=og, pen=pen, opn=opn, pen_rate=pen_rate))

    out_stats = []
    for s in range(2):
        o = 1 - s
        df = sides[s]
        P = len(df)
        e = ev[s]
        on, off = ons[s], offs[s]
        onp = (on[:, None, :] < e["t"][..., None]) & (e["t"][..., None] <= off[:, None, :])     # (n, g, P)
        xg = df.xg_share.to_numpy()
        xa = df.xa_share.to_numpy()
        pen_w = np.where(df.pen_order.to_numpy() == 1, 50.0, xg)
        o_xg = oth[s].get("others_xg", 0.1)
        o_xa = oth[s].get("others_xa", 0.1)
        # scorers
        w_open = onp * xg
        w_pen = onp * pen_w
        scorer = np.where(e["pen"], _weighted_pick(rng, w_pen, o_xg * 0.3), _weighted_pick(rng, w_open, o_xg))
        scorer = np.where(e["opn"] | e["pen"], scorer, P + 1)              # OG / invalid -> not a player of this side
        goals = _count(scorer, e["opn"] | e["pen"], n, P)
        ob = e["opn"] & (scorer < P) & (rng.random(scorer.shape) < df.ob_share.to_numpy()[np.minimum(scorer, P - 1)])
        ob_goals = _count(scorer, ob, n, P)
        # assists (open play only, not the scorer)
        has_ast = e["opn"] & (rng.random(scorer.shape) < sim["assist_rate"])
        w_a = onp * xa
        sc_onehot = (np.arange(P)[None, None, :] == scorer[..., None])
        w_a = np.where(sc_onehot, 0.0, w_a)
        assister = _weighted_pick(rng, w_a, o_xa)
        assists = _count(assister, has_ast, n, P)
        # missed penalties (no goal): taker on the pitch
        n_miss = np.minimum(rng.poisson(e["pen_rate"] * (1 - sim["penalty_conversion"]), n), 2)
        t_miss = _goal_times(rng, (n, 2), 0.0)
        miss_valid = np.arange(2)[None, :] < n_miss[:, None]
        onp_m = (on[:, None, :] < t_miss[..., None]) & (t_miss[..., None] <= off[:, None, :])
        taker_m = _weighted_pick(rng, onp_m * pen_w, o_xg * 0.3)
        pen_missed = _count(taker_m, miss_valid, n, P)
        # penalties won (fouls only) among this side's on-pitch players
        pen_events_t = np.concatenate([np.where(e["pen"], e["t"], np.nan), np.where(miss_valid, t_miss, np.nan)], axis=1)
        pen_valid = np.isfinite(pen_events_t) & (rng.random(pen_events_t.shape) < sim["penalty_won_by_foul_share"])
        tt = np.nan_to_num(pen_events_t, nan=-5.0)
        onp_p = (on[:, None, :] < tt[..., None]) & (tt[..., None] <= off[:, None, :])
        winner = _weighted_pick(rng, onp_p * (xg + 0.5 * xa), o_xg + 0.5 * o_xa)
        pen_won = _count(winner, pen_valid, n, P)
        out_stats.append(dict(goals=goals, ob_goals=ob_goals, assists=assists, pen_missed=pen_missed, pen_won=pen_won,
                              pen_all_t=pen_events_t, miss_valid=miss_valid, t_miss=t_miss))

    # ---------------------------------------------------------------- defensive side of each event
    results = []
    for s in range(2):
        o = 1 - s
        df = sides[s]
        P = len(df)
        pos = df.pos.to_numpy()
        on, off, minutes = ons[s], offs[s], mins[s]
        eo = ev[o]                                   # goals scored by the opponent (incl. OGs credited to us)
        onp_c = (on[:, None, :] < eo["t"][..., None]) & (eo["t"][..., None] <= off[:, None, :])
        conceded = (onp_c & eo["valid"][..., None]).sum(axis=1)
        # own goals by this side's players = opponent goals flagged OG
        og_w = onp_c * np.array([OG_POS_W[p] for p in pos])
        og_pick = _weighted_pick(rng, og_w, 0.004)
        own_goals = _count(og_pick, eo["og"], n, P)
        # penalties conceded by this side (all opponent penalties, scored or missed)
        so = out_stats[o]
        tt = np.nan_to_num(so["pen_all_t"], nan=-5.0)
        onp_pc = (on[:, None, :] < tt[..., None]) & (tt[..., None] <= off[:, None, :])
        pc_pick = _weighted_pick(rng, onp_pc * np.array([PEN_CONC_W[p] for p in pos]), 0.15)
        pen_conceded = _count(pc_pick, np.isfinite(so["pen_all_t"]), n, P)
        # penalty saves: opponent misses that are saves, credited to our keeper on the pitch
        is_gk = (pos == "GK")
        tm = so["t_miss"]
        onp_ps = (on[:, None, :] < tm[..., None]) & (tm[..., None] <= off[:, None, :]) & is_gk[None, None, :]
        saved = so["miss_valid"] & (rng.random(tm.shape) < 0.7)
        ps_pick = _weighted_pick(rng, onp_ps.astype(float), 1e-9)
        pen_saves = _count(ps_pick, saved, n, P)
        # saves (keeper): opponent non-goal shots on target ∝ opponent expected goals
        lam_opp = lh if s == 1 else la
        team_saves = rng.poisson(sim.get("saves_per_xgc", 2.0) * lam_opp, n)
        frac = np.where(is_gk[None, :], minutes / 90.0, 0.0)
        saves = rng.binomial(team_saves[:, None].repeat(P, axis=1), np.clip(frac, 0, 1))
        # recoveries: Gamma-Poisson (negative binomial) around the per-90 rate
        ratio = np.array([rec_ratio[p] for p in pos])
        lam_own = la if s == 1 else lh
        opp_factor = (lam_opp / max(lam_own, 0.1)) ** 0.10
        mean_rec = df.rec90.to_numpy() * ratio * opp_factor * minutes / 90.0
        size = np.array([NB_SIZE[p] for p in pos])
        gam = rng.gamma(size, np.maximum(mean_rec, 1e-6) / size)
        recov = rng.poisson(gam) * (minutes > 0)
        # cards
        m90 = minutes / 90.0
        yellow = (rng.random((n, P)) < 1 - np.exp(-df.yc90.to_numpy() * m90)) & (minutes > 0)
        red = (rng.random((n, P)) < df.red90.to_numpy() * m90) & (minutes > 0)
        cs = (conceded == 0) & (minutes >= 60)
        results.append(dict(conceded=conceded, own_goals=own_goals, pen_conceded=pen_conceded, pen_saves=pen_saves,
                            saves=saves, recov=recov, yellow=yellow, red=red, cs=cs))

    # ---------------------------------------------------------------- Player of the Match
    hg, ag = goals_side
    win = [hg > ag, ag > hg]
    draw = hg == ag
    beta = pot["noise_scale"]
    scores_all, owners = [], []
    for s in range(2):
        df = sides[s]
        st, r = out_stats[s], results[s]
        is_def = df.pos.isin(["GK", "DEF"]).to_numpy()
        perf = (pot["w_goal"] * st["goals"] + pot["w_assist"] * st["assists"] + pot["w_cs_def"] * (r["cs"] & is_def)
                + pot["w_save3"] * (r["saves"] // 3) + pot["w_recov3"] * (r["recov"] // 3)
                + (pot["w_win"] * win[s] + pot["w_draw"] * draw)[:, None] + 0.15 * np.log1p(df.price.to_numpy()))
        perf = np.where(mins[s] >= 45, perf + rng.gumbel(0, beta, perf.shape), -np.inf)
        scores_all.append(perf)
        owners.append(np.full(len(df), s))
        # unlisted players of this side, as one pseudo-player (max of k Gumbels => +beta*ln k)
        k = max(oth[s].get("others_on_pitch", 0.0), 0.0)
        o_goals = goals_side[s] - st["goals"].sum(axis=1)
        o_perf = (pot["w_goal"] * np.clip(o_goals - ev[s]["og"].sum(axis=1), 0, None) + pot["w_win"] * win[s]
                  + pot["w_draw"] * draw + rng.gumbel(0, beta, n) + beta * np.log(max(k, 1e-3)))
        o_perf = np.where(rng.random(n) < min(k, 1.0), o_perf, -np.inf)
        scores_all.append(o_perf[:, None])
        owners.append(np.array([-1 - s]))
    allsc = np.concatenate(scores_all, axis=1)
    best = np.argmax(allsc, axis=1)
    Ph = len(sides[0])
    potm = [np.zeros((n, len(sides[0])), bool), np.zeros((n, len(sides[1])), bool)]
    rows = np.arange(n)
    hb = best < Ph
    potm[0][rows[hb], best[hb]] = True
    ab = (best > Ph) & (best < Ph + 1 + len(sides[1]))
    potm[1][rows[ab], best[ab] - Ph - 1] = True
    potm_other = ~(hb | ab)

    # ---------------------------------------------------------------- points
    out = []
    for s in range(2):
        df = sides[s]
        st, r = out_stats[s], results[s]
        pts = np.zeros((len(df), n), dtype=np.int16)
        for j, pos in enumerate(df.pos.to_numpy()):
            sa = StatArrays(minutes=mins[s][:, j], goals=st["goals"][:, j], ob_goals=st["ob_goals"][:, j],
                            assists=st["assists"][:, j], cs=r["cs"][:, j], conceded=r["conceded"][:, j],
                            saves=r["saves"][:, j], pen_saves=r["pen_saves"][:, j], recoveries=r["recov"][:, j],
                            potm=potm[s][:, j], pen_won=st["pen_won"][:, j], pen_conceded=r["pen_conceded"][:, j],
                            pen_missed=st["pen_missed"][:, j], yellow=r["yellow"][:, j].astype(int),
                            red=r["red"][:, j].astype(int), own_goals=r["own_goals"][:, j])
            pts[j] = points(pos, sa)
        out.append(dict(ids=df.id.tolist(), points=pts, minutes=mins[s].T.astype(np.uint8), goals=st["goals"].T.astype(np.int8),
                        assists=st["assists"].T.astype(np.int8), cs=r["cs"].T, recov=r["recov"].T.astype(np.int16),
                        potm=potm[s].T, started=starts[s].T))
    return dict(sides=out, scores=np.stack([hg, ag], axis=1), potm_other=potm_other)


def simulate_matchday(fixtures: pd.DataFrame, players: pd.DataFrame, others: dict, n_sims: int | None = None,
                      seed: int | None = None, rec_ratio: dict | None = None) -> SimResult:
    """fixtures needs columns: home, away, lh, la, rho, day."""
    cfg = settings().raw
    n = n_sims or cfg["simulation"]["n_sims"]
    rng = np.random.default_rng(seed if seed is not None else cfg["simulation"]["seed"])
    ids, pts, mins, gls, ast, css, rec, pot, sta, day, fix = [], [], [], [], [], [], [], [], [], [], []
    scores, pother = [], []
    for fi, f in enumerate(fixtures.itertuples()):
        h = players[players.team == f.home]
        a = players[players.team == f.away]
        res = simulate_fixture(rng, n, f.lh, f.la, f.rho, h, a, others, cfg, rec_ratio,
                               sh=float(getattr(f, "sh", 0.0) or 0.0), sa=float(getattr(f, "sa", 0.0) or 0.0))
        for sd in res["sides"]:
            ids += sd["ids"]
            pts.append(sd["points"])
            mins.append(sd["minutes"])
            gls.append(sd["goals"])
            ast.append(sd["assists"])
            css.append(sd["cs"])
            rec.append(sd["recov"])
            pot.append(sd["potm"])
            sta.append(sd["started"])
            day += [int(f.day)] * len(sd["ids"])
            fix += [fi] * len(sd["ids"])
        scores.append(res["scores"])
        pother.append(res["potm_other"])

    def cat(lst, dtype=None):
        arr = np.concatenate(lst, axis=0) if lst else np.zeros((0, n))
        return arr.astype(dtype) if dtype else arr

    return SimResult(player_ids=ids, points=cat(pts, np.int16), minutes=cat(mins, np.uint8), goals=cat(gls, np.int8),
                     assists=cat(ast, np.int8), cs=cat(css, bool), recoveries=cat(rec, np.int16), potm=cat(pot, bool),
                     started=cat(sta, bool), day=np.array(day), fixture=np.array(fix), scores=np.stack(scores),
                     potm_other=np.stack(pother), meta=dict(n_sims=n))


def summarise_players(sim: SimResult, players: pd.DataFrame) -> pd.DataFrame:
    idx = sim.index()
    rows = []
    pl = players.set_index("id")
    for pid, i in idx.items():
        pts = sim.points[i].astype(float)
        played = sim.minutes[i] > 0
        started = sim.started[i]
        rows.append(dict(
            id=pid, xpts=pts.mean(), p10=np.percentile(pts, 10), p50=np.percentile(pts, 50), p90=np.percentile(pts, 90),
            p_haul=float((pts >= 10).mean()), p_blank=float((pts <= 2).mean()), p_play=float(played.mean()),
            p_start_sim=float(started.mean()), xpts_if_start=float(pts[started].mean()) if started.any() else 0.0,
            p_goal=float((sim.goals[i] > 0).mean()), p_assist=float((sim.assists[i] > 0).mean()),
            xgoals=float(sim.goals[i].mean()), xassists=float(sim.assists[i].mean()),
            xrec=float(sim.recoveries[i].mean()), xmins=float(sim.minutes[i].mean()), p_cs=float(sim.cs[i].mean()),
            p_potm=float(sim.potm[i].mean()),
            day=int(sim.day[i]), fixture=int(sim.fixture[i])))
    out = pd.DataFrame(rows)
    return out.merge(pl[["name", "team", "pos", "price"]], left_on="id", right_index=True, how="left")
