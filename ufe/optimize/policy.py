"""In-matchday decision policy: captain switches, manual substitutions and the auto-sub trade-off.

UCL Fantasy lets you (after a day's games finish) move the armband to a player yet to play and
swap up to four starters for bench players yet to play, but any manual change switches off
auto-subs for the rest of the matchday. The decision after day d compares a realised score with
the expected score of a later player:

    switch captain C -> K    iff  x_C < E[Y_K] - autosub_cover - margin
    sub starter i -> bench j iff  x_i < E[Y_j] - autosub_cover - margin

E[Y] is unconditional unless later line-ups are already known. Fixtures are simulated
independently of each other, so these expectations are exact under the model; the simulation
then measures how many points the policy adds over playing passively.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import rules
from ..models.simulate import SimResult
from ..rules import manual_actions_allowed

POS_ORDER = ["GK", "DEF", "MID", "FWD"]


def _bounds():
    lo, hi = rules()["lineup"]["min"], rules()["lineup"]["max"]
    return np.array([lo[p] for p in POS_ORDER]), np.array([hi[p] for p in POS_ORDER])


def _valid(counts: np.ndarray) -> np.ndarray:
    lo, hi = _bounds()
    return np.all((counts >= lo) & (counts <= hi), axis=1)


@dataclass
class PolicyReport:
    ev_passive: float
    ev_active: float
    p_captain_switch: float
    avg_manual_subs: float
    rules: list[dict]
    dist_passive: list[float]
    dist_active: list[float]

    @property
    def uplift(self) -> float:
        return self.ev_active - self.ev_passive

    def to_dict(self) -> dict:
        return dict(ev_passive=round(self.ev_passive, 2), ev_active=round(self.ev_active, 2), uplift=round(self.uplift, 2),
                    p_captain_switch=round(self.p_captain_switch, 3), avg_manual_subs=round(self.avg_manual_subs, 2),
                    rules=self.rules, dist_passive=self.dist_passive, dist_active=self.dist_active)


def auto_subs(pts: np.ndarray, played: np.ndarray, pos_arr: np.ndarray, xi: list[int], bench: list[int],
              day: np.ndarray | None = None, only_days_gt: int | None = None) -> np.ndarray:
    """Vectorised auto-substitution (bench order, formation-valid, keepers for keepers).
    Returns the points added by substitutes per simulation."""
    N = pts.shape[1]
    xi_arr = np.array(xi)
    dead = ~played[xi_arr].T                         # (N, 11)
    if only_days_gt is not None:
        dead &= (day[xi_arr] > only_days_gt)[None, :]
    eye = np.eye(4, dtype=int)
    counts = np.tile(eye[pos_arr[xi_arr]].sum(axis=0), (N, 1))
    added = np.zeros(N)
    rows_all = np.arange(N)
    for b in bench:
        pb = pos_arr[b]
        chosen = np.full(N, -1)
        for k, j in enumerate(xi):
            pj = pos_arr[j]
            if (pj == 0) != (pb == 0):
                continue
            ok = played[b] & dead[:, k] & (chosen < 0) & _valid(counts - eye[pj] + eye[pb])
            chosen = np.where(ok, k, chosen)
        hit = chosen >= 0
        if hit.any():
            r = rows_all[hit]
            dead[r, chosen[hit]] = False
            np.add.at(counts, (r, pos_arr[xi_arr[chosen[hit]]]), -1)
            counts[r, pb] += 1
            added[hit] += pts[b, hit]
    return added


def evaluate(sim: SimResult, players: pd.DataFrame, squad: list[str], xi: list[str], captain: str,
             bench: list[str], md: int, margin: float = 0.25, lineups_known: bool = False) -> PolicyReport:
    idx = sim.index()
    missing = [p for p in squad if p not in idx]
    if missing:
        raise KeyError(f"no fixture this matchday for: {missing}")
    pos_of = players.set_index("id").pos
    names = players.set_index("id").name
    pos_arr = np.zeros(len(sim.player_ids), dtype=int)
    for p in squad:
        pos_arr[idx[p]] = POS_ORDER.index(pos_of[p])
    pts = sim.points.astype(float)
    played = sim.minutes > 0
    day = sim.day
    XI = [idx[p] for p in xi]
    BE = [idx[p] for p in bench]
    CAP = idx[captain]
    N = pts.shape[1]
    R = np.arange(N)
    max_subs = rules()["manual_subs"]["max_per_md"]

    passive = pts[XI].sum(axis=0) + auto_subs(pts, played, pos_arr, XI, BE) + pts[CAP]
    pct = [10, 25, 50, 75, 90]
    days = sorted(set(int(day[i]) for i in XI + BE))
    if not manual_actions_allowed(md) or len(days) < 2:
        why = ("No in-matchday changes on this matchday: every game kicks off at the same time."
               if not manual_actions_allowed(md) else "All your players play on the same day, so there is nothing to switch.")
        q = np.percentile(passive, pct).round(1).tolist()
        return PolicyReport(float(passive.mean()), float(passive.mean()), 0.0, 0.0, [dict(kind="info", text=why)], q, q)

    ev = pts.mean(axis=1)
    st = sim.started
    ev_start = (pts * st).sum(axis=1) / np.maximum(st.sum(axis=1), 1)
    ev_bench = (pts * ~st).sum(axis=1) / np.maximum((~st).sum(axis=1), 1)

    def expv(j: np.ndarray) -> np.ndarray:
        if lineups_known:
            return np.where(st[j, R], ev_start[j], ev_bench[j])
        return ev[j]

    eye = np.eye(4, dtype=int)
    cur_xi = np.tile(np.array(XI), (N, 1))
    cur_be = np.tile(np.array(BE), (N, 1))
    counts = np.tile(eye[pos_arr[np.array(XI)]].sum(axis=0), (N, 1))
    cap = np.full(N, CAP)
    switched = np.zeros(N, dtype=bool)
    changed = np.zeros(N, dtype=bool)
    nsubs = np.zeros(N, dtype=int)
    for d in days[:-1]:
        cover = float(auto_subs(pts, played, pos_arr, XI, BE, day, d).mean())
        # captain switch
        best_k, best_e = np.full(N, -1), np.full(N, -np.inf)
        for s in range(11):
            j = cur_xi[:, s]
            e = np.where(day[j] > d, expv(j), -np.inf)
            better = e > best_e
            best_k, best_e = np.where(better, j, best_k), np.where(better, e, best_e)
        need = np.where(changed, 0.0, cover)
        go = (day[cap] <= d) & ~switched & (best_k >= 0) & (best_e - pts[cap, R] > margin + need)
        cap = np.where(go, best_k, cap)
        switched |= go
        changed |= go
        # manual subs, greedily, up to the cap
        for _ in range(max_subs):
            need = np.where(changed, 0.0, cover)
            best_gain = np.full(N, -np.inf)
            best_s, best_b = np.full(N, -1), np.full(N, -1)
            for b in range(cur_be.shape[1]):
                jb = cur_be[:, b]
                pb = pos_arr[jb]
                eb = np.where(day[jb] > d, expv(jb), -np.inf)
                for s in range(11):
                    js = cur_xi[:, s]
                    ps = pos_arr[js]
                    ok = ((day[js] <= d) & ((ps == 0) == (pb == 0)) & (js != cap) & (nsubs < max_subs)
                          & _valid(counts - eye[ps] + eye[pb]))
                    gain = np.where(ok, eb - pts[js, R], -np.inf)
                    better = gain > best_gain
                    best_gain = np.where(better, gain, best_gain)
                    best_s = np.where(better, s, best_s)
                    best_b = np.where(better, b, best_b)
            go = best_gain > margin + need
            if not go.any():
                break
            r = R[go]
            s_i, b_i = best_s[go], best_b[go]
            out_p, in_p = cur_xi[r, s_i].copy(), cur_be[r, b_i].copy()
            cur_xi[r, s_i], cur_be[r, b_i] = in_p, out_p
            np.add.at(counts, (r, pos_arr[out_p]), -1)
            np.add.at(counts, (r, pos_arr[in_p]), 1)
            nsubs[go] += 1
            changed |= go
    active = np.where(changed, pts[cur_xi, R[:, None]].sum(axis=1) + pts[cap, R], passive)

    # readable decision rules for the first decision point
    d0 = days[0]
    cover0 = float(auto_subs(pts, played, pos_arr, XI, BE, day, d0).mean())
    out_rules: list[dict] = []
    if int(day[CAP]) == d0:
        later = [j for j in XI if day[j] > d0]
        if later:
            k = max(later, key=lambda j: ev[j])
            thr = ev[k] - cover0 - margin
            out_rules.append(dict(kind="captain", captain=sim.player_ids[CAP], to=sim.player_ids[k],
                                  threshold=round(float(thr), 2), ev_to=round(float(ev[k]), 2),
                                  text=f"If {names[sim.player_ids[CAP]]} scores {max(int(np.ceil(thr)) - 1, 0)} or fewer on day {d0}, "
                                       f"move the armband to {names[sim.player_ids[k]]} (day {int(day[k])}, expected {ev[k]:.1f})."))
    bench_later = sorted([b for b in BE if day[b] > d0], key=lambda b: -ev[b])
    for s in [j for j in XI if day[j] == d0 and j != CAP]:
        cands = [b for b in bench_later if (pos_arr[b] == 0) == (pos_arr[s] == 0)]
        if not cands:
            continue
        b = cands[0]
        thr = ev[b] - cover0 - margin
        if thr > 1:
            out_rules.append(dict(kind="sub", out=sim.player_ids[s], into=sim.player_ids[b], threshold=round(float(thr), 2),
                                  text=f"If {names[sim.player_ids[s]]} scores {max(int(np.ceil(thr)) - 1, 0)} or fewer, "
                                       f"bring on {names[sim.player_ids[b]]} (day {int(day[b])}, expected {ev[b]:.1f})."))
    out_rules.append(dict(kind="autosub_cost", value=round(cover0, 2),
                          text=f"A manual change switches off auto-subs; that cover is worth {cover0:.2f} points after day {d0}."))
    return PolicyReport(float(passive.mean()), float(active.mean()), float(switched.mean()), float(nsubs.mean()), out_rules,
                        np.percentile(passive, pct).round(1).tolist(), np.percentile(active, pct).round(1).tolist())


def head_to_head(sim: SimResult, mine: dict, rival: dict) -> dict:
    """P(my XI outscores a rival XI) this matchday, using the joint simulation (shared players cancel)."""
    idx = sim.index()
    pts = sim.points.astype(float)

    def total(t):
        tot = sum(pts[idx[p]] for p in t["xi"] if p in idx)
        return tot + (pts[idx[t["captain"]]] if t["captain"] in idx else 0)

    a, b = total(mine), total(rival)
    diff = a - b
    return dict(p_win=float((diff > 0).mean()), p_draw=float((diff == 0).mean()), exp_margin=float(diff.mean()),
                margin_p10=float(np.percentile(diff, 10)), margin_p90=float(np.percentile(diff, 90)))


def live_decisions(sim: SimResult, players: pd.DataFrame, squad: list[str], xi: list[str], captain: str,
                   bench: list[str], md: int, day_done: int, realised: dict[str, float], margin: float = 0.25,
                   lineups_known: bool = False, sent_off: set[str] | None = None) -> dict:
    """Concrete calls after `day_done` finishes, using real points for players who have played.

    Compares whole plans, because any manual change (a sub, a captain switch or a bench reorder)
    switches off auto-subs for the rest of the matchday:
      passive      no change; auto-subs still cover later-day starters who do not play
      captain only move the armband to a later-day starter
      subs only    up to 4 swaps (incoming players must not have played yet; sent-off players stay)
      both         subs first, then the best captain among the new XI
    Later-day players are valued at their expected points (zero if ruled out), played ones at
    their real points."""
    idx = sim.index()
    pos_of = players.set_index("id").pos
    names = players.set_index("id").name
    sent_off = set(sent_off or ())
    pts = sim.points.astype(float)
    played = sim.minutes > 0
    pos_arr = np.zeros(len(sim.player_ids), dtype=int)
    for p in squad:
        pos_arr[idx[p]] = POS_ORDER.index(pos_of[p])
    day = sim.day
    st = sim.started
    ev = pts.mean(axis=1)
    ev_start = (pts * st).sum(axis=1) / np.maximum(st.sum(axis=1), 1)
    XI, BE = [idx[p] for p in xi], [idx[p] for p in bench]
    cover = float(auto_subs(pts, played, pos_arr, XI, BE, day, day_done).mean())
    later = {p for p in squad if day[idx[p]] > day_done}
    e = {p: (float(ev_start[idx[p]] if lineups_known else ev[idx[p]]) if p in later else float(realised.get(p, 0.0)))
         for p in squad}
    if not manual_actions_allowed(md):
        return dict(actions=[dict(kind="info", text="No manual changes are possible on this matchday.")], cover=cover,
                    plans={})

    def total(xi_, cap):
        return sum(e[p] for p in xi_) + e[cap]

    def best_cap(xi_, cap):
        cands = [cap] + [p for p in xi_ if p in later]
        return max(cands, key=lambda p: (e[p], p == cap))

    # greedy subs (each step takes the largest gain that keeps a legal formation)
    cur_xi, swaps = list(xi), []
    pool = [b for b in bench if b in later]
    counts = np.eye(4, dtype=int)[[pos_arr[idx[p]] for p in cur_xi]].sum(axis=0)
    for _ in range(rules()["manual_subs"]["max_per_md"]):
        best = None
        for b in pool:
            for s_ in cur_xi:
                if s_ in sent_off:
                    continue
                ps, pb = pos_arr[idx[s_]], pos_arr[idx[b]]
                if (ps == 0) != (pb == 0):
                    continue
                newc = counts.copy()
                newc[ps] -= 1
                newc[pb] += 1
                if not _valid(newc[None, :])[0]:
                    continue
                g = e[b] - e[s_]
                if best is None or g > best[0]:
                    best = (g, s_, b)
        if best is None or best[0] <= margin:
            break
        g, s_, b = best
        swaps.append((s_, b, g))
        cur_xi[cur_xi.index(s_)] = b
        pool.remove(b)
        counts[pos_arr[idx[s_]]] -= 1
        counts[pos_arr[idx[b]]] += 1

    cap_only = best_cap(xi, captain)
    cap_both = best_cap(cur_xi, captain)
    plans = {
        "passive": total(xi, captain) + cover,
        "captain only": total(xi, cap_only) if cap_only != captain else -1e9,
        "subs only": total(cur_xi, captain) if swaps else -1e9,
        "both": total(cur_xi, cap_both) if swaps and cap_both != captain else -1e9,
    }
    choice = max(plans, key=lambda k: plans[k])
    if choice != "passive" and plans[choice] - plans["passive"] <= margin:
        choice = "passive"
    actions = []
    x_c = e[captain]
    if choice in ("captain only", "both"):
        k = cap_only if choice == "captain only" else cap_both
        actions.append(dict(kind="captain", do=True, frm=captain, to=k, gain=round(e[k] - x_c, 2),
                            text=f"Move the armband: {names[captain]} scored {x_c:.0f}, {names[k]} expects {e[k]:.1f}."))
    if choice in ("subs only", "both"):
        for s_, b, g in swaps:
            was = f"{realised.get(s_, 0):.0f} pts" if s_ not in later else f"expects {e[s_]:.1f}"
            actions.append(dict(kind="sub", do=True, out=s_, into=b, gain=round(g, 2),
                                text=f"Sub {names[s_]} ({was}) for {names[b]} (expects {e[b]:.1f})."))
    if choice == "passive":
        actions.append(dict(kind="hold", do=False,
                            text=f"No change: keeping auto-sub cover ({cover:.2f} pts) is worth more."))
    for p in sorted(sent_off & set(xi)):
        actions.append(dict(kind="info", do=False, text=f"{names[p]} was sent off, so he cannot be substituted."))
    return dict(actions=actions, cover=round(cover, 2), choice=choice,
                plans={k: round(v, 2) for k, v in plans.items() if v > -1e8})


def best_captain(sim: SimResult, players: pd.DataFrame, squad: list[str], xi: list[str], bench: list[str], md: int,
                 ev: pd.Series, k: int = 4) -> tuple[str, PolicyReport, list[dict]]:
    """Choose the captain by in-matchday policy value (includes the switch option), not by mean alone."""
    cands = sorted(xi, key=lambda p: -float(ev.get(p, 0.0)))[:k]
    best_c, best_rep, table = None, None, []
    for c in cands:
        rep = evaluate(sim, players, squad, xi, c, bench, md)
        table.append(dict(id=c, ev_passive=round(rep.ev_passive, 2), ev_policy=round(rep.ev_active, 2)))
        if best_rep is None or rep.ev_active > best_rep.ev_active:
            best_c, best_rep = c, rep
    return best_c, best_rep, table
