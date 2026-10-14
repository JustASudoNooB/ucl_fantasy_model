"""Is the model's lineup just the tipsters' lineup? And how sure is it?

    python scripts/verify_lineup.py            # writes data/processed/verification.json

1. MD2 retro: re-solve MD2 on the full official price list with only pre-deadline information
   (team model fitted before 13 Oct; MD2 team news), and compare with the published XI, Fantasy
   Football Scout's MD2 picks and the most-owned ("template") squad.
2. MD3: best squad from scratch, the template squad scored by the same model, overlap and the
   expected-points gap, and differentials (high expected points, low ownership).
3. Robustness: the MD3 optimum under 3 simulation seeds and under parameter uncertainty
   (team ratings drawn from their posterior sd, player shares and start probabilities jittered),
   reported as how often each player is picked.
4. Transfers for MD3-MD5 from the squad shown on the published dashboard (1-3 free transfers).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ufe.optimize import policy as policy_mod  # noqa: E402
from ufe.optimize import transfers as tr_mod  # noqa: E402
from ufe.optimize.milp import Model  # noqa: E402
from ufe.optimize.squad import best_squad, best_xi, pickable  # noqa: E402
from ufe.pipeline import PROC, Engine  # noqa: E402
from ufe.report import export as ex  # noqa: E402
from ufe.config import rules  # noqa: E402

OUT = PROC / "verification.json"
PUBLISHED_MD2 = dict(  # the squad on the dashboard published 13 Oct 17:44 IST (pre-deadline build)
    xi=["ARS-david-raya", "LIV-virgil-van-dijk", "BAY-dayot-upamecano", "ARS-gabriel-magalhaes", "BAY-michael-olise",
        "BAR-lamine-yamal", "LIV-dominik-szoboszlai", "ARS-bukayo-saka", "BAY-harry-kane", "VFB-deniz-undav",
        "VFB-ermedin-demirovic"],
    bench=["SHA-valeriy-bondar", "SHA-mykola-matviyenko", "RBL-ridle-baku", "AEK-thomas-strakosha"],
    captain="BAY-michael-olise")
FFS_MD2 = ["INT-josep-martinez", "AVL-zion-suzuki", "ARS-gabriel-magalhaes", "BAY-dayot-upamecano", "INT-federico-dimarco",
           "VFB-maximilian-mittelstadt", "SCP-sergi-altimira", "BAR-lamine-yamal", "BAY-michael-olise", "ARS-bukayo-saka",
           "LIV-dominik-szoboszlai", "AEK-razvan-marin", "BAY-harry-kane", "BVB-serhou-guirassy", "VFB-deniz-undav"]


def resolve(ids: list[str], players: pd.DataFrame) -> list[str]:
    """Map ids from older builds onto the current pool (same club + surname)."""
    from ufe.names import norm
    have = set(players.id)
    out = []
    for i in ids:
        if i in have:
            out.append(i)
            continue
        team, rest = i.split("-", 1)
        sur = rest.split("-")[-1]
        cand = players[(players.team == team) & players.name.map(lambda n: norm(n).split()[-1] == sur if n else False)]
        out.append(cand.id.iloc[0] if len(cand) else i)
    return out


def template_squad(players: pd.DataFrame, ev: pd.Series, md: int) -> list[str]:
    """The crowd's squad: the most-owned legal squad among players expected to be available
    (start probability >= 0.3 after team news), ties broken by expected points."""
    pl = pickable(players)
    pl = pl[pl.id.isin(ev.index) & (pl.p_start >= 0.3)].reset_index(drop=True)
    ids = list(pl.id)
    score = pl.own_pct.fillna(0.0).to_numpy() + 0.01 * ev.reindex(ids).fillna(0).to_numpy()
    m = Model()
    x = m.block("x", len(ids))
    m.maximize(x, score)
    pos = pl.pos.to_numpy()
    for p, k in rules()["squad"]["positions"].items():
        m.add_sum(x[np.where(pos == p)[0]], lo=k, hi=k)
    m.add_sum(x, pl.price.to_numpy(), hi=ex.budget(md) + 1e-6)
    for t in pl.team.unique():
        m.add_sum(x[np.where(pl.team.to_numpy() == t)[0]], hi=ex.club_cap(md))
    sol, _ = m.solve(time_limit=60)
    return [ids[i] for i in range(len(ids)) if sol[x[i]] > 0.5]


def score_squad(e: Engine, md: int, squad: list[str], ev: pd.Series) -> dict:
    players = e.players_for(md)
    bx = best_xi(squad, players, ev)
    cap, rep, _ = policy_mod.best_captain(e.simulate(md), players, squad, bx["xi"], bx["bench"], md, ev)
    return dict(squad=squad, xi=bx["xi"], bench=bx["bench"], formation=bx.get("formation"), captain=cap,
                ev_passive=round(rep.ev_passive, 2), ev_active=round(rep.ev_active, 2))


def perturbed_engine(base: Engine, rng: np.random.Generator, md: int, n_sims: int) -> Engine:
    e = copy.copy(base)
    e._sims, e._news = {}, {}
    tm = copy.deepcopy(base.team_model)
    tm.attack = tm.attack + rng.normal(0, 1, len(tm.attack)) * tm.sd_attack
    tm.defence = tm.defence + rng.normal(0, 1, len(tm.defence)) * tm.sd_defence
    players, others = base.player_tables
    p = players.copy()
    w_g = p.w_obs.fillna(0).clip(0, 1).to_numpy()
    w_a = p.get("w_obs_xa", p.w_obs).fillna(0).clip(0, 1).to_numpy()
    p["xg_share"] = p.xg_share * np.exp(rng.normal(0, 1, len(p)) * (0.35 * (1 - w_g) + 0.12))
    p["xa_share"] = p.xa_share * np.exp(rng.normal(0, 1, len(p)) * (0.35 * (1 - w_a) + 0.12))
    z = np.log(p.p_start.clip(0.01, 0.99) / (1 - p.p_start.clip(0.01, 0.99))) + rng.normal(0, 0.4, len(p))
    p["p_start"] = np.where(p.p_start > 0, 1 / (1 + np.exp(-z)), 0.0)
    e.__dict__["team_model"] = tm
    e.__dict__["player_tables"] = (p, others)
    e.n_sims = n_sims
    e.seed = int(rng.integers(1, 10_000))
    return e


def names(players: pd.DataFrame, ids) -> list[str]:
    nm = players.set_index("id").name
    return [str(nm.get(i, i)) for i in ids]


def main() -> None:
    t0 = time.time()
    report = {}
    # ------------------------------------------------------------------ 1. MD2 retro, pre-deadline information
    e2 = Engine(as_of="2026-10-13", n_sims=12000)
    md = 2
    proj2 = e2.projections(md)
    pl2 = e2.players_for(md)
    ev2 = proj2.set_index("id").xpts
    best2 = best_squad(pl2, ev2, ex.budget(md), ex.club_cap(md))
    cap2, rep2, _ = policy_mod.best_captain(e2.simulate(md), pl2, best2.squad, best2.xi, best2.bench, md, ev2)
    pub = dict(PUBLISHED_MD2)
    pub_xi, pub_bench = resolve(pub["xi"], pl2), resolve(pub["bench"], pl2)
    pub_sc = score_squad(e2, md, pub_xi + pub_bench, ev2)
    ffs = resolve(FFS_MD2, pl2)
    tmpl2 = template_squad(pl2, ev2, md)
    tmpl2_sc = score_squad(e2, md, tmpl2, ev2)
    report["md2_retro"] = dict(
        note="Full official price list, information available before the MD2 deadline (13 Oct 18:45 CEST).",
        best=dict(formation=best2.formation, xi=names(pl2, best2.xi), bench=names(pl2, best2.bench), captain=names(pl2, [cap2])[0],
                  ev_passive=round(rep2.ev_passive, 2), ev_active=round(rep2.ev_active, 2), cost=best2.cost),
        published=dict(xi=names(pl2, pub_xi), ev_passive=pub_sc["ev_passive"], ev_active=pub_sc["ev_active"]),
        template=dict(xi=names(pl2, tmpl2_sc["xi"]), ev_passive=tmpl2_sc["ev_passive"], ev_active=tmpl2_sc["ev_active"]),
        overlap=dict(published_vs_ffs_xi=len(set(pub_xi) & set(ffs)), published_vs_template_xi=len(set(pub_xi) & set(tmpl2_sc["xi"])),
                     newbest_vs_published_xi=len(set(best2.xi) & set(pub_xi)), newbest_vs_ffs_xi=len(set(best2.xi) & set(ffs)),
                     newbest_vs_template_xi=len(set(best2.xi) & set(tmpl2_sc["xi"]))))
    print(json.dumps(report["md2_retro"], indent=1), f"\n[{time.time()-t0:.0f}s]", flush=True)

    # ------------------------------------------------------------------ 2. MD3 from scratch vs template
    e = Engine(as_of="2026-10-14", n_sims=12000)
    md = 3
    proj = e.projections(md)
    pl = e.players_for(md)
    ev = proj.set_index("id").xpts
    best = best_squad(pl, ev, ex.budget(md), ex.club_cap(md))
    cap, rep, cap_table = policy_mod.best_captain(e.simulate(md), pl, best.squad, best.xi, best.bench, md, ev)
    tmpl = template_squad(pl, ev, md)
    tsc = score_squad(e, md, tmpl, ev)
    own = pl.set_index("id").own_pct
    diffs = proj.merge(pl[["id", "own_pct"]], on="id", how="left")
    diffs = diffs[(diffs.own_pct.fillna(0) < 5) & diffs.id.isin(pickable(pl).id)].sort_values("xpts", ascending=False).head(10)
    report["md3"] = dict(
        best=dict(formation=best.formation, xi=names(pl, best.xi), bench=names(pl, best.bench), captain=names(pl, [cap])[0],
                  ev_passive=round(rep.ev_passive, 2), ev_active=round(rep.ev_active, 2), cost=best.cost,
                  rules=[r["text"] for r in rep.rules[:3]],
                  own=[None if pd.isna(own.get(i)) else float(own.get(i)) for i in best.xi]),
        template=dict(xi=names(pl, tsc["xi"]), bench=names(pl, tsc["bench"]), captain=names(pl, [tsc["captain"]])[0],
                      ev_passive=tsc["ev_passive"], ev_active=tsc["ev_active"]),
        overlap_xi=len(set(best.xi) & set(tsc["xi"])),
        differentials=[dict(name=r.name, team=r.team, pos=r.pos, price=r.price, xpts=round(r.xpts, 2),
                            own=None if pd.isna(r.own_pct) else float(r.own_pct)) for r in diffs.itertuples()])
    print(json.dumps(report["md3"], indent=1), f"\n[{time.time()-t0:.0f}s]", flush=True)

    # ------------------------------------------------------------------ 3. robustness
    picks: dict[str, int] = {}
    xi_picks: dict[str, int] = {}
    runs = []
    for s in (101, 202, 303):
        es = copy.copy(e)
        es._sims = {}
        es.seed = s
        evs = es.projections(md).set_index("id").xpts
        b = best_squad(pl, evs, ex.budget(md), ex.club_cap(md))
        runs.append(dict(kind="seed", seed=s, ev_xi=round(b.ev_xi, 2), xi_overlap=len(set(b.xi) & set(best.xi))))
    rng = np.random.default_rng(2026)
    R = 10
    for k in range(R):
        ep = perturbed_engine(e, rng, md, n_sims=5000)
        evp = ep.projections(md).set_index("id").xpts
        b = best_squad(ep.players_for(md), evp, ex.budget(md), ex.club_cap(md))
        for i in b.squad:
            picks[i] = picks.get(i, 0) + 1
        for i in b.xi:
            xi_picks[i] = xi_picks.get(i, 0) + 1
        runs.append(dict(kind="uncertainty", k=k, ev_xi=round(b.ev_xi, 2), xi_overlap=len(set(b.xi) & set(best.xi))))
        print(f"  replicate {k}: XI overlap {len(set(b.xi) & set(best.xi))}/11  [{time.time()-t0:.0f}s]", flush=True)
    freq = sorted(((names(pl, [i])[0], c / R, xi_picks.get(i, 0) / R) for i, c in picks.items()), key=lambda x: -x[1])
    report["robustness"] = dict(runs=runs, replicates=R, pick_rate=[dict(name=n, squad=round(a, 2), xi=round(b, 2)) for n, a, b in freq])
    # ------------------------------------------------------------------ 4. transfers from the published squad
    sq = resolve(PUBLISHED_MD2["xi"] + PUBLISHED_MD2["bench"], pl)
    sq = [i for i in sq if i in set(pl.id)]
    mds = [3, 4, 5]
    evm = e.ev_matrix(mds)
    bank = round(100.0 - float(pl.set_index("id").loc[sq].price.sum()), 1)
    tro = []
    for ft in (2, 3):
        for o in tr_mod.options(pl, evm, sq, bank=max(bank, 0.0), free_now=ft, mds=mds, max_k=3):
            tro.append(dict(free_transfers=ft, n=o.n_transfers, buy=names(pl, o.buy), sell=names(pl, o.sell), hits=o.hits,
                            horizon_ev=round(o.horizon_ev, 2), gain_vs_roll=round(o.gain_vs_roll, 2),
                            md3_ev=round(o.ev_by_md.get(3, 0.0), 2)))
            print("  transfers", tro[-1], flush=True)
    report["transfers_from_published"] = dict(squad=names(pl, sq), bank=bank, horizon=mds, options=tro)
    report["generated_at"] = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
