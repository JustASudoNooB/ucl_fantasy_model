"""The neural network applied to Champions League players: the challenger engine.

For a matchday the same features as the Premier League training rows are built for every player
in the pool, from what the engine knows before kick-off:
  position, price (UCL price on the FPL scale; relative and rank inside the club), home/away,
  the engine's expected goals for and against in that fixture, rest days since the club's last
  league match, start history (this season's league starts where known, else the history
  module's start rate, else this season's UCL XIs), the last two UCL XIs, and xG / xA per 90 as
  a share of team xG (history module), with the minutes behind them.
The network's start probability replaces the start prior (injury and status multipliers still
apply, team news and confirmed XIs still update it), and its goal and assist rates per 90 become
shares of the team's expected goals. The XI-history update is skipped, because the network
already sees the last two UCL XIs. Everything downstream (simulation, points, optimiser) is the
current engine, so the two models differ only in the player priors.

`blend` mixes the two: 0 = current model, 1 = network only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluate import grade
from ..models.player_rates import ASSISTS_PER_GOAL, fpl_equivalent_price
from ..pipeline import Engine
from .data import FEATURES, POS
from .train import load_model

CUPS = {"cl", "el", "conf", "ucl_current"}


def _xi_ids(md: int, players: pd.DataFrame) -> tuple[set, set]:
    x = grade.load_xis(md) if md >= 1 else None
    if x is None:
        return set(), set()
    return {i for i in grade._match_names(x, players) if i}, set(x.team)


def features(engine: Engine, md: int, players: pd.DataFrame) -> pd.DataFrame:
    """One row per player in `players` (same order), columns = data.FEATURES."""
    p = players.reset_index(drop=True)
    ft = engine._base_fixture_table(md)
    side = {}
    for r in ft.itertuples():
        side[r.home] = (1.0, float(r.lh), float(r.la), str(r.date)[:10])
        side[r.away] = (0.0, float(r.la), float(r.lh), str(r.date)[:10])
    f = pd.DataFrame(index=p.index)
    for c in POS:
        f[f"pos_{c}"] = (p.pos == c).astype(float)
    price = p.price.astype(float).fillna(4.5)
    f["log_price"] = np.log(fpl_equivalent_price(price))
    tmax = price.groupby(p.team).transform("max")
    f["price_rel"] = price / tmax.clip(lower=0.1)
    f["price_rank"] = price.groupby(p.team).rank(ascending=False, pct=True)
    f["home"] = [side.get(t, (0.5,))[0] for t in p.team]
    f["lam_for"] = [side.get(t, (0, 1.4))[1] for t in p.team]
    f["lam_against"] = [side.get(t, (0, 1.4, 1.4))[2] for t in p.team]
    # rest days since the club's last league match before this one
    m = engine.matches
    dom = m[~m.comp.isin(CUPS)]
    rest = []
    for t in p.team:
        day = side.get(t, (0, 0, 0, None))[3]
        if day is None:
            rest.append(7.0)
            continue
        d = dom[((dom.home_id == t) | (dom.away_id == t)) & (dom.date < pd.Timestamp(day))].date
        rest.append(float(np.clip((pd.Timestamp(day) - d.max()).days, 2, 14)) if len(d) else 7.0)
    f["rest_days"] = rest
    # start history: league starts this season > history start rate > this season's UCL XIs
    s1, t1 = _xi_ids(md - 1, p)
    s2, t2 = _xi_ids(md - 2, p)
    ucl_n, ucl_s = np.zeros(len(p)), np.zeros(len(p))
    for k in range(1, md):
        sk, tk = _xi_ids(k, p)
        on = p.team.isin(tk).to_numpy()
        ucl_n += on
        ucl_s += on & p.id.isin(sk).to_numpy()
    dom_ok = pd.to_numeric(p.get("dom_team_matches"), errors="coerce").fillna(0) >= 3
    dom_rate = pd.to_numeric(p.get("dom_starts"), errors="coerce") / pd.to_numeric(p.get("dom_team_matches"), errors="coerce")
    obs_ok = pd.to_numeric(p.get("obs_start_n"), errors="coerce").fillna(0) >= 3
    obs_rate = pd.to_numeric(p.get("obs_start_rate"), errors="coerce")
    sr = np.where(dom_ok, dom_rate, np.where(obs_ok & obs_rate.notna(), obs_rate, np.where(ucl_n >= 2, ucl_s / np.maximum(ucl_n, 1), np.nan)))
    f["has_start_hist"] = np.isfinite(sr).astype(float)
    f["start_rate"] = np.nan_to_num(np.clip(sr, 0, 1))
    f["started_last1"] = p.id.isin(s1).astype(float).to_numpy()
    f["started_last2"] = p.id.isin(s2).astype(float).to_numpy()
    # attacking history (per-90 share of team xG, as in training)
    xs = pd.to_numeric(p.get("obs_xg_share"), errors="coerce")
    xa = pd.to_numeric(p.get("obs_xa_share"), errors="coerce")
    n90 = pd.to_numeric(p.get("minutes_hist"), errors="coerce").fillna(0) / 90.0
    ok = xs.notna() & (n90 >= 5)
    f["has_att_hist"] = ok.astype(float)
    f["xg_share"] = np.where(ok, xs.clip(0, 1.5), 0.0)
    f["xa_share"] = np.where(ok, xa.fillna(0).clip(0, 1.5), 0.0)
    f["log_n90"] = np.log1p(n90.clip(upper=114))
    return f[FEATURES].fillna(0.0)


class Challenger(Engine):
    """The engine with learned player priors.

    One model: `blend` is its weight (0 = current model, 1 = the learned model only).
    Several: `nets` = {name: (model, weight)}; the current model keeps the rest of the weight."""
    blend: float = 1.0

    def __init__(self, *a, blend: float = 1.0, net=None, nets: dict | None = None, **k):
        super().__init__(*a, **k)
        self.blend = float(blend)
        self.nets = nets
        if nets is None:
            self.net = net or load_model()
            if self.net is None:
                raise FileNotFoundError("no trained network: run `python -m ufe ml train` first")

    @staticmethod
    def _priors(net, F: pd.DataFrame, players: pd.DataFrame) -> dict:
        """One learned model's start probability and attacking shares for every player."""
        pr = net.predict(F.to_numpy(float))
        lam = np.clip(F.lam_for.to_numpy(float), 0.3, None)
        av = (players["avail"] if "avail" in players else pd.Series(1.0, index=players.index)).to_numpy(float)
        return dict(p_start=np.clip(pr["p_start"], 0, 0.98) * av,
                    xg_share=np.clip(pr["g90"] / lam, 0.0, 0.8),
                    xa_share=np.clip(pr["a90"] / (ASSISTS_PER_GOAL * lam), 0.0, 0.8))

    def _xi_update(self, md: int, players: pd.DataFrame) -> pd.DataFrame:
        cur = Engine._xi_update(self, md, players)          # the current model's priors (with XI history)
        parts = self.nets if self.nets is not None else ({"net": (self.net, self.blend)} if self.blend > 0 else {})
        parts = {k: v for k, v in parts.items() if v[1] > 0}
        if not parts:
            return cur
        F = features(self, md, players)
        w0 = max(1.0 - sum(w for _, w in parts.values()), 0.0)
        out = cur.copy()
        acc = {c: w0 * cur[c].to_numpy(float) for c in ("p_start", "xg_share", "xa_share")}
        for net, w in parts.values():
            pri = self._priors(net, F, players)
            for c in acc:
                acc[c] = acc[c] + w * pri[c]
        for c, v in acc.items():
            out[c] = v
        return out
