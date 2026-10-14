"""Hierarchical, time-decayed Poisson team model (Dixon–Coles family).

log λ_home = μ + h_g + a_home − d_away        (h_g = home advantage of the competition group)
log λ_away = μ + a_away − d_home

Every club has attack a and defence d on one cross-league scale. Clubs are shrunk toward the
mean of their league group, and league means are shrunk toward 0. Domestic matches identify
differences inside a league; UEFA matches identify the gaps between leagues. That is what
lets a Norwegian or Azerbaijani club with almost no data get a sensible rating.

Fitting is penalised maximum likelihood (MAP) with analytic gradients (L-BFGS); a full refit
on ~11k matches takes a few seconds on a laptop CPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from ..config import team_lookup

EURO_COMPS = {"cl", "el", "conf", "ucl_current"}


def _group_of_team(team_id: str, tier: int) -> str:
    tl = team_lookup()
    country = tl[team_id]["country"] if team_id in tl else team_id.split(":")[0]
    return f"{country}-{max(tier, 1)}"


@dataclass
class TeamModel:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    mu: float
    home_adv: dict[str, float]
    groups: dict[str, str]                 # team -> league group
    group_att: dict[str, float]
    group_def: dict[str, float]
    rho: float
    sd_attack: np.ndarray
    sd_defence: np.ndarray
    fitted_through: str
    n_matches: int
    hyper: dict = field(default_factory=dict)
    home_att: dict = field(default_factory=dict)    # team -> extra log-goal rate scored at home
    home_def: dict = field(default_factory=dict)    # team -> extra log-goal rate prevented at home

    def __post_init__(self):
        self.index = {t: i for i, t in enumerate(self.teams)}

    # ------------------------------------------------------------------ prediction
    def _ad(self, team: str) -> tuple[float, float, float, float]:
        if team in self.index:
            i = self.index[team]
            return self.attack[i], self.defence[i], self.sd_attack[i], self.sd_defence[i]
        g = _group_of_team(team, 1)
        sd = self.hyper.get("prior_sd_team", 0.35)
        return self.group_att.get(g, 0.0), self.group_def.get(g, 0.0), sd, sd

    def lambdas(self, home: str, away: str, neutral: bool = False, comp_group: str = "EUR") -> tuple[float, float]:
        ah, dh, _, _ = self._ad(home)
        aa, da, _, _ = self._ad(away)
        h = 0.0 if neutral else self.home_adv.get(comp_group, self.home_adv.get("EUR", 0.25))
        xa = 0.0 if neutral else float(self.home_att.get(home, 0.0))
        xd = 0.0 if neutral else float(self.home_def.get(home, 0.0))
        return float(np.exp(self.mu + h + xa + ah - da)), float(np.exp(self.mu + aa - dh - xd))

    def log_sd(self, home: str, away: str, scale: float = 1.0) -> tuple[float, float]:
        """Posterior sd of log(home goal rate) and log(away goal rate) (diagonal approximation)."""
        _, _, sah, sdh = self._ad(home)
        _, _, saa, sda = self._ad(away)
        return float(scale * np.hypot(sah, sda)), float(scale * np.hypot(saa, sdh))

    def rating(self, team: str) -> float:
        """Single strength number: expected goal difference vs an average opponent (neutral)."""
        a, d, _, _ = self._ad(team)
        return float(np.exp(self.mu + a) - np.exp(self.mu - d))

    def uncertainty(self, team: str) -> float:
        _, _, sa, sd = self._ad(team)
        return float(np.sqrt(sa ** 2 + sd ** 2))

    # ------------------------------------------------------------------ persistence
    def to_json(self, path: Path) -> None:
        payload = dict(teams=self.teams, attack=self.attack.tolist(), defence=self.defence.tolist(), mu=self.mu,
                       home_adv=self.home_adv, groups=self.groups, group_att=self.group_att, group_def=self.group_def,
                       rho=self.rho, sd_attack=self.sd_attack.tolist(), sd_defence=self.sd_defence.tolist(),
                       fitted_through=self.fitted_through, n_matches=self.n_matches, hyper=self.hyper,
                       home_att=self.home_att, home_def=self.home_def)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "TeamModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(teams=d["teams"], attack=np.array(d["attack"]), defence=np.array(d["defence"]), mu=d["mu"],
                   home_adv=d["home_adv"], groups=d["groups"], group_att=d["group_att"], group_def=d["group_def"],
                   rho=d["rho"], sd_attack=np.array(d["sd_attack"]), sd_defence=np.array(d["sd_defence"]),
                   fitted_through=d["fitted_through"], n_matches=d["n_matches"], hyper=d.get("hyper", {}),
                   home_att=d.get("home_att", {}), home_def=d.get("home_def", {}))


def match_weights(df: pd.DataFrame, as_of: pd.Timestamp, half_life_days: float, european_weight: float,
                  current_ucl_weight: float, form_mix: float = 0.0, form_half_life: float = 60.0) -> np.ndarray:
    """Exponential time decay; with form_mix > 0 a second, short half-life component puts extra
    weight on the last few weeks (recent form) while the long component keeps the class."""
    age = (as_of - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    w = np.exp(-np.log(2) * age / half_life_days)
    if form_mix > 0:
        w = (1 - form_mix) * w + form_mix * np.exp(-np.log(2) * age / form_half_life)
    comp = df["comp"].to_numpy()
    w = np.where(np.isin(comp, ["cl", "el", "conf"]), w * european_weight, w)
    w = np.where(comp == "ucl_current", w * current_ucl_weight, w)
    tier = df["tier"].to_numpy()
    w = np.where(tier == 2, w * 0.6, w)     # second tiers mainly anchor promoted clubs
    return w


def fit(df: pd.DataFrame, as_of: str | pd.Timestamp | None = None, *, half_life_days: float = 300,
        european_weight: float = 1.6, current_ucl_weight: float = 2.0, prior_sd_team: float = 0.35,
        prior_sd_league: float = 0.45, fit_rho: bool = True, min_weight: float = 1e-3,
        home_sd_team: float = 0.0, form_mix: float = 0.0, form_half_life: float = 60.0) -> TeamModel:
    """Fit on all matches strictly before `as_of` (walk-forward safe).

    home_sd_team > 0 adds club-specific home effects (extra goals scored and extra goals prevented
    at home, each ~ N(0, home_sd_team^2) around the league's home advantage): the home/away record.
    form_mix > 0 adds a short-term recency component to the match weights (recent form)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max() + pd.Timedelta(days=1)
    d = df[df["date"] < as_of].copy()
    w = match_weights(d, as_of, half_life_days, european_weight, current_ucl_weight, form_mix, form_half_life)
    keep = w > min_weight
    d, w = d[keep].reset_index(drop=True), w[keep]

    # team -> group (league of most-weighted appearances)
    tiers = d["tier"].to_numpy()
    g_rows = pd.DataFrame({"team": np.concatenate([d.home_id, d.away_id]),
                           "tier": np.concatenate([tiers, tiers]), "w": np.concatenate([w, w])})
    g_rows["group"] = [_group_of_team(t, tr) for t, tr in zip(g_rows.team, g_rows.tier)]
    dom = g_rows[g_rows.tier > 0]
    groups = dom.groupby(["team", "group"]).w.sum().reset_index().sort_values("w").drop_duplicates("team", keep="last")
    team_group = dict(zip(groups.team, groups.group))
    for t in g_rows.team.unique():
        team_group.setdefault(t, _group_of_team(t, 1))

    teams = sorted(team_group)
    tix = {t: i for i, t in enumerate(teams)}
    group_names = sorted(set(team_group.values()))
    gix = {g: i for i, g in enumerate(group_names)}
    team_g = np.array([gix[team_group[t]] for t in teams])

    # home-advantage groups: domestic league group, or EUR for UEFA comps
    hgroups = ["EUR"] + group_names
    hix = {g: i for i, g in enumerate(hgroups)}
    comp = d["comp"].to_numpy()
    hg_idx = np.array([hix["EUR"] if c in EURO_COMPS else hix[_group_of_team(h, t)]
                       for c, h, t in zip(comp, d.home_id, tiers)])
    neutral = d["neutral"].to_numpy(dtype=bool)
    hi = d.home_id.map(tix).to_numpy()
    ai = d.away_id.map(tix).to_numpy()
    yh = d.hg.to_numpy(dtype=float)
    ya = d.ag.to_numpy(dtype=float)

    T, G, H = len(teams), len(group_names), len(hgroups)
    # theta = [mu | h(H) | a(T) | d(T) | A(G) | D(G) | xa(T) | xd(T)]   (xa, xd only if home_sd_team > 0)
    o_h, o_a, o_d = 1, 1 + H, 1 + H + T
    o_A, o_D = 1 + H + 2 * T, 1 + H + 2 * T + G
    use_home = home_sd_team > 0
    o_xa, o_xd = 1 + H + 2 * T + 2 * G, 1 + H + 3 * T + 2 * G
    n = 1 + H + 2 * T + 2 * G + (2 * T if use_home else 0)
    sh2 = home_sd_team ** 2 if use_home else 1.0
    st2, sl2 = prior_sd_team ** 2, prior_sd_league ** 2
    h_prior_mu, h_prior_sd2 = 0.25, 0.15 ** 2
    not_neutral = (~neutral).astype(float)

    def nll(theta: np.ndarray):
        mu = theta[0]
        h = theta[o_h:o_h + H]
        a = theta[o_a:o_a + T]
        dd = theta[o_d:o_d + T]
        A = theta[o_A:o_A + G]
        D = theta[o_D:o_D + G]
        eh = mu + h[hg_idx] * not_neutral + a[hi] - dd[ai]
        ea = mu + a[ai] - dd[hi]
        if use_home:
            xa_, xd_ = theta[o_xa:o_xa + T], theta[o_xd:o_xd + T]
            eh = eh + xa_[hi] * not_neutral
            ea = ea - xd_[hi] * not_neutral
        lh, la = np.exp(eh), np.exp(ea)
        f = -np.sum(w * (yh * eh - lh + ya * ea - la))
        rh, ra = w * (lh - yh), w * (la - ya)
        g = np.zeros(n)
        if use_home:
            f += 0.5 * (np.sum(xa_ ** 2) + np.sum(xd_ ** 2)) / sh2
            g[o_xa:o_xa + T] = np.bincount(hi, weights=rh * not_neutral, minlength=T) + xa_ / sh2
            g[o_xd:o_xd + T] = -np.bincount(hi, weights=ra * not_neutral, minlength=T) + xd_ / sh2
        g[0] = rh.sum() + ra.sum()
        g[o_h:o_h + H] = np.bincount(hg_idx, weights=rh * not_neutral, minlength=H)
        g[o_a:o_a + T] = np.bincount(hi, weights=rh, minlength=T) + np.bincount(ai, weights=ra, minlength=T)
        g[o_d:o_d + T] = -np.bincount(ai, weights=rh, minlength=T) - np.bincount(hi, weights=ra, minlength=T)
        # priors
        da_ = a - A[team_g]
        dd_ = dd - D[team_g]
        f += 0.5 * (np.sum(da_ ** 2) + np.sum(dd_ ** 2)) / st2 + 0.5 * (np.sum(A ** 2) + np.sum(D ** 2)) / sl2
        f += 0.5 * np.sum((h - h_prior_mu) ** 2) / h_prior_sd2
        g[o_a:o_a + T] += da_ / st2
        g[o_d:o_d + T] += dd_ / st2
        g[o_A:o_A + G] += -np.bincount(team_g, weights=da_, minlength=G) / st2 + A / sl2
        g[o_D:o_D + G] += -np.bincount(team_g, weights=dd_, minlength=G) / st2 + D / sl2
        g[o_h:o_h + H] += (h - h_prior_mu) / h_prior_sd2
        return f, g

    theta0 = np.zeros(n)
    theta0[0] = np.log(max((yh.mean() + ya.mean()) / 2, 0.5))
    theta0[o_h:o_h + H] = h_prior_mu
    res = minimize(nll, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "maxfun": 6000})
    th = res.x
    mu = float(th[0])
    h = th[o_h:o_h + H]
    a = th[o_a:o_a + T]
    dd = th[o_d:o_d + T]
    A = th[o_A:o_A + G]
    D = th[o_D:o_D + G]

    # diagonal Fisher-information approximation of posterior sd per team
    eh = mu + h[hg_idx] * not_neutral + a[hi] - dd[ai]
    ea = mu + a[ai] - dd[hi]
    xa_fit = th[o_xa:o_xa + T] if use_home else np.zeros(T)
    xd_fit = th[o_xd:o_xd + T] if use_home else np.zeros(T)
    eh = eh + xa_fit[hi] * not_neutral
    ea = ea - xd_fit[hi] * not_neutral
    lh, la = np.exp(eh), np.exp(ea)
    info_a = np.bincount(hi, weights=w * lh, minlength=T) + np.bincount(ai, weights=w * la, minlength=T) + 1 / st2
    info_d = np.bincount(ai, weights=w * lh, minlength=T) + np.bincount(hi, weights=w * la, minlength=T) + 1 / st2
    sd_a, sd_d = 1 / np.sqrt(info_a), 1 / np.sqrt(info_d)

    rho = -0.06
    if fit_rho:
        rho = _fit_rho(yh.astype(int), ya.astype(int), lh, la, w)

    return TeamModel(teams=teams, attack=a, defence=dd, mu=mu,
                     home_adv={g: float(h[i]) for g, i in hix.items()}, groups=team_group,
                     group_att={g: float(A[i]) for g, i in gix.items()}, group_def={g: float(D[i]) for g, i in gix.items()},
                     rho=float(rho), sd_attack=sd_a, sd_defence=sd_d, fitted_through=str((as_of - pd.Timedelta(days=1)).date()),
                     n_matches=int(len(d)),
                     hyper=dict(half_life_days=half_life_days, european_weight=european_weight,
                                current_ucl_weight=current_ucl_weight, prior_sd_team=prior_sd_team,
                                prior_sd_league=prior_sd_league, home_sd_team=home_sd_team, form_mix=form_mix,
                                form_half_life=form_half_life, converged=bool(res.success),
                                fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds")),
                     home_att={t: float(xa_fit[i]) for t, i in tix.items()} if use_home else {},
                     home_def={t: float(xd_fit[i]) for t, i in tix.items()} if use_home else {})


def _dc_tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    tau = np.ones_like(lh)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1 - lh[m00] * la[m00] * rho
    tau[m01] = 1 + lh[m01] * rho
    tau[m10] = 1 + la[m10] * rho
    tau[m11] = 1 - rho
    return np.clip(tau, 1e-9, None)


def _fit_rho(x, y, lh, la, w) -> float:
    base = poisson.logpmf(x, lh) + poisson.logpmf(y, la)

    def obj(r):
        return -np.sum(w * (base + np.log(_dc_tau(x, y, lh, la, r))))

    out = minimize_scalar(obj, bounds=(-0.25, 0.15), method="bounded")
    return float(out.x)
