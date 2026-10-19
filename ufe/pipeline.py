"""End-to-end engine: data -> team model -> fixture probabilities -> player rates -> joint
simulation -> optimisation. Every step reads overrides from data/processed when present:

  odds_md{N}.csv        bookmaker 1X2 (+ over 2.5) prices -> market-implied goal rates, blended in
  availability.csv      news / injury layer: player id, p_available, reason, source, updated
  lineups_md{N}.csv     confirmed (or predicted) line-ups: player id, role (start | bench | out)
  players_feed.csv      official UCL Fantasy player list (replaces the bundled demo pool)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, rules, settings, team_lookup
from .ingest import fpl as fpl_mod
from .ingest import team_news
from .ingest.openfootball import add_current_ucl, load_matches
from .models import player_rates
from .models.scoreline import blend_lambdas, implied_lambdas, match_probs, remove_margin
from .models.simulate import SimResult, simulate_matchday, summarise_players
from .models.team_ratings import TeamModel, fit
from .models.tournament import simulate_league_phase
from .models.player_rates import slug
from .timeutil import IST, day_index, kickoff_dt

SEED = ROOT / "data" / "seed"
PROC = ROOT / "data" / "processed"


def load_fixtures() -> pd.DataFrame:
    synced = PROC / "fixtures.csv"
    f = pd.read_csv(synced if synced.exists() else SEED / "fixtures_2026_27.csv")
    f["kickoff_cet"] = f["kickoff_cet"].fillna("21:00").replace("", "21:00")
    f["fixture_id"] = [f"MD{m}-{h}-{a}" for m, h, a in zip(f.md, f.home, f.away)]
    return f


def base_pool() -> pd.DataFrame:
    """Players with published prices: the bundled list plus extra verified prices."""
    p = pd.read_csv(SEED / "players_2026_27.csv")
    extra = SEED / "prices_extra_2026_27.csv"
    if extra.exists():
        p = pd.concat([p, pd.read_csv(extra)], ignore_index=True)
    p["price_status"] = "verified"
    p["note"] = p["note"].fillna("") if "note" in p else ""
    p["id"] = [f"{t}-{slug(n)}" for t, n in zip(p.team, p.name)]
    return p.drop_duplicates(subset=["id"])


OFFICIAL = SEED / "official_squads_2026_27.csv"
# official availability flags -> start-probability multipliers when no fresher team news exists.
# 'Suspended' on the mirror also covers unregistered players and stale one-match bans, so it is
# discounted rather than zeroed; journalists' predicted XIs (team news) override all of these.
STATUS_MULT = {"Available": 1.0, "Doubtful": 0.65, "Injured": 0.15, "Suspended": 0.4}


def legacy_pool() -> pd.DataFrame:
    """Article prices plus players added from team news (before the full price list existed)."""
    p = base_pool()
    added = SEED / "players_added_2026_27.csv"
    if added.exists():
        a = pd.read_csv(added)
        a["note"] = a["note"].fillna("")
        a["id"] = [f"{t}-{slug(n)}" for t, n in zip(a.team, a.name)]
        p = pd.concat([p, a[~a.id.isin(p.id)]], ignore_index=True)
    return p


def load_pool() -> pd.DataFrame:
    """Every player the game lists, with official prices.

    1. data/seed/official_squads_2026_27.csv (every club page of the game's price list,
       reconciled with the article pool by scripts/import_official_squads.py), plus the article
       players that page did not show (kept with their article price or flagged 'unknown'),
       or the article pool alone when that file is missing;
    2. then data/processed/players_feed.csv (the game's own feed, `ufe sync --feed`) laid over it
       when it looks complete; a partial or malformed feed is ignored with a printed note."""
    return _overlay_feed(_official_pool())


FEED_MIN_PLAYERS, FEED_MIN_TEAMS = 300, 30


def _read_feed() -> pd.DataFrame | None:
    """The synced game feed, or None when it is missing or does not look like the full player list
    (a partial or wrong-matchday response must never replace the pool)."""
    feed = PROC / "players_feed.csv"
    if not feed.exists():
        return None
    try:
        f = pd.read_csv(feed)
    except Exception as e:                                   # empty or truncated file
        print(f"players_feed.csv ignored: {e}")
        return None
    need = {"name", "team", "pos", "price"}
    if not need <= set(f.columns) or len(f) < FEED_MIN_PLAYERS or f.team.nunique() < FEED_MIN_TEAMS:
        print(f"players_feed.csv ignored: {len(f)} rows, {f.team.nunique() if 'team' in f else 0} clubs, "
              f"columns {sorted(f.columns)}")
        return None
    return f[f.price.notna() & f.team.notna() & f.name.notna()]


def _initial_surname(a: str, b: str) -> float:
    """90 when two spellings share the surname and the first initial ("J. Bellingham" / "Jude
    Bellingham"), or one is the surname alone and it is the other's surname; else 0."""
    from .names import norm
    x, y = norm(a).replace(".", " ").split(), norm(b).replace(".", " ").split()
    if not x or not y or x[-1] != y[-1] or len(x[-1]) < 3:
        return 0.0
    if len(x) == 1 or len(y) == 1:
        return 90.0
    return 90.0 if x[0][0] == y[0][0] else 0.0


def _overlay_feed(pool: pd.DataFrame) -> pd.DataFrame:
    """Official feed prices, positions, ownership and status laid over the pool by club + name,
    keeping the pool's player ids (every model, news file and manager file keys on them).
    Feed players the pool lacks are added with ids built the same way."""
    f = _read_feed()
    if f is None:
        return pool
    pool = pool.copy()
    ids, near = {}, set()
    short = f["short"] if "short" in f else f["name"]
    for t, grp in pool.groupby("team"):
        cand = list(zip(grp.id, grp.name))
        for i in f.index[f.team == t]:
            names = {str(f.at[i, "name"]), str(short[i])}
            full = str(f.at[i, "name"])
            sc = [(max(max(team_news._score(n, str(c[1])) for n in names), _initial_surname(full, str(c[1]))), c[0])
                  for c in cand]
            if sum(_initial_surname(full, str(c[1])) > 0 for c in cand) > 1:   # two players share it
                sc = [(max(team_news._score(n, str(c[1])) for n in names), c[0]) for c in cand]
            s, best = max(sc, default=(0, None))
            if s >= 86:
                ids[i] = best
            elif s >= 70:
                near.add(i)                      # probably a pool player under another spelling
    f = f.assign(id=[ids.get(i) for i in f.index])
    hit = f[f.id.notna()].drop_duplicates("id").set_index("id")
    on = pool.id.isin(hit.index)
    pool.loc[on, "price"] = pool.loc[on, "id"].map(hit.price).values
    pool.loc[on, "pos"] = pool.loc[on, "id"].map(hit.pos).values
    if "selected" in hit:
        pool.loc[on, "own_pct"] = pd.to_numeric(pool.loc[on, "id"].map(hit.selected), errors="coerce").values
    pool.loc[on, "price_status"] = "verified"
    pool.loc[on, "source"] = "ucl_feed"
    new = f[f.id.isna() & ~f.index.isin(near)].copy()
    if len(new):
        new["id"] = [f"{t}-{slug(n)}" for t, n in zip(new.team, new.name)]
        new = new[~new.id.isin(pool.id)].assign(price_status="verified", source="ucl_feed", note="")
        pool = pd.concat([pool, new[[c for c in pool.columns if c in new.columns]]], ignore_index=True)
    return pool.drop_duplicates(subset=["id"]).reset_index(drop=True)


def _official_pool() -> pd.DataFrame:
    p = legacy_pool()
    if not OFFICIAL.exists():
        return p
    o = pd.read_csv(OFFICIAL)
    o = o[o.price.notna()].copy()
    legacy = p.drop_duplicates("id").set_index("id")
    moved = o["moved_from"] if "moved_from" in o else pd.Series(index=o.index, dtype=object)
    o["note"] = [legacy.note.get(m if isinstance(m, str) else i, "") for i, m in zip(o.id, moved)]
    o["note"] = o["note"].fillna("")
    o["price_status"] = "official"
    keep = p[~p.id.isin(o.id) & ~p.id.isin(moved.dropna())]
    cols = ["id", "name", "team", "pos", "price", "source", "as_of", "note", "price_status", "status", "own_pct",
            "pts_to_date", "start_pct", "g90", "a90", "sot90", "kp90", "mins_sample"]
    o = o[[c for c in cols if c in o]]
    out = pd.concat([o, keep], ignore_index=True)
    out["status"] = out["status"].fillna("") if "status" in out else ""
    return out.drop_duplicates(subset=["id"]).reset_index(drop=True)


@dataclass
class Engine:
    as_of: str = "2026-10-13"
    n_sims: int | None = None
    seed: int | None = None
    _sims: dict = field(default_factory=dict)
    _news: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- data + models
    @cached_property
    def cfg(self) -> dict:
        return settings().raw

    @cached_property
    def fixtures(self) -> pd.DataFrame:
        return load_fixtures()

    @cached_property
    def matches(self) -> pd.DataFrame:
        return add_current_ucl(load_matches(), self.fixtures)

    @cached_property
    def team_model(self) -> TeamModel:
        tm = self.cfg["team_model"]
        return fit(self.matches, as_of=self.as_of, half_life_days=tm["half_life_days"],
                   european_weight=tm["european_weight"], current_ucl_weight=tm["current_ucl_weight"],
                   prior_sd_team=tm["prior_sd_team"], prior_sd_league=tm["prior_sd_league"],
                   form_mix=float(tm.get("form_mix", 0.0)), form_half_life=float(tm.get("form_half_life", 60.0)),
                   home_sd_team=float(tm.get("home_sd_team", 0.0)))

    @cached_property
    def fpl(self) -> fpl_mod.FPLData:
        return fpl_mod.load()

    @cached_property
    def pool(self) -> pd.DataFrame:
        return load_pool()

    @cached_property
    def domestic_gpm(self) -> dict[str, float]:
        """Each UCL club's expected goals per domestic league match (team model vs its league
        rivals, home and away averaged): the denominator for per-90 goal shares."""
        m, tm = self.matches, self.team_model
        cups = {"cl", "el", "conf", "ucl_current"}
        out = {}
        for t in team_lookup():
            rows = m[((m.home_id == t) | (m.away_id == t)) & ~m.comp.isin(cups)]
            if rows.empty:
                continue
            latest = rows.season.max()
            comp = rows[rows.season == latest].comp.mode().iloc[0]
            lg = m[(m.comp == comp) & (m.season == latest)]
            rivals = (set(lg.home_id) | set(lg.away_id)) - {t}
            grp = tm.groups.get(t, "EUR")
            lam = [(tm.lambdas(t, r, comp_group=grp)[0] + tm.lambdas(r, t, comp_group=grp)[1]) / 2
                   for r in rivals if r in tm.index]
            if lam:
                out[t] = float(np.mean(lam))
        return out

    @cached_property
    def player_tables(self) -> tuple[pd.DataFrame, dict]:
        strength = {t: self.team_model.rating(t) for t in team_lookup()}
        use_form = bool((self.cfg.get("player_form") or {}).get("enabled", True))
        players, others = player_rates.build(self.pool, self.fpl, team_strength=strength, team_gpm=self.domestic_gpm,
                                             use_current_form=use_form)
        extra = [c for c in ("status", "own_pct", "pts_to_date", "start_pct", "g90", "a90") if c in self.pool]
        if extra:
            players = players.merge(self.pool[["id"] + extra], on="id", how="left")
            mult = players["status"].map(STATUS_MULT).fillna(1.0) if "status" in players else 1.0
            players["p_start"] = players.p_start * mult
            players["p_sub"] = players.p_sub * mult
        av = PROC / "availability.csv"
        if av.exists():
            a = pd.read_csv(av).set_index("id")
            m = players.id.isin(a.index)
            players.loc[m, "p_start"] = players.loc[m, "p_start"] * players.loc[m, "id"].map(a.p_available).clip(0, 1)
            players.loc[m, "p_sub"] = players.loc[m, "p_sub"] * players.loc[m, "id"].map(a.p_available).clip(0, 1)
        return players, others

    @property
    def players(self) -> pd.DataFrame:
        return self.player_tables[0]

    @property
    def others(self) -> dict:
        return self.player_tables[1]

    # ---------------------------------------------------------------- fixtures with probabilities
    def fixture_table(self, md: int) -> pd.DataFrame:
        f = self.fixtures[self.fixtures.md == md].copy()
        di = day_index(f.date.tolist())
        f["day"] = f.date.map(di)
        f["kickoff_ist"] = [kickoff_dt(d, k).astimezone(IST).strftime("%a %d %b %H:%M")
                            for d, k in zip(f.date, f.kickoff_cet)]
        odds = self._odds(md)
        lam, src = [], []
        w = self.cfg["market"]["blend_weight"]
        for r in f.itertuples():
            model = self.team_model.lambdas(r.home, r.away)
            market = None
            if odds is not None and r.fixture_id in odds.index:
                o = odds.loc[r.fixture_id]
                ph, pdr, pa = remove_margin([o.home_odds, o.draw_odds, o.away_odds], self.cfg["market"]["margin_method"])
                p_over = None
                if "over25_odds" in o and pd.notna(o.get("over25_odds")) and pd.notna(o.get("under25_odds")):
                    p_over = float(remove_margin([o.over25_odds, o.under25_odds], "proportional")[0])
                market = implied_lambdas(ph, pdr, pa, p_over, rho=self.team_model.rho)
            lam.append(blend_lambdas(model, market, w))
            src.append("model+market" if market else "model")
        f["lh"], f["la"] = zip(*lam) if lam else ([], [])
        f["rho"] = self.team_model.rho
        f["lambda_source"] = src
        k = float(self.cfg["team_model"].get("predictive_sd_scale", 1.0))
        sds = [self.team_model.log_sd(r.home, r.away, k) for r in f.itertuples()]
        shrink = [(1 - w) if s_ == "model+market" else 1.0 for s_ in src]     # market prices carry their own
        f["sh"] = [a * c for (a, _), c in zip(sds, shrink)] if sds else []
        f["sa"] = [b * c for (_, b), c in zip(sds, shrink)] if sds else []
        f = self._lineup_adjust(md, f)
        return f.reset_index(drop=True)

    def _odds(self, md: int) -> pd.DataFrame | None:
        p = PROC / f"odds_md{md}.csv"
        return pd.read_csv(p).set_index("fixture_id") if p.exists() else None

    def _lineup_adjust(self, md: int, f: pd.DataFrame) -> pd.DataFrame:
        """Scale each side's goal rate by who is expected to play (team news / confirmed XIs).

        Attacking weight per player w = xG share + 0.6 * xA share. The change in expected weight on
        the pitch, dS = sum((p_start_prior - p_start_now) * w), lowers the rate by `strength_adjust`
        per unit (replacements recover the rest), clipped to `strength_clip`. Without news the
        prior and current tables are identical and nothing changes."""
        cfg = self.cfg.get("news") or {}
        k = float(cfg.get("strength_adjust", 0.5))
        lo, hi = cfg.get("strength_clip", [0.88, 1.06])
        prior = self.players.set_index("id")
        now = self.players_for(md).set_index("id")
        o0, o1 = self.others, self.others_for(md)
        w = prior.xg_share.fillna(0) + 0.6 * prior.xa_share.fillna(0)
        s0 = (prior.p_start * w).groupby(prior.team).sum()
        s1 = (now.p_start.reindex(prior.index).fillna(prior.p_start) * w).groupby(prior.team).sum()
        ph = lambda o, t: float(o.get(t, {}).get("others_xg", 0.0)) + 0.6 * float(o.get(t, {}).get("others_xa", 0.0))
        mult = {}
        for t in s0.index:
            S0, S1 = float(s0[t]) + ph(o0, t), float(s1[t]) + ph(o1, t)
            mult[t] = float(np.clip(1 - k * (S0 - S1) / max(S0, 1e-6), lo, hi))
        mult = pd.Series(mult)
        f = f.copy()
        f["att_mult_home"] = [float(mult.get(t, 1.0)) for t in f.home]
        f["att_mult_away"] = [float(mult.get(t, 1.0)) for t in f.away]
        f["lh"] = f.lh * f.att_mult_home
        f["la"] = f.la * f.att_mult_away
        return f

    def match_probabilities(self, md: int) -> list[dict]:
        out = []
        for r in self.fixture_table(md).itertuples():
            mp = match_probs(r.lh, r.la, r.rho, sh=r.sh, sa=r.sa)
            d = mp.to_dict()
            d.update(fixture_id=r.fixture_id, md=md, home=r.home, away=r.away, date=r.date, kickoff_cet=r.kickoff_cet,
                     kickoff_ist=r.kickoff_ist, day=int(r.day), status=r.status, lambda_source=r.lambda_source,
                     home_goals=None if pd.isna(r.home_goals) else int(r.home_goals),
                     away_goals=None if pd.isna(r.away_goals) else int(r.away_goals))
            out.append(d)
        return out

    # ---------------------------------------------------------------- simulation
    def players_for(self, md: int) -> pd.DataFrame:
        """Player table with this matchday's confirmed/predicted line-ups applied (if provided)."""
        players = self.players.copy()
        # 1) this matchday's predicted line-ups + injuries (Bayes update at full strength)
        news = team_news.load_for(md, players)
        logs, covered = [], set()
        if news is not None:
            k = float(news.attrs.get("strength", 1.0))
            cfg = None if k >= 0.999 else {**team_news.softened(team_news._cfg(), k), "_carried": True}
            players, log = team_news.apply(players, news, cfg)
            logs.append(log)
            covered = set(news.team)
        # 2) clubs without news for this matchday reuse the latest earlier news, softened
        hit = team_news.latest_news_before(md)
        carried_from = None
        if hit is not None:
            old = team_news.match(team_news.parse(hit[1]), players)
            old = old[~old.team.isin(covered)]
            if len(old):
                k = float(team_news._cfg().get("carry_forward", 0.5)) ** max(md - hit[0], 1)
                cfg = {**team_news.softened(team_news._cfg(), k), "_carried": True}
                players, log = team_news.apply(players, old, cfg)
                logs.append(log)
                carried_from = hit[0]
        if logs:
            unmatched = news[news.id.isna()] if news is not None else pd.DataFrame()
            self._news[md] = dict(log=pd.concat(logs, ignore_index=True), unmatched=unmatched, carried_from=carried_from)
        lp = PROC / f"lineups_md{md}.csv"                 # confirmed line-ups override predictions
        if lp.exists():
            lu = pd.read_csv(lp)
            role = dict(zip(lu.id, lu.role))
            teams_with = set(players[players.id.isin(lu.id)].team)
            for i, r in players.iterrows():
                if r.team not in teams_with:
                    continue
                rl = role.get(r.id, "out")
                players.at[i, "p_start"] = 1.0 if rl == "start" else 0.0
                players.at[i, "p_sub"] = r.p_sub if rl == "bench" else 0.0
        return players

    def others_for(self, md: int) -> dict:
        """Unlisted-player block for one matchday. When team news or confirmed XIs name the
        starters, the unlisted players' expected starts shrink to what the named players leave
        uncovered (never below 25% of the prior, which keeps their substitute minutes)."""
        now = self.players_for(md)
        prior = self.players
        n_now = now.groupby("team").p_start.sum()
        n_pri = prior.groupby("team").p_start.sum()
        out = {}
        for team, o in self.others.items():
            o = dict(o)
            k0 = float(o.get("others_on_pitch", 0.0))
            if k0 > 0 and abs(float(n_now.get(team, 0)) - float(n_pri.get(team, 0))) > 1e-6:
                k1 = max(0.0, 11.0 - float(n_now.get(team, 0.0)))
                r = min(max(k1 / k0, 0.25), 1.0)
                o["others_on_pitch"] = k0 * r
                o["others_xg"] = float(o.get("others_xg", 0.1)) * r
                o["others_xa"] = float(o.get("others_xa", 0.1)) * r
            out[team] = o
        return out

    def simulate(self, md: int) -> SimResult:
        if md not in self._sims:
            self._sims[md] = simulate_matchday(self.fixture_table(md), self.players_for(md), self.others_for(md),
                                               n_sims=self.n_sims, seed=(self.seed or self.cfg["simulation"]["seed"]) + md)
        return self._sims[md]

    def projections(self, md: int) -> pd.DataFrame:
        s = summarise_players(self.simulate(md), self.players_for(md))
        s["value"] = s.xpts / s.price
        return s.sort_values("xpts", ascending=False).reset_index(drop=True)

    def ev_matrix(self, mds: list[int]) -> pd.DataFrame:
        cols = {}
        for md in mds:
            p = self.projections(md).set_index("id").xpts
            cols[md] = p
        return pd.DataFrame(cols).reindex(self.players.id).fillna(0.0)

    # ---------------------------------------------------------------- tournament
    def league_table_projection(self, n_sims: int = 20000) -> pd.DataFrame:
        f = self.fixtures[self.fixtures.md <= 8]
        lam = {(r.home, r.away): self.team_model.lambdas(r.home, r.away) for r in f.itertuples()}
        noise = {t: self.team_model.uncertainty(t) * 0.5 for t in team_lookup()}
        return simulate_league_phase(f, lam, self.team_model.rho, n_sims=n_sims, rating_noise=noise)

    def save_team_model(self) -> Path:
        p = PROC / "team_model.json"
        self.team_model.to_json(p)
        return p
