"""The models that predict every matchday, built the same way so they can be compared.

  statistical   the current engine: team model + bookmaker blend, XI history, team news
  network       neural network (ml/nn.py) for each player's start, goals and assists
  bookmaker     match goal rates from bookmaker prices only (compare/market.py)
  trees         gradient-boosted decision trees (ml/trees.py), same inputs as the network
Everything after the player and match inputs (simulation, points, optimiser) is shared, so the
models differ only where their descriptions say.
"""
from __future__ import annotations

from ..config import ROOT
from ..pipeline import Engine

PROC = ROOT / "data" / "processed"
SHARE = ("cfg", "fixtures", "matches", "team_model", "fpl", "pool", "domestic_gpm", "player_tables")

MODELS = {
    "statistical": dict(name="Statistical", kind="hand-built",
                        desc="Poisson model of each club's goals from its matches (time-weighted, league and season "
                             "levels), blended 65/35 with bookmaker prices; player starts from previous UCL XIs and "
                             "team news; each player's share of his club's chances from his history."),
    "network": dict(name="Neural network", kind="learned",
                    desc="A small neural network (two hidden layers, 32 and 16 units) trained on Premier League "
                         "player-matches predicts each player's chance of starting and goals and assists per 90. "
                         "Match goal rates are the statistical model's."),
    "bookmaker": dict(name="Bookmaker", kind="market",
                      desc="Each match's goal rates come from bookmaker prices alone (result odds and the goals line, "
                           "median across bookmakers). Players, starts and chance shares are the statistical model's, "
                           "so this tests whether the team model beats the market."),
    "trees": dict(name="Decision trees", kind="learned",
                  desc="Gradient-boosted decision trees (LightGBM) on the same Premier League rows and inputs as the "
                       "network: a few hundred small trees per output. Match goal rates are the statistical model's."),
}


def available() -> list[str]:
    out = ["statistical", "bookmaker"]
    if (PROC / "ml_model.json").exists():
        out.insert(1, "network")
    if (PROC / "trees_model.json").exists():
        out.append("trees")
    return [k for k in MODELS if k in out]


def build(key: str, as_of: str, n_sims: int | None = None, seed: int | None = None, base: Engine | None = None) -> Engine:
    """An engine for model `key`. With `base`, the slow shared pieces (data, team model, player
    tables) are reused from it instead of being rebuilt."""
    if key == "statistical":
        e = Engine(as_of=as_of, n_sims=n_sims, seed=seed)
    elif key == "bookmaker":
        from .market import MarketEngine
        e = MarketEngine(as_of=as_of, n_sims=n_sims, seed=seed)
    elif key in ("network", "trees"):
        from ..ml.ucl import Challenger
        net = None
        if key == "trees":
            from ..ml.trees import load_model
            net = load_model()
            if net is None:
                raise FileNotFoundError("no tree model: run `python -m ufe models train` first")
        e = Challenger(as_of=as_of, n_sims=n_sims, seed=seed, blend=1.0, net=net)
    else:
        raise KeyError(key)
    if base is not None:
        for k in SHARE:
            if k in base.__dict__:
                e.__dict__[k] = base.__dict__[k]
    e.model_key = key
    return e
