"""Scoring and squad rules, driven entirely by config/rules_2026_27.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .config import rules

POSITIONS = ("GK", "DEF", "MID", "FWD")


def stage_of(md: int) -> str:
    for row in rules()["matchdays"]:
        if row["md"] == md:
            return row["stage"]
    raise KeyError(md)


def club_cap(md: int) -> int:
    return rules()["club_cap"][stage_of(md)]


def budget(md: int) -> float:
    b = rules()["squad"]["budget"]
    return b["league_phase"] if stage_of(md) == "league" else b["knockout"]


def free_transfers(md: int) -> tuple[int | None, int]:
    """(free transfers, max carry). None = unlimited."""
    row = rules()["transfers"]["by_md"][md]
    free = row["free"]
    return (None if free == "unlimited" else int(free)), int(row.get("carry", 0))


def chip_allowed(chip: str, md: int) -> bool:
    return md not in rules()["chips"][chip]["blocked_mds"]


def manual_actions_allowed(md: int) -> bool:
    return md not in rules()["manual_subs"]["unavailable_mds"]


def formations() -> list[tuple[int, int, int]]:
    """All valid (DEF, MID, FWD) outfield shapes."""
    lo, hi = rules()["lineup"]["min"], rules()["lineup"]["max"]
    out = []
    for d, m, f in product(range(lo["DEF"], hi["DEF"] + 1), range(lo["MID"], hi["MID"] + 1),
                           range(lo["FWD"], hi["FWD"] + 1)):
        if d + m + f == 10:
            out.append((d, m, f))
    return out


def formation_label(f: tuple[int, int, int]) -> str:
    return "-".join(str(x) for x in f)


@dataclass
class StatArrays:
    """Per-player event counts, each an array over simulations (or a single realised match)."""
    minutes: np.ndarray
    goals: np.ndarray
    ob_goals: np.ndarray          # goals from outside the box
    assists: np.ndarray
    cs: np.ndarray                # clean-sheet eligible (bool)
    conceded: np.ndarray          # goals conceded while on the pitch
    saves: np.ndarray
    pen_saves: np.ndarray
    recoveries: np.ndarray
    potm: np.ndarray              # bool
    pen_won: np.ndarray
    pen_conceded: np.ndarray
    pen_missed: np.ndarray
    yellow: np.ndarray
    red: np.ndarray
    own_goals: np.ndarray


def points(pos: str, s: StatArrays) -> np.ndarray:
    """Fantasy points for one player given stat arrays (vectorised over simulations)."""
    sc = rules()["scoring"]
    played = s.minutes > 0
    pts = np.where(played, sc["appearance"], 0).astype(np.int32)
    pts += np.where(s.minutes >= 60, sc["sixty_minutes"], 0)
    pts += s.goals * sc["goal"][pos]
    pts += s.ob_goals * sc["goal_outside_box_bonus"]
    pts += s.assists * sc["assist"]
    pts += np.where(s.cs & (s.minutes >= sc["clean_sheet_min_minutes"]), sc["clean_sheet"][pos], 0)
    if sc["goals_conceded_penalty"][pos]:
        pts += (s.conceded // sc["goals_conceded_step"]) * sc["goals_conceded_penalty"][pos]
    if pos == "GK":
        pts += s.saves // sc["saves_step"]
        pts += s.pen_saves * sc["penalty_save"]
    pts += s.recoveries // sc["recoveries_step"]
    pts += np.where(s.potm, sc["player_of_the_match"], 0)
    pts += s.pen_won * sc["penalty_won"]
    pts += s.pen_conceded * sc["penalty_conceded"]
    pts += s.pen_missed * sc["penalty_missed"]
    pts += s.yellow * sc["yellow_card"]
    pts += s.red * sc["red_card"]
    pts += s.own_goals * sc["own_goal"]
    return np.where(played, pts, 0)


def validate_squad(players: list[dict], md: int, budget_limit: float | None = None) -> list[str]:
    """Return a list of rule violations for a 15-man squad (empty list = valid)."""
    errs = []
    want = rules()["squad"]["positions"]
    counts = {p: 0 for p in POSITIONS}
    for p in players:
        counts[p["pos"]] += 1
    if len(players) != rules()["squad"]["size"]:
        errs.append(f"squad has {len(players)} players, needs {rules()['squad']['size']}")
    for p, n in want.items():
        if counts[p] != n:
            errs.append(f"{p}: {counts[p]} (needs {n})")
    lim = budget(md) if budget_limit is None else budget_limit
    cost = sum(p["price"] for p in players)
    if cost > lim + 1e-9:
        errs.append(f"cost €{cost:.1f}m exceeds €{lim:.1f}m")
    cap = club_cap(md)
    per_team: dict[str, int] = {}
    for p in players:
        per_team[p["team"]] = per_team.get(p["team"], 0) + 1
    for t, n in per_team.items():
        if n > cap:
            errs.append(f"{n} players from {t} (cap {cap})")
    return errs


def valid_xi(pos_list: list[str]) -> bool:
    lo, hi = rules()["lineup"]["min"], rules()["lineup"]["max"]
    if len(pos_list) != rules()["lineup"]["size"]:
        return False
    for p in POSITIONS:
        n = pos_list.count(p)
        if n < lo[p] or n > hi[p]:
            return False
    return True
