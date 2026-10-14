"""Configuration loading: rules, settings, sources, teams, and tuned parameters."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Settings:
    raw: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        p = ROOT / self.raw["paths"][key]
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def rules() -> dict:
    return _load_yaml("rules_2026_27.yaml")


@lru_cache(maxsize=1)
def sources() -> dict:
    return _load_yaml("sources.yaml")


@lru_cache(maxsize=1)
def teams() -> list[dict]:
    return _load_yaml("teams_2026_27.yaml")["teams"]


def settings() -> Settings:
    """Defaults from settings.yaml, overridden by tuned values in data/processed/params.json."""
    base = _load_yaml("settings.yaml")
    tuned_path = ROOT / base["paths"]["processed"] / "params.json"
    if tuned_path.exists():
        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
        base = deep_update(base, tuned.get("settings", {}))
    return Settings(base)


def team_lookup() -> dict[str, dict]:
    return {t["id"]: t for t in teams()}
