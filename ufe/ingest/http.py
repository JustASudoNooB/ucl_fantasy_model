"""Small, polite HTTP layer: caching, retries, rate limiting, provenance logging."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from ..config import ROOT

UA = "ucl-fantasy-engine/0.9 (personal, non-commercial; contact via GitHub)"
LOG = ROOT / "data" / "raw" / "fetch_log.jsonl"
_last_call: dict[str, float] = {}


def _throttle(host: str, min_gap: float) -> None:
    t = _last_call.get(host, 0.0)
    wait = min_gap - (time.monotonic() - t)
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


def _log(url: str, status: int, dest: Path | None) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"url": url, "status": status, "dest": str(dest) if dest else None,
                             "fetched_at": datetime.now(timezone.utc).isoformat()}) + "\n")


def get(url: str, *, headers: dict | None = None, params: dict | None = None, timeout: float = 20.0,
        retries: int = 3, min_gap: float = 1.0) -> httpx.Response:
    host = httpx.URL(url).host
    h = {"User-Agent": UA, "Accept": "*/*"}
    h.update(headers or {})
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle(host, min_gap)
        try:
            r = httpx.get(url, headers=h, params=params, timeout=timeout, follow_redirects=True)
            _log(str(r.url), r.status_code, None)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            return r
        except httpx.HTTPError as exc:  # network errors: back off and retry
            last_exc = exc
            time.sleep(2 ** attempt)
    if last_exc:
        raise last_exc
    return r  # type: ignore[possibly-undefined]


def fetch_to_file(url: str, dest: Path, *, force: bool = False, ok_missing: bool = False,
                  max_age_hours: float | None = None, **kw) -> bool:
    """Download url to dest unless a fresh cached copy exists. Returns True if dest exists after."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if max_age_hours is None:
            return True
        age = (time.time() - dest.stat().st_mtime) / 3600
        if age < max_age_hours:
            return True
    try:
        r = get(url, **kw)
    except httpx.HTTPError:
        return dest.exists()
    if r.status_code == 200 and r.content:
        dest.write_bytes(r.content)
        _log(url, 200, dest)
        return True
    if r.status_code == 404 and ok_missing:
        return dest.exists()
    return dest.exists()


def get_json(url: str, **kw):
    r = get(url, **kw)
    r.raise_for_status()
    return r.json()
