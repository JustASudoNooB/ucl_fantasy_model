"""Render the Matchday Command Center: one self-contained HTML file with the data embedded."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("command_center.html")
MARK = "/*__UFE_DATA__*/null"


SKELETON = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


PARTS = Path(__file__).with_name("parts")
PART_MARKS = (("/*__UFE_PARTS_CSS__*/", "*.css"), ("<!--__UFE_PARTS_HTML__-->", "*.html"), ("/*__UFE_PARTS_JS__*/", "*.js"))
# Left out of the public edition: saved teams, the mini-league, live calls and plans made for them
PRIVATE = ("managers", "league", "replan", "live", "verification")


def assemble(template: Path = TEMPLATE) -> str:
    """The template with the files in parts/ placed at their markers (styles, sections, script), in name order."""
    html = template.read_text(encoding="utf-8")
    for mark, pattern in PART_MARKS:
        if mark in html:
            html = html.replace(mark, "\n".join(p.read_text(encoding="utf-8") for p in sorted(PARTS.glob(pattern))))
    return html


def public_copy(data: dict) -> dict:
    """The same data without anything about saved teams: safe to hand to other people."""
    d = {k: v for k, v in data.items() if k not in PRIVATE}
    d["meta"] = dict(data.get("meta") or {}, public=True)
    if isinstance(d.get("models"), dict):
        m = {k: v for k, v in d["models"].items() if k != "forecasts"}
        if isinstance(m.get("leaderboard"), dict):
            m["leaderboard"] = {k: v for k, v in m["leaderboard"].items() if k != "forecasts"}
        if isinstance(m.get("models"), list):           # each model's XI and captain for the saved teams
            m["models"] = [{k: v for k, v in x.items() if k != "picks"} if isinstance(x, dict) else x for x in m["models"]]
        d["models"] = m
    return d


def render(data: dict, out: Path, template: Path = TEMPLATE, standalone: bool = True, public: bool = False) -> Path:
    """standalone=True adds the document skeleton for opening the file directly in a browser;
    the hosted (artifact) copy is published without it because the host adds its own."""
    html = assemble(template)
    if standalone:
        html = SKELETON + html
    payload = json.dumps(public_copy(data) if public else data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    if MARK not in html:
        raise ValueError("template is missing the data marker")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(MARK, payload), encoding="utf-8")
    return out
