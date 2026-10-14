"""Render the Matchday Command Center: one self-contained HTML file with the data embedded."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("command_center.html")
MARK = "/*__UFE_DATA__*/null"


SKELETON = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def render(data: dict, out: Path, template: Path = TEMPLATE, standalone: bool = True) -> Path:
    """standalone=True adds the document skeleton for opening the file directly in a browser;
    the hosted (artifact) copy is published without it because the host adds its own."""
    html = template.read_text(encoding="utf-8")
    if standalone:
        html = SKELETON + html
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    if MARK not in html:
        raise ValueError("template is missing the data marker")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(MARK, payload), encoding="utf-8")
    return out
