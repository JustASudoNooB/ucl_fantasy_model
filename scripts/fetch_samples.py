"""Save raw copies of the pages the free-data parsers read (run in CI, where the sites are reachable;
the files go to the `samples` branch so parsers can be written and tested against real markup).

  python scripts/fetch_samples.py samples/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ufe.ingest.http import get  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
OUT.mkdir(parents=True, exist_ok=True)
BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/129.0 Safari/537.36", "Accept-Language": "en-GB,en;q=0.9"}
report = {}


def save(name: str, url: str, headers: dict | None = None):
    try:
        r = get(url, headers={**BROWSER, **(headers or {})}, retries=1, timeout=30)
        (OUT / name).write_bytes(r.content)
        report[name] = dict(url=url, status=r.status_code, bytes=len(r.content), type=r.headers.get("content-type"))
        return r
    except Exception as e:  # noqa: BLE001
        report[name] = dict(url=url, error=f"{type(e).__name__}: {e}"[:300])
        return None


import os
for i, u in enumerate(x.strip() for x in os.environ.get("EXTRA_URLS", "").split(",") if x.strip()):
    save(f"extra_{i}.html", u)                 # pages asked for by hand (e.g. past odds articles)
if os.environ.get("EXTRA_ONLY") == "1":
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    raise SystemExit(0)
save("starting11_ucl.html", "https://starting11.com/champions-league")
for lg in ("la-liga", "bundesliga", "serie-a", "ligue-1", "eredivisie", "super-lig", "primeira-liga",
           "belgian-pro-league", "premier-league"):
    save(f"s11_setpieces_{lg}.html", f"https://starting11.com/set-piece-takers/{lg}")
save("starting11_home.html", "https://starting11.com/")
for tag, url in (("rg_news_us.html", "https://rg.org/news/soccer"), ("rg_news_ca.html", "https://rg.org/en-ca/news/soccer")):
    r = save(tag, url)
    if r is not None and r.status_code == 200:
        links = sorted(set(re.findall(r'href="([^"]*champions-league[^"]*odds[^"]*)"', r.text)))
        report[tag]["ucl_odds_links"] = links[:30]
save("rg_odds_ca.html", "https://rg.org/en-ca/news/soccer/barcelona-vs-villa-fenerbahce-vs-liverpool-champions-league-odds")
save("rg_odds_us.html", "https://rg.org/news/soccer/barcelona-vs-aston-villa-atletico-vs-bayern-champions-league-odds")
save("understat_laliga.html", "https://understat.com/league/La_liga/2026")
save("understat_laliga_api.json", "https://understat.com/getLeagueData/La_liga/2026",
     headers={"X-Requested-With": "XMLHttpRequest", "Referer": "https://understat.com/league/La_liga/2026"})
r = save("fpl_bootstrap.json", "https://fantasy.premierleague.com/api/bootstrap-static/")
if r is not None and r.status_code == 200:
    ev = [e for e in r.json().get("events", []) if e.get("finished")]
    report["fpl_finished_gws"] = [e["id"] for e in ev]
    for e in ev[-2:]:
        save(f"fpl_live_gw{e['id']}.json", f"https://fantasy.premierleague.com/api/event/{e['id']}/live/")
    (OUT / "fpl_bootstrap.json").write_text(json.dumps({k: v for k, v in r.json().items() if k in ("events", "teams")}))
save("fpl_fixtures.json", "https://fantasy.premierleague.com/api/fixtures/")
meta = ROOT / "data" / "processed" / "feed_meta.json"
if meta.exists():
    url = json.loads(meta.read_text()).get("url")
    r = save("feed_raw.json", url, headers={"Accept": "application/json", "Referer": "https://gaming.uefa.com/en/uclfantasy/"})
    if r is None or r.status_code != 200:
        import time
        time.sleep(5)
        r = save("feed_raw.json", url, headers={"Accept": "application/json", "Referer": "https://gaming.uefa.com/en/uclfantasy/"})
    if r is not None and r.status_code == 200:
        try:
            data = r.json()
            from ufe.ingest.ucl_fantasy import _find_player_list
            recs = _find_player_list(data) or []
            (OUT / "feed_raw.json").write_text(json.dumps(dict(top_keys=list(data)[:20] if isinstance(data, dict) else None,
                                                               n=len(recs), first=recs[:3]), ensure_ascii=False, indent=1))
        except Exception as e:  # noqa: BLE001
            report["feed_raw.json"]["parse_error"] = str(e)[:200]
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
