# UCL Fantasy Engine

A personal decision engine for UEFA Champions League Fantasy 2026/27. It predicts every
match, every player's points distribution, line-ups, scorers and Player of the Match, then
turns those predictions into decisions: squad, starting XI, captain, transfers, chips, and
what to change after each day of a matchday. It runs on a laptop CPU, uses only free data,
and improves after every matchday from the results you feed back.

Open `data/runs/dashboard.html` in a browser for the Matchday Command Center.

---

## Quick start

```bash
# Python 3.11+ (Windows, macOS or Linux)
python -m venv .venv
.venv\Scripts\activate            # Windows      (macOS/Linux: source .venv/bin/activate)
pip install -e .                   # or: pip install -r requirements.txt

python -m ufe probe                # which free sources work from your network
python -m ufe sync --all           # results history, Premier League events, game feed, Understat
python -m ufe tune                 # optional: re-tune the team model (≈4 min)
python -m ufe dashboard --mds 3-8  # projections + Command Center -> data/runs/dashboard.html
```

Free keys that switch on extra layers (set as environment variables, all optional):

| Variable | Get it at | Adds |
|---|---|---|
| `UFE_FOOTBALL_DATA_KEY` | football-data.org (free tier includes the UCL) | kick-off times, results |
| `UFE_ODDS_API_KEY` | the-odds-api.com (free tier) | bookmaker prices, blended into match probabilities |
| `UFE_TELEGRAM_TOKEN`, `UFE_TELEGRAM_CHAT` | @BotFather on Telegram | deadline and line-up alerts on your phone |

## Your weekly rhythm (times in IST)

| When | Command | What you get |
|---|---|---|
| Any time | `python -m ufe schedule` | leaves a loop running that does everything below and pings Telegram |
| T-72h | `python -m ufe sync --all` | fresh fixtures, results, prices, xG, news |
| T-48h | `python -m ufe predict --md N` | projections + a frozen pre-deadline snapshot (used to grade the model honestly) |
| T-30h | save UEFA.com's predicted line-ups as `data/seed/news/mdN_uefa.txt`, then `python -m ufe teamnews --md N --file data/seed/news/mdN_uefa.txt` | start chances updated from the named XIs, injuries and bans; missing starters added to the pool |
| T-24h | `python -m ufe advise --md N --squad data/my/squad.csv --bank 0.5 --ft 2` | best XI, captain plan, transfer options with gains net of hits, chip table |
| T-6h | `python -m ufe sync --odds N` then `predict` again | market-blended probabilities |
| T-1h | `python -m ufe lineup --md N --team BAY --start "Neuer, Kim, ..."` | confirmed XIs replace predicted ones; re-run `predict` |
| After day 1 | add the evening's events to `data/seed/results/mdN_events.csv`, then `python -m ufe --as-of <day-2 date> replan --md N --squad data/my/squad.csv --xi data/my/xi.csv --captain Olise` | refit on day 1, refreshed day-2 forecast (frozen for grading), your day-1 points, the captain switch and manual subs (`--points app.csv` overrides the event estimate; `--lineups-known` once XIs are out) |
| After day 1 (quick) | `python -m ufe live --md N --day 1 --squad data/my/squad.csv --captain Olise --points day1.csv` | the same call without the refit |
| After each day | `python -m ufe grade --md N` | every match graded (result, scoreline, scorers, assisters, POTM, XIs), day 1 and day 2 separately |
| After the MD | `python -m ufe results --md N --results results.csv --players players.csv` | scores the snapshot, updates calibration, logs a model version |

`data/templates/` has a CSV template for every file above.

## What is inside

```
config/         rules (every scoring, transfer, chip and club-cap rule), settings, sources, teams
data/seed/      bundled 2026/27 fixtures and results, the full price list (953 players, official_squads_2026_27.csv),
                article prices with URLs, FBref history, team news and actual line-ups
data/raw/       cached downloads (results, FPL archive)          data/processed/  model outputs, params, logs
data/runs/      dashboards and per-matchday snapshots            data/templates/  CSV templates
ufe/ingest/     openfootball, FPL archive, official game feed, football-data.org, ClubElo, odds,
                Understat, news (RSS + optional local LLM), line-ups, source probe
ufe/models/     team_ratings (hierarchical Dixon–Coles), scoreline, player_rates, simulate, tournament
ufe/optimize/   milp (HiGHS via SciPy), squad, transfers (multi-matchday), policy (in-matchday), chips
ufe/evaluate/   metrics, backtest, tune, feedback (the learning loop)
ufe/report/     export (snapshots, dashboard data), render, command_center.html
ufe/alerts/     Telegram                                          ufe/scheduler.py  the self-running loop
tests/          pytest suite                                      schedule/  cron, Windows, GitHub Actions
```

## How the model works

1. **Team model.** Each club has an attack and defence rating on one cross-league scale,
   fitted on ~11,000 results from 13 leagues plus three seasons of UEFA competitions, with
   time decay. Clubs shrink toward their league's mean and leagues toward zero, so a club with
   almost no data (Viking, Sabah) still gets a sensible rating. Dixon–Coles corrects low
   scores. Match probabilities average over the uncertainty in each rating (a club with little
   data gets wider score distributions). Bookmaker odds, when present, are converted to goal
   rates and blended in.
2. **Line-ups and minutes.** Start probability from each club's "slots" by position, shared
   by price and history (start rate per match the player was fit for). Published predicted
   line-ups update it with a logistic model calibrated on real XIs (scripts/calibrate_lineups.py:
   on MD2 Tuesday a named player started 74% of the time, not 93%); injured and suspended players
   go to zero, the previous XI and older news count at half strength, and a confirmed XI sets it
   to 1 or 0. A side's goal rate moves with the attacking share missing from its expected XI.
   Minutes, substitution times and 60-minute thresholds follow Premier League distributions.
3. **Player events.** Each player's share of his club's xG and xA: Opta history from FBref (big
   five leagues) and FPL (Premier League) blended with a price prior by sample size; goals and
   assists per 90 for players outside the big five; price alone otherwise. Recoveries
   (role-adjusted, converted to UEFA's definition), cards, penalties (designated takers), own
   goals, saves.
4. **Joint simulation.** 12,000 runs per match: scoreline, every goal's minute and type, scorer
   and assister among players on the pitch at that minute, goals conceded while each defender
   was on, Player of the Match as the best performance score. Points follow the rules file
   exactly. Team-mates' outcomes stay correlated.
5. **Decisions.** Integer programming picks squads, XIs and multi-matchday transfer plans (with
   free-transfer rollover and hits). The in-matchday policy decides captain switches and manual
   subs after each day and prices the auto-sub cover you give up. The chip planner scores
   Limitless and Wildcard on every eligible matchday.
6. **Learning loop.** After each matchday the frozen pre-deadline snapshot is graded (log loss,
   RPS, Brier, calibration, points error), the recovery conversion is updated, every player's
   UCL results feed back into his rates, and a registry entry records the change.

## How accurate is it

Walk-forward backtests (the model only sees matches before each round):

| Season | Matches | RPS (lower is better) | Base-rate RPS | Log loss (base) | Result picked right | Exact score |
|---|---|---|---|---|---|---|
| 2024/25 UCL | 189 | 0.198 | 0.240 | 0.888 (1.006) | 58.2% | 7.4% |
| 2025/26 UCL | 189 | 0.194 | 0.235 | 0.908 (1.024) | 58.7% | 8.5% |

Clean-sheet probabilities are calibrated to within 2–3 percentage points on average. The team
model's three settings were tuned on these same seasons; the score surface is flat (RPS moved
under 2% across 45 combinations), so the optimism from tuning is small.

Line-ups (MD2 Tuesday, 144 players from six clubs, leave-one-club-out): log loss 0.567 for the
calibrated logistic update vs 0.678 for the Bayes rule it replaced.

First live results (MD2 Tuesday, frozen pre-deadline forecast, model 0.10): 5 of 9 results right,
log loss 1.09 against 0.95 for base rates (Brugge's win at Inter and Viking's draw with Bayern
were given 12% and 11%); no exact scores (the actual score was on average the 11th likeliest);
10 of 32 goals scored by a player in the forecast's top 3 for his match; Player of the Match in
the top 3 in 5 of 9; 158 of 198 actual starters inside the predicted XIs. Player-level skill vs
a flat rate (Brier): P(goal) +13%, P(assist) +4%, P(start) +37%. Nine matches say almost nothing;
`ufe grade` accumulates every matchday, day by day.

Team form and player form (adopted 14 Oct) were tested walk-forward before going in; club home/away
effects were tested and rejected. Numbers in DECISIONS.md.

`python scripts/verify_lineup.py` checks the pick against the tipsters and the most-owned team
and re-solves it under 3 seeds and 10 parameter redraws (results on the dashboard's Check panel).

Run `python -m ufe backtest` to reproduce. Bookmakers sit near the ceiling for match results;
matching them is the realistic target. The compounding edge in Fantasy comes from decisions
(captain switching, manual subs, transfer timing, chips), which the engine prices explicitly.

## Known limits

- Prices come from a third-party mirror of the game's club pages (uclfantasy.com, read 14 Oct):
  953 players, matching 375 of 377 prices published in articles. A few squad players were cut
  from those pages; they stay in the simulation but are never bought. Slovan Bratislava's prices
  are not published yet. On your machine `python -m ufe sync --feed` loads the game's own feed.
  If discovery fails, copy the feed URL from your browser's DevTools (Network tab, filter
  "players") into `config/sources.yaml` → `ucl_fantasy_feed.players_url`.
- The line-up model is calibrated on one matchday (six clubs, just after an international
  break). Refit it after each matchday with `python scripts/calibrate_lineups.py --md N` once
  the actual XIs are in `data/seed/lineups/mdN_actual_xis.txt`.
- No bookmaker odds unless an odds file or key is present; the team model alone can disagree
  with the market (for MD3 it makes Arsenal favourites in Munich).
- Domestic results in the open data end on 20 Sep 2026 and FBref history early in 2025/26.
  This season's league numbers per player are bundled for 8 clubs (StatMuse, 14 Oct); the other
  clubs use earlier seasons until `data/processed/domestic_form.csv` is filled on your machine.
- Day-1 fantasy points for the half-time call are rebuilt from match reports plus estimated ball
  recoveries; pass the app's numbers with `--points` when they differ.
- Adapters for the game feed, football-data.org, odds, Understat, FotMob and RSS were written to
  their documented or observed formats but could not be reached from the build machine. Run
  `python -m ufe probe` first; anything that fails says why.
- Ball recoveries: UEFA counts more than Opta. The conversion factor is provisional and is
  re-estimated from your uploaded results.
- Unknown rule details are listed in `config/rules_2026_27.yaml` → `unknowns`, with the default
  the engine assumes for each.
