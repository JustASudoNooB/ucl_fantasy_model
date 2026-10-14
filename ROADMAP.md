# Roadmap

Gates are measurable. A phase is done when its gate passes, not when its code exists.

## Done on 14 Oct
- [x] Full price list: 953 players with official prices and availability (mirror of the game's club pages)
- [x] Line-up model calibrated on real XIs (scripts/calibrate_lineups.py); start rates per fit match
- [x] Verification report: tipster overlap, crowd comparison, seed and parameter robustness (scripts/verify_lineup.py)

## Done on 14 Oct (evening)
- [x] MD2 Tuesday verified: scores, every goal, assist, card, sub and POTM (data/seed/results/md2_events.csv), all 18 XIs
- [x] Match-by-match grading (`ufe grade --md N`), split by matchday day; retro forecasts graded alongside
- [x] Half-time re-plan (`ufe replan`), day-2 snapshot frozen, in-browser version in the dashboard
- [x] Team form term adopted; club home/away effects rejected (scripts/backtest_form_homeaway.py)
- [x] Player domestic form adopted (scripts/backtest_player_form.py); 2026/27 league data for 8 clubs

## Next for form (laptop)
- [ ] `ufe sync --form`: FPL element-summary (Premier League, match by match) and Understat (big five) into
      data/processed/domestic_form.csv, so every club has this season's numbers
- [ ] With match-by-match rows, use the last-three-matches start rate (backtest: log loss 0.528 vs 0.566)
- [ ] Domestic results after 20 Sep into the team model (openfootball lags)

## Phase 0 — on your laptop (this week)
- [ ] `python -m ufe probe` passes for openfootball, FPL archive, ClubElo, RSS; game feed found
- [ ] `python -m ufe sync --feed` writes `data/processed/players_feed.csv` with 700+ players
- [ ] Squad entered in `data/my/squad.csv`; `advise` runs without errors
- **Gate:** dashboard built from the official feed (pool size shown in the banner > 700)

## Phase 1 — MD3 (deadline Tue 20 Oct, 22:15 IST)
- [ ] Upload MD2 results with a `started` column, so the predicted-XI likelihood ratios start learning
- [ ] Before each deadline: save the published predicted line-ups as `data/seed/news/mdN_*.txt`, run `ufe teamnews --md N --file ...`
- [ ] Upload MD2 results: `ufe results --md 2 --results ... --players ...`
- [ ] Odds key set; MD3 projections show "model + market" on fixture cards
- [ ] Telegram alerts working
- **Gate:** MD3 scoreboard row exists; recovery ratio updated from real MD2 data

## Phase 2 — MD4 to MD6 (3 Nov – 9 Dec)
- [ ] News scanner reviewed weekly (`ufe news --md N`), false positives logged in ERROR_LOG.md
- [ ] Confirmed line-ups captured for day-1 matches before each deadline
- [ ] Understat shares replace price priors for big-five-league players
- **Gate:** player-level Brier for P(goal) beats a price-only baseline over MD3-MD5

## Phase 3 — winter break (10 Dec – 18 Jan)
- [ ] Re-tune team model with this season's league phase included (`ufe tune`)
- [ ] POTM conditional-logit fit once 40+ matches logged
- [ ] Start-probability Platt calibration once 300+ player-matches logged
- [ ] Chip plan for MD7-MD11 decided from the league-phase race and the chip table
- **Gate:** calibration error on clean sheets and P(goal) under 3 percentage points

## Phase 4 — knockouts (from MD9, 16 Feb)
- [ ] Add KO fixtures after each draw (`ufe sync --fixtures` or edit `data/processed/fixtures.csv`)
- [ ] Use two-leg advancement probabilities in transfer planning (club caps rise by stage)
- [ ] Full rebuilds on MD9 and MD11 (unlimited transfers)
- **Gate:** knockout-stage rank inside the top 5% (overall percentile tracked in the scoreboard)

## Data gaps found on 13-14 Oct (MD2 review)
- [ ] Domestic results after 20 Sep 2026 (openfootball lags); a second free results source
- [ ] Ownership for every player (the mirror shows the top 100 by price and the biggest movers)
- [ ] MD2 Wednesday XIs and all MD1 XIs, to refit the line-up model on 36 clubs instead of 6
- [ ] A free bookmaker-odds source for every fixture (no single page listed all 18 MD2 prices)
- [ ] Penalty takers for the 27 clubs without one on file (`data/seed/news/set_pieces.csv`)
- [ ] Second predicted line-ups source, to estimate each source's hit rate separately

## Ideas backlog (only if a backtest shows a gain)
- Referee card rates; weather; travel distance
- Player-specific outside-the-box shot shares from event data
- Mini-league mode: rivals' squads → maximise P(finish first) instead of expected points
- Bayesian (PyMC) version of the team model for proper posterior uncertainty
