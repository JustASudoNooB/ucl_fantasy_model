# Error log

After every matchday, record the biggest misses and what they teach. Patterns here become
roadmap items; one-offs stay as notes.

| MD | What happened | What the model said | Cause (data / model / variance) | Action |
|---|---|---|---|---|
| MD2 (pre-deadline review) | The recommended XI looked like the tipster teams online: 8 of 11 matched Fantasy Football Scout's MD2 picks | Best XI by expected points, exact MILP optimum | Data: the player list was built from tipster articles (236 of 325 players were budget defenders from one article; most attackers came from "players to buy" pieces), so the optimiser chose from a pool tipsters had already filtered | Pool widened from UEFA's predicted line-ups (325 → 651); unpriced players simulated but never picked; break-even prices reported |
| MD2 (pre-deadline review) | Szoboszlai (doubtful, hamstring) was in the XI and the captain fallback; Strakosha (out, hand) was on the bench | Both ~90% to start | Data: no team-news layer in the bundled build | Predicted line-ups and injuries now update start chances by Bayes' rule (`ufe teamnews`) |
| MD2 (14 Oct review) | Wirtz was priced €6.5m in the widened pool; the game says €8.5m. 17 of 27 FPL-mapped prices were wrong | Picked Wirtz as a cheap premium | Data: estimated prices treated as buyable | Full official price list loaded (953 players); estimates no longer bought once it exists |
| MD2 Tue | Saka benched (came on 65'), Upamecano and Davies benched, Gyökeres/Madueke/Merino started unpredicted | Named starters at 0.93-0.98 | Model: Bayes LR 9.3 assumed 93% of predicted starters start; real rate 74% (49/66) | Logistic recalibration (leave-one-club-out log loss 0.678 → 0.567), p_max 0.95 |
| MD2 Tue | Inter 3-4 Club Brugge (Brugge win given 12%), Viking 3-3 Bayern (draw 11%), Sabah 2-0 Slavia (27%) | 5 of 9 results right; log loss 1.09 vs base 0.95 | Variance (9 games); favourites' calibration over 378 backtest matches is within a few points | None; watch whether the model's strong favourites keep under-delivering over MD2-MD4 |
| MD2 Tue | Olise sent off (second yellow, 73'): about 0 points as captain; banned for MD3 | Captain, 8.8 expected | Variance (cards are rare); the switch rule covers it | Live call: move the armband to a Wednesday player; Olise marked suspended for MD3 |
| MD2 Tue | 5 of 32 goals were scored by players missing from the frozen forecast's pool (Fermín López, Vermant, Bonny, Adeyemi, Oluwaseyi); 21 of 198 actual starters were missing too | Their goals fell to the "unlisted players" block | Data: the 13 Oct pool came from tipster articles | Full official list since 14 Oct; v0.11 retro has every scorer in the pool |
| MD2 Tue | Mouriño (Villarreal centre-back) scored and won POTM: given 0.7% to score, 0.1% POTM | Defender, low share | Variance; centre-backs' set-piece goals are rare events | None |
| MD2 Tue | MUN and ATM line-ups in the actual-XI file came from a pre-match live blog (Maguire, Cunha, Pubill marked as starters) | n/a | Data entry | Re-read from UEFA.com's line-up page and teleprensa; files corrected 14 Oct 17:15 IST |
| Build | Rebuilding the dashboard after a deadline would have overwritten that matchday's frozen snapshot | n/a | Code | Snapshots are write-once after the deadline |
| Build | Start rates from history counted injury spells as rotation (Kvaratskhelia 0.31, Courtois 0.64) | Low priors the news layer could not lift | Model | Start rates per fit match (starts / (apps + 0.35 × missed)) |

Checklist after each upload of results:
- Largest point misses: were they minutes (benched, subbed) or events (goals, POTM)?
- Any confirmed starter we gave under 50% start probability? Why?
- Any player flagged OUT by the news scanner who played? (false positive → add a rule exception)
- Recovery ratio after update vs before (data/processed/params.json → history)
| Ops, 17 Oct | GitHub Actions `matchday` run failed at 23:25 IST in the dashboard step (26 s after start); the four earlier runs passed | n/a | Code: the first run inside 72 h of the MD3 deadline fired `sync --all` for the first time on the runner. Most likely the official game feed answered there (it is blocked from this workspace), and `load_pool` replaced the whole pool with `players_feed.csv`, which has no `id` column, so the build stopped with `KeyError: ['id']` (reproduced with a synthetic feed; the run log itself could not be downloaded) | Feed now overlays prices, positions and ownership on the pool by club + name and keeps the pool's ids; partial feeds (<300 players or <30 clubs) are ignored; each sync source fails on its own; sync gets the current matchday; workflow retries the dashboard on committed data, pins ubuntu-24.04 and moves to Node 24 actions (v0.12.1) |
| MD3 | 74 goals against 63.3 forecast (MD2: 78 v 59.2); 59 players scored against 51.9 expected | Every goal-based number low by 10-25% | Model: one intercept with a 500-day half-life could not follow a season scoring 4.09 a game | Season and UEFA-season goal-level terms (v0.14); MD4 forecast 3.80 a game |
| MD3 | Martínez (AVL) given 0.93 to start, Suzuki started MD1-MD3; 272 of 396 actual starters inside the clubs' top 11s | Wrong keeper ranked first for Villa | Model: start rates ignored this season's UCL XIs | Beta update from the last two UCL XIs (v0.14): 289 in a retro run |
| MD3 | Hand-entered 1X2 prices implied 57.7 goals for the 18 games (actual 74) | With the 0.65 blend the forecast would have dropped to 61.7 | Model: 1X2 prices barely identify total goals | Without an over/under price the blend uses the market's supremacy only (v0.14) |
| MD3 | 15 actual starters unmatched (Tasos Douvikas, Provod, João Mário, ...) | Graded as missing; start model could not learn from them | Code: spellings differ from the pool | Unique-surname rule (3 fixed); 12 are missing from the pool |
| MD3 day 2 | COM 4-1 MUN, LIL 2-4 GAL, SCP 2-3 LAS, BAY 2-2 ARS; log loss 1.085 v base 0.902 | Day 2 results 6 of 9 | Variance plus no odds; day 1 was 0.745 v 1.031 | None beyond the goal-level fix |

