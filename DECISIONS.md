# Decisions log

Each entry: what was decided, why, and what evidence would reverse it.

**2026-10-13 · Expected points is the objective, prediction accuracy is an input.**
Match-result accuracy saturates near the bookmakers; Fantasy edges come from decisions.
Reverse if: never; this is the brief.

**2026-10-13 · Hierarchical Poisson (Dixon–Coles) team model over gradient boosting.**
Few UCL matches per season (189); a structured model with shrinkage generalises better and fits
in seconds. Walk-forward RPS 0.194–0.198 vs 0.235–0.240 for base rates.
Reverse if: a feature model beats it on the same walk-forward backtest.

**2026-10-13 · Tuned settings: half-life 500 days, UEFA-match weight 1.0, team prior sd 0.5.**
27 + 18 combinations tried; the score surface is flat (RPS within 2%).
Reverse if: re-tuning with 2026/27 data moves the optimum.

**2026-10-13 · Joint event-level simulation instead of independent per-player formulas.**
Captaincy, manual subs and mini-league risk depend on correlations (keeper and defenders share
clean sheets; scorer and assister share goals). Independent formulas cannot price them.

**2026-10-13 · SciPy's HiGHS MILP instead of PuLP/CBC.**
No separate solver install; works the same on Windows, macOS and Linux.

**2026-10-13 · UEFA/Opta recovery ratio starts at GK 0.35, DEF 1.5, MID 1.3, FWD 1.1.**
Evidence: UEFA leaderboards (2020/21) show top defenders at 10–16 recoveries per match and no
keepers among leaders; Opta averages are 3.6 (DEF) and 8.2 (GK) per 90.
Reverse: automatically, from uploaded results (`feedback.update_params`).

**2026-10-13 · Price-based priors map UCL prices onto the FPL scale (€4.5m≈£4.5m … €11m≈£15.5m).**
Only within-club differences are used, with slopes fitted on Premier League 2025/26 with team
fixed effects (xG share: DEF 0.25, MID 0.31, FWD 0.07 per £m) and caps at elite levels.
Reverse if: the official feed's per-player UCL stats show a different gradient.

**2026-10-13 · Captain chosen by policy value, not expected points alone.**
An early-kickoff captain carries a switch option; the policy simulation values it
(MD2: captaining a Tuesday player with a Wednesday fallback was worth ~+2 points).

**2026-10-13 · FPL injury flags from archived snapshots are ignored.**
The bundled 2026/27 snapshot is from August; stale flags removed fit players.
Live FPL API pulls may use them (`use_fpl_availability=True`).

**2026-10-13 · The candidate pool must not come from tipster content alone.**
An optimiser is only as independent as the list it chooses from. The bundled list came from
"players to buy" articles, so the MD2 optimum overlapped 8 of 11 with a tipster team. Now every
predicted starter is in the pool: published price where one exists, an FPL-mapped estimate for
Premier League players (fit on 51 players listed in both games; MAE 0.38 defenders/keepers,
0.59 midfielders/forwards), otherwise "unknown" (simulated, never bought). For each estimated or
unknown player that matters, the break-even price says whether he belongs in the best squad.
Reverse if: the official feed is loaded (it supersedes all of this).

**2026-10-13 · Predicted line-ups update start chances by Bayes' rule.**
Named starter: prior odds × 9.3 (0.93 / 0.10); fit but not named: × 0.078; out or suspended: 0;
doubtful: capped at 0.62 (named) or 0.15 (not named). Predicted XIs from one source (UEFA.com).
Reverse: automatically. Each uploaded matchday adds counts of named/started players and the
likelihood ratios are re-derived with a Beta prior worth ~100 players (`feedback.update_params`).

**2026-10-13 · Match probabilities integrate over rating uncertainty.**
Posterior-predictive score matrix (7×7 Gauss–Hermite over each side's log goal rate, sd from the
Fisher information). Walk-forward UCL 2024/25–2025/26 (378 matches): log loss 0.8980 → 0.8970,
RPS 0.19576 → 0.19559, Brier 0.5264 → 0.5258, clean-sheet ECE 0.0235 → 0.0214; scale 0.5–1.5
all beat the point estimate. Small, consistent gain; kept at scale 1.0.
Reverse if: a re-tune including 2026/27 prefers scale 0.

**2026-10-13 · Team strength follows the expected XI (not yet testable).**
A side's goal rate falls by half of the attacking share missing from its expected XI, clipped to
−12%/+6%. No historical line-ups exist to backtest it; MD2 multipliers range 0.96–1.05.
Reverse if: after MD2–MD4, matches with large adjustments score worse than without them
(compare snapshot xG with and without `news.strength_adjust`).

**2026-10-13 · Unlisted-player block shrinks when the named XI is complete.**
With predicted starters named, the "unlisted" pseudo-player keeps only the starts the named
players leave uncovered (floor 25% for substitutes). Without this, unlisted players kept
starter-level scoring weight next to a full named XI.

**2026-10-14 · The full official price list replaces the article pool.**
Every club page of the game (953 players, read through the uclfantasy.com mirror) is matched to the
old pool by club and name (abbreviations, accents, word order; transfers matched across clubs only
on initial + exact surname). Audit: article prices agreed for 375 of 377 players; FPL-mapped
estimates were wrong for 17 of 27, so estimated prices are no longer bought once the list exists.
Players missing from the pages stay in the simulation with their old status, never bought.
Reverse if: the game's own feed is loaded on the laptop (it supersedes the mirror).

**2026-10-14 · Line-up news uses a calibrated logistic update, not Bayes' rule.**
On MD2 Tuesday UEFA's predicted XIs named 49 of 66 actual starters (74%). Fitted on 144 players:
logit P(start) = a + b·logit(prior) + c·named. Leave-one-club-out log loss: Bayes rule 0.678,
logistic with b free (0.26) 0.553, with b fixed at 0.5 0.567. Kept b = 0.5 (a = −0.94, c = 1.86)
until more line-ups are logged; p_max 0.95. Older news (a previous XI, or the last matchday's
predictions carried forward) blends halfway toward the prior in logit space.
Reverse: re-run scripts/calibrate_lineups.py after each matchday; adopt the refit when its
leave-one-club-out loss is lower.

**2026-10-14 · History start rates are per match the player was fit for.**
starts / (appearances + 0.35 × matches missed entirely): about a third of missed matches are
assumed to be fit-but-unused, the rest injuries and bans. Starts per team match had treated
injury spells as rotation (Kvaratskhelia 0.31, Courtois 0.64) and the news layer could not undo
it from a low prior. Reverse if: a season-ahead test on FBref shows the old definition predicts
next-season starts of fit players better.

**2026-10-14 · Per-90 goals and assists for players with no Opta history.**
For players outside the big five without FPL/FBref history (Suárez, Osimhen, Tresoldi…), the
mirror's goals and assists per 90 become shares of the club's domestic goal rate (team model vs
its league rivals), with weights n/(n+25) and n/(n+40) (actual goals are noisier than xG, which
uses 12 and 20). Reverse if: the official feed's per-match UCL stats disagree systematically.

**2026-10-14 · Live calls compare whole plans.**
Any manual change switches off auto-subs, so "captain only", "subs only" and "both" are each
valued against "passive" (auto-subs on). Later-day starters who are ruled out can be subbed by
hand; sent-off players cannot be subbed (rules file). Reverse: never; it follows the rules.

**2026-10-14 · Frozen snapshots are write-once after the deadline.**
`export.snapshot` returns the existing file once a matchday's deadline has passed (force=True to
override), so a post-deadline rebuild cannot leak results into grading.

**2026-10-14 (evening) · Team form: a short recency term in the match weights.**
w = 0.7 · 0.5^(age/500) + 0.3 · 0.5^(age/90). Walk-forward on 378 UCL matches (refit before every
round) and 5,714 top-flight domestic matches (refit every 14 days), 2024/25 + 2025/26:
UCL RPS 0.19580 → 0.19566, log loss 0.89748 → 0.89721; domestic RPS 0.20138 → 0.20129, log loss
0.98894 → 0.98882. Better on all four, so adopted (rule: improve UCL and domestic, RPS and log
loss). Also tried: (0.3, 30 days) worse on domestic log loss; (0.5, 90) worse on domestic log loss;
(0.3, 150) better on domestic, worse on UCL than (0.3, 90). Reverse if: the next season's walk-forward
prefers form_mix 0.

**2026-10-14 (evening) · Club-specific home/away effects: rejected.**
Extra home-scoring and home-defending terms per club, shrunk toward the league home advantage
(prior sd 0.05 / 0.1): UCL RPS 0.19598 / 0.19644 vs 0.19580 without; domestic change ±0.00004.
A club's home/away split over one or two seasons is mostly noise. Records stay on the dashboard
for reading, not in the model. Reverse if: a larger sample (three-plus seasons) shows a gain.

**2026-10-14 (evening) · Players: this season's domestic numbers enter the rates.**
Premier League 2025/26 walk-forward (FPL archive, 10,582 player-matches from GW4): per 90, this
season's xG weighted equal to last season's cut goal log loss 0.2682 → 0.2653 (W=2: 0.2654;
within-season decay with half-life 5 or 10 matches: no better). Assists: this season at half weight
0.2766 → 0.2752. Starts (13,054 player-matches of players seen in the last three games): last
season's rate 0.668, this season's rate 0.566, both with last season counted as 3 matches 0.566
(early season 0.514), last three matches alone 0.528. Adopted: W_CUR_XG 1, W_CUR_XA 0.5 (assists
only, no xA), start rate = (starts + 3 × last rate) / (fit matches + 3) with prior weight 4.
Data here: StatMuse (Opta) league tables for 8 clubs; elsewhere earlier seasons until synced.

**2026-10-14 (evening) · Half-time re-plan and per-day grading.**
After the first evening: refit through it (as_of = day-2 date), refresh day 2, score day-1 players
from match events (+ estimated recoveries) and choose among whole plans. The day-2 forecast is
frozen (snapshot_day2.json, write-once after the first day-2 kick-off) so day 2 is graded twice:
pre-deadline and half-time. Retro forecasts (today's code, pre-match inputs, as_of = matchday
date) are graded alongside and labelled; they never replace a frozen forecast.

**2026-10-26 · Goal-level terms in the team model (MD3 review).**
UCL goals per game: 2.98, 3.00, 3.27, 3.44 over the last four seasons, 4.09 after three 2026/27
matchdays; top-flight domestic 3.10 against 2.79 last season. The 500-day half-life moves the single
intercept slowly, so every matchday was under-forecast (MD2 78 goals v 59.2, MD3 74 v 63.3). Added
log-scale shifts with N(0, sd^2) priors: every match of this season, and this season's UEFA matches
on top (a UEFA-for-all-seasons shift was tested too and did nothing). Walk-forward
(scripts/backtest_goal_level.py), season 0.2 + cup_season 0.2 against none: UCL 2026/27 log loss
0.8418 -> 0.8351, total-goals log score 2.140 -> 2.085, goal bias -0.69 -> -0.31 a match; domestic
2026/27 log loss 0.9541 -> 0.9517; UCL 2024-26 log loss +0.0003 better, goal log score 0.004 worse;
domestic 2025/26 unchanged. MD4 forecast 3.80 goals a match (3.39 before). Reverse if: the
2027/28 walk-forward prefers sd 0.

**2026-10-26 · Start probabilities learn from the previous UCL XIs.**
Start rates came from domestic and last-season history, and never looked at who started this
season's UCL games: Aston Villa's keeper was Martínez at 0.93 while Suzuki started MD1-MD3.
Now, before any team news: p = (kappa * p_model + started_last + 0.7 * started_before) /
(kappa + 1 + 0.7), on the start probability before injury multipliers (a ruled-out player stays
out). Engine test, update before news: start log loss MD2 0.388 -> 0.358, MD3 0.483 -> 0.422; actual
starters inside each club's top 11: 311 -> 317 and 262 -> 289 (of 395 and 392). kappa 1.0 best on
both matchdays (1.5 close; 2.5 and 4 worse). Older team news is no longer carried forward once the
XIs of that matchday are on file. Reverse if: after MD6 the update loses on held-out matchdays.

**2026-10-26 · Odds without an over/under price set the supremacy only.**
A 1X2 price fixes the favourite's edge well and total goals poorly (the draw price carries it, with a
weak 2.8-goal anchor). MD3 with the hand-entered rg.org prices: 1X2-implied totals 57.7 goals,
model 70.6, actual 74. Blending the market's home/away ratio at the model's total: result log loss
0.857 -> 0.869 (18 games, noise level), goals 61.7 -> 70.4. Fantasy points follow goals, so the
total stays with the model unless an over/under 2.5 price exists (then the full blend applies).
Reverse if: over a season, 1X2-implied totals beat the model's.

**2026-10-26 · Name matching: a surname unique within the club counts.**
When no spelling scores 86+, a surname that exactly one player of that club shares is accepted
("Tasos Douvikas" -> Anastasios Douvikas); two players with the surname (the Haugens) means no
match. Three more MD3 starters matched; the other 12 unmatched are players missing from the pool.

**2026-11-02 · Free team news and odds instead of waiting for keys.**
starting11.com publishes an "Oracle XI" for all clubs of the next matchday with a confidence tag per
player, plus absences and doubts; rg.org publishes every game's 1X2 and goal-line prices from up to
ten bookmakers (US and Canadian editions). Both are server-rendered and reachable from GitHub
Actions, so the matchday workflow now writes news_md{N}.txt and odds_md{N}.csv itself before every
dashboard build and commits them. Tags scale each named player's update (nailed on 1.0, probable 0.8,
toss-up 0.5, punt 0.3, doubt 0.5) and the file counts at strength 0.7: the site is an algorithm with
no measured hit rate yet (158 of its 353 MD4 picks are toss-ups). These values are priors, to be
replaced from MD4 with scripts/calibrate_news_labels.py. Reverse if: MD4-MD5 show the tags do not
separate starters (then strength 0.4 and equal tags).

**2026-11-02 · Goal lines other than 2.5.**
Bookmakers now quote 3.5 for most UCL games (4.5 for Barcelona v Villa). implied_lambdas fits any
line; integer lines are Asian (a push refunds, so P(over) / (1 − P(exactly the line))). With a line
present the market's total is blended at the usual 0.65; without one the supremacy-only rule stays.

**2026-11-02 · Penalty takers from this season's evidence.**
Only 13 clubs had a designated taker (FPL order or notes), so 23 clubs' penalties went by xG share.
Now scored per player: UCL penalties taken this season 3.0, league penalties taken this season 2.0
(starting11.com tables, 28 clubs), last season's attempts at the club 0.7, the season before 0.3,
the existing order 1.5, set_pieces.csv 100. Top score of 1.4 or more = taker, runner-up = backup
(weight 8 against the taker's 50 in the simulation). 30 of 36 clubs now have one (not BOD, LAS, SAB,
SLA, SLO, VIK). Changes that matter: Raphinha is Barcelona's taker (3/3 in the league; Yamal 1/2 is
the backup), Grimaldo is Atlético's (2/2), Da Cunha is Como's (4/4; Nico Paz 0/2 backup).
LaLiga's order matches analiticafantasy.com's 2026/27 guide for Barcelona, Real Madrid and Betis.

**2026-11-02 · Keepers' previous XIs count four times as much (kappa_gk 0.25).**
scripts/tune_xi_history.py on MD2 and MD3: keeper start log loss 0.197 -> 0.179, outfield unchanged,
best overall at kappa 1.0, decay 0.5-0.7 (0.7 kept). 0.25 is the edge of the grid; not pushed lower
on two matchdays.

**2026-11-02 · Domestic XIs in the start model: tested, rejected.**
The five English clubs' league starters from FPL (GW6 before MD2, GW7 before MD3) added to the Beta
update at weight 0.3 / 0.5 / 1.0 / 1.5: start log loss MD2 0.416 -> 0.421 / 0.425 / 0.434 / 0.443,
MD3 0.517 -> 0.519 / 0.524 / 0.541 / 0.559. Managers rotate between league and UCL games, so the
weekend XI does not predict the midweek one. Not added. Reverse if: a larger sample shows a gain
(or a negative weight, i.e. rested-at-the-weekend players starting midweek, is worth a test).

**2026-11-02 · Live call from game-feed points; mini-league mode.**
Between a matchday's days the workflow computes each manager file's day-1 points as the feed's
season totals minus the copy saved before the deadline (feed_pre_md{N}.csv, kept by data-sync),
then runs the existing whole-plan policy (captain switch, up to four subs, auto-sub cover) and
reports the chance the new captain beats the points already banked. The XI is the file's md{N}
block, else the plan frozen at the deadline, else the model's best XI. Mini-league mode ranks your
captain and transfer options by P(top after MD8) from the joint simulation (rivals assumed to play
their best expected XI), so a differential captain can rank above the expected-points pick when you
trail. It switches on when every manager file has league_points.

**2026-11-03 · Neural-network challenger (ufe/ml/), shown beside the model, not yet in the picks.**
A multi-task network in NumPy (two hidden layers, 32 and 16 units; outputs: start, goals per 90,
assists per 90; an average of three training runs) trained on 108,984 Premier League player-matches
from the FPL archive (2022/23 GW16, when FPL began recording starts and xG, to 2026/27 GW6), with
features that can be built the same way for UCL players (position, price inside the club, team and
opponent expected goals, rest days, start history, the last two XIs, xG/xA share of team xG).
Against a linear model on the same features, on seasons it never saw:
2025/26 start log loss 0.2825 v 0.2947, goals 0.2579 v 0.2602, assists 0.2670 v 0.2676;
2026/27 0.3475 v 0.3545, 0.2798 v 0.2822, 0.2731 v 0.2753 (better on all six).
In the engine on UCL matchdays, pre-match information only (ufe/ml/evaluate.py), start log loss
model / 50-50 blend / network: MD1 0.541 / 0.528 / 0.621, MD2 0.358 / 0.353 / 0.354, MD3 0.430 /
0.418 / 0.431; goal log loss MD2 0.1651 / 0.1652 / 0.1670, MD3 0.1756 / 0.1731 / 0.1710; expected v
event points correlation MD2 0.518 / 0.524 / 0.527, MD3 0.427 / 0.433 / 0.436. The blend beats the
model on 7 of 8 comparisons (one tie); the network alone does not (MD1 starts: it has no UCL XIs
to go on). settings ml.blend stays 0 for MD4, as promised before the deadline; switch to 0.5 for
MD5 if MD4's grading agrees. Retrained every Monday by .github/workflows/ml.yml.


**2026-11-03 · Four models side by side (ufe/compare/): statistical, neural network, bookmaker, decision trees.**
Every model predicts every matchday through the same simulation, points and optimiser, so they
differ only in their inputs. Bookmaker: match goal rates from prices alone (1X2 plus the goals line;
1X2 only: same supremacy, total at the UCL two-year average; no prices: the statistical rates,
flagged), no line-up adjustment (prices carry the news); players as the statistical model. It tests
whether the team model adds anything over the market. Decision trees: LightGBM (free, MIT licence)
on the network's rows and features (start: binary; goals and assists: Poisson with a log-minutes
offset), trees chosen on 2025/26, scored once on 2026/27. Premier League 2026/27 test, trees /
network / linear: start log loss 0.3432 / 0.3475 / 0.3545, goals 0.2794 / 0.2797 / 0.2822, assists 0.2717 / 0.2731 / 0.2753
(trees best on all three). UCL MD1-3 rebuilt with what was known at each deadline (compare/leaderboard.py reads the
pre-deadline copies of news and prices; MD2's news file had been updated on day 2, so MD2 uses the
UEFA predicted line-ups saved before its deadline; MD1 prices backfilled from rg.org's preview
articles; MD2's articles cover only 7 games, so the bookmaker model skips MD2). On the matchdays
every model has (MD1 and MD3 for starts and results, MD3 for goals and points): start log loss
statistical = bookmaker 0.481, network 0.527, trees 0.571 (the learned models have no UCL XIs to
use at MD1); goal log loss trees 0.171, network 0.171, statistical 0.176, bookmaker 0.178; points
correlation trees 0.443, network 0.433, bookmaker 0.424, statistical 0.424; result log loss
bookmaker 0.739 v 0.753; total goals off by 10.3 a matchday for the bookmaker (1X2-only prices with
an average total) v 2.2. Scores won (matchday x score, ties split): trees 5.2, network 4.7,
statistical 4.2, bookmaker 4.0. Too few matchdays to pick a winner; nothing changes in the picks.
Deadline forecasts for every manager's saved team, from every model, are written in the three
hours before each deadline and frozen after it (forecast_{manager}.json); MD4's was computed just
after the deadline from inputs frozen before it, and says so. Reverse if: after MD6 one model leads
on most scores, then it (or a blend weighted by accuracy) feeds the picks.


**2026-11-11 · Error audit on MD1-4 against UEFA's own numbers; combined model feeds the picks (v0.18.0).**
UEFA's game feed gives each player's points per matchday and season totals of minutes, goals,
assists, clean sheets, recoveries and awards. The statistical model was rebuilt as of each matchday
with pre-deadline inputs and compared (scratch audit, 10-11 Nov). Found and fixed:
1. Recoveries: 5.85 per 90 assumed for midfielders v 3.51 real, forwards 2.40 v 1.36, keepers 2.85
   v 0. recovery_ratio refitted (MID 0.78, FWD 0.64, GK 0); each player's own rate now updates.
2. Assists inside a club: forwards 49 v 27 expected, midfielders 132 v 150 (assist_pos_mult).
   Starters per club by position 3.96 / 4.78 / 1.26, not 4.1 / 3.6 / 2.3 (player_rates.SLOTS).
3. Outfield players at 90%+ to start started 83% of the time; expected starters per matchday ran
   from 356 to 400 against 396. Each club's start chances now sum to 10.8, outfield capped at 0.85.
4. The status column was a mid-October copy: 25 fit players marked injured, 59 UEFA-flagged injured
   players treated as available. Availability now comes from UEFA's flags at each sync (MD4: 0 of
   85 flagged injured started, 1 of 121 not in squad, 8 of 25 doubtful).
5. Absences in a weaker predicted line-up file were treated as certain. MD4 (third-party file,
   strength 0.7): 45 listed out but unflagged by UEFA, 7 started; 6 listed suspended, 4 started;
   69 also flagged by UEFA, 0 started. team_news.absence_mult: unconfirmed out 0.4, suspended 0.85
   of the prior chance (flat between 0.3 and 0.6; 0.443 -> 0.370 on MD4, 0.371 with the numbers
   chosen on the other clubs). Players given under 0.5% who started: 15 of 242 before, 0 of 168 now.
Measured on MD1-4, statistical model on 10 Nov -> combined model now: start log loss 0.447 -> 0.407
(by matchday 0.529 / 0.359 / 0.433 / 0.469 -> 0.517 / 0.353 / 0.393 / 0.366); expected v official
points, all players: correlation 0.521 -> 0.532, RMSE 2.345 -> 2.326; starters: correlation 0.363 ->
0.400, bias +0.40 -> +0.06. Points scored by the 20 highest-rated players per matchday: 498 -> 490
over four matchdays, no change (the two lists share 88% of names; noise is about 40 points).
The constants in 1-3 and 5 were fitted on these four matchdays, so MD5 is the first clean test.
Combined (mean of the statistical, network and trees player priors, weights fixed at a third
each before grading) v statistical, same inputs: start log loss lower on all four matchdays (paired
difference 0.0071, 95% interval 0.0015 to 0.0129, club-matchdays resampled); squared points error
lower by 0.026 (interval -0.032 to 0.083); best four keepers, 12 defenders, 12 midfielders, 8
forwards per matchday 750 v 722 points. engine.model: combined.
Tested and not changed: (a) form: a starter's over- or under-performance against the model did not
carry to his next match (958 player-matches, slope +0.016, s.e. 0.043), so UCL results move a
player's rates slowly (ucl_obs, k 10 + 8 x history weight); (b) the predicted line-up labels: no
set of weights beat the current one on MD4 (0.3699 to 0.3701 v 0.3704, worse leave-one-club-out);
(c) players priced 8.0+: 6.1 expected v 7.1 scored per start, from 18 players: goals per 90 0.64 v
0.68, assists 0.31 v 0.36, awards 8.1 v 14; start rate at the cap 0.82 v 0.85. No structural cause,
left alone; (d) total goals 268 predicted v 291 (z about 1.3 with 72 matches; the bookmaker's
totals are lower still), left alone; team clean sheets 28.9 v 27.
`ufe calibrate` (weekly) refits 1-2 from the feed, at most 15% a run.
Reverse if: after MD6 the statistical model's start log loss or points RMSE beats the combined
model's on MD5 and MD6 both; then engine.model goes back to statistical.

