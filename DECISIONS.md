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
