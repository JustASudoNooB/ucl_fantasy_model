# CSV templates

Copy a template, fill it, and pass it to the matching command. Names are matched fuzzily within
the club, so "Olise" with team BAY is enough; ids (see `python -m ufe players --search olise`)
always match exactly. Club codes are in `config/teams_2026_27.yaml`.

| File | Used by | Notes |
|---|---|---|
| `squad.csv` | `advise`, `live`, the scheduler (copy to `data/my/squad.csv`) | 15 rows: 2 GK, 5 DEF, 5 MID, 3 FWD |
| `results_md2.csv` | `results --results` | 90-minute score; fill both goal columns |
| `player_results_md2.csv` | `results --players` | any subset of columns; more columns = more learning |
| `day1_points.csv` | `live --points` | points of your players whose games have finished |
| `odds_md3.csv` | copy to `data/processed/odds_md3.csv` | decimal odds; median across bookmakers is best |
| `lineups_md3.csv` | copy to `data/processed/lineups_md3.csv` or use `ufe lineup` | role = start / bench |
| `availability.csv` | copy to `data/processed/availability.csv` | p_available multiplies start and sub chances |
