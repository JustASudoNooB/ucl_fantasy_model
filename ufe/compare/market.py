"""Bookmaker model: every match's goal rates from the bookmakers' prices alone.

  with a goals line (over/under):   home and away goal rates that reproduce the 1X2 and the line
  1X2 prices only:                  the same supremacy, total set to the Champions League average
                                    (all UCL matches in the two years before `as_of`)
  no prices for the fixture:        the statistical model's rates, marked "model (no odds)"
Nothing else differs from the statistical engine: the same players, start probabilities (XI
history, team news) and chance shares. Prices already carry team news, so there is no line-up
adjustment of the rates and no extra rating uncertainty. Comparing the two answers one question:
does the team model add anything the market has not priced in?
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from ..models.scoreline import implied_lambdas, remove_margin, rescale_total
from ..pipeline import Engine

UCL = {"cl", "ucl_current"}


class MarketEngine(Engine):
    @cached_property
    def ucl_avg_total(self) -> float:
        m = self.matches
        t = pd.Timestamp(self.as_of)
        c = m[m.comp.isin(UCL) & (m.date < t) & (m.date >= t - pd.Timedelta(days=730))].dropna(subset=["hg", "ag"])
        return float((c.hg + c.ag).mean()) if len(c) >= 50 else 3.0

    def fixture_table(self, md: int) -> pd.DataFrame:
        return self._base_fixture_table(md).reset_index(drop=True)

    def _base_fixture_table(self, md: int) -> pd.DataFrame:
        f = Engine._base_fixture_table(self, md)
        odds = self._odds(md)
        meth = self.cfg["market"]["margin_method"]
        lh, la, src = list(f.lh), list(f.la), list(f.lambda_source)
        sh, sa = list(f.sh), list(f.sa)
        for i, r in enumerate(f.itertuples()):
            if odds is None or r.fixture_id not in odds.index:
                src[i] = "model (no odds)"
                continue
            o = odds.loc[r.fixture_id]
            ph, pdr, pa = remove_margin([o.home_odds, o.draw_odds, o.away_odds], meth)
            p_over, line = None, 2.5
            if pd.notna(o.get("total_line", np.nan)) and pd.notna(o.get("over_odds", np.nan)) and pd.notna(o.get("under_odds", np.nan)):
                p_over, line = float(remove_margin([o.over_odds, o.under_odds], "proportional")[0]), float(o.total_line)
            elif pd.notna(o.get("over25_odds", np.nan)) and pd.notna(o.get("under25_odds", np.nan)):
                p_over = float(remove_margin([o.over25_odds, o.under25_odds], "proportional")[0])
            lam = implied_lambdas(ph, pdr, pa, p_over, rho=self.team_model.rho, line=line)
            if p_over is None:
                lam, src[i] = rescale_total(lam, self.ucl_avg_total), "market (1X2, average total)"
            else:
                src[i] = "market"
            lh[i], la[i], sh[i], sa[i] = float(lam[0]), float(lam[1]), 0.0, 0.0
        f["lh"], f["la"], f["lambda_source"], f["sh"], f["sa"] = lh, la, src, sh, sa
        return f

    def odds_coverage(self, md: int) -> float:
        f = self._base_fixture_table(md)
        return float(f.lambda_source.str.startswith("market").mean()) if len(f) else 0.0
