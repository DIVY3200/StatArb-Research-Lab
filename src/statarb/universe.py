"""Training-period pair screening with false-discovery control.

The screen aligns all candidate series on their common dates, keeps only the first
``train_fraction`` of that common sample, and runs one Engle-Granger test per unordered
pair (the first-listed ticker is the dependent variable). Benjamini-Hochberg adjusted
p-values are reported alongside raw p-values.

Caveats reported with every screen:

* Every additional pair tested raises the chance that some pass by luck. With ``m``
  independent tests at level ``alpha`` about ``alpha * m`` false positives are expected.
* Benjamini-Hochberg controls the false-discovery rate among the *formally* tested pairs;
  it does not account for informal pre-screening, parameter searches or the choice of universe.
* A universe built from today's listed companies suffers from survivorship bias: firms that
  were delisted, merged or failed are missing from the history.
* Screening never looks at validation or test data, and must not be re-run on them to pick
  a winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .cointegration import benjamini_hochberg, engle_granger_test
from .data_loader import AssetData, generate_synthetic_pair


@dataclass
class ScreenResult:
    table: pd.DataFrame
    n_tickers: int
    n_pairs_tested: int
    n_common_obs: int
    train_start: str
    train_end: str
    significance: float
    fdr_alpha: float
    criterion: str
    warnings: list[str] = field(default_factory=list)


def screen_pairs(
    prices: dict[str, pd.Series],
    train_fraction: float = 0.6,
    significance: float = 0.05,
    fdr_alpha: float = 0.05,
    min_obs: int = 252,
    autolag: str | None = "AIC",
) -> ScreenResult:
    """Screen all unordered pairs of cleaned price series using training data only."""
    if len(prices) < 2:
        raise ValueError("At least two tickers are needed for a pair screen.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    wide = pd.concat(prices, axis=1, join="inner").sort_index()
    if wide.isna().any().any() or (wide <= 0).any().any():
        raise ValueError("Price series must be cleaned (positive, non-missing) before screening.")
    n_train = int(np.floor(len(wide) * train_fraction))
    if n_train < min_obs:
        raise ValueError(
            f"Only {n_train} common training observations across the universe (minimum {min_obs}). "
            "Use fewer tickers with longer overlapping histories, or a longer date range."
        )
    train = np.log(wide.iloc[:n_train])
    warnings: list[str] = []
    shortest = min(len(s) for s in prices.values())
    if len(wide) < 0.9 * shortest:
        warnings.append(
            f"Aligning {len(prices)} tickers on common dates kept {len(wide)} rows; some series lost more than 10% "
            "of their history."
        )

    rows = []
    for dep, ind in combinations(list(wide.columns), 2):
        row: dict[str, object] = {"dependent": dep, "independent": ind}
        try:
            result = engle_granger_test(
                train[dep], train[ind], significance=significance, autolag=autolag, min_obs=min_obs,
                name_a=dep, name_b=ind, check_integration=False,
            )
            returns = train[[dep, ind]].diff().dropna()
            row.update(
                hedge_ratio=result.hedge_ratio, eg_statistic=result.test_statistic, p_value=result.p_value,
                return_correlation=float(returns[dep].corr(returns[ind])), error="",
            )
        except ValueError as exc:
            row.update(hedge_ratio=np.nan, eg_statistic=np.nan, p_value=np.nan, return_correlation=np.nan, error=str(exc))
        rows.append(row)
    table = pd.DataFrame(rows)
    valid = table["p_value"].notna()
    table["bh_adjusted_p"] = np.nan
    table["passes_raw"] = False
    table["passes_bh"] = False
    if valid.any():
        rejected, adjusted = benjamini_hochberg(table.loc[valid, "p_value"].to_numpy(), fdr_alpha)
        table.loc[valid, "bh_adjusted_p"] = adjusted
        table.loc[valid, "passes_bh"] = rejected
        table.loc[valid, "passes_raw"] = table.loc[valid, "p_value"] < significance
    table = table.sort_values("p_value", na_position="last").reset_index(drop=True)

    m = int(valid.sum())
    expected_false = significance * m
    warnings.append(
        f"{m} pairs were tested. Even if none were truly cointegrated, about {expected_false:.1f} would be expected "
        f"to pass at the raw {significance:.0%} level by chance."
    )
    warnings.append(
        "Survivorship bias: a universe of currently listed tickers omits delisted and merged companies."
    )
    return ScreenResult(
        table=table,
        n_tickers=len(prices),
        n_pairs_tested=m,
        n_common_obs=int(len(wide)),
        train_start=str(wide.index[0].date()),
        train_end=str(wide.index[n_train - 1].date()),
        significance=significance,
        fdr_alpha=fdr_alpha,
        criterion=(
            f"Engle-Granger p-value on the first {train_fraction:.0%} of common dates; raw threshold {significance:.0%}, "
            f"Benjamini-Hochberg FDR {fdr_alpha:.0%}."
        ),
        warnings=warnings,
    )


def synthetic_universe(seed: int = 7, n_obs: int = 1500) -> dict[str, AssetData]:
    """Six labelled synthetic assets: two cointegrated pairs and one independent pair."""
    assets: dict[str, AssetData] = {}
    specs = [
        (True, "SYNTH_U1", seed),
        (True, "SYNTH_U2", seed + 1),
        (False, "SYNTH_U3", seed + 2),
    ]
    for cointegrated, prefix, s in specs:
        a, b = generate_synthetic_pair(n_obs=n_obs, seed=s, cointegrated=cointegrated, prefix=prefix)
        assets[a.ticker] = a
        assets[b.ticker] = b
    return assets


__all__ = ["ScreenResult", "screen_pairs", "synthetic_universe"]
