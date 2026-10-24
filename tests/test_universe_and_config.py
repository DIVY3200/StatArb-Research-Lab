import numpy as np
import pytest

from statarb.config import DEFAULT_CONFIG_PATH, ResearchConfig, load_config
from statarb.universe import screen_pairs, synthetic_universe


@pytest.fixture(scope="module")
def universe_prices():
    return {ticker: asset.frame["Adj Close"] for ticker, asset in synthetic_universe(seed=7, n_obs=1200).items()}


def test_screen_counts_and_reports_corrections(universe_prices):
    result = screen_pairs(universe_prices, train_fraction=0.6, significance=0.05)
    assert result.n_tickers == 6 and result.n_pairs_tested == 15
    table = result.table
    assert {"p_value", "bh_adjusted_p", "passes_raw", "passes_bh"}.issubset(table.columns)
    assert (table["bh_adjusted_p"] >= table["p_value"] - 1e-12).all()
    assert table["passes_bh"].sum() <= table["passes_raw"].sum()
    assert any("expected" in w for w in result.warnings) and any("Survivorship" in w for w in result.warnings)
    # The two planted cointegrated pairs have the smallest raw p-values (fixed seed).
    top_two = set(zip(table["dependent"].iloc[:2], table["independent"].iloc[:2]))
    assert top_two == {("SYNTH_U1_A", "SYNTH_U1_B"), ("SYNTH_U2_A", "SYNTH_U2_B")}
    # With 15 tests the BH correction is stricter than the raw threshold: here it removes every
    # raw discovery, including the independent SYNTH_U3 pair that passes at 5% purely by chance.
    u3 = table.loc[(table["dependent"] == "SYNTH_U3_A") & (table["independent"] == "SYNTH_U3_B")].iloc[0]
    assert u3["passes_raw"] and not u3["passes_bh"]


def test_screen_uses_training_data_only(universe_prices):
    base = screen_pairs(universe_prices, train_fraction=0.6)
    n_train = int(np.floor(1200 * 0.6))
    tampered = {k: v.copy() for k, v in universe_prices.items()}
    for series in tampered.values():
        series.iloc[n_train:] *= np.linspace(1, 3, len(series) - n_train)
    after = screen_pairs(tampered, train_fraction=0.6)
    np.testing.assert_allclose(base.table["p_value"].to_numpy(), after.table["p_value"].to_numpy())


def test_screen_requires_two_tickers(universe_prices):
    with pytest.raises(ValueError):
        screen_pairs({"X": next(iter(universe_prices.values()))})


def test_default_config_file_loads_and_round_trips():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.strategy.entry_z == 2.0 and config.strategy.exit_z == 0.5
    assert config.costs.commission_bps > 0  # the main backtest uses non-zero costs by default
    assert ResearchConfig.from_dict(config.to_dict()) == config
    assert len(config.fingerprint()) == 12


@pytest.mark.parametrize("overrides, message", [
    ({"strategy": {"entry_zz": 1.0}}, "Unknown keys"),
    ({"strategy": {"execution_lag": 0}}, "execution_lag"),
    ({"splits": {"train": 0.7}}, "sum to 1"),
    ({"strategy": {"exit_z": 2.5}}, "exit_z"),
    ({"data": {"source": "bloomberg"}}, "data.source"),
])
def test_invalid_config_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        ResearchConfig().with_overrides(overrides)
