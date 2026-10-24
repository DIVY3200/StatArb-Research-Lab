import json

import numpy as np
import pandas as pd
import pytest

from statarb.config import ResearchConfig, load_config
from statarb.data_validation import AlignedPair
from statarb.evaluation import (
    chronological_split,
    run_pair_research,
    run_walk_forward,
    save_research_run,
    select_parameters,
    walk_forward_schedule,
)


def _with_prices(pair: AlignedPair, prices: pd.DataFrame) -> AlignedPair:
    return AlignedPair(prices, pair.ticker_a, pair.ticker_b, pair.price_column_a, pair.price_column_b, pair.report,
                       pair.validation_a, pair.validation_b, pair.metadata_a, pair.metadata_b)


# --------------------------------------------------------------------------- splits
def test_chronological_split_is_contiguous_ordered_and_complete():
    index = pd.bdate_range("2015-01-01", periods=1000)
    splits = chronological_split(index, 0.6, 0.2, 0.2)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    assert (train.start_idx, train.end_idx) == (0, 600)
    assert (val.start_idx, val.end_idx) == (600, 800)
    assert (test.start_idx, test.end_idx) == (800, 1000)
    assert train.end_date < val.start_date and val.end_date < test.start_date
    assert train.n_obs + val.n_obs + test.n_obs == len(index)


@pytest.mark.parametrize("props", [(0.6, 0.3, 0.2), (0.0, 0.5, 0.5), (0.8, 0.2, 0.0)])
def test_invalid_split_proportions(props):
    with pytest.raises(ValueError):
        chronological_split(pd.bdate_range("2015-01-01", periods=500), *props)


def test_split_rejects_unsorted_index_and_tiny_blocks():
    index = pd.bdate_range("2015-01-01", periods=500)
    with pytest.raises(ValueError, match="strictly increasing"):
        chronological_split(index[::-1])
    with pytest.raises(ValueError, match="minimum"):
        chronological_split(pd.bdate_range("2015-01-01", periods=60))


# --------------------------------------------------------------------------- walk-forward
def test_walk_forward_schedule_rolling_and_expanding():
    rolling = walk_forward_schedule(1000, 300, 100, expanding=False, min_eval_obs=20)
    assert [w.eval_start_idx for w in rolling] == [300, 400, 500, 600, 700, 800, 900]
    for w in rolling:
        assert w.fit_end_idx == w.eval_start_idx  # fit strictly precedes evaluation
        assert w.fit_end_idx - w.fit_start_idx == 300
    for prev, nxt in zip(rolling, rolling[1:]):
        assert prev.eval_end_idx == nxt.eval_start_idx  # contiguous, non-overlapping
    expanding = walk_forward_schedule(1000, 300, 100, expanding=True)
    assert all(w.fit_start_idx == 0 for w in expanding)
    partial = walk_forward_schedule(1010, 300, 100, min_eval_obs=20)
    assert partial[-1].eval_end_idx == 1000  # a 10-bar tail is too short to evaluate
    with pytest.raises(ValueError, match="Not enough data"):
        walk_forward_schedule(310, 300, 100)


def test_walk_forward_run_is_chronological_and_chained(coint_pair):
    config = ResearchConfig().with_overrides({"walk_forward": {"train_window": 300, "test_window": 150}})
    wf = run_walk_forward(coint_pair.prices, config)
    assert wf.daily.index.is_monotonic_increasing and not wf.daily.index.has_duplicates
    assert (pd.to_datetime(wf.windows["fit_end"]) < pd.to_datetime(wf.windows["eval_start"])).all()
    starts, ends = wf.windows["start_equity"].to_numpy(), wf.windows["end_equity"].to_numpy()
    np.testing.assert_allclose(starts[1:], ends[:-1])  # capital carried forward
    assert wf.daily.index[0] == coint_pair.prices.index[300]
    assert wf.metrics["final_equity"] == pytest.approx(ends[-1])


def test_walk_forward_skips_windows_that_fail_cointegration(indep_pair):
    config = ResearchConfig().with_overrides({"walk_forward": {"train_window": 300, "test_window": 150}})
    wf = run_walk_forward(indep_pair.prices, config)
    untraded = wf.windows.loc[~wf.windows["traded"]]
    assert (untraded["cointegrated"] == False).all()  # noqa: E712
    flat_days = wf.daily.loc[wf.daily["window"].isin(untraded["window"])]
    assert (flat_days["position"] == 0).all()


# --------------------------------------------------------------------------- research pipeline integrity
def test_test_period_hidden_until_revealed(coint_pair, config):
    hidden = run_pair_research(coint_pair, config, include_walk_forward=False)
    splits = hidden.splits
    assert "test" not in hidden.periods
    assert hidden.research_end_idx == splits["validation"].end_idx
    assert len(hidden.zscore) == splits["validation"].end_idx
    revealed = run_pair_research(coint_pair, config, reveal_test=True, include_walk_forward=False)
    assert "test" in revealed.periods and len(revealed.zscore) == len(coint_pair.prices)


def test_validation_and_grid_unaffected_by_test_period_data(coint_pair, config):
    base = run_pair_research(coint_pair, config, include_walk_forward=True)
    prices = coint_pair.prices.copy()
    start = base.splits["test"].start_idx
    prices.iloc[start:] = prices.iloc[start:] * np.array([1.5, 0.6])
    tampered = run_pair_research(_with_prices(coint_pair, prices), config, include_walk_forward=True)
    pd.testing.assert_frame_equal(base.validation_grid, tampered.validation_grid)
    pd.testing.assert_frame_equal(base.periods["validation"].backtest.daily, tampered.periods["validation"].backtest.daily)
    assert base.selected_params == tampered.selected_params
    pd.testing.assert_frame_equal(base.walk_forward.daily, tampered.walk_forward.daily)


def test_validation_selected_parameters_feed_the_test_period(coint_pair):
    config = ResearchConfig().with_overrides({"evaluation": {"test_parameter_source": "validation_selected",
                                                             "min_trades_for_selection": 1}})
    result = run_pair_research(coint_pair, config, reveal_test=True, include_walk_forward=False)
    assert result.selected_params is not None
    assert result.test_params["source"] == "validation_selected"
    assert result.periods["test"].rules.entry_z == result.selected_params["entry_z"]


def test_select_parameters_respects_minimum_trades():
    grid = pd.DataFrame({"entry_z": [1.5, 2.0], "exit_z": [0.5, 0.5], "completed_trades": [1, 5],
                         "sharpe_ratio": [3.0, 1.0], "cumulative_return": [0.1, 0.05], "calmar_ratio": [1, 1]})
    assert select_parameters(grid, "sharpe", min_trades=3)["entry_z"] == 2.0
    assert select_parameters(grid, "sharpe", min_trades=10) is None


def test_failed_cointegration_is_warned(indep_pair, config):
    result = run_pair_research(indep_pair, config, include_walk_forward=False)
    assert not result.model.coint.reject_null
    assert any("does not pass" in w for w in result.warnings)
    assert any("SYNTHETIC" in w for w in result.warnings)


def test_save_research_run_snapshot(coint_pair, config, tmp_path):
    result = run_pair_research(coint_pair, config, reveal_test=True)
    run_dir = save_research_run(result, tmp_path)
    expected = {"config_snapshot.yaml", "run_metadata.json", "statistical_tests.json", "aligned_prices.csv",
                "daily_portfolio_test.csv", "trades_validation.csv", "signals_train.csv", "positions_test.csv",
                "strategy_metrics.csv", "walk_forward_windows.csv", "validation_parameter_grid.csv",
                "cost_sensitivity_test.csv"}
    assert expected.issubset({p.name for p in run_dir.iterdir()})
    assert load_config(run_dir / "config_snapshot.yaml") == config
    meta = json.loads((run_dir / "run_metadata.json").read_text())  # strict JSON (no NaN tokens)
    assert meta["asset_a"]["is_synthetic"] is True
    assert len(meta["aligned_prices_sha256"]) == 64
    tests = json.loads((run_dir / "statistical_tests.json").read_text())
    assert tests["label"] == "Training-period estimates"
