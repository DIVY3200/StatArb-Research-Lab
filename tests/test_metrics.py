import numpy as np
import pandas as pd
import pytest

from conftest import signal_frame
from statarb.backtester import BacktestConfig, run_backtest
from statarb.metrics import (
    annualized_volatility,
    benchmark_returns,
    cagr,
    calmar_ratio,
    comparison_table,
    cumulative_return,
    daily_risk_free,
    drawdown_series,
    max_drawdown,
    return_metrics,
    sharpe_ratio,
    strategy_metrics,
    trade_statistics,
)


def _r(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_cumulative_return_and_cagr():
    r = _r([0.001] * 252)
    assert cumulative_return(r) == pytest.approx(1.001**252 - 1)
    assert cagr(r, 252, min_years=1.0) == pytest.approx(1.001**252 - 1)
    two_years = _r([0.0005] * 504)
    assert cagr(two_years, 252) == pytest.approx(1.0005**252 - 1)


def test_volatility_and_sharpe_known_values():
    r = _r([0.01, -0.01] * 50)
    assert annualized_volatility(r, 252) == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    r2 = _r([0.002, 0.0] * 100)
    expected = r2.mean() / r2.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r2, 0.0, 252) == pytest.approx(expected)
    rf = 0.03
    rf_d = (1 + rf) ** (1 / 252) - 1
    assert daily_risk_free(rf, 252) == pytest.approx(rf_d)
    assert sharpe_ratio(r2, rf, 252) == pytest.approx((r2 - rf_d).mean() / (r2 - rf_d).std(ddof=1) * np.sqrt(252))


def test_drawdown_hand_calculated():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])
    np.testing.assert_allclose(drawdown_series(equity).to_numpy(), [0, 0, -0.25, 0, -0.1])
    assert max_drawdown(equity) == pytest.approx(-0.25)
    # Drawdowns measured from returns include the starting capital.
    assert return_metrics(_r([-0.1, 0.05]))["max_drawdown"] == pytest.approx(-0.1)


def test_undefined_metrics_are_nan_not_infinite():
    empty = _r([])
    assert np.isnan(cumulative_return(empty))
    assert np.isnan(sharpe_ratio(_r([0.01])))
    flat = _r([0.0] * 300)
    assert np.isnan(sharpe_ratio(flat))  # zero volatility: undefined, not +/-inf
    assert np.isnan(calmar_ratio(flat))  # no drawdown
    assert np.isnan(cagr(_r([0.001] * 100), 252, min_years=1.0))  # shorter than one year
    assert np.isnan(cagr(_r([-1.0] + [0.0] * 300)))  # wiped out
    assert np.isnan(annualized_volatility(_r([0.02])))
    assert np.isnan(max_drawdown(pd.Series([1.0])))
    for value in return_metrics(flat).values():
        assert not np.isinf(value)


def test_trade_statistics_with_and_without_trades():
    empty = pd.DataFrame(columns=["status", "net_pnl", "net_return", "gross_return", "holding_days"])
    stats = trade_statistics(empty)
    assert stats["completed_trades"] == 0 and np.isnan(stats["win_rate"]) and np.isnan(stats["avg_holding_days"])
    trades = pd.DataFrame({
        "status": ["closed", "closed", "closed", "open"],
        "net_pnl": [100.0, -50.0, 25.0, 999.0],
        "net_return": [0.01, -0.005, 0.0025, 0.1],
        "gross_return": [0.011, -0.004, 0.003, 0.1],
        "holding_days": [5, 10, 3, 1],
    })
    stats = trade_statistics(trades)
    assert stats["completed_trades"] == 3  # the open trade is excluded
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["avg_holding_days"] == pytest.approx(6.0)
    assert stats["profit_factor"] == pytest.approx(125 / 50)
    winners_only = trades.iloc[[0, 2]]
    assert np.isnan(trade_statistics(winners_only)["profit_factor"])


def test_benchmarks_use_same_dates_and_start_flat():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 45.0]},
                          index=pd.bdate_range("2021-01-01", periods=3))
    bench = benchmark_returns(prices, {"A": "AAA", "B": "BBB"}, rf_annual=0.0)
    assert bench.index.equals(prices.index)
    assert (bench.iloc[0] == 0).all()
    assert cumulative_return(bench["Buy & hold AAA"]) == pytest.approx(0.21)
    assert cumulative_return(bench["Buy & hold BBB"]) == pytest.approx(-0.10)
    assert cumulative_return(bench["50/50 buy & hold"]) == pytest.approx(0.5 * 1.21 + 0.5 * 0.9 - 1)
    assert cumulative_return(bench["Cash (risk-free)"]) == 0.0
    table = comparison_table(bench["Buy & hold AAA"], bench["Buy & hold AAA"], bench)
    assert "Pairs strategy (net)" in table.columns and "sharpe_ratio" in table.index


def test_strategy_metrics_consistent_with_backtest_equity():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-01", periods=300)
    prices = pd.DataFrame({"A": 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(300))),
                           "B": 80 * np.exp(np.cumsum(0.01 * rng.standard_normal(300)))}, index=idx)
    targets = np.repeat([0, 1, 0, -1, 0, 1], 50)
    result = run_backtest(prices, signal_frame(idx, targets), 1.0, BacktestConfig(commission_bps=5, slippage_bps=5))
    m = strategy_metrics(result)
    assert m["cumulative_return"] == pytest.approx(result.equity.iloc[-1] / 1_000_000 - 1)
    assert m["max_drawdown"] == pytest.approx(result.daily["drawdown"].min())
    assert m["gross_cumulative_return"] >= m["cumulative_return"]
    assert m["completed_trades"] == 3
    assert 0 < m["pct_time_invested"] < 1
    assert m["total_transaction_costs"] == pytest.approx(result.daily["trade_cost"].sum())
