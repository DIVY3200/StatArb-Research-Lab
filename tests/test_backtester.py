import numpy as np
import pandas as pd
import pytest

from conftest import signal_frame
from statarb.backtester import BacktestConfig, run_backtest
from statarb.portfolio import CostModel, target_leg_weights

CAPITAL = 1_000_000.0


def _prices(a, b):
    index = pd.bdate_range("2021-01-04", periods=len(a))
    return pd.DataFrame({"A": a, "B": b}, index=index, dtype=float)


def _cfg(**overrides):
    base = dict(initial_capital=CAPITAL, commission_bps=0.0, slippage_bps=0.0, borrow_bps_annual=0.0,
                rebalance_tolerance=None, max_gross_exposure=1.0)
    base.update(overrides)
    return BacktestConfig(**base)


def _random_prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    a = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    b = 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    return _prices(a, b)


# --------------------------------------------------------------------------- execution timing
@pytest.mark.parametrize("lag, expected", [
    (1, [0, 0, 1, 1, 1, 0, 0, -1, -1, 0]),
    (2, [0, 0, 0, 1, 1, 1, 0, 0, -1, 0]),
])
def test_positions_follow_signals_after_execution_lag(lag, expected):
    prices = _random_prices(10)
    targets = [0, 1, 1, 1, 0, 0, -1, -1, 0, 0]
    result = run_backtest(prices, signal_frame(prices.index, targets), 1.0, _cfg(execution_lag=lag))
    assert result.daily["position"].tolist() == expected  # final bar is always flat


def test_signal_cannot_earn_the_move_that_produced_it():
    # A jumps 10% on bar 3. A signal on bar 2 is executed at bar 3's close: the jump is not earned.
    prices = _prices([100, 100, 100, 110, 110, 110], [100] * 6)
    late = run_backtest(prices, signal_frame(prices.index, [0, 0, 1, 1, 1, 0]), 1.0, _cfg())
    assert late.daily["pnl"].abs().sum() == 0.0
    assert late.equity.iloc[-1] == CAPITAL
    # A signal on bar 1 is executed at bar 2's close (price 100) and does earn the bar-3 jump.
    early = run_backtest(prices, signal_frame(prices.index, [0, 1, 1, 1, 1, 0]), 1.0, _cfg())
    assert early.daily["pnl"].iloc[3] == pytest.approx(0.5 * CAPITAL / 100 * 10)


def test_execution_lag_zero_is_rejected():
    with pytest.raises(ValueError, match="execution_lag"):
        BacktestConfig(execution_lag=0)


@pytest.mark.parametrize("cut", [30, 61, 99])
def test_no_look_ahead_truncation_invariance(cut):
    """Results up to bar k must not depend on any price after bar k."""
    prices = _random_prices(120, seed=3)
    rng = np.random.default_rng(4)
    targets = np.repeat(rng.choice([-1, 0, 1], size=24), 5)
    cfg = _cfg(commission_bps=3.0, slippage_bps=2.0, borrow_bps_annual=80.0, rebalance_tolerance=0.05,
               close_at_period_end=False)
    full = run_backtest(prices, signal_frame(prices.index, targets), 0.9, cfg)
    part = run_backtest(prices.iloc[:cut], signal_frame(prices.index[:cut], targets[:cut]), 0.9, cfg)
    pd.testing.assert_frame_equal(full.daily.iloc[:cut], part.daily)


# --------------------------------------------------------------------------- two-leg accounting
def test_position_direction_and_hedge_ratio_leg_weights():
    prices = _prices([100.0] * 6, [40.0] * 6)
    beta = 0.8
    long_ = run_backtest(prices, signal_frame(prices.index, [1, 1, 1, 1, 1, 1]), beta, _cfg())
    row = long_.daily.iloc[2]
    assert row["units_a"] > 0 and row["units_b"] < 0  # long spread = long A, short B
    assert row["notional_b"] / row["notional_a"] == pytest.approx(-beta)
    assert row["gross_exposure"] / row["equity"] == pytest.approx(1.0)
    short = run_backtest(prices, signal_frame(prices.index, [-1] * 6), beta, _cfg())
    row = short.daily.iloc[2]
    assert row["units_a"] < 0 and row["units_b"] > 0  # short spread = short A, long B
    assert row["notional_b"] / row["notional_a"] == pytest.approx(-beta)


def test_target_leg_weights_convention():
    assert target_leg_weights(1, 1.0, 1.0) == (0.5, -0.5)
    wa, wb = target_leg_weights(-1, 0.5, 1.5)
    assert wa == pytest.approx(-1.0) and wb == pytest.approx(0.5)
    assert abs(wa) + abs(wb) == pytest.approx(1.5)
    assert target_leg_weights(0, 2.0, 1.0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        target_leg_weights(2, 1.0, 1.0)


def test_both_leg_returns_are_aligned():
    prices = _prices([100, 100, 101, 101], [50, 50, 50, 49])
    result = run_backtest(prices, signal_frame(prices.index, [1, 1, 1, 1]), 1.0, _cfg())
    # Entered at bar 1: 5,000 units of A (+$500k) and -10,000 units of B (-$500k).
    assert result.daily["units_a"].iloc[1] == pytest.approx(5_000)
    assert result.daily["units_b"].iloc[1] == pytest.approx(-10_000)
    np.testing.assert_allclose(result.daily["pnl"].to_numpy(), [0, 0, 5_000, 10_000])
    assert result.equity.iloc[-1] == pytest.approx(CAPITAL + 15_000)


def test_transaction_costs_charged_on_both_legs_entry_and_exit():
    prices = _prices([100, 100, 101, 101], [50, 50, 50, 49])
    result = run_backtest(prices, signal_frame(prices.index, [1, 1, 1, 1]), 1.0,
                          _cfg(commission_bps=5.0, slippage_bps=5.0))
    rate = 0.001
    budget = CAPITAL / (1 + rate)  # gross notional after paying the entry cost
    entry_cost = budget * rate
    exit_notional = 0.5 * budget / 100 * 101 + 0.5 * budget / 50 * 49
    exit_cost = exit_notional * rate
    np.testing.assert_allclose(result.daily["trade_cost"].to_numpy(), [0, entry_cost, 0, exit_cost])
    assert result.daily["gross_leverage"].iloc[1] == pytest.approx(1.0)
    trade = result.trades.iloc[0]
    assert trade["transaction_costs"] == pytest.approx(entry_cost + exit_cost)
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - entry_cost - exit_cost)
    assert result.equity.iloc[-1] == pytest.approx(CAPITAL + trade["net_pnl"])


def test_cost_model_rates():
    model = CostModel(commission_bps=4, slippage_bps=6, borrow_bps_annual=252, periods_per_year=252)
    assert model.trade_cost(1_000, -3_000) == pytest.approx(4.0)
    assert model.borrow_cost(-10_000) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        CostModel(commission_bps=-1)


def test_borrow_cost_accrues_daily_on_short_leg():
    prices = _prices([100.0] * 6, [100.0] * 6)
    result = run_backtest(prices, signal_frame(prices.index, [1] * 6), 1.0, _cfg(borrow_bps_annual=252.0))
    # Short notional $500k x 1bp/day, charged on bars 2-5 (the position exists from bar 1's close).
    np.testing.assert_allclose(result.daily["borrow_cost"].to_numpy(), [0, 0, 50, 50, 50, 50])
    assert result.equity.iloc[-1] == pytest.approx(CAPITAL - 200)
    assert result.trades.iloc[0]["borrow_costs"] == pytest.approx(200)


def test_zero_cost_diagnostic_net_equals_gross():
    prices = _random_prices(80, seed=9)
    targets = np.repeat([0, 1, 0, -1], 20)
    result = run_backtest(prices, signal_frame(prices.index, targets), 1.1, _cfg())
    np.testing.assert_allclose(result.returns.to_numpy(), result.gross_returns.to_numpy(), atol=1e-15)


def test_accounting_identities_with_costs():
    prices = _random_prices(150, seed=5)
    targets = np.repeat(np.random.default_rng(6).choice([-1, 0, 1], size=15), 10)
    result = run_backtest(prices, signal_frame(prices.index, targets), 0.7,
                          _cfg(commission_bps=5, slippage_bps=5, borrow_bps_annual=100, rebalance_tolerance=0.05))
    d = result.daily
    equity_prev = np.concatenate([[CAPITAL], d["equity"].to_numpy()[:-1]])
    np.testing.assert_allclose(d["equity"], equity_prev + d["pnl"] - d["trade_cost"] - d["borrow_cost"])
    np.testing.assert_allclose(d["cash"] + d["notional_a"] + d["notional_b"], d["equity"])
    assert (d["net_return"] <= d["gross_return"] + 1e-15).all()
    np.testing.assert_allclose(d["cumulative_costs"], (d["trade_cost"] + d["borrow_cost"]).cumsum())
    assert result.trades["net_pnl"].sum() == pytest.approx(d["equity"].iloc[-1] - CAPITAL)
    np.testing.assert_allclose(d["drawdown"], d["equity"] / d["equity"].cummax() - 1)


def test_drift_rebalance_restores_gross_limit():
    n = 25
    prices = _prices([100.0] * n, 100 * 1.02 ** np.arange(n))  # the short leg rallies: leverage drifts up
    result = run_backtest(prices, signal_frame(prices.index, [1] * n), 1.0, _cfg(rebalance_tolerance=0.10))
    events = result.daily["event"]
    assert (events == "rebalance").sum() >= 1
    first = events[events == "rebalance"].index[0]
    assert result.daily.loc[first, "gross_leverage"] == pytest.approx(1.0)
    before = result.daily["gross_leverage"].shift(1).loc[first]
    assert before > 1.10
    assert result.trades.iloc[0]["rebalances"] >= 1


# --------------------------------------------------------------------------- final period handling
def test_open_position_is_closed_on_final_bar_with_costs():
    prices = _random_prices(10)
    result = run_backtest(prices, signal_frame(prices.index, [0, 1] + [1] * 8), 1.0, _cfg(commission_bps=10))
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "end_of_period"
    assert trade["exit_date"] == prices.index[-1]
    assert result.daily["position"].iloc[-1] == 0
    assert result.daily["trade_cost"].iloc[-1] > 0


def test_signal_on_last_bars_never_trades():
    prices = _random_prices(10)
    for targets in ([0] * 9 + [1], [0] * 8 + [1, 1]):  # last-bar signal; signal whose execution is the last bar
        result = run_backtest(prices, signal_frame(prices.index, targets), 1.0, _cfg(commission_bps=10))
        assert result.trades.empty
        assert result.daily["trade_cost"].sum() == 0
        assert result.equity.iloc[-1] == CAPITAL


def test_last_observation_adds_no_return_and_open_trades_are_reported():
    prices = _random_prices(10)
    result = run_backtest(prices, signal_frame(prices.index, [1] * 10), 1.0, _cfg(close_at_period_end=False))
    assert result.daily["net_return"].iloc[0] == 0.0
    assert len(result.daily) == len(prices)  # no synthetic extra bar after the data
    assert result.trades.iloc[0]["status"] == "open"
    assert any("still open" in w for w in result.warnings)


def test_trade_log_timestamps_and_holding_period():
    prices = _random_prices(12)
    targets = [0, 0, 1, 1, 1, 0, 0, 0, -1, -1, 0, 0]
    events = ["", "", "entry_long", "", "", "exit", "", "", "entry_short", "", "exit", ""]
    result = run_backtest(prices, signal_frame(prices.index, targets, events), 1.0, _cfg(execution_lag=1))
    t1, t2 = result.trades.iloc[0], result.trades.iloc[1]
    assert t1["entry_signal_date"] == prices.index[2] and t1["entry_date"] == prices.index[3]
    assert t1["exit_signal_date"] == prices.index[5] and t1["exit_date"] == prices.index[6]
    assert t1["holding_days"] == 3 and t1["exit_reason"] == "exit"
    assert t2["direction"] == "short_spread" and t2["holding_days"] == 2


def test_input_validation():
    prices = _random_prices(10)
    sig = signal_frame(prices.index, [0, 1] + [1] * 8)
    with pytest.raises(ValueError, match="Short selling is disabled"):
        run_backtest(prices, sig, 1.0, _cfg(short_selling_allowed=False))
    with pytest.raises(ValueError, match="identical index"):
        run_backtest(prices.iloc[1:], sig, 1.0, _cfg())
    bad = prices.copy()
    bad.iloc[3, 0] = -1
    with pytest.raises(ValueError, match="positive"):
        run_backtest(bad, sig, 1.0, _cfg())
    with pytest.raises(ValueError, match="-1, 0 or"):
        run_backtest(prices, signal_frame(prices.index, [2] * 10), 1.0, _cfg())
