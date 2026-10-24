"""Execution and cost realism: next-open fills, NSE cost schedule, futures shorting, lots, cash interest."""

import numpy as np
import pandas as pd
import pytest

from conftest import make_asset, make_frame, signal_frame
from statarb.backtester import BacktestConfig, run_backtest
from statarb.config import ResearchConfig
from statarb.data_validation import adjusted_open, prepare_pair
from statarb.portfolio import CostModel, MarketCostModel, lot_scale

CAPITAL = 1_000_000.0
ZERO = dict(commission_bps=0.0, slippage_bps=0.0, borrow_bps_annual=0.0, rebalance_tolerance=None)


def _frame(values):
    return pd.DataFrame(values, index=pd.bdate_range("2024-01-01", periods=len(values["A"])), dtype=float)


# --------------------------------------------------------------------------- NSE cost schedule
def test_nse_rates_by_hand():
    m = MarketCostModel(profile="nse", slippage_bps=5.0, brokerage_flat_futures=20.0, dp_charge_cash_sell=15.0)
    gst_cash = (0.297 + 0.01) * 0.18
    gst_fut = (0.173 + 0.01) * 0.18
    assert m.rate_bps("cash", "buy") == pytest.approx(10 + 0.297 + 0.01 + gst_cash + 1.5 + 5)
    assert m.rate_bps("cash", "sell") == pytest.approx(10 + 0.297 + 0.01 + gst_cash + 5)
    assert m.rate_bps("futures", "buy") == pytest.approx(0.173 + 0.01 + gst_fut + 0.2 + 5)
    assert m.rate_bps("futures", "sell") == pytest.approx(0.173 + 0.01 + gst_fut + 2.0 + 5)
    assert m.flat_fee("futures", "buy") == pytest.approx(20 * 1.18)
    assert m.flat_fee("cash", "sell") == pytest.approx(15.0) and m.flat_fee("cash", "buy") == 0.0
    assert m.leg_cost(1_000_000, "cash", "sell") == pytest.approx(1_000_000 * m.rate_bps("cash", "sell") / 1e4 + 15)
    assert m.leg_cost(0.0, "cash", "buy") == 0.0
    roll = m.roll_cost(-500_000)
    assert roll == pytest.approx(500_000 * (m.rate_bps("futures", "buy") + m.rate_bps("futures", "sell") + 2.0) / 1e4
                                 + 2 * 20 * 1.18)


def test_generic_market_model_matches_original_cost_model():
    m = MarketCostModel(profile="generic", commission_bps=4, slippage_bps=6)
    c = CostModel(commission_bps=4, slippage_bps=6)
    assert m.leg_cost(1_000, "cash", "buy") + m.leg_cost(-3_000, "futures", "sell") == pytest.approx(c.trade_cost(1_000, -3_000))
    assert MarketCostModel(short_instrument="futures").instrument(-1.0) == "futures"
    assert MarketCostModel().instrument(-1.0) == "cash"


def test_lot_scale():
    assert lot_scale(-1050.0, 100.0) == pytest.approx(1000 / 1050)
    assert lot_scale(-50.0, 100.0) is None
    assert lot_scale(-50.0, None) == 1.0


# --------------------------------------------------------------------------- next-open execution
def test_open_fill_earns_only_after_the_fill():
    closes = _frame({"A": [100, 100, 104, 104], "B": [100, 100, 100, 100]})
    opens = _frame({"A": [100, 100, 102, 104], "B": [100, 100, 100, 100]})
    sig = signal_frame(closes.index, [0, 1, 1, 0])
    result = run_backtest(closes, sig, 1.0, BacktestConfig(execution_price="open", **ZERO), opens=opens)
    qa = 0.5 * CAPITAL / 102  # bought at bar 2's open, not its close
    assert result.daily["units_a"].iloc[2] == pytest.approx(qa)
    np.testing.assert_allclose(result.daily["pnl"].to_numpy(), [0, 0, qa * (104 - 102), 0])
    assert result.trades.iloc[0]["entry_price_a"] == 102.0
    assert result.trades.iloc[0]["exit_price_a"] == 104.0  # final bar liquidated at its open
    close_mode = run_backtest(closes, sig, 1.0, BacktestConfig(**ZERO))
    assert close_mode.daily["pnl"].abs().sum() == 0.0  # filled at bar 2's close: the 100 -> 104 move is missed


def test_open_mode_overnight_gap_not_earned_by_new_position():
    closes = _frame({"A": [100, 100, 100, 100], "B": [100, 100, 100, 100]})
    opens = _frame({"A": [100, 100, 110, 100], "B": [100, 100, 100, 100]})  # gap up at bar 2's open
    result = run_backtest(closes, signal_frame(closes.index, [0, 1, 1, 0]), 1.0,
                          BacktestConfig(execution_price="open", **ZERO), opens=opens)
    qa = 0.5 * CAPITAL / 110
    assert result.daily["pnl"].iloc[2] == pytest.approx(qa * (100 - 110))  # only open -> close after buying


@pytest.mark.parametrize("cut", [20, 45, 70])
def test_open_mode_truncation_invariance(cut):
    rng = np.random.default_rng(5)
    n = 90
    closes = _frame({"A": 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))),
                     "B": 50 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))})
    opens = closes * np.exp(0.003 * rng.standard_normal((n, 2)))
    targets = np.repeat(rng.choice([-1, 0, 1], size=18), 5)
    cfg = BacktestConfig(execution_price="open", commission_bps=3, slippage_bps=2, borrow_bps_annual=40,
                         close_at_period_end=False, cash_interest_rate=0.05)
    full = run_backtest(closes, signal_frame(closes.index, targets), 0.9, cfg, opens=opens)
    part = run_backtest(closes.iloc[:cut], signal_frame(closes.index[:cut], targets[:cut]), 0.9, cfg,
                        opens=opens.iloc[:cut])
    pd.testing.assert_frame_equal(full.daily.iloc[:cut], part.daily)


def test_open_mode_requires_opens_and_falls_back_to_close():
    closes = _frame({"A": [100, 100, 104, 104], "B": [100, 100, 100, 100]})
    sig = signal_frame(closes.index, [0, 1, 1, 0])
    with pytest.raises(ValueError, match="opening prices"):
        run_backtest(closes, sig, 1.0, BacktestConfig(execution_price="open", **ZERO))
    opens = _frame({"A": [100, 100, np.nan, 104], "B": [100, 100, 100, 100]})
    result = run_backtest(closes, sig, 1.0, BacktestConfig(execution_price="open", **ZERO), opens=opens)
    assert result.trades.iloc[0]["entry_price_a"] == 104.0  # no usable open on bar 2 -> close
    assert any("no usable opening price" in w for w in result.warnings)


def test_adjusted_open_uses_close_adjustment_factor():
    frame = make_frame([100.0, 110.0])
    frame["Adj Close"] = [50.0, 55.0]
    frame["Open"] = [98.0, -1.0]
    adj = adjusted_open(frame, "Adj Close")
    assert adj.iloc[0] == pytest.approx(49.0) and np.isnan(adj.iloc[1])
    assert adjusted_open(frame.drop(columns="Open"), "Adj Close") is None


def test_prepare_pair_carries_opens(coint_assets):
    pair = prepare_pair(*coint_assets)
    assert pair.opens is not None and pair.opens.index.equals(pair.prices.index)
    assert (pair.opens > 0).all().all()


# --------------------------------------------------------------------------- futures shorting, rolls, lots
def _flat_prices(n=12):
    return _frame({"A": [100.0] * n, "B": [50.0] * n})


def test_futures_short_round_trip_costs_by_hand():
    prices = _flat_prices(6)
    cfg = BacktestConfig(cost_profile="nse", short_instrument="futures", slippage_bps=5.0, borrow_bps_annual=50.0,
                         rebalance_tolerance=None, futures_roll_every_bars=100)
    m = cfg.market_cost_model()
    result = run_backtest(prices, signal_frame(prices.index, [1] * 6), 1.0, cfg)
    proportional = 0.5 * (m.rate_bps("cash", "buy") + m.rate_bps("futures", "sell")) / 1e4
    budget = CAPITAL / (1 + proportional)
    entry = budget * proportional
    exit_cost = 0.5 * budget * (m.rate_bps("cash", "sell") + m.rate_bps("futures", "buy")) / 1e4
    trade = result.trades.iloc[0]
    assert trade["transaction_costs"] == pytest.approx(entry + exit_cost)
    assert result.daily["borrow_cost"].sum() == 0.0  # futures shorts pay no borrow fee
    assert result.daily["futures_margin"].iloc[2] == pytest.approx(0.2 * 0.5 * budget)


def test_futures_rolls_charged_on_schedule():
    prices = _flat_prices(12)
    cfg = BacktestConfig(cost_profile="nse", short_instrument="futures", rebalance_tolerance=None,
                         futures_roll_every_bars=3, slippage_bps=0.0)
    result = run_backtest(prices, signal_frame(prices.index, [1] * 12), 1.0, cfg)
    # Entered at bar 1 (lag 1); held through bars 2..10; liquidated at bar 11 -> rolls at bars 4, 7, 10.
    rolled = result.daily.index[result.daily["event"].str.contains("roll")]
    assert list(rolled) == list(prices.index[[4, 7, 10]])
    assert result.trades.iloc[0]["rolls"] == 3
    short_notional = result.daily["short_exposure"].iloc[4]
    m = cfg.market_cost_model()
    assert result.daily["trade_cost"].iloc[4] == pytest.approx(m.roll_cost(-short_notional))


def test_futures_lot_sizes_round_both_legs_and_block_small_positions():
    prices = _flat_prices(6)
    cfg = BacktestConfig(cost_profile="generic", short_instrument="futures", rebalance_tolerance=None,
                         commission_bps=0.0, slippage_bps=0.0, futures_lot_size_b=3000.0)
    result = run_backtest(prices, signal_frame(prices.index, [1] * 6), 1.0, cfg)
    units_b = result.daily["units_b"].iloc[2]  # unrounded short: 10,000 units of B at 50
    assert units_b == pytest.approx(-9000.0)
    assert result.daily["notional_a"].iloc[2] / -result.daily["notional_b"].iloc[2] == pytest.approx(1.0)
    blocked = run_backtest(prices, signal_frame(prices.index, [1] * 6), 1.0,
                           BacktestConfig(short_instrument="futures", futures_lot_size_b=50_000.0, **ZERO))
    assert blocked.trades.empty and "lot_size_block" in set(blocked.daily["event"])
    assert any("smaller than one futures lot" in w for w in blocked.warnings)


# --------------------------------------------------------------------------- cash interest / risk-free rate
def test_cash_interest_on_idle_capital():
    prices = _flat_prices(11)
    rate = 0.06
    result = run_backtest(prices, signal_frame(prices.index, [0] * 11), 1.0,
                          BacktestConfig(cash_interest_rate=rate, **ZERO))
    daily = (1 + rate) ** (1 / 252) - 1
    assert result.equity.iloc[-1] == pytest.approx(CAPITAL * (1 + daily) ** 10)
    assert result.daily["net_return"].iloc[1] == pytest.approx(daily)
    np.testing.assert_allclose(result.daily["gross_return"], result.daily["net_return"])


def test_interest_base_includes_short_proceeds_and_excludes_long_cash():
    prices = _flat_prices(5)
    result = run_backtest(prices, signal_frame(prices.index, [1] * 5), 1.0,
                          BacktestConfig(cash_interest_rate=0.05, **ZERO))
    d = result.daily
    daily = 1.05 ** (1 / 252) - 1
    base = d["equity"].iloc[1] - d["long_exposure"].iloc[1] + d["short_exposure"].iloc[1]
    assert d["interest"].iloc[2] == pytest.approx(daily * base)
    identity = d["equity"].shift(1) + d["pnl"] + d["interest"] - d["trade_cost"] - d["borrow_cost"]
    np.testing.assert_allclose(d["equity"].iloc[1:], identity.iloc[1:])


def test_research_config_wires_realism_options():
    cfg = ResearchConfig().with_overrides({
        "strategy": {"execution_price": "open"},
        "costs": {"profile": "nse", "short_instrument": "futures", "futures_lot_sizes": {"INFY.NS": 400}},
        "portfolio": {"cash_interest": True}, "metrics": {"risk_free_rate": 0.06},
    })
    bt = BacktestConfig.from_research_config(cfg, tickers=("TCS.NS", "INFY.NS"))
    assert (bt.execution_price, bt.cost_profile, bt.short_instrument) == ("open", "nse", "futures")
    assert bt.cash_interest_rate == 0.06 and bt.futures_lot_size_b == 400 and bt.futures_lot_size_a is None
    assert BacktestConfig.from_research_config(ResearchConfig()).cash_interest_rate == 0.0
    with pytest.raises(ValueError):
        ResearchConfig().with_overrides({"costs": {"profile": "lse"}})


def test_make_asset_pair_with_opens_for_prepare(coint_assets):
    a, b = coint_assets
    frame = a.frame.copy()
    frame.loc[frame.index[10], "Open"] = 0.0
    pair = prepare_pair(make_asset("AAA", frame), make_asset("BBB", b.frame))
    assert np.isnan(pair.opens["A"].iloc[10])
    assert any("lack a usable opening price" in w for w in pair.report.warnings)


# --------------------------------------------------------------------------- calendar-year risk-free schedule
def test_annual_rate_lookup_carries_forward():
    from statarb.metrics import annual_rate_for_year

    schedule = {2016: 0.01, 2018: 0.03, 2015: 0.002}
    assert annual_rate_for_year(schedule, 2014) == 0.002  # before the schedule: first rate
    assert annual_rate_for_year(schedule, 2017) == 0.01   # gap: latest earlier year
    assert annual_rate_for_year(schedule, 2030) == 0.03   # after: last rate
    assert annual_rate_for_year({"2016": 0.01}, 2016) == 0.01
    with pytest.raises(ValueError):
        annual_rate_for_year({}, 2016)


def test_sharpe_with_a_year_schedule_matches_hand_calculation():
    from statarb.metrics import daily_risk_free, sharpe_ratio, sortino_ratio

    index = pd.bdate_range("2019-12-20", periods=12)
    r = pd.Series(np.linspace(-0.002, 0.004, 12), index=index)
    schedule = {2019: 0.02, 2020: 0.05}
    rf = np.where(index.year == 2019, daily_risk_free(0.02), daily_risk_free(0.05))
    excess = r - rf
    assert sharpe_ratio(r, schedule) == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
    assert sharpe_ratio(r, {2019: 0.03, 2020: 0.03}) == pytest.approx(sharpe_ratio(r, 0.03))
    assert np.isfinite(sortino_ratio(r, schedule))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        sharpe_ratio(r.reset_index(drop=True), schedule)


def test_schedule_config_and_cash_interest_by_year():
    cfg = ResearchConfig().with_overrides({"metrics": {"risk_free_by_year": {2020: 0.05, 2019: 0.02}},
                                           "portfolio": {"cash_interest": True}})
    assert cfg.metrics.risk_free_by_year == ((2019, 0.02), (2020, 0.05))
    assert cfg.metrics.risk_free == {2019: 0.02, 2020: 0.05}
    assert ResearchConfig().metrics.risk_free == 0.0
    with pytest.raises(ValueError):
        ResearchConfig().with_overrides({"metrics": {"risk_free_by_year": {2020: 0.9}}})
    bt = BacktestConfig.from_research_config(cfg, **ZERO)
    assert bt.cash_interest_by_year == ((2019, 0.02), (2020, 0.05))
    closes = pd.DataFrame({"A": 100.0, "B": 100.0}, index=pd.bdate_range("2019-12-30", periods=4))
    result = run_backtest(closes, signal_frame(closes.index, [0, 0, 0, 0]), 1.0, bt)
    daily = result.daily
    # flat book: each bar earns its own year's daily rate on the prior equity (2019-12-31 vs 2020-01-01/02)
    for i in range(1, 4):
        year_rate = 0.02 if daily.index[i].year == 2019 else 0.05
        expected = daily["equity"].iloc[i - 1] * ((1 + year_rate) ** (1 / 252) - 1)
        assert daily["interest"].iloc[i] == pytest.approx(expected)


# --------------------------------------------------------------------------- risk-free accounting at period starts
def test_flat_period_earns_exactly_the_risk_free_rate():
    from statarb.metrics import strategy_metrics

    closes = pd.DataFrame({"A": 100.0, "B": 100.0}, index=pd.bdate_range("2023-12-20", periods=30))
    cfg = BacktestConfig(cash_interest_rate=0.05, **ZERO)
    alone = run_backtest(closes, signal_frame(closes.index, [0] * 30), 1.0, cfg)
    assert alone.daily["interest"].iloc[0] == 0.0 and alone.daily["net_return"].iloc[0] == 0.0  # no holding period
    assert np.isnan(strategy_metrics(alone, 0.05)["sharpe_ratio"])  # zero excess return, zero volatility
    chained = run_backtest(closes, signal_frame(closes.index, [0] * 30), 1.0, cfg, accrue_first_bar=True)
    assert chained.daily["interest"].iloc[0] == pytest.approx(CAPITAL * (1.05 ** (1 / 252) - 1))
    assert run_backtest(closes, signal_frame(closes.index, [0] * 30), 1.0, BacktestConfig(**ZERO),
                        accrue_first_bar=True).daily["interest"].sum() == 0.0  # no rate, nothing accrues


def test_flat_chained_walk_forward_has_zero_excess_return(coint_pair):
    from statarb.evaluation import run_walk_forward
    from statarb.metrics import daily_risk_free_series, period_excess_returns

    schedule = {2015: 0.01, 2017: 0.03, 2019: 0.05}
    cfg = ResearchConfig().with_overrides({"strategy": {"entry_z": 50.0}, "portfolio": {"cash_interest": True},
                                           "metrics": {"risk_free_by_year": schedule},
                                           "walk_forward": {"train_window": 500, "test_window": 250}})
    wf = run_walk_forward(coint_pair.prices, cfg)
    daily = wf.daily
    assert (daily["position"] == 0).all() and len(wf.windows) > 2
    rf = daily_risk_free_series(schedule, daily.index).to_numpy()
    assert daily["net_return"].iloc[0] == 0.0
    np.testing.assert_allclose(daily["net_return"].to_numpy()[1:], rf[1:], rtol=1e-9)  # window starts included
    np.testing.assert_allclose(period_excess_returns(daily["net_return"], schedule), 0.0, atol=1e-15)
    assert np.isnan(wf.metrics["sharpe_ratio"])


def test_starts_flat_metrics_match_plain_metrics_at_zero_rate():
    from statarb.metrics import return_metrics

    r = pd.Series([0.0, 0.01, -0.004, 0.002, 0.003], index=pd.bdate_range("2024-01-01", periods=5))
    np.testing.assert_equal(return_metrics(r, 0.0, starts_flat=True), return_metrics(r, 0.0))  # NaN-aware
    shifted = return_metrics(r, 0.03, starts_flat=True)
    assert shifted["sharpe_ratio"] != return_metrics(r, 0.03)["sharpe_ratio"]  # first value is not charged the rate
