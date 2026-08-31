"""Walk-forward optimization engine.

Prevents curve-fitting by never scoring a parameter set on the same data it
trades: for each fold, strategy parameters are grid-searched on a trailing
`train_weeks` window and then applied, untouched, to the following blind
`test_weeks` window. Only the out-of-sample (test window) returns are
stitched together into the final walk-forward equity curve.

For comparison, `run_retail_insample` reproduces what a retail trader
typically does instead: optimize once on the *entire* dataset and trade that
single curve-fit parameter set across all of it (in-sample).
"""
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Optional

import numpy as np
import pandas as pd

from metrics import bar_returns_with_costs, extract_trades, sharpe_ratio, total_return
from strategy import generate_positions


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict = field(default_factory=dict)
    train_sharpe: float = 0.0
    n_train_bars: int = 0
    n_test_bars: int = 0
    # Per-fold OOS aggregates — computed here (not reconstructed later by bucketing
    # oos_trades.csv rows against fold test windows, which is only approximate).
    # These are what let a caller check "% of folds profitable" and "median fold
    # OOS return/Sharpe" directly instead of dividing by the single curve-fit
    # Retail In-Sample number, whose small-CAGR cases make that ratio degenerate.
    n_oos_trades: int = 0
    oos_return: float = 0.0
    oos_sharpe: float = 0.0


def build_grid(trend_mas, dip_lookbacks, dip_fracs, session) -> list[dict]:
    """Assemble the searched params into `strategy.generate_positions()` kwargs.

    Three params are searched, and they are the strategy's only three degrees
    of freedom:
      - `trend_ma` - length (in bars) of the simple moving average that
        defines "we are in an uptrend". Close must be above it to arm.
      - `dip_lookback` - how far back (in bars) the rolling high is taken from
        when measuring how deep the current pullback is.
      - `dip_frac` - how deep that pullback has to be, expressed in units of
        the trailing average daily range, before the dip is worth buying.

    One param is fixed and threaded into every combo as-is, never searched:
    `session` (required by CLAUDE.md).

    The volatility unit itself is not here at all: `strategy.RANGE_WINDOW`
    (96 bars, one UTC day at 15min) and `strategy.VOL_WINDOW` (480 bars, ~5
    days) are module constants deliberately kept off the search axes, so the
    depth threshold keeps exactly one width degree of freedom (`dip_frac`).
    There is no stop param and no target param either - the exit is the
    session flatten, nothing else.

    No cross-param filter is applied and none is needed: every combination is
    a valid strategy (there is no degenerate pair the way a `fast_ma >=
    slow_ma` pair would be), so no CLI override can silently empty the grid.
    `dip_lookback` may exceed or undercut `trend_ma` freely; both orderings
    are meaningful.

    Type discipline (see CLAUDE.md and `_max_lookback_bars()` below):
      - `trend_ma` and `dip_lookback` are cast to `int` **on purpose**. They
        are genuine bar-count lookbacks, so they are exactly what
        `_max_lookback_bars()` is meant to see when it sizes the
        pre-test-window warm-up buffer.
      - `dip_frac` is cast to `float` **on purpose**. It is a fraction of the
        volatility unit, not a bar count, so it must never feed that buffer.
        The cast is defensive: a grid point hand-written as `1` would
        otherwise arrive as an `int` and be read as a bar count.
      - Consequently `_max_lookback_bars()` returns max(trend_mas +
        dip_lookbacks) = 1920 at the intended grid, the buffer is
        max((1920 + 5) * 3, bars_per_day + 5) = 5,775 bars, and
        `run_walk_forward()`'s fold-skip guard is `len(train_df) < 1930`.
        A 12-week train window at 15min is ~7,700 bars, so no fold is skipped
        - but this strategy IS TIMEFRAME-SENSITIVE: at 1h a 12-week train
        window is only ~1,930 bars (right on the guard, so real folds get
        silently skipped) and at 4h/1d every fold is skipped. Run it at
        `--timeframe 15min`.
      - Hand-check for the leg `_max_lookback_bars()` cannot see (the module
        constants above never appear in the grid): `strategy.volatility_unit`
        needs RANGE_WINDOW + VOL_WINDOW = 96 + 480 = 576 bars of history
        before it is non-NaN. The smallest buffer this grid can produce is
        (960 + 5) * 3 = 2,895 bars, which clears 576 at every grid corner. If
        it ever did not, the failure is a NaN volatility unit -> comparison
        False -> no entry on the first bars of a test window: missed trades,
        never lookahead.
    """
    grid = []
    for trend_ma, dip_lookback, dip_frac in product(trend_mas, dip_lookbacks, dip_fracs):
        grid.append(
            {
                "trend_ma": int(trend_ma),
                "dip_lookback": int(dip_lookback),
                "dip_frac": float(dip_frac),
                "session": session,
            }
        )
    return grid


def _score_params(df: pd.DataFrame, params: dict, ann_factor: float) -> float:
    position = generate_positions(df, **params)
    ret = bar_returns_with_costs(df["Close"], position)
    if (position.diff().abs().fillna(0.0) > 0).sum() < 2:
        return -np.inf
    std = ret.std()
    if std == 0 or np.isnan(std):
        return -np.inf
    return float(ret.mean() / std * np.sqrt(ann_factor))


def optimize(df: pd.DataFrame, grid: list[dict], ann_factor: float) -> tuple[Optional[dict], float]:
    best_params, best_score = None, -np.inf
    for params in grid:
        score = _score_params(df, params, ann_factor)
        if score > best_score:
            best_score, best_params = score, params
    return best_params, best_score


def _max_lookback_bars(grid: list[dict]) -> int:
    """Largest integer-valued (i.e. bar-count-style) param across the grid.

    Deliberately generic rather than keyed off specific param names (e.g.
    "rsi_period", "donchian_n") so it doesn't need updating in lockstep every
    time strategy.py's param list changes — see CLAUDE.md. Float params
    (multipliers like atr_mult) and string params (session, target_mode)
    are naturally excluded by the isinstance check, leaving only genuine
    lookback-window sizes.
    """
    max_val = 0
    for params in grid:
        for v in params.values():
            if isinstance(v, int) and not isinstance(v, bool) and v > max_val:
                max_val = v
    return max_val


def _bars_per_day(df: pd.DataFrame) -> int:
    """Median bar count in one calendar day of `df`.

    Used only as a generic warm-up floor: some indicators reset on a
    calendar-day boundary (e.g. a session VWAP) rather than a fixed N-bar
    lookback, so an N-bar-derived buffer alone isn't guaranteed to reach back
    to that boundary. Flooring the buffer at "at least one full day" covers
    that case without hardcoding which param (if any) is day-anchored.
    """
    if df.empty:
        return 0
    counts = df.groupby(df.index.normalize()).size()
    return int(counts.median()) if len(counts) else 0


def generate_folds(start: pd.Timestamp, end: pd.Timestamp, train_weeks: int, test_weeks: int) -> list[Fold]:
    folds = []
    train_start = start
    i = 0
    while True:
        train_end = train_start + pd.Timedelta(weeks=train_weeks)
        test_start = train_end
        test_end = test_start + pd.Timedelta(weeks=test_weeks)
        if test_end > end:
            break
        folds.append(Fold(index=i, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
        train_start = train_start + pd.Timedelta(weeks=test_weeks)
        i += 1
    return folds


def _simulate_window(
    df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp, params: dict, buffer_bars: int
) -> tuple[pd.Series, pd.Series]:
    """Compute position/returns over [window_start, window_end) using indicator
    history from before window_start as warm-up (carried state, not re-fit)."""
    start_idx = df.index.searchsorted(window_start)
    end_idx = df.index.searchsorted(window_end, side="right")
    calc_start_idx = max(0, start_idx - buffer_bars)
    calc = df.iloc[calc_start_idx:end_idx]
    position = generate_positions(calc, **params)
    ret = bar_returns_with_costs(calc["Close"], position)
    mask = (calc.index >= window_start) & (calc.index < window_end)
    return ret[mask], position[mask]


def run_walk_forward(
    df: pd.DataFrame,
    train_weeks: int,
    test_weeks: int,
    grid: list[dict],
    ann_factor: float,
    progress_callback: Optional[Callable[[int, int, Fold], None]] = None,
) -> tuple[pd.Series, pd.DataFrame, list[Fold]]:
    folds = generate_folds(df.index[0], df.index[-1], train_weeks, test_weeks)
    if not folds:
        raise ValueError(
            "Not enough history for even one fold — shorten the train/test window or pick a wider date range."
        )

    max_lookback = _max_lookback_bars(grid)
    day_bars = _bars_per_day(df)
    buffer_bars = max((max_lookback + 5) * 3, day_bars + 5)

    oos_returns = []
    oos_trades = []
    for fold in folds:
        train_df = df.loc[(df.index >= fold.train_start) & (df.index < fold.train_end)]
        fold.n_train_bars = len(train_df)
        if len(train_df) < max_lookback + 10:
            if progress_callback:
                progress_callback(fold.index + 1, len(folds), fold)
            continue

        best_params, best_score = optimize(train_df, grid, ann_factor)
        fold.best_params = best_params or {}
        fold.train_sharpe = best_score if best_params else 0.0

        if best_params:
            test_ret, test_pos = _simulate_window(df, fold.test_start, fold.test_end, best_params, buffer_bars)
            fold.n_test_bars = len(test_ret)
            fold_trades = extract_trades(df["Close"], test_pos)
            fold_trades = fold_trades.copy()
            fold_trades["fold"] = fold.index + 1  # 1-based, matches fold_table.csv's "fold" column
            fold.n_oos_trades = int(len(fold_trades))
            fold.oos_return = total_return(test_ret)
            fold.oos_sharpe = sharpe_ratio(test_ret, ann_factor)
            oos_returns.append(test_ret)
            oos_trades.append(fold_trades)

        if progress_callback:
            progress_callback(fold.index + 1, len(folds), fold)

    oos_series = pd.concat(oos_returns).sort_index() if oos_returns else pd.Series(dtype=float)
    oos_series = oos_series[~oos_series.index.duplicated(keep="first")]
    trades_df = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "net_return", "fold"]
    )
    return oos_series, trades_df, folds


def run_retail_insample(df: pd.DataFrame, grid: list[dict], ann_factor: float) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Optimize once on the whole dataset and trade that single (curve-fit)
    parameter set across the whole dataset — the naive retail approach."""
    best_params, best_score = optimize(df, grid, ann_factor)
    if not best_params:
        empty = pd.Series(dtype=float)
        return empty, pd.DataFrame(), {}
    position = generate_positions(df, **best_params)
    ret = bar_returns_with_costs(df["Close"], position)
    trades = extract_trades(df["Close"], position)
    return ret, trades, best_params
