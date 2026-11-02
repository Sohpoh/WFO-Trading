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
    # Pre-OOS purge: the last `purge_days` training bars nearest the test window
    # are excluded from grid search (dropped, not trained on) to prevent any
    # label warm-up bleeding test information backward. None means no purge.
    train_end_effective: Optional[pd.Timestamp] = None
    # Trial population for the Deflated Sharpe Ratio: the train Sharpe of every
    # valid grid combo scored in this fold (used to estimate sqrt(V[SR])).
    grid_train_sharpes: list = field(default_factory=list)
    # Rank-stability diagnostic: logit of where this fold's IS-selected combo
    # ranked OOS among the top-K candidates (see tools/statistics.py). None when
    # rank-stability was disabled or not computable.
    oos_rank_logit: Optional[float] = None


def build_grid(range_lookbacks, squeeze_ratios, target_atr_mults, stop_atr_mults, size_caps, session) -> list[dict]:
    """Assemble the searched params into `strategy.generate_positions()` kwargs.

    Five params are searched for the long-only volatility-squeeze Donchian
    breakout + bounded ATR target + inverse-vol sizing:

      - `range_lookback` — the Donchian channel trailing window in bars (grid
        {8,12,24,48} = 2h/3h/1d/2d at 15min). A genuine bar-count lookback,
        so it is threaded as a plain `int` **on purpose** (feeds the warm-up
        buffer). The build_grid parameter is `range_lookbacks` — the
        grid-search naming convention is "<DEFAULT_PARAMS key> + 's'" (see
        tools/check_strategy.py).
      - `squeeze_ratio` — the vol-ratio threshold at/below which the setup is
        armed (grid {0.8,0.9,1.0,1.1}). A ratio threshold (float), NOT a
        lookback, so it must not feed `_max_lookback_bars()`. The build_grid
        parameter is `squeeze_ratios`.
      - `target_atr_mult` — the ATR multiplier for the bounded profit target
        (grid {1.0,1.5,2.0,3.0}). An ATR multiplier (float), NOT a lookback.
        The build_grid parameter is `target_atr_mults`.
      - `stop_atr_mult` — the ATR multiplier for the below-entry stop (grid
        {0.75,1.0,1.5,2.0}; 2.0 is the control anchor carried over from the
        prior fixed value). An ATR multiplier (float), NOT a lookback, so it
        must not feed `_max_lookback_bars()`. The build_grid parameter is
        `stop_atr_mults`.
      - `size_cap` — the inverse-vol exposure cap: size is
        clip(size_cap * ATR(960)/ATR(14), 0.5, size_cap) (grid
        {1.0,1.5,2.0,3.0}). An exposure-cap multiplier (float), NOT a lookback,
        so it must not feed `_max_lookback_bars()`. The build_grid parameter is
        `size_caps`.

    The strategy's fixed params are never searched, but the three ATR windows
    are genuine bar-count lookbacks, so they are threaded into every combo as
    plain `int`s **on purpose** so `_max_lookback_bars()` accounts for them:
      - `atr_slow_period` (`strategy.ATR_SLOW_PERIOD` = 960 bars ~ 10 days at
        15min) — the slow squeeze-ratio ATR.
      - `atr_fast_period` (`strategy.ATR_FAST_PERIOD` = 96 bars ~ 1 day) — the
        fast squeeze-ratio ATR.
      - `atr_stop_period` (`strategy.ATR_STOP_PERIOD` = 14 bars) — the ATR
        window for the stop/target distance.

    `session` is fixed and threaded into every combo as-is, never searched
    (required by CLAUDE.md).

    WARM-UP — `ATR_SLOW_PERIOD` (960) is the largest int in the grid, so
    `_max_lookback_bars()` returns 960 and `run_walk_forward()` sizes the
    buffer to `max((960 + 5) * 3, day_bars + 5)` = 2895 bars (~30 days) at
    15min. The ATR(960) Wilder warm-up, the Donchian window (at most 48 bars)
    and ATR(14) are all covered with headroom; the fold-skip guard of
    `max_lookback + 10` = 970 bars is far below any 12-week train window at
    15min, so no fold is silently skipped.
    """
    from strategy import ATR_FAST_PERIOD, ATR_SLOW_PERIOD, ATR_STOP_PERIOD

    grid = []
    for rl, sr, tam, sam, sc in product(range_lookbacks, squeeze_ratios, target_atr_mults, stop_atr_mults, size_caps):
        grid.append(
            {
                "range_lookback": int(rl),
                "squeeze_ratio": float(sr),
                "target_atr_mult": float(tam),
                "stop_atr_mult": float(sam),
                "size_cap": float(sc),
                "atr_fast_period": int(ATR_FAST_PERIOD),
                "atr_slow_period": int(ATR_SLOW_PERIOD),
                "atr_stop_period": int(ATR_STOP_PERIOD),
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


def _score_all(df: pd.DataFrame, grid: list[dict], ann_factor: float) -> list[tuple[dict, float]]:
    """Score every grid combo, sorted by train Sharpe descending.

    A single pass is shared by ``optimize`` (best combo), the DSR trial
    population (all scores), and rank-stability (top-K), so the grid is never
    scored more than once per fold.
    """
    scored = [(params, _score_params(df, params, ann_factor)) for params in grid]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def optimize(df: pd.DataFrame, grid: list[dict], ann_factor: float) -> tuple[Optional[dict], float]:
    scored = _score_all(df, grid, ann_factor)
    if not scored or scored[0][1] == -np.inf:
        return None, -np.inf
    return scored[0][0], scored[0][1]


def optimize_top_k(df: pd.DataFrame, grid: list[dict], ann_factor: float, k: int) -> list[tuple[dict, float]]:
    """Top-K valid (params, train_sharpe) combos, for rank-stability diagnostics."""
    scored = _score_all(df, grid, ann_factor)
    return [(p, s) for p, s in scored[:k] if s != -np.inf]


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


def generate_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_weeks: int,
    test_weeks: int,
    purge_days: int = 0,
) -> list[Fold]:
    """Build sliding train/test folds.

    ``purge_days`` implements a pre-OOS purge (Bailey & López de Prado style):
    the last N calendar days of the nominal train window — the bars sitting
    immediately next to the test window — are dropped from grid search, so no
    indicator warm-up or overlapping state can leak test information backward.
    ``test_start`` stays where it is; only the effective training end moves.
    """
    folds = []
    train_start = start
    i = 0
    while True:
        train_end = train_start + pd.Timedelta(weeks=train_weeks)
        test_start = train_end
        test_end = test_start + pd.Timedelta(weeks=test_weeks)
        if test_end > end:
            break
        train_end_effective = train_end - pd.Timedelta(days=purge_days) if purge_days > 0 else None
        folds.append(Fold(
            index=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_end_effective=train_end_effective,
        ))
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
    purge_days: int = 0,
    rank_stability_k: int = 0,
) -> tuple[pd.Series, pd.DataFrame, list[Fold]]:
    folds = generate_folds(df.index[0], df.index[-1], train_weeks, test_weeks, purge_days)
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
        train_end = fold.train_end_effective if fold.train_end_effective is not None else fold.train_end
        train_df = df.loc[(df.index >= fold.train_start) & (df.index < train_end)]
        fold.n_train_bars = len(train_df)
        if len(train_df) < max_lookback + 10:
            if progress_callback:
                progress_callback(fold.index + 1, len(folds), fold)
            continue

        # Score the grid once per fold; reuse for the best combo, the DSR trial
        # population (all scores), and rank-stability (top-K).
        scored = _score_all(train_df, grid, ann_factor)
        valid = [(p, s) for p, s in scored if s != -np.inf]
        if not valid:
            if progress_callback:
                progress_callback(fold.index + 1, len(folds), fold)
            continue

        best_params, best_score = valid[0]
        fold.best_params = best_params
        fold.train_sharpe = best_score
        fold.grid_train_sharpes = [s for _, s in valid]

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

        # Rank-stability: where does the IS-selected combo rank OOS among the
        # top-K IS trials?  A logit <= 0 means argmax selection landed in the
        # bottom half out-of-sample (selection overfit, not a real param surface).
        if rank_stability_k > 1 and len(valid) >= 2:
            from tools.statistics import selected_trial_rank_logit  # local import avoids engine->tools load ordering
            top_k = valid[:rank_stability_k]
            oos_sharpes = []
            for p, _ in top_k:
                r, _ = _simulate_window(df, fold.test_start, fold.test_end, p, buffer_bars)
                oos_sharpes.append(sharpe_ratio(r, ann_factor))
            fold.oos_rank_logit = selected_trial_rank_logit(fold.oos_sharpe, oos_sharpes)

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
