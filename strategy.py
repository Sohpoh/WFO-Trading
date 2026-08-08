"""Bollinger Band mean-reversion fade, with a std-scaled stop/target.

Rules (see mean-reversion.md's Bollinger Bands section):
  - SHORT when price closes above the upper band (overbought), expecting
    reversion back to the middle band.
  - LONG when price closes below the lower band (oversold), expecting
    reversion back to the middle band.
  - FLAT otherwise, or once the stop or target is hit while in a trade.
  - Target: the middle band (SMA) itself, read once as a fixed absolute
    level at the entry bar — this is a mean-reversion fade, not a breakout,
    so the target is the mean, not a projected move.
  - Stop: `stop_std_mult` standard deviations beyond entry, in the same std
    units as the bands themselves (rather than mixing in an unrelated ATR
    series) — keeps stop and target distances commensurate.
  - Zero-param trend guard (reuses `bb_period`, no new tunable): the top
    pitfall mean-reversion.md calls out is "shorting strength in a strong
    uptrend". A short is suppressed while the middle band is still rising
    (mid_slope > 0) and a long is suppressed while it's still falling
    (mid_slope < 0) — i.e. only fade the band when the mean itself isn't
    trending in the same direction as the "overextension".

This module only decides *when the strategy wants to enter and where its
stop/target sit*. Because those exits are path-dependent (a stop or target
can fire mid-trade, not just on the next opposing signal), plain
`apply_session_constraint()` can't express them — see
`session.apply_session_constraint_with_stops()`'s docstring for why that
walk (and the session-boundary force-flatten) has to live in session.py
rather than here. This file only ever computes raw signals/distances; it
has no session awareness of its own, consistent with every other strategy
in this codebase.

Band/mid/std are computed on the full continuous-Globex df (not
session-scoped), same as this codebase's prior VWAP anchor — so
`bb_period`'s rolling window can span into the prior evening's bars for
early-session entries. Intentional, not a bug.

Caveat carried from session.py: stops/targets are *detected* off intrabar
High/Low but the exit still prices at that bar's Close (this app has no
finer-than-bar price series and `metrics.py` costs everything off Close), so
a stop does not cap the realized loss on the triggering bar — see
`apply_session_constraint_with_stops()` for details before describing
results as risk-capped.
"""
import pandas as pd

from session import apply_session_constraint_with_stops


def compute_bollinger(df: pd.DataFrame, bb_period: int) -> tuple[pd.Series, pd.Series]:
    """Middle band (SMA) and rolling std from the `bb_period` bars *before*
    the current one (`.shift(1)` first, matching `strategy.py`'s existing
    Donchian convention, so the band never includes its own signal bar)."""
    prior_close = df["Close"].shift(1)
    mid = prior_close.rolling(bb_period, min_periods=bb_period).mean()
    std = prior_close.rolling(bb_period, min_periods=bb_period).std()
    return mid, std


def generate_positions(
    df: pd.DataFrame,
    bb_period: int,
    bb_std_mult: float,
    stop_std_mult: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    mid, std = compute_bollinger(df, bb_period)
    upper = mid + bb_std_mult * std
    lower = mid - bb_std_mult * std

    # Zero-param trend guard: don't fade a band whose own mean is still
    # moving in the direction of the "overextension" (mean-reversion.md's
    # top pitfall — "shorting strength in a strong uptrend"). Reuses
    # bb_period rather than adding a new tunable.
    slope_lag = max(bb_period // 4, 1)
    mid_slope = mid - mid.shift(slope_lag)

    short_signal = ((close > upper) & ~(mid_slope > 0)).fillna(False)
    long_signal = ((close < lower) & ~(mid_slope < 0)).fillna(False)

    stop_distance = stop_std_mult * std

    # Target is the mean itself (mid) for both directions — a mean-reversion
    # fade, not a breakout, so there's no per-direction anchor to resolve
    # the way Donchian's channel-height measured-move needed one.
    target_price = mid.where(long_signal | short_signal)

    return apply_session_constraint_with_stops(
        close=close,
        high=df["High"],
        low=df["Low"],
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    "bb_period": 20,
    "bb_std_mult": 2.0,
    "stop_std_mult": 1.5,
    "session": "New York",
}
