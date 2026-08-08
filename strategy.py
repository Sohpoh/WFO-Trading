"""Overnight-range breakout: day-anchored range, ATR stop, R-multiple target.

Rules (invented variation of the documented Opening Range Breakout in
es-futures.md / nq-futures.md — the range is the *day so far* rather than a
fixed opening window):

  - ATR = rolling mean of True Range over `atr_period` bars, where
    TR = max(H-L, |H-C_prev|, |L-C_prev|). ATR is masked to NaN where it is
    non-positive or still warming up, which propagates into both the breakout
    buffer (no signal) and the stop distance (`stop_ok` False in the walk)
    from one place.
  - Day-anchored range, computed on the full continuous df with no session
    awareness: `day = index.normalize()` (UTC calendar day), then
        day_high = High.groupby(day).cummax().groupby(day).shift(1)
        day_low  = Low.groupby(day).cummin().groupby(day).shift(1)
    i.e. the running extremes of the day so far *excluding* the current bar.
    The second, intra-group `.shift(1)` is what stops the prior day's extreme
    leaking across the boundary — it yields NaN on each day's first bar
    instead of the previous day's final cummax.
  - Breakout levels: upper = day_high + breakout_atr_buffer * ATR,
    lower = day_low - breakout_atr_buffer * ATR.
  - Entry is a CROSSING of the level, not the level condition itself:
        long_signal  = (Close > upper) & (Close.shift(1) <= upper.shift(1))
        short_signal = (Close < lower) & (Close.shift(1) >= lower.shift(1))
    The crossing guard matters because
    `session.apply_session_constraint_with_stops()` skips same-bar re-entry
    after a stop (`continue`) but re-evaluates on the very next bar — a
    persistent level condition would re-open the same trade bar after bar for
    as long as price stayed beyond the band. A crossing gives exactly one
    entry per excursion. Zero-param structural guard.
  - Long and short are mutually exclusive by construction: buffer >= 0 and
    day_high >= day_low, so upper >= lower and Close cannot be both above
    upper and below lower on the same bar.
  - No flip on an opposite signal: the walk only opens when flat, so an
    opposing breakout while in a trade is a no-op.
  - Exits are path-dependent and owned entirely by session.py's walk: a stop
    `stop_atr_mult * ATR` from the entry Close, a profit target at `target_r`
    multiples of that same stop distance, and the forced flatten on the
    session's last bar.

Why the target needs explicit per-side resolution: unlike the prior
Bollinger strategy (whose mid-band target was direction-correct by
construction), this target is symmetric around the entry Close, so it is
resolved here with an explicit `np.where` on the short signal —
`Close - target_r*stop` for shorts, `Close + target_r*stop` for longs — to
satisfy the walk's side-validity check (`target > close` for a long,
`target < close` for a short).

Payoff shape is deliberately low-win-rate / high-R: `target_r >= 2` means the
average winner is a multiple of the average loser, since the literal ORB
target (~0.53 avg-win/avg-loss) is smaller than `metrics.py`'s ~0.102%
round-trip cost. Iteration 1's mid-band fade lost with a 34.5% win rate
because price kept extending rather than reverting; this trades the other
side of that same observation.

The UTC calendar day used for the range anchor rolls at 19:00/20:00 ET, i.e.
in the middle of the Globex evening — so by the time the New York session
opens, `day_high`/`day_low` already span the whole overnight. That is the
intended "overnight range". A consequence: on each UTC day's first two bars
`upper`/`upper.shift(1)` are NaN so no entry can fire, but that dead zone
lands in the overnight, never inside a New York session.

The range keeps ratcheting *during* the session (running cummax, not frozen
at the open) by design — a later breakout must clear every extreme printed so
far that day, not just the opening range's.

This module only decides *when the strategy wants to enter and where its
stop/target sit*; it has no session awareness of its own. See
`session.apply_session_constraint_with_stops()`'s docstring for why the
path-dependent walk and the session-boundary flatten live there.

Caveat carried from session.py: stops/targets are *detected* off intrabar
High/Low but the exit still prices at that bar's Close (this app has no
finer-than-bar price series and `metrics.py` costs everything off Close), so
a stop does not cap the realized loss on the triggering bar — see
`apply_session_constraint_with_stops()` for details before describing
results as risk-capped.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops


def compute_atr(df: pd.DataFrame, atr_period: int) -> pd.Series:
    """Rolling-mean ATR over `atr_period` bars, NaN while warming up.

    Masked to NaN where non-positive so it can never produce a zero-width
    breakout buffer or a zero-width stop.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(atr_period, min_periods=atr_period).mean()
    return atr.where(atr > 0)


def compute_day_range(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(day_high, day_low) — running extremes of the UTC day so far, excluding
    the current bar. NaN on each day's first bar (see module docstring)."""
    day = df.index.normalize()
    day_high = df["High"].groupby(day).cummax().groupby(day).shift(1)
    day_low = df["Low"].groupby(day).cummin().groupby(day).shift(1)
    return day_high, day_low


def generate_positions(
    df: pd.DataFrame,
    atr_period: int,
    breakout_atr_buffer: float,
    stop_atr_mult: float,
    target_r: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    atr = compute_atr(df, atr_period)
    day_high, day_low = compute_day_range(df)

    upper = day_high + breakout_atr_buffer * atr
    lower = day_low - breakout_atr_buffer * atr

    prev_close = close.shift(1)
    # Crossing, not level — one entry per excursion (see module docstring).
    long_signal = ((close > upper) & (prev_close <= upper.shift(1))).fillna(False)
    short_signal = ((close < lower) & (prev_close >= lower.shift(1))).fillna(False)

    stop_distance = stop_atr_mult * atr

    # Symmetric around the entry Close, so unlike the prior strategy's
    # mid-band target this one genuinely needs a per-side split to stay on
    # the valid side of the walk's check.
    target_price = pd.Series(
        np.where(
            short_signal,
            close - target_r * stop_distance,
            close + target_r * stop_distance,
        ),
        index=close.index,
    )

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
    "atr_period": 14,
    "breakout_atr_buffer": 0.25,
    "stop_atr_mult": 2.0,
    "target_r": 2.0,
    "session": "New York",
}
