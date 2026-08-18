"""Rolling N-bar range breakout, flip exit — no stops, winners run to the close.

Bar-indexed Donchian channel on the full continuous frame, traded as a plain
flip strategy: long when Close breaks out above the prior `range_lookback`
bars' high (plus a buffer), short on the mirror-image break below, and no
exit at all other than the opposite break or the session's forced flatten.

Rules (all computed with no session awareness of their own):

  - Channel, shifted so the current bar can never define the level it must
    break (that `.shift(1)` is load-bearing — without it the bar's own High
    is part of `upper` and the comparison is lookahead):
        upper = High.rolling(range_lookback).max().shift(1)
        lower = Low.rolling(range_lookback).min().shift(1)
        width = upper - lower
  - Raw entries:
        +1.0 where Close >  upper + buffer_frac * width
        -1.0 where Close <  lower - buffer_frac * width
        NaN  elsewhere
    `buffer_frac` scales the required overshoot by the channel's own width,
    so the filter is volatility-adaptive and scale-free rather than a fixed
    number of points. `buffer_frac = 0` is the plain touch-the-level break.
  - The trigger is Close-based on purpose. `apply_session_constraint()` has
    no intrabar machinery — every bar is priced off its Close — so a
    High-touch trigger would book a fill at a price the bar's own high says
    was already exceeded, i.e. an unfillable trade. Close-based is the only
    honest formulation on this path.
  - Long and short are mutually exclusive for free: `upper >= lower` always
    and `buffer_frac >= 0`, so `upper + buffer_frac*width >= lower -
    buffer_frac*width` and a single Close cannot satisfy both inequalities.

Exit is a plain flip — deliberately no stop and no target:

  - The position forward-fills from the breakout bar until either the
    opposite band breaks (a reversal, 2 cost legs) or the forced flatten on
    the session's last bar (1 leg). The ffilled 0 then propagates through
    the overnight gap, so every session starts flat.
  - A repeat same-direction signal while already positioned is a true no-op
    with zero extra cost legs (it just re-writes the same 1.0/-1.0 into a
    series that is already forward-filled to that value), satisfying the
    single-position contract by construction.
  - Dropping the path-dependent stop/target walk is the whole point of this
    iteration, not an omission. On that walk a stop is *detected* intrabar
    but the exit still prices at the triggering bar's Close, so it never
    actually capped the loss on that bar — it only decided when to leave,
    and then allowed a fresh in-session re-entry that paid two more cost
    legs. Removing it removes the extra legs and lets a winner run
    uncapped to the session flatten instead of being clipped at a fixed
    R-multiple, which is how gross-per-trade grows rather than merely
    turnover shrinking.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged
here. The design bets on a larger average gross move per trade (one
directional hold per session run instead of several capped ones), not on a
cheaper toll.

Warm-up: `range_lookback` bars for the rolling extremes plus the one-bar
shift. That's a genuine bar-count lookback, so it is passed as a plain `int`
from `build_grid()` and correctly sizes the engine's pre-test-window buffer;
`buffer_frac` is a fraction and is passed as a `float` so it cannot.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint()`; see its docstring for that contract.
"""
import pandas as pd

from session import apply_session_constraint


def donchian_channel(df: pd.DataFrame, range_lookback: int) -> tuple[pd.Series, pd.Series]:
    """Prior-`range_lookback`-bar high/low, shifted one bar.

    The shift excludes the current bar from its own breakout level — without
    it, `Close > upper` would be comparing against a maximum that already
    contains this bar's High (lookahead).
    """
    upper = df["High"].rolling(range_lookback, min_periods=range_lookback).max().shift(1)
    lower = df["Low"].rolling(range_lookback, min_periods=range_lookback).min().shift(1)
    return upper, lower


def generate_positions(
    df: pd.DataFrame,
    range_lookback: int,
    buffer_frac: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    upper, lower = donchian_channel(df, range_lookback)
    width = upper - lower

    # Mutually exclusive by construction (upper >= lower, buffer_frac >= 0).
    long_signal = (close > upper + buffer_frac * width).fillna(False)
    short_signal = (close < lower - buffer_frac * width).fillna(False)

    entries = pd.Series(index=df.index, dtype=float)  # all-NaN = "no signal"
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "range_lookback": 48,
    "buffer_frac": 0.05,
    "session": "New York",
}
