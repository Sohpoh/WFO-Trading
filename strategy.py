"""Rolling N-bar range breakout + volatility-regime gate, flip exit — no stops.

Bar-indexed Donchian channel on the full continuous frame, traded as a plain
flip strategy: long when Close breaks out above the prior `range_lookback`
bars' high (plus a buffer), short on the mirror-image break below, and no
exit at all other than the opposite break or the session's forced flatten.

New in this iteration, and the *only* change from the plain breakout: a
zero-parameter volatility-regime gate that suppresses every raw entry while
realized volatility is contracting. The breakout core, the grid and the exit
rule are byte-identical to the previous iteration, so any difference in
results is attributable to the gate alone.

Rules (all computed with no session awareness of their own):

  - Channel, shifted so the current bar can never define the level it must
    break (that `.shift(1)` is load-bearing — without it the bar's own High
    is part of `upper` and the comparison is lookahead):
        upper = High.rolling(range_lookback).max().shift(1)
        lower = Low.rolling(range_lookback).min().shift(1)
        width = upper - lower
  - Raw breakout signals:
        long  where Close >  upper + buffer_frac * width
        short where Close <  lower - buffer_frac * width
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
    The gate below only ever removes signals, so it cannot break that.
  - Volatility-regime gate (hardcoded, NOT grid-searched — see below):
        TR       = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        atr_fast = TR.rolling(ATR_FAST_BARS).mean()
        atr_slow = TR.rolling(ATR_SLOW_BARS).mean()
        vol_ok   = atr_fast >= atr_slow
    A raw signal only becomes an entry where `vol_ok`; elsewhere the bar
    contributes NaN (no signal), exactly as if the break hadn't happened.
    The gate is applied symmetrically to long and short. Gating only
    *fresh-from-flat* entries would need the forward-filled position state,
    which the sparse entries-series contract has no room for — that's
    `apply_session_constraint()`'s job, not this module's.
  - Why a slow ratio rather than an absolute vol threshold: a ratio is
    scale-free (no points/percent constant to fit or to re-fit as NQ's price
    level doubles) and, because volatility clusters, the fast/slow crossing
    is a multi-day regime switch rather than a bar-by-bar chop filter. The
    thesis is that a range-breakout edge is regime-conditional — it pays in
    expanding-volatility stretches and bleeds two cost legs per whipsaw in
    calm grind stretches — so the gate targets trade *quality*, not power.
  - Both windows are hardcoded constants, deliberately outside the grid, so
    this iteration adds zero degrees of freedom for the optimizer to overfit.

Exit is a plain flip — deliberately no stop and no target:

  - The position forward-fills from the breakout bar until either the
    opposite band breaks (a reversal, 2 cost legs) or the forced flatten on
    the session's last bar (1 leg). The ffilled 0 then propagates through
    the overnight gap, so every session starts flat.
  - Explicit consequence of gating the *exit* side too: when the opposite
    break happens in a suppressed (contracting-vol) regime, the position does
    not reverse — it simply carries to the session flatten. That is part of
    the thesis, not an oversight: the flip-flop reversals in chop are exactly
    the two-cost-leg whipsaws the gate exists to remove.
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

The gate's own warm-up is 384 bars (`ATR_SLOW_BARS` exactly — TR's first bar
degrades to High-Low rather than NaN, so `Close.shift(1)` costs nothing here),
and it is *invisible* to
`wfo_engine._max_lookback_bars()` because it is a module constant rather than
a grid param. That is safe for the intended grid: the longest searched
`range_lookback` is 192, so buffer_bars = max((192+5)*3, day_bars+5) = 591
> 384, and the slow leg is fully warm before the first test-window bar. The
rolling windows use the strict default `min_periods` on purpose, so an
incompletely warmed slow leg is NaN and the gate fails *closed* (no entries)
rather than silently degrading to an always-true no-op. Caveat for whoever
changes the grid next: a longest searched lookback below 123 shrinks the
buffer under 384 and would start the test window with the gate closed —
if you do that, raise the buffer (e.g. by threading a fixed int lookback
param through `build_grid()`) rather than loosening `min_periods`.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint()`; see its docstring for that contract.
"""
import pandas as pd

from session import apply_session_constraint

# Volatility-regime gate windows, in bars. Hardcoded on purpose: the gate adds
# no searchable degrees of freedom, so the only change versus the previous
# iteration is the filter itself. Sized for 15min bars — 96 bars is ~one full
# 24h Globex day and 384 is ~four of them, i.e. a slow, multi-day regime
# switch rather than a bar-by-bar chop filter.
ATR_FAST_BARS = 96
ATR_SLOW_BARS = 384


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range: max(H-L, |H-C_prev|, |L-C_prev|).

    `prev_close` is NaN on the first bar, so two of the three legs are NaN
    there; `.max(axis=1)` skips NaNs by default, leaving TR[0] = High - Low.
    That degenerate-but-sane first value is intentional (no fillna needed),
    which is why the gate's warm-up is exactly `ATR_SLOW_BARS` bars, not one
    more.
    """
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def vol_regime_ok(df: pd.DataFrame) -> pd.Series:
    """True where fast realized vol >= slow realized vol (expanding regime).

    Strict `min_periods` (pandas' default = window) so an unwarmed slow leg is
    NaN and the comparison is False — the gate fails closed (suppress) rather
    than open (trade ungated), which is the conservative direction.
    """
    tr = true_range(df)
    atr_fast = tr.rolling(ATR_FAST_BARS, min_periods=ATR_FAST_BARS).mean()
    atr_slow = tr.rolling(ATR_SLOW_BARS, min_periods=ATR_SLOW_BARS).mean()
    return (atr_fast >= atr_slow).fillna(False)


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

    # Mutually exclusive by construction (upper >= lower, buffer_frac >= 0);
    # the gate only ever removes signals, so it preserves that.
    vol_ok = vol_regime_ok(df)
    long_signal = (close > upper + buffer_frac * width).fillna(False) & vol_ok
    short_signal = (close < lower - buffer_frac * width).fillna(False) & vol_ok

    entries = pd.Series(index=df.index, dtype=float)  # all-NaN = "no signal"
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "range_lookback": 48,
    "buffer_frac": 0.05,
    "session": "New York",
}
