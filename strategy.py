"""Rolling N-bar range breakout + vol-regime gate, failed-breakout stop at half
the channel width — no profit target.

Bar-indexed Donchian channel on the full continuous frame: long when Close
breaks out above the prior `range_lookback` bars' high (plus a buffer), short
on the mirror-image break below, both ANDed with a single hardcoded
volatility-regime gate.

The entry side — signal, both searched grids, and the ATR-ratio gate — is
byte-identical to the last *accepted* iteration of this family (rolling range
breakout + volatility-regime gate, flip exit). Exactly one thing changes this
iteration: **the exit**. The variance-ratio persistence gate tried in the
immediately preceding iteration is removed outright (it cut trade count
further and came back with a worse overfit gap), so this run is attributable
to the new exit alone.

Why change the exit and nothing else. The two levers already spent off the
accepted iteration are exhausted in the same direction: a wider `buffer_frac`
grid and then a second (persistence) gate both traded fewer, more selective
breakouts, and both cut OOS trade count hard while getting worse — "be more
selective about which breakouts to take" is now tested and failed twice. The
opposite lever is ruled out by arithmetic on this family's own results: the
vol-regime gate removed roughly a third of the ungated iteration's trades and
about a tenth of its total return, i.e. ~-0.10% net per removed trade, which
after backing out the ~0.102% round-trip toll charged in `metrics.py` is ~0.00%
gross. Those trades carried no directional information in *either* direction,
so neither loosening the gate nor fading them can pay. What is left is gross
P&L per trade on the entries already being taken — i.e. the exit.

And the accepted iteration's exit was, in practice, no exit: reading its OOS
trade log, essentially every trade closes at the forced session flatten. The
flip almost never fires, so losers are held exactly as long as winners and the
left tail runs past -1.7%.

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
  - The trigger is Close-based on purpose: this codebase has no intrabar price
    series, every bar is priced off its Close, so a High-touch trigger would
    book a fill at a price the bar's own high says was already exceeded.
  - Long and short are mutually exclusive for free: `upper >= lower` always
    and `buffer_frac >= 0`, so `upper + buffer_frac*width >= lower -
    buffer_frac*width` and a single Close cannot satisfy both inequalities.
    The gate below only ever *removes* signals, so it cannot break that.
  - Volatility-regime gate (hardcoded, NOT grid-searched):
        TR       = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        atr_fast = TR.rolling(ATR_FAST_BARS).mean()
        atr_slow = TR.rolling(ATR_SLOW_BARS).mean()
        vol_ok   = atr_fast >= atr_slow
    Why a ratio rather than an absolute vol threshold: a ratio is scale-free
    (no points/percent constant to fit or to re-fit as NQ's price level
    doubles) and, because volatility clusters, the fast/slow crossing is a
    multi-day regime switch rather than a bar-by-bar chop filter. Strict
    `min_periods` on both legs so an unwarmed gate is NaN and therefore False
    — it fails *closed* (suppress the trade) rather than degrading to an
    always-true no-op.
  - The gate's windows are hardcoded module constants, deliberately outside
    the grid, so it adds zero degrees of freedom for the optimizer to overfit.

Exit — NEW this iteration: a failed-breakout stop, no target.

  - stop_distance = STOP_WIDTH_FRAC * width, with STOP_WIDTH_FRAC = 0.5 a
    module constant that is *not* grid-searched (zero added degrees of
    freedom, and no grid/CLI plumbing change). The economic reading: a
    breakout that immediately gives back half the channel it just broke out
    of is a failed breakout, and is cut. Quoting it in channel widths keeps
    it in the same unit `buffer_frac` already uses, so it is volatility- and
    price-level-adaptive by construction rather than a fitted point value.
  - Sizing sanity on this dataset: ATR96 runs ~0.13% of price (checked on
    both a 2023 and a 2025 slice), so a ~24h channel gives a stop of roughly
    0.5-0.7% of price. That sits deliberately *between* the accepted
    iteration's ordinary noise losers (-0.03% to -0.3%, which should survive
    untouched) and its tail losers (-0.5% to -1.8%, which should be cut).
  - No profit target. `apply_session_constraint_with_stops()` requires a
    non-NaN, correctly-sided target on every entry bar, so the target is set
    to a deliberately unreachable 100 * stop_distance from the entry Close.
    One session cannot travel 50 channel widths, so the profit side is
    unchanged from the accepted iteration: winners still run to the forced
    session-close flatten.
  - Consequence for reversals, stated explicitly: the stop/target walk never
    flips directly from long to short — it only opens when flat. But a stop
    at half the channel width always triggers long before price could reach
    the opposite band, so what used to be a single flip now shows up as a
    stop-out followed by a fresh opposite entry: two trades, and the same two
    cost legs. Nothing is silently held through a reversal.
  - Fill-price honesty (see the delegate's own docstring): a stop hit is
    *detected* intrabar on High/Low against the stored level, but the exit is
    priced at that bar's Close. So the stop caps *when* you exit, not the
    realized loss on the triggering bar — results must not be described as
    guaranteeing a maximum loss of stop_distance.
  - Why the earlier abandoned stop attempt in this repo does not transfer:
    that one hung the stop off a static day-anchored range level that was
    re-triggerable at the same price all session, so it churned. This channel
    ratchets after every breakout — a re-entry needs a *new* extreme beyond
    the run's peak — so the same-price churn path is closed.
  - Degenerate bars fail closed for free: an unwarmed or zero `width` makes
    stop_distance NaN/0 and the target NaN, and the delegate refuses to open
    a position without a strictly positive stop distance and a valid,
    correctly-sided target.

Pre-registered direction (so the read of the result is not made after the
fact): OOS trade count should come in at or *above* the accepted iteration's,
since stops add exits and permit later re-entries within the same session. A
count materially below it is an implementation smell — most likely a NaN/zero
width suppressing entries, or a mis-sided target series — not a verdict on the
stop.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here
and out of scope for this file. Note that this iteration *adds* cost legs (a
stopped-out trade that re-enters pays two round trips where the flip version
paid one), so the stop has to save more than that to show up as an
improvement.

Warm-up: `range_lookback` bars for the rolling extremes plus the one-bar
shift. That's a genuine bar-count lookback, so it is passed as a plain `int`
from `build_grid()` and correctly sizes the engine's pre-test-window buffer;
`buffer_frac` is a fraction and is passed as a `float` so it cannot inflate
that buffer. `STOP_WIDTH_FRAC` never enters the grid at all, so it is
invisible to the buffer arithmetic — correctly, since it introduces no new
lookback (it reuses the channel width already computed).

The gate's own warm-up is likewise invisible to
`wfo_engine._max_lookback_bars()` (module constants aren't grid params), so
check it by hand: with the persistence gate gone the binding leg is
ATR_SLOW_BARS = 384 bars exactly (TR's first bar degrades to High-Low rather
than NaN, so `Close.shift(1)` costs nothing). With the intended grid the
engine buffers max((192 + 5) * 3, day_bars + 5) = 591 bars > 384, so the gate
is fully warm before the first test-window bar. Caveat for whoever changes the
grid next: keep 192 at the top of the `range_lookback` grid (the binding floor
on the longest searched lookback is 123) or raise the buffer, rather than
loosening `min_periods`.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint_with_stops()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Volatility-regime gate windows, in bars. Hardcoded on purpose: the gate adds
# no searchable degrees of freedom. Sized for 15min bars — 96 bars is ~one
# full 24h Globex day and 384 is ~four of them, i.e. a slow, multi-day regime
# switch rather than a bar-by-bar chop filter.
ATR_FAST_BARS = 96
ATR_SLOW_BARS = 384

# Failed-breakout stop, quoted as a fraction of the breakout channel's own
# width. Hardcoded, NOT grid-searched: 0.5 is the "gave back half of what it
# broke out of" boundary, chosen for its economic reading rather than fitted,
# so it costs zero degrees of freedom.
STOP_WIDTH_FRAC = 0.5

# Multiple of the stop distance used as the (deliberately unreachable) profit
# target. The stop/target walk requires a non-NaN, correctly-sided target on
# every entry bar; 100 stop-widths inside one session is not attainable, so
# this reproduces "no target — winners run to the session flatten".
UNREACHABLE_TARGET_MULT = 100.0


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range: max(H-L, |H-C_prev|, |L-C_prev|).

    `prev_close` is NaN on the first bar, so two of the three legs are NaN
    there; `.max(axis=1)` skips NaNs by default, leaving TR[0] = High - Low.
    That degenerate-but-sane first value is intentional (no fillna needed),
    which is why this gate's warm-up is exactly `ATR_SLOW_BARS` bars, not one
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

    # Failed-breakout stop: half the channel width, symmetric, resolved to a
    # side by the delegate. NaN/0 width (warm-up or a degenerate flat channel)
    # propagates here and makes the delegate refuse the entry.
    stop_distance = STOP_WIDTH_FRAC * width

    # No real target — an unreachable level on the correct side of the entry
    # Close, so the profit side is governed only by the session flatten. Only
    # read on actual entry bars, so the value on non-signal bars is irrelevant
    # (but is still sided off `long_signal` for readability).
    reach = UNREACHABLE_TARGET_MULT * stop_distance
    target_price = pd.Series(
        np.where(long_signal, close + reach, close - reach),
        index=df.index,
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
    "range_lookback": 48,
    "buffer_frac": 0.05,
    "session": "New York",
}
