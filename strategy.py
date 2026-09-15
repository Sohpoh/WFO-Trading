"""NY 5min overnight-range breakout — Donchian channel, vol-regime gate,
failed-breakout stop (session port of iteration 15's 5min config).

A one-axis slice change on this repo's only demonstrated edge. Accepted
iterations 6/10/11 (NQ 15min, New York) paired a Donchian breakout with two
*zero-parameter* overlays — an ATR volatility-regime gate and a
channel-width-quoted failed-breakout stop — and cleared the gate. Iteration 15
ported that identical entry primitive to 5min bars in the *London* overnight
window and got `no-edge`, its own reasoning concluding "the London session
itself, not the entry primitive, is the binding constraint." This iteration
changes exactly one economic thing — the session, London -> New York — and
holds everything else byte-identical: symbol NQ, 5min bars, both grids, the
288/1152-bar vol gate, the 0.5x-channel stop, no target, and the 12/3 train/test
schedule. A clean pass localizes the edge to the NY session where the 15min
version lives; a clean fail says the breakout edge was 15min-specific rather
than session-specific.

Why the wall-clock-matched 5min form of iteration 6/10/11's gate: the 15min run
used ATR windows of 96/384 bars (24h vs 96h). At 5min those same wall-clock
horizons are 288/1152 bars, so the gate windows are re-expressed 3x (see
constants below). Leaving them at 96/384 would silently redefine the gate as an
8h-vs-32h measure, which is a different indicator wearing the same name.

Rules (all computed on continuous, session-unaware bars):

  - Donchian channel over the *previous* `range_lookback` bars, current bar
    excluded:
        upper = High.rolling(n, min_periods=n).max().shift(1)
        lower = Low.rolling(n, min_periods=n).min().shift(1)
        channel_width = upper - lower
    The shift is what makes it a breakout rather than a tautology — without it
    the current bar's own High is in its own channel ceiling and Close can
    never exceed it. Strict `min_periods` (pandas' default = window) leaves
    unwarmed bars NaN so every comparison against them is False: the signal
    fails *closed*.

  - Breakout with a proportional buffer, so the required push scales with how
    wide the range already is:
        LONG  where Close > upper + buffer_frac * channel_width
        SHORT where Close < lower - buffer_frac * channel_width
    Mutually exclusive by construction: for any channel_width >= 0 the long
    threshold sits at or above the short threshold, so a single Close cannot
    clear both. The delegate's long-first if/elif tie-break never binds.

  - Volatility-regime gate, AND-ed into both sides. Carried over *unchanged in
    meaning* from accepted iterations 6/10/11, with its two windows scaled 3x
    for the 5min bar so it still measures 24h against 96h:
        ATR_fast = mean true range over atr_fast = 288 bars   (24h)
        ATR_slow = mean true range over atr_slow = 1152 bars  (96h)
        gate     = ATR_fast >= ATR_slow
    Both are fixed (never grid-searched) values — zero added degrees of
    freedom. Strict `min_periods` again means the gate fails closed while
    unwarmed.

Exit — path-dependent stop, *no* profit target, plus the forced session
flatten, all owned by `apply_session_constraint_with_stops()`:

  - stop_distance = STOP_WIDTH_MULT * channel_width of the trigger bar, with
    STOP_WIDTH_MULT = 0.5 a hardcoded module constant that is NOT
    grid-searched. This is the settled dimension from accepted iterations
    10/11 and is unchanged in *price* terms here, since channel_width spans
    the same wall-clock horizon as it did at 15min. The economic reading is
    "a breakout that gives back half the range it broke out of was a failed
    breakout", and it self-scales with `range_lookback` without refitting.

  - No target. The delegate refuses any entry whose target is NaN or on the
    wrong side of the entry Close, so "no target" has to be expressed as an
    unreachable level rather than omitted:
        LONG  -> Close + TARGET_WIDTH_MULT * channel_width
        SHORT -> Close - TARGET_WIDTH_MULT * channel_width
    with TARGET_WIDTH_MULT = 1000.0. Winners therefore run to the 16:00 ET
    session flatten. It is a `float` on purpose and never enters the grid — an
    int 1000 in a param dict would buy 3000 bars of meaningless warm-up via
    `_max_lookback_bars()`.

Contract notes, stated rather than assumed:

  - Repeat signals while already in position are no-ops: the delegate only
    opens when flat *and* in-session, which is what keeps the {-1,0,1}
    position contract intact (no pyramiding, never both sides at once).

  - Degenerate bars fail closed. A zero channel_width makes stop_distance 0
    and puts the target exactly at the Close; the delegate refuses to open
    without a strictly positive stop distance and a strictly correctly-sided
    target.

  - **Fill-model caveat, unchanged and stated up front.** A stop hit is
    *detected* intrabar on High/Low against the stored level, but the exit is
    priced at that bar's Close. Realized losses can therefore exceed
    stop_distance; results must not be described as capping loss at the stop.
    Relatedly, the walk never flips long->short directly — a stopped-out trade
    followed by an opposite-side entry is two trades and two round-trip cost
    legs.

Cost note: `metrics.py`'s per-leg toll is cost model v2 (0.001% fee + 0.005%
slippage, ~1.2bp round trip, recalibrated Aug 2026) and is out of scope for
this file. Iteration 15 was judged under the old ~10.2bp model and its own
caveat flagged that a fixed RTH slippage understated true overnight spreads;
cost v2 removes that cost-to-excursion objection, and a New York session trades
during RTH anyway, so no overnight-spread caveat applies here. Nothing about
costs is implemented in this file.

Warm-up: the binding requirement is the *gate*, not the channel —
atr_slow (1152) + 1 for the rolling mean's shift + 1 for TR's own
`Close.shift(1)` = 1154 bars, versus range_lookback + 1 <= 577 for the channel.
`range_lookback`, `atr_fast`, and `atr_slow` are genuine bar-count lookbacks
and are threaded through `build_grid()` as plain `int`s **on purpose**, so they
feed `_max_lookback_bars()` — whose max over the grid (1152 here) sizes the
engine's pre-test-window warm-up buffer to max((1152 + 5) * 3, day_bars + 5) =
3471 bars at 5min, comfortably covering the 1154-bar gate warm-up even if the
`range_lookback` grid later shrinks. `buffer_frac` is a ratio (dimensionless)
and is passed as a `float` **on purpose** so it can never inflate that buffer;
`STOP_WIDTH_MULT` and `TARGET_WIDTH_MULT` are module constants that never enter
the grid at all.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint_with_stops()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Volatility-regime gate windows, in bars. Fixed, NOT grid-searched — the
# zero-parameter overlay carried unchanged (in wall-clock meaning) from accepted
# iterations 6/10/11. Scaled 3x from the 15min-era 96/384 so that at 5min they
# still measure the same horizons: 288 bars = 24h, 1152 bars = 96h. They are
# threaded through `build_grid()` as fixed ints so `_max_lookback_bars()`
# accounts for the gate's warm-up (see the module docstring's warm-up note).
ATR_FAST_BARS = 288
ATR_SLOW_BARS = 1152

# Failed-breakout stop, quoted in units of the trigger bar's Donchian channel
# width. Hardcoded, NOT grid-searched — the settled dimension from accepted
# iterations 10/11. A dimensionless ratio (not a lookback), so it never enters
# the grid and never feeds `_max_lookback_bars()`.
STOP_WIDTH_MULT = 0.5

# "No target", expressed as an unreachable level because the delegate refuses
# any entry with a NaN or wrong-sided target. A float on purpose so it could
# never be mistaken for a bar-count lookback if it ever entered a param dict.
TARGET_WIDTH_MULT = 1000.0


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder true range: max(H-L, |H-C_prev|, |L-C_prev|).

    `skipna=False` keeps the first bar NaN (its `Close.shift(1)` is NaN) rather
    than silently falling back to High-Low.
    """
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)


def average_true_range(df: pd.DataFrame, period: int) -> pd.Series:
    """True range, simple-mean averaged over `period` bars, shifted one bar.

    The shift excludes the current bar from its own average — strictly more
    conservative than the gate needs (the gate compares two averages, so a
    same-bar contribution would partially cancel), and it keeps both legs of
    the ratio anchored to the same as-of-previous-bar information set.

    Strict `min_periods` keeps every unwarmed bar NaN, so comparisons against
    it are False during warm-up and the gate fails closed.
    """
    return true_range(df).rolling(period, min_periods=period).mean().shift(1)


def generate_positions(
    df: pd.DataFrame,
    range_lookback: int,
    buffer_frac: float,
    session: str | None = "New York",
    atr_fast: int = ATR_FAST_BARS,
    atr_slow: int = ATR_SLOW_BARS,
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Donchian channel over the previous `range_lookback` bars, current bar
    # excluded by the shift (otherwise Close could never exceed its own High).
    upper = high.rolling(range_lookback, min_periods=range_lookback).max().shift(1)
    lower = low.rolling(range_lookback, min_periods=range_lookback).min().shift(1)
    channel_width = upper - lower

    # Zero-parameter volatility-regime gate: only trade breakouts while
    # short-horizon volatility is at or above its longer-horizon baseline.
    # atr_fast/atr_slow are fixed ints (never grid-searched) — genuine
    # bar-count lookbacks threaded through build_grid() so the warm-up buffer
    # covers the gate's 1154-bar need.
    atr_fast_series = average_true_range(df, atr_fast)
    atr_slow_series = average_true_range(df, atr_slow)
    vol_regime = (atr_fast_series >= atr_slow_series).fillna(False)

    # Proportional-buffer breakout. Mutually exclusive by construction for any
    # channel_width >= 0. `.astype(bool)` is deliberate: the delegate indexes
    # these arrays with plain truthiness, and a stray object-dtype NaN is
    # truthy — this is what makes an unwarmed bar fail closed rather than open.
    long_signal = (
        (close > upper + buffer_frac * channel_width).fillna(False) & vol_regime
    ).astype(bool)
    short_signal = (
        (close < lower - buffer_frac * channel_width).fillna(False) & vol_regime
    ).astype(bool)

    # Failed-breakout stop, in channel-width units — symmetric, resolved to a
    # side by the delegate. A zero/NaN width makes the delegate refuse the entry.
    stop_distance = STOP_WIDTH_MULT * channel_width

    # "No target": an unreachable level, so the only exits are the stop and the
    # forced session flatten. Absolute and already direction-resolved, which is
    # the shape the delegate expects. Only read on actual entry bars.
    target_price = pd.Series(
        np.where(
            long_signal,
            close + TARGET_WIDTH_MULT * channel_width,
            close - TARGET_WIDTH_MULT * channel_width,
        ),
        index=df.index,
    )

    return apply_session_constraint_with_stops(
        close=close,
        high=high,
        low=low,
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    # range_lookback is a plain int because it IS a genuine bar-count lookback
    # (the Donchian channel window) and must feed wfo_engine's warm-up buffer;
    # buffer_frac is a float because it is a dimensionless ratio (NOT a
    # lookback) and must NOT feed the buffer. Both are grid-searched
    # ({72,144,288,576} bars, {0.05,0.10,0.15,0.20}). atr_fast/atr_slow are
    # fixed plain-int lookbacks (288 / 1152) — never grid-searched, but threaded
    # through build_grid() as fixed params so _max_lookback_bars() accounts for
    # the 1154-bar vol-gate warm-up. STOP_WIDTH_MULT (0.5) and TARGET_WIDTH_MULT
    # (1000.0) are module constants that never enter the grid. `session` is a
    # fixed param. These are also the concrete set the sanity checker runs
    # generate_positions() against.
    "range_lookback": 144,
    "buffer_frac": 0.10,
    "session": "New York",
    "atr_fast": ATR_FAST_BARS,
    "atr_slow": ATR_SLOW_BARS,
}
