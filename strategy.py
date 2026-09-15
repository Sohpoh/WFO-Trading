"""NY 5min overnight-range breakout — Donchian channel, vol-regime gate,
persistence-confirmed entry, failed-breakout stop (variation of iteration 56).

Iteration 56 (this repo's NY 5min session port of the accepted 15min breakout)
was rejected as overfit-gap, not no-edge: median train Sharpe 2.98 vs median
fold OOS Sharpe -0.31, 48.7% fold consistency, and borderline robustness (top 5
of 142 trades = 99.4% of the +12.2% return). Its fold_table.csv showed no
grid-boundary pinning on either axis (range_lookback 72/144/288/576 in
13/12/15/8 folds; buffer_frac 0.05/0.1/0.15/0.2 in 5/23/11/9), so the searched
region was not the limiter and a param-range extension was out. The defect that
shape exposes is 5min-specific: a single bar's Close can pierce the channel on
a noise spike and reverse, producing false breakouts the optimizer curve-fits
in-sample. This iteration retires `buffer_frac` (the diffuse proportional-margin
axis the optimizer could not settle on) 1-for-1 and replaces it with
`confirm_bars` — a persistence requirement that the Close stays beyond the
broken channel level for N consecutive bars before entry. Requiring the break
to hold for 2-3 bars filters one-bar wicks while keeping genuine order-flow
breakouts, raising win rate rather than merely trading less (the difference
from iterations 8/9's failed magnitude-threshold filters). This is the one
lever iteration 4 proposed but never actually ran (it errored on a build_grid
naming issue).

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

  - Persistence-confirmed breakout against the RAW channel level (no
    proportional buffer — `buffer_frac` is retired):
        break_above = Close > upper
        break_below = Close < lower
        LONG  where break_above has been True for `confirm_bars` consecutive
              bars ending on the current bar
        SHORT where break_below has been True for `confirm_bars` consecutive
              bars ending on the current bar
    `confirm_bars = 1` degenerates to the plain single-bar breakout (the old
    buffer_frac = 0 case) and is retained as a control. Mutually exclusive by
    construction for any channel_width >= 0: upper >= lower, so a single Close
    cannot be simultaneously above upper and below lower, hence no bar can
    satisfy both persistence counts at once.

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
    10/11 and is unchanged in *price* terms here. The economic reading is
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
this file. Nothing about costs is implemented in this file.

Warm-up: the binding requirement is the *gate*, not the channel —
atr_slow (1152) + 1 for the rolling mean's shift + 1 for TR's own
`Close.shift(1)` = 1154 bars. The persistence-confirmed channel needs
range_lookback + confirm_bars bars (range_lookback + 1 for the shifted rolling
extreme, plus confirm_bars - 1 more so the last bar of the persistence window
sees a warmed channel), at most 576 + 3 = 579. `range_lookback`, `confirm_bars`,
`atr_fast`, and `atr_slow` are genuine bar-count lookbacks and are threaded
through `build_grid()` as plain `int`s **on purpose**, so they feed
`_max_lookback_bars()` — whose max over the grid (1152 here) sizes the engine's
pre-test-window warm-up buffer to max((1152 + 5) * 3, day_bars + 5) = 3471 bars
at 5min, comfortably covering the 1154-bar gate warm-up. `confirm_bars` is an
int *because* it is a rolling lookback (a `confirm_bars`-bar persistence
window), not an integer threshold: a huge grid value there genuinely would
require that much extra warm-up, so feeding the buffer is correct. `buffer_frac`
is gone entirely. `STOP_WIDTH_MULT` and `TARGET_WIDTH_MULT` are module constants
that never enter the grid at all.

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
    confirm_bars: int,
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

    # Persistence-confirmed breakout against the raw channel level (no
    # proportional buffer — `buffer_frac` is retired). `break_above`/`break_below`
    # are clean bools; the rolling `confirm_bars`-window sum counts consecutive
    # beyond-level bars, and `.eq(confirm_bars)` is True only when the whole
    # window (including the current bar) is beyond the level. `confirm_bars = 1`
    # degenerates to the plain single-bar breakout. `.astype(bool)` is
    # deliberate: the delegate indexes these arrays with plain truthiness, and a
    # stray object-dtype NaN is truthy — this is what makes an unwarmed bar fail
    # closed rather than open.
    break_above = close.gt(upper).fillna(False).astype(bool)
    break_below = close.lt(lower).fillna(False).astype(bool)
    confirmed_above = (
        break_above.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars)
    )
    confirmed_below = (
        break_below.rolling(confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars)
    )

    long_signal = (confirmed_above & vol_regime).astype(bool)
    short_signal = (confirmed_below & vol_regime).astype(bool)

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
    # range_lookback and confirm_bars are plain ints because they ARE genuine
    # bar-count lookbacks (the Donchian channel window and the persistence
    # window over it) and must feed wfo_engine's warm-up buffer. Both are
    # grid-searched ({72,144,288,576} bars, {1,2,3} bars). atr_fast/atr_slow are
    # fixed plain-int lookbacks (288 / 1152) — never grid-searched, but threaded
    # through build_grid() as fixed params so _max_lookback_bars() accounts for
    # the 1154-bar vol-gate warm-up. STOP_WIDTH_MULT (0.5) and TARGET_WIDTH_MULT
    # (1000.0) are module constants that never enter the grid. `session` is a
    # fixed param. These are also the concrete set the sanity checker runs
    # generate_positions() against.
    "range_lookback": 144,
    "confirm_bars": 2,
    "session": "New York",
    "atr_fast": ATR_FAST_BARS,
    "atr_slow": ATR_SLOW_BARS,
}
