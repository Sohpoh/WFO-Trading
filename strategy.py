"""London volatility-ignition continuation — expansion-bar entry, trigger-bar stop, no target.

A *continuation* strategy, deliberately confined to the quiet 02:00-05:00 ET
London window. Where this repo's retired fade (iteration 13) sold the poke that
was rejected back inside a channel, this buys the bar that breaks the quiet:
one bar whose true range is a large multiple of the prevailing baseline.

The edge thesis is volatility.md's clustering claim taken literally — "If
volatility was high yesterday, it's likely to be elevated today"; "After a gap
or large move, expect continued volatility before markets settle" — combined
with its regime table ("Momentum: often works better in higher volatility —
more pronounced trends"). In a structurally quiet session, a bar that is 2-3x
the prevailing ATR is not noise, it is the arrival of real order flow, and
nq-futures.md names exactly this window as an NQ volatility source ("Fed
announcements (2 AM ET)", "Asian market closes (2-3 AM ET affecting tech)").

Rules (all computed on continuous bars, with no session awareness of their own):

  - True range and its baseline:
        TR    = max(H - L, |H - C_prev|, |L - C_prev|)
        ATR   = TR.rolling(atr_period, min_periods=atr_period).mean().shift(1)
    The shift excludes the current bar from its own baseline — without it, a
    large bar would inflate the very average it is being compared against
    (lookahead, and it would mechanically damp its own signal). Strict
    `min_periods` (pandas' default = window) means an unwarmed bar is NaN and
    every comparison against it is False: the signal fails *closed*.

  - Ignition: bar_range = High - Low, and
        ignition = (bar_range >= expansion_mult * ATR) AND bar_range > 0 AND ATR > 0
    The two guards are not redundant. `bar_range > 0` is required because the
    close-location test below degenerates on a flat bar (a Close is
    simultaneously in the "top quartile" and the "bottom quartile" of a
    zero-width range), which would break the mutual exclusivity the delegate's
    stop/target walk documents as an assumption. `ATR > 0` is required because
    `bar_range >= mult * 0` is true on *every* bar, so a dead-flat baseline
    (possible on thin overnight tape) would otherwise turn every bar into an
    ignition.

  - Direction from close location *within the ignition bar's own range* — the
    bar has to have closed where it went, not merely have been wide:
        LONG  where ignition AND Close >= Low + CLOSE_LOC_TOP * bar_range
        SHORT where ignition AND Close <= Low + CLOSE_LOC_BOT * bar_range
    CLOSE_LOC_TOP = 0.75 / CLOSE_LOC_BOT = 0.25 are hardcoded module constants,
    NOT grid-searched — zero added degrees of freedom. Given the `bar_range > 0`
    guard the two sides are then mutually exclusive by construction (a Close
    cannot sit in both the top and the bottom quartile of a strictly positive
    range), so the delegate's long-first if/elif tie-break never binds.

  - Repeat signals while already in position are no-ops: the delegate's walk
    only opens when flat, which is also what keeps the {-1,0,1} position
    contract intact (no pyramiding).

Exit — path-dependent stop, *no* profit target, plus the session flatten, all
owned by `apply_session_constraint_with_stops()`:

  - stop_distance = STOP_BAR_MULT * bar_range of the trigger bar, with
    STOP_BAR_MULT = 1.0 a hardcoded module constant that is NOT grid-searched.
    It is quoted in trigger-bar units rather than ATR units on purpose: entry
    is at the Close of a bar whose range is >= 2x ATR with the Close in the
    top/bottom quartile, so a conventional ATR-quoted stop of 1-1.5x ATR would
    sit *inside* the trigger bar and be hit by that bar's own noise. At
    STOP_BAR_MULT = 1.0, the long stop lands at `close - bar_range`, i.e. about
    0.25 * bar_range *below* the trigger bar's Low (since the Close is in the
    top quartile) — the economic reading is "a full retrace of the ignition bar
    means the ignition failed" — and it self-scales with `expansion_mult`
    without needing to be re-fitted. Iterations 10 and 11 were both accepted
    with a hardcoded width-quoted stop, so this dimension is treated as settled
    in this repo and no search budget is spent on it.

  - No target. The delegate refuses any entry whose target is NaN or on the
    wrong side of the entry Close, so "no target" has to be expressed as an
    unreachable level rather than omitted:
        LONG  -> Close + TARGET_BAR_MULT * bar_range
        SHORT -> Close - TARGET_BAR_MULT * bar_range
    with TARGET_BAR_MULT = 1000.0. Winners therefore run to the forced session
    flatten — the exit shape of accepted iterations 10/11. (It is a float
    constant and never enters the grid; an int 1000 in a param dict would buy
    3000 bars of meaningless warm-up via `_max_lookback_bars()`.)

Two behaviours of the delegate, stated rather than assumed:

  - Degenerate bars fail closed twice over. An unwarmed ATR makes the ignition
    comparison False; a zero `bar_range` makes stop_distance 0 and the target
    exactly equal to the Close, and the delegate refuses to open without a
    strictly positive stop distance and a strictly correctly-sided target.

  - **Fill-model caveat, stated up front.** A stop hit is *detected* intrabar on
    High/Low against the stored level, but the exit is priced at that bar's
    Close. Volatility clustering — the very thesis of this entry — makes the bar
    following an ignition bar often large too, so realized losses will
    systematically *exceed* stop_distance. Results must not be described as
    capping loss at the stop. Relatedly, the walk never flips long->short
    directly: a stopped-out trade followed by an opposite-side entry is two
    trades and two round-trip cost legs.

Iteration 6's ATR-ratio volatility-*regime* gate is deliberately omitted so this
family's first run is attributable to the ignition entry alone; it is the
reserved zero-param lever for the next in-family variation.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here and
out of scope for this file. Unlike the fade it replaces, the reward leg is not
structurally capped — there is no target, so a winner's size is bounded only by
how far the move runs before the London flatten (2-3 hours after entry at most).

Warm-up: `atr_period` bars for the rolling mean plus the one-bar shift, plus one
more for TR's own `Close.shift(1)`, i.e. `atr_period + 2`. That's a genuine
bar-count lookback, so `atr_period` is passed as a plain `int` from
`build_grid()` and correctly sizes the engine's pre-test-window buffer;
`expansion_mult` is a ratio and is passed as a `float` so it can never inflate
that buffer. With the intended grid topping out at atr_period = 288, the engine
buffers max((288 + 5) * 3, day_bars + 5) = 879 bars, comfortably above the 290
needed, so the ATR is fully warm at each test window's first bar. The binding
floor is (max_lookback + 5) * 3 >= max_lookback + 2, which always holds — raise
the buffer rather than loosening `min_periods` if that ever changes.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint_with_stops()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Where the ignition bar's Close has to sit inside its own High-Low range for
# the move to count as directional. Hardcoded, NOT grid-searched: top/bottom
# quartile is an economic reading ("it closed where it went"), not a fitted
# value, so it costs zero degrees of freedom.
CLOSE_LOC_TOP = 0.75
CLOSE_LOC_BOT = 0.25

# Stop distance, quoted in units of the trigger bar's own range (NOT ATR — see
# the module docstring: an ATR-quoted stop would sit inside the trigger bar).
# Hardcoded, NOT grid-searched.
STOP_BAR_MULT = 1.0

# "No target", expressed as an unreachable level because the delegate refuses
# any entry with a NaN or wrong-sided target. A float on purpose so it could
# never be mistaken for a bar-count lookback if it ever entered a param dict.
TARGET_BAR_MULT = 1000.0


def average_true_range(df: pd.DataFrame, atr_period: int) -> pd.Series:
    """Wilder true range, simple-mean averaged over `atr_period`, shifted one bar.

    The shift excludes the current bar from its own baseline — an ignition bar
    must be measured against the volatility that preceded it, not against an
    average it has already contributed to.

    `skipna=False` on the row-wise max keeps the first bar's TR NaN (its
    `Close.shift(1)` is NaN) rather than silently falling back to High-Low, and
    strict `min_periods` keeps every unwarmed bar NaN, so comparisons against
    the ATR are False during warm-up and the signal fails closed.
    """
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    return tr.rolling(atr_period, min_periods=atr_period).mean().shift(1)


def generate_positions(
    df: pd.DataFrame,
    atr_period: int,
    expansion_mult: float,
    session: str | None = "London",
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    atr = average_true_range(df, atr_period)
    bar_range = high - low

    # Volatility ignition: this bar's range is a large multiple of the
    # prevailing baseline. Both guards matter — see the module docstring.
    ignition = (
        (bar_range >= expansion_mult * atr) & (bar_range > 0) & (atr > 0)
    ).fillna(False)

    # Direction from where the bar closed inside its own range. Mutually
    # exclusive by construction given `bar_range > 0`.
    long_signal = ignition & (close >= low + CLOSE_LOC_TOP * bar_range)
    short_signal = ignition & (close <= low + CLOSE_LOC_BOT * bar_range)

    # Stop in trigger-bar units — symmetric, resolved to a side by the
    # delegate. A zero/NaN range makes the delegate refuse the entry.
    stop_distance = STOP_BAR_MULT * bar_range

    # "No target": an unreachable level, so the only exits are the stop and the
    # forced session flatten. Absolute and already direction-resolved, which is
    # the shape the delegate expects. Only read on actual entry bars.
    target_price = pd.Series(
        np.where(
            long_signal,
            close + TARGET_BAR_MULT * bar_range,
            close - TARGET_BAR_MULT * bar_range,
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
    "atr_period": 96,
    "expansion_mult": 2.4,
    "session": "London",
}
