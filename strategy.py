"""London overnight-drift hold — sqrt(L)-scaled drift *state*, plain flip exit.

Three London primitives have now been tested in this repo (iterations 13/14/15)
and all three were *event*-triggered: a fade, an ignition and a breakout, each
firing at some moment inside the 02:00-05:00 ET window and each exiting at the
05:00 flatten. Backing `metrics.py`'s ~10.2bps round-trip toll out of their
reported numbers leaves gross expectancy of roughly +1bp, -1bp and -9bps per
trade — i.e. the entries carried little to no directional information, and the
naive sign-flip of the worst of them is only ~+9bps gross, ~-1bp net.

So this iteration does not argue "drift instead of breakout". It argues
*excursion per toll*. An event trigger pays the full fixed cost on whatever
remains of an already-short 3-hour window after the trigger fires (iteration
15's OOS trade log shows entries scattered 06:00-09:50 UTC against exits
clustered at the flatten). A *state* signal is instead already true or false at
the window's first bar, so the day-trade gate opens the position on that first
in-session bar and holds it to the forced flatten: one round trip, maximum
excursion per toll. That is the only lever the cost arithmetic leaves open on
this window.

Stated plainly rather than hidden behind the new family name: the entry
statistic is a close cousin of rejected iteration 7's t-stat drift — drift
divided by (ATR * sqrt(L)) and a mean-over-sigma t-stat are the same statistic
up to a sqrt(L) scaling the threshold grid absorbs. The load-bearing
differences are (a) the session, (b) the removal of iteration 6's
volatility-regime gate, and (c) the one-round-trip turnover design. Iteration 7
was rejected on overfit gap with in-sample PF 1.09, not on no-edge, so the
primitive was never shown to be information-free.

Rules (all computed on continuous, session-unaware bars):

  - Volatility scale: `atr` = Wilder true range, simple-mean averaged over
    ATR_BARS = 288 bars (24h at 5min), strict `min_periods`, `.shift(1)` so the
    current bar is excluded from its own scale. ATR_BARS is a hardcoded module
    constant and is NOT grid-searched — zero added degrees of freedom.

  - Drift over the searched horizon:
        drift = Close - Close.shift(drift_lookback)

  - Scale the drift by the random-walk-typical move over the *same* horizon, so
    one threshold is meaningful across every lookback:
        z = drift / (atr * sqrt(drift_lookback))
    `.where(atr > 0)` masks degenerate zero-volatility bars: without it a zero
    denominator gives +/-inf, and +inf clears any threshold, so a dead bar
    would fire at every grid setting. Masked bars are NaN and fail closed.

  - Entries, a persistent state rather than an event:
        LONG  where z >=  drift_mult
        SHORT where z <= -drift_mult
        NaN   elsewhere ("no signal this bar")
    Mutually exclusive for any drift_mult > 0 (the whole intended grid). Strict
    `min_periods` leaves unwarmed bars NaN, so every comparison against them is
    False and the signal fails closed. Under a random walk E|z| ~ 0.8, so the
    grid 0.75/1.0/1.25/1.6 spans roughly 45%/32%/23%/11% of bars qualifying:
    every combo selects an at-or-above-typical overnight move — the "big
    excursion" the cost arithmetic requires — and none of the 16 combos is
    degenerate.

Exit — plain flip, no stop and no target. The raw entries series is handed
straight to `apply_session_constraint()` (deliberately NOT the stops variant).
Position is the forward-fill of the gated entries, so a trade ends only on a
sign flip of z or on the forced 05:00 ET flatten that the delegate owns. This
is the one exit dimension never varied across the London attempts — 13, 14 and
15 all used a path-dependent stop and all three reported average loss larger
than average win — and it is what preserves the one-round-trip-per-session cost
profile the whole thesis rests on.

Why the grid starts at 72 bars: every searched `drift_lookback` is >= 72 bars
(6h), i.e. at least 2x the 3-hour window, so the trailing drift window can
never fully roll over inside a single session. Mid-session sign flips — which
would cost two extra legs on exactly the sessions that matter most — therefore
stay rare. 36 bars is excluded from the grid for that reason, not for fit.

Contract notes:

  - Repeat same-side signals while already in position are no-ops by
    construction: the position is a forward-fill of a {1.0, -1.0} series, so
    re-asserting the same value cannot add exposure. The {-1, 0, 1} contract
    holds with no pyramiding and no simultaneous long/short.

  - Iteration 6's volatility-regime gate is deliberately gone. Its hardcoded
    1152-bar slow leg is what forced iteration 15's grid top to 576 and left
    7 in-sample / 65 out-of-sample trades — per-fold parameter selection on
    one or two recurring trades of noise. Without it the binding warm-up here
    is only ~433 bars.

Warm-up: `drift_lookback` (<= 432) + 1 for `Close.shift()` = 433 bars binds,
against ATR's 288 + 1 (true range's own `Close.shift(1)`) + 1 (the mean's
shift) = 290. `drift_lookback` is a genuine bar-count lookback and is passed as
a plain `int` from `build_grid()` so it feeds the engine's warm-up buffer;
`drift_mult` is a threshold in sigma units and is passed as a `float` so it can
never inflate that buffer. At the intended grid top the engine buffers
max((432 + 5) * 3, day_bars + 5) = 1311 bars >= 433, so both legs are warm at
every test window's first bar. **Caveat: ATR_BARS is a module constant and is
invisible to that arithmetic** — the buffer is sized purely by the largest int
in the grid. Narrowing the searched `drift_lookback` set below ~96 would drop
the int-derived buffer under ATR's 290 bars, leaving the scale NaN (signal
closed) at the start of every test window; the engine's one-full-day floor
happens to cover that by three bars at 5min, which is not margin worth relying
on. Keep the grid top at 432.

Cost caveat to carry into evaluation (NOT changed here — `metrics.py` is out of
scope for this file): the cost model charges a fixed 0.05% slippage regardless
of session, while overnight/international spreads are materially wider than the
regular-hours figure. This backtest therefore *understates* true London cost.
Pre-registered honest hurdle: on qualifying sessions the remaining NQ London
excursion is ~40-60 points at ~15k, so gross must clear ~15 points (10.2bps)
per trade — roughly a 57-60% directional hit rate on a symmetric payoff. If
this fails at the *gross* level too, the binding constraint is the window
itself rather than the entry primitive, and the London steer should be retired
rather than given a sixth attempt.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Volatility scale for the drift statistic, in bars. Hardcoded, NOT
# grid-searched — 288 bars = 24h at 5min, the same wall-clock horizon the
# accepted iterations used for their fast volatility leg.
ATR_BARS = 288


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

    The shift excludes the current bar from its own scale, so the denominator
    of `z` is anchored to strictly prior information — the current bar's own
    range can't shrink the yardstick it is being measured against.

    Strict `min_periods` keeps every unwarmed bar NaN, so comparisons against
    it are False during warm-up and the signal fails closed.
    """
    return true_range(df).rolling(period, min_periods=period).mean().shift(1)


def scaled_drift(df: pd.DataFrame, drift_lookback: int) -> pd.Series:
    """Trailing `drift_lookback`-bar drift in random-walk-typical-move units.

    A random walk with per-bar scale `atr` moves ~`atr * sqrt(L)` over L bars,
    so dividing by that makes the statistic comparable across every lookback
    in the grid and lets a single threshold grid apply to all of them.

    `.where(atr > 0)` drops zero-volatility bars, which would otherwise divide
    by zero and produce a +/-inf that clears any threshold.
    """
    close = df["Close"]
    drift = close - close.shift(drift_lookback)
    atr = average_true_range(df, ATR_BARS)
    scale = atr * np.sqrt(drift_lookback)
    return (drift / scale).where(atr > 0)


def generate_positions(
    df: pd.DataFrame,
    drift_lookback: int,
    drift_mult: float,
    session: str | None = "London",
) -> pd.Series:
    z = scaled_drift(df, drift_lookback)

    # Mutually exclusive for drift_mult > 0 (the whole intended grid). NaN
    # (unwarmed or zero-volatility) compares False on both sides: fails closed.
    long_signal = (z >= drift_mult).fillna(False)
    short_signal = (z <= -drift_mult).fillna(False)

    entries = pd.Series(index=df.index, dtype=float)  # all-NaN = "no signal"
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "drift_lookback": 144,
    "drift_mult": 1.0,
    "session": "London",
}
