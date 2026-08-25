"""Trailing-range location trend-hold — stochastic-location state signal, ATR stop, no target.

The pivot here is horizon/cost-per-trade, not a new indicator. `metrics.py`
charges a flat ~10.2bps round trip (0.001% fee + 0.05% slippage per leg,
2 legs) regardless of timeframe, and all 24 prior iterations ran 5min/15min
bars — iterations 21/22/24 bled roughly -11bps net per trade on ~0bps gross
across 383-946 trades. `transaction-costs.md`'s documented remedy ("reduce
trade frequency; lower-frequency strategies can afford higher per-trade
costs") has never been applied in this log. At 1h bars the New York session
holds only ~6 tradable bars, so a wide deadband yields on the order of one
round trip per session held ~6h against an ES RTH range near 110bps — the
toll is ~9% of a typical session excursion instead of ~40% at 15min.

The signal is a *level/state*, not an event. Iterations 6/10/11 fired only on
new extremes and were killed by the gate's leave-top-5-out concentration
check: a handful of breakout days carried the whole curve. Being long simply
*because* price currently sits in the top zone of its trailing range trades
most sessions, so P&L is spread across many trades and deleting the best five
removes ~1% of the sample rather than ~3%.

ES rather than NQ (23 of 24 prior iterations used NQ) is the closest thing to
untouched data against `loop-review.md`'s named multiple-testing problem, and
ES's tighter spread makes `metrics.py`'s flat 5bps slippage more realistic.

Rules (all computed on the full continuous frame with no session awareness of
their own; every input is a strictly backward rolling window, so unwarmed bars
are NaN and the signal fails *closed*):

  - Location inside the trailing range (Lane's stochastic %K, expressed 0-1):

        lo_t  = rolling_min(Low,  loc_lookback)   ending at t
        hi_t  = rolling_max(High, loc_lookback)   ending at t
        loc_t = (Close_t - lo_t) / (hi_t - lo_t)

    The window ends at t and includes bar t's own High/Low, so `loc_t` is
    bounded to [0, 1] and is fully knowable at t's Close — nothing after t is
    touched. Strict `min_periods` keeps unwarmed bars NaN; a degenerate
    (zero-width) window is masked to NaN too, since it would divide by zero.

  - Raw entries, a *level* condition that re-arms every bar rather than a
    crossing:
        +1.0 where loc_t >= loc_threshold
        -1.0 where loc_t <= 1 - loc_threshold
         NaN in the deadband between them.
    A repeat same-direction signal while already in that position is a no-op,
    per the position contract. The two sides are mutually exclusive *because
    every threshold in the grid is > 0.5* — if the grid is ever widened to
    include 0.5 or below, both legs could be True on the same bar and this
    construction would need an explicit tie-break.

  - Stop: `stop_atr_mult * ATR(atr_period)` measured from the entry Close,
    symmetric — the delegate resolves it below the entry for longs and above
    for shorts. `volatility.md`'s guidance (scale the stop with measured
    volatility, wide enough that ordinary noise doesn't trip it) is what a
    2.0x ATR stop encodes.

  - No profit target. Winners run to the forced session flatten, which is the
    "trend-hold" half of the idea: at 1h the session is only ~6 bars, so the
    hold is naturally bounded by the clock rather than by a level. The
    delegate refuses any entry whose target is NaN or on the wrong side of the
    entry Close, so "no target" has to be expressed as an unreachable level —
    the same convention iteration 14 used.

Exit is path-dependent, so this delegates to
`session.apply_session_constraint_with_stops()` (both legs supplied on every
entry bar, as that function requires). `session.py`'s forced flatten on the
session's last bar remains the backstop — no position survives the session.

Known divergence from the idea as specified: the spec called for an
opposite-zone signal to *flip* an open position. The delegate only opens when
it is flat (by design — it owns the session bookkeeping a flip would have to
respect), and faking a flip here would require session awareness inside
`strategy.py`, which the contract forbids. So an opposite-zone signal does not
reverse an open trade; the trade ends at its ATR stop or at the session
flatten, and the opposite side can only be taken from flat on a later in-
session bar. A corollary worth stating plainly: because the signal is a level
that re-arms every bar, a stopped-out trade whose zone condition still holds
will simply re-enter the same direction on the next in-session bar, so the
realized rate can exceed the spec's assumed one round trip per session.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):
  - `loc_lookback` and `atr_period` are genuine bar counts, so both are passed
    as plain `int` and correctly size the pre-test-window warm-up buffer
    (max 48 -> (48+5)*3 = 159 bars, ample for either).
  - `loc_threshold` and `stop_atr_mult` are unitless multipliers, not bar
    counts, so both are passed as `float` and are correctly ignored by that
    buffer sizing.

Cost note: `metrics.py`'s ~0.102% round-trip is unchanged and out of scope —
the whole point of moving to 1h is to amortize that fixed toll over a larger
per-trade excursion, not to re-price it.

This module decides only *when* the strategy wants to be long or short and at
what levels it wants out. All day-trade gating and the end-of-session flatten
are delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd


from session import apply_session_constraint_with_stops

# "No target", expressed as an unreachable level because the delegate refuses
# any entry with a NaN or wrong-sided target. A float on purpose so it could
# never be mistaken for a bar-count lookback if it ever entered a param dict.
NO_TARGET_ATR_MULT = 1000.0


def average_true_range(df: pd.DataFrame, atr_period: int) -> pd.Series:
    """Wilder true range, simple-mean averaged over `atr_period`, shifted one bar.

    The shift excludes the current bar from its own volatility baseline, so the
    stop is sized off strictly prior information — the entry bar's own range
    can't widen or narrow the stop it is about to be given.

    `skipna=False` on the row-wise max keeps the first bar's TR NaN (its
    `Close.shift(1)` is NaN) rather than silently falling back to High-Low, and
    strict `min_periods` keeps every unwarmed bar NaN, so the delegate's
    `stop_distance > 0` guard refuses entries during warm-up.
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


def range_location(df: pd.DataFrame, loc_lookback: int) -> pd.Series:
    """Close's position inside its trailing `loc_lookback`-bar High/Low range, 0-1.

    Lane's stochastic %K on a 0-1 scale. The rolling window ends at (and
    includes) the current bar, so the value is bounded to [0, 1] and knowable
    at that bar's Close. NaN until the window is full, and NaN on a degenerate
    zero-width range — both cases make every downstream comparison False, so
    the signal fails closed.
    """
    n = int(loc_lookback)
    lo = df["Low"].rolling(n, min_periods=n).min()
    hi = df["High"].rolling(n, min_periods=n).max()
    width = hi - lo
    return ((df["Close"] - lo) / width).where(width > 0)


def generate_positions(
    df: pd.DataFrame,
    loc_lookback: int,
    loc_threshold: float,
    stop_atr_mult: float = 2.0,
    atr_period: int = 14,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    loc = range_location(df, loc_lookback)
    thr = float(loc_threshold)

    # Level (not crossing) conditions: the state re-arms on every bar price
    # spends in the zone. `fillna(False)` is load-bearing — the delegate reads
    # these as raw numpy values and `bool(np.nan)` is True, so a NaN left in
    # the array would fire an entry on an unwarmed bar.
    # Mutually exclusive only because every grid threshold is > 0.5.
    long_signal = (loc >= thr).fillna(False)
    short_signal = (loc <= 1.0 - thr).fillna(False)

    # Protective stop only, in ATR units from the entry Close. Symmetric — the
    # delegate resolves the side. NaN/zero ATR makes it refuse the entry.
    atr = average_true_range(df, atr_period)
    stop_distance = float(stop_atr_mult) * atr

    # "No target": an unreachable level, so the only exits are the ATR stop and
    # the forced session flatten. Absolute and already direction-resolved,
    # which is the shape the delegate expects. Only read on actual entry bars.
    target_price = pd.Series(
        np.where(
            long_signal,
            close + NO_TARGET_ATR_MULT * atr,
            close - NO_TARGET_ATR_MULT * atr,
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
    "loc_lookback": 24,
    "loc_threshold": 0.80,
    "stop_atr_mult": 2.0,
    "atr_period": 14,
    "session": "New York",
}
