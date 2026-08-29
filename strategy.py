"""Money-flow pressure trend state (Chaikin Money Flow), held to the flatten.

Thirty-one prior iterations mined price and price-derived volatility. Volume
appeared only twice, and never as a *direction*: as a relative-volume event
trigger (iterations 18/19) and as a price weighting inside VWAP (26-28). The
information channel here is new to this project — the **intrabar close
location weighted by volume**, i.e. where each bar settles inside its own
High-Low range, scaled by how much traded in that bar, and summed over a
rolling window. That is the accumulation/distribution reading, and summed as
Chaikin Money Flow it is orthogonal to every rolling-mean / band / regression
z-score that iterations 21 and 24 concluded is information-free on this data.

The holding *geometry* is deliberately inherited from iteration 31 while its
statistic is discarded: a persistent state signal with **no extremity
condition at entry**, held until the opposite state fires or the session
flattens. Iteration 31 failed at PF 0.9166 because there was no edge to
concentrate, not because the geometry concentrated it, so the geometry is
re-used rather than re-litigated. Thresholds are set loose on purpose: 8/9
showed selectivity collapsing trade counts 183 -> 100 -> 74 straight into an
overfit gap, so the design target is many small symmetric bets, not a few
tail-selected ones.

Rules (all computed on the full continuous frame with no session awareness of
their own; every window is strictly backward-looking and includes only the
current *completed* bar, so there is no lookahead):

  - Money-flow volume, per bar:

        MFM_t = ((Close_t - Low_t) - (High_t - Close_t)) / (High_t - Low_t)
              = (2*Close_t - High_t - Low_t) / (High_t - Low_t)     in [-1, 1]

        MFV_t = MFM_t * Volume_t

    ZERO-RANGE BARS ARE FORCED TO EXACTLY 0.0, NOT NaN. A zero-range bar
    carries no directional information, and NaN would be actively harmful
    here rather than merely absent: `rolling(n, min_periods=n).sum()` treats
    NaN as *missing*, so a single NaN drops its window below `min_periods` and
    NaNs out the next `n` CMF values. ES 15min has plenty of zero-range
    overnight/holiday bars, so letting them through as NaN would silently gut
    the signal at large `cmf_lookback` without raising anything. The bar's
    Volume still counts in the denominator (per spec), which only dampens CMF
    toward zero — the conservative direction.

  - Chaikin Money Flow over a strictly-warm window:

        CMF_t = rolling_sum(MFV, cmf_lookback, min_periods=cmf_lookback)
                / rolling_sum(Volume, cmf_lookback, min_periods=cmf_lookback)

    NaN wherever the window is unwarmed, and NaN wherever the volume
    denominator is <= 0 (an all-zero-volume window would otherwise divide by
    zero). NaN compares False on both sides below, so unwarmed bars fail
    *closed* with no separate validity mask.

  - Raw entries — a STATE, not a crossing:

        CMF_t >=  entry_pressure  -> LONG   (1.0)
        CMF_t <= -entry_pressure  -> SHORT (-1.0)
        otherwise                 -> NaN    (no *new* signal)

    Every qualifying bar is marked, not just the bar that first crosses. NaN
    means "no new signal", so `apply_session_constraint()`'s ffill holds the
    prior position through the neutral zone rather than closing it. The two
    sides are mutually exclusive for any `entry_pressure > 0`, which the whole
    intended grid satisfies.

Exit is a plain flip — no stop, no target, no path dependence — so this hands
the raw entries series to `session.apply_session_constraint()` unchanged. A
position flips only when CMF crosses the opposite threshold; otherwise it is
closed by `session.py`'s forced flatten on the session's last bar. Winners are
therefore uncapped (iteration 27 showed hard-capping a continuation payoff at
2R flipping per-trade economics from +0.4bps to -11.6bps) and losers are
truncated by the opposite-side crossing or by that flatten.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):

  - `cmf_lookback` is a plain `int` **on purpose** — it is a genuine bar-count
    lookback, so it is exactly what should size the pre-test-window warm-up
    buffer. At the grid's top end (96) that gives
    `buffer_bars = max((96 + 5) * 3, day_bars + 5)` = 303 bars, comfortably
    covering a 96-bar rolling window.
  - `entry_pressure` is unitless (CMF is a ratio in [-1, 1]), never a bar
    count, so it is passed as `float` and is correctly ignored by the buffer
    sizing.
  - There is **no `history_bars` hint** and none is needed: unlike iteration
    31 this signal is a plain rolling window, not day-anchored, so the
    engine's generic int-scan sizes it correctly on its own. This also
    dissolves iteration 31's coverage hazard — the fold-skip guard
    (`max_lookback + 10`) drops from 1450 bars to 106, so coarse timeframes no
    longer skip every fold.

Cost note: `metrics.py`'s ~0.102% round trip (0.001% fee + 0.05% slippage per
leg, 2 legs) is unchanged and out of scope. ES is chosen on idea-breadth
hygiene grounds (it appears in only 2 of 31 iterations, so it is the closest
thing to untouched data against the named multiple-testing hazard), not as a
cost argument — ES ~1.08% vs NQ ~1.3% ATR-to-price makes the fixed toll
comparable on both.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to `session.py`;
see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint


def chaikin_money_flow(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Rolling Chaikin Money Flow over `lookback` completed bars.

    Strictly backward-looking and inclusive of the current bar only. Returns
    NaN on unwarmed bars (strict `min_periods`) and wherever the rolling
    volume denominator is <= 0.

    Zero-range bars contribute exactly 0.0 to the numerator — never NaN, which
    would drop the containing window below `min_periods` and blank the next
    `lookback` values. See the module docstring.
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    rng = high - low
    # (Close - Low) - (High - Close), algebraically identical to the spec form.
    with np.errstate(invalid="ignore", divide="ignore"):
        mfm = (2.0 * close - high - low) / rng
    # Zero-range (or otherwise non-finite) bars carry no directional
    # information -> exactly 0.0, applied BEFORE the rolling sum.
    mfm = mfm.where(rng > 0, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    mfv = mfm * volume

    n = int(lookback)
    mfv_sum = mfv.rolling(n, min_periods=n).sum()
    vol_sum = volume.rolling(n, min_periods=n).sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        cmf = mfv_sum / vol_sum
    # Guard the degenerate denominator; unwarmed bars are already NaN from the
    # strict min_periods above, and both fail closed downstream.
    return cmf.where(vol_sum > 0)


def generate_positions(
    df: pd.DataFrame,
    cmf_lookback: int,
    entry_pressure: float,
    session: str | None = "New York",
) -> pd.Series:
    cmf = chaikin_money_flow(df, cmf_lookback)

    # STATE, not crossing: every bar sitting on the pressure side is marked, so
    # the position is re-asserted rather than fired once. NaN CMF (unwarmed, or
    # a dead-volume window) compares False both ways and stands aside.
    thr = abs(float(entry_pressure))
    with np.errstate(invalid="ignore"):
        long_state = (cmf >= thr).to_numpy()
        short_state = (cmf <= -thr).to_numpy()

    # Raw, session-unaware entries. NaN means "no new signal" — the delegate's
    # ffill holds the prior position through the neutral zone instead of
    # closing it. Explicit float64 so that ffill/fillna arithmetic stays
    # numeric.
    entries = pd.Series(np.nan, index=df.index, dtype=float)
    entries[long_state] = 1.0
    entries[short_state] = -1.0

    # session.py alone decides which of those bars are tradable and force-
    # flattens on the session's last bar. Plain flip, no stops/targets, so
    # apply_session_constraint_with_stops() is deliberately not used.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "cmf_lookback": 24,
    "entry_pressure": 0.10,
    "session": "New York",
}
