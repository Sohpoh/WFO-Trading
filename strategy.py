"""Vol-normalized intraday drift momentum + volatility-regime gate, flip exit.

A time-series-momentum continuation strategy on the full continuous frame,
traded as a plain flip: long when the trailing N-bar drift is large *relative
to its own noise*, short on the mirror image, and no exit at all other than
the opposite-side signal or the session's forced flatten.

The change from the previous iteration is the entry primitive only. The
breakout channel is gone; what replaces it is a normalized drift statistic
(a t-stat on the mean log return). The thesis is that continuation is a
property of the *rate* of drift relative to its own volatility, not of a
single bar poking through a boundary — so this enters mid-trend rather than
only at range edges, and is far less whipsaw-prone at the boundary itself.
The volatility-regime gate below is carried over byte-identical.

Rules (all computed with no session awareness of their own):

  - Log returns and their drift t-statistic:
        r      = log(Close).diff()
        mu     = r.rolling(N).mean()
        sd     = r.rolling(N).std()
        mom_t  = (mu / sd) * sqrt(N)
    with N = `mom_lookback`. This is the N-bar cumulative move expressed in
    units of its own realized volatility. The sqrt(N) scaling is what makes
    it comparable across lookbacks, so one `entry_t` grid is valid for every
    N rather than needing a per-N threshold.
  - No skip period. The usual S=1 skip is justified by *monthly-horizon*
    equity short-term reversal, which does not transfer to 15min NQ, so it
    would be a mis-citation here.
  - No lookahead: `r` is a backward difference and the rolling window ends on
    the current bar's own return, which is knowable at that bar's Close. The
    entry is priced at that same Close, exactly as the previous iteration's
    breakout was.
  - Raw signals:
        long  where mom_t >=  entry_t
        short where mom_t <= -entry_t
    `entry_t` is strictly positive on the intended grid, so the two are
    mutually exclusive by construction. (Caveat for the CLI, which will
    accept it: at `entry_t = 0` exactly, both conditions are true when
    mom_t == 0 and short wins by assignment order below. The grid never
    visits that point.)
  - Degenerate windows are dropped: a stretch of identical nonzero returns
    gives sd == 0 and mu/sd == +/-inf, which would fire a signal at *any*
    threshold. `mom_t` is therefore masked to NaN wherever sd is not
    strictly positive. (sd == 0 with mu == 0 is already NaN via 0/0, but
    the explicit mask covers both cases.)
  - Volatility-regime gate (hardcoded, NOT grid-searched — unchanged from
    the previous iteration):
        TR       = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        atr_fast = TR.rolling(ATR_FAST_BARS).mean()
        atr_slow = TR.rolling(ATR_SLOW_BARS).mean()
        vol_ok   = atr_fast >= atr_slow
    A raw signal only becomes an entry where `vol_ok`; elsewhere the bar
    contributes NaN (no signal), exactly as if the drift hadn't reached the
    threshold. The gate is applied symmetrically to long and short, and only
    ever *removes* signals, so it cannot break their mutual exclusivity.
    Gating only *fresh-from-flat* entries would need the forward-filled
    position state, which the sparse entries-series contract has no room for
    — that's `apply_session_constraint()`'s job, not this module's.
  - Why a slow ratio rather than an absolute vol threshold: a ratio is
    scale-free (no points/percent constant to fit or to re-fit as NQ's price
    level doubles) and, because volatility clusters, the fast/slow crossing
    is a multi-day regime switch rather than a bar-by-bar chop filter. It
    adds zero searchable degrees of freedom.

Exit is a plain flip — deliberately no stop and no target:

  - The position forward-fills from the entry bar until either the opposite
    threshold fires (a reversal, 2 cost legs) or the forced flatten on the
    session's last bar (1 leg). The ffilled 0 then propagates through the
    overnight gap, so every session starts flat.
  - Explicit consequence of gating the *exit* side too: when the opposite
    signal happens in a suppressed (contracting-vol) regime, the position
    does not reverse — it carries to the session flatten. That is part of
    the thesis, not an oversight.
  - A repeat same-direction signal while already positioned is a true no-op
    with zero extra cost legs (it just re-writes the same 1.0/-1.0 into a
    series that is already forward-filled to that value), satisfying the
    single-position contract by construction.
  - A bar-count holding period is deliberately NOT used: `session.py`'s
    contract only supports 1.0/-1.0/NaN entries with 0.0 written by the
    session module itself, so a timed exit is not one of the two permitted
    exit shapes.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here.
The flip-only exit books strictly fewer cost legs than the stop/target
variants did, since there is no early exit followed by an in-session
re-entry.

Warm-up: `mom_lookback + 1` bars — N returns for the rolling mean/std, plus
the one bar that `log().diff()` costs. That's a genuine bar-count lookback,
so `mom_lookback` is passed as a plain `int` from `build_grid()` and
correctly sizes the engine's pre-test-window buffer; `entry_t` is a
threshold in t-units, not a bar count, and is passed as a `float` so it
cannot inflate that buffer.

The gate's own warm-up is 384 bars (`ATR_SLOW_BARS` exactly — TR's first bar
degrades to High-Low rather than NaN, so `Close.shift(1)` costs nothing
here), and it is *invisible* to `wfo_engine._max_lookback_bars()` because it
is a module constant rather than a grid param. That is safe for the intended
grid: the longest searched `mom_lookback` is 192, so
buffer_bars = max((192+5)*3, day_bars+5) = 591, which clears both 384 and
the 193-bar momentum warm-up before the first test-window bar. The rolling
windows use strict `min_periods` on purpose, so an incompletely warmed slow
leg is NaN and the gate fails *closed* (no entries) rather than silently
degrading to an always-true no-op. Caveat for whoever changes the grid next:
a longest searched `mom_lookback` below 123 shrinks the buffer under 384 and
would start every test window with the gate closed — i.e. zero trades, with
no error to explain it. If you do that, raise the buffer (e.g. by threading
a fixed int lookback param through `build_grid()`) rather than loosening
`min_periods`.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Volatility-regime gate windows, in bars. Hardcoded on purpose: the gate adds
# no searchable degrees of freedom, so the only change versus the previous
# iteration is the entry primitive itself. Sized for 15min bars — 96 bars is
# ~one full 24h Globex day and 384 is ~four of them, i.e. a slow, multi-day
# regime switch rather than a bar-by-bar chop filter.
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


def drift_tstat(close: pd.Series, mom_lookback: int) -> pd.Series:
    """t-statistic of the trailing `mom_lookback`-bar mean log return.

    (mu / sd) * sqrt(N) — the N-bar drift measured in units of its own
    realized volatility, and scale-free across N so a single threshold grid
    is valid for every lookback.

    Strict `min_periods` on both legs so an unwarmed window is NaN (no
    signal) rather than a partially-estimated one. Windows with a
    non-positive/undefined `sd` are masked out: N identical nonzero returns
    would otherwise give +/-inf here and fire at any threshold.
    """
    r = np.log(close).diff()
    mu = r.rolling(mom_lookback, min_periods=mom_lookback).mean()
    sd = r.rolling(mom_lookback, min_periods=mom_lookback).std()
    return ((mu / sd) * (mom_lookback ** 0.5)).where(sd > 0)


def generate_positions(
    df: pd.DataFrame,
    mom_lookback: int,
    entry_t: float,
    session: str | None = "New York",
) -> pd.Series:
    mom_t = drift_tstat(df["Close"], mom_lookback)

    # Mutually exclusive for entry_t > 0 (the whole intended grid); the gate
    # only ever removes signals, so it preserves that.
    vol_ok = vol_regime_ok(df)
    long_signal = (mom_t >= entry_t).fillna(False) & vol_ok
    short_signal = (mom_t <= -entry_t).fillna(False) & vol_ok

    entries = pd.Series(index=df.index, dtype=float)  # all-NaN = "no signal"
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "mom_lookback": 48,
    "entry_t": 1.0,
    "session": "New York",
}
