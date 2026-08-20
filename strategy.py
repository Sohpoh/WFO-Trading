"""Rolling N-bar range breakout + vol-regime gate + variance-ratio persistence
gate, flip exit — no stops.

Bar-indexed Donchian channel on the full continuous frame, traded as a plain
flip strategy: long when Close breaks out above the prior `range_lookback`
bars' high (plus a buffer), short on the mirror-image break below, and no exit
at all other than the opposite gated break or the session's forced flatten.

The breakout core and the exit are byte-identical to the last *accepted*
iteration of this family (rolling range breakout + volatility-regime gate,
flip exit). Two things change this iteration:

  1. The widened `buffer_frac` grid of the immediately preceding iteration is
     retired outright and the searched range is restored to the accepted
     iteration's values, byte-identical, as a control:

         buffer_frac:  0.15 / 0.20 / 0.25 / 0.35  ->  0.0 / 0.05 / 0.10 / 0.15

     That widening was tested and failed on its own terms — OOS trade count
     roughly halved while OOS Sharpe fell by more than half and max drawdown
     exceeded the whole period's return — and its fold table showed
     `buffer_frac` splitting bimodally across *both* edges of the new grid
     rather than pinning to one, i.e. not an unfinished boundary but a param
     the folds disagree about. `range_lookback` (24/48/96/192) is likewise
     unchanged from the accepted iteration.

  2. A second hardcoded, NOT grid-searched gate is ANDed onto the entry
     condition: a variance-ratio persistence filter (below). Restoring both
     grids to their accepted values is what makes this new gate the only
     moving part, so any change in results is attributable to it.

Why a *second* gate rather than a retune: the accepted iteration's fold table
still selected combos with clearly negative train Sharpe in some folds and
~zero in others *with the vol-regime gate live*. Those are volatile-but-choppy
stretches — exactly the failure mode an ATR-magnitude gate structurally cannot
see, because it measures how *big* the bars are, not whether their moves
accumulate in a direction. The variance ratio separates the two: for a random
walk the variance of q-bar returns is q times the variance of 1-bar returns
(VR ~ 1); a trending/persistent series overshoots that (VR > 1) and a
mean-reverting/choppy one undershoots it (VR < 1). The 1.0 threshold is the
random-walk boundary itself, not a fitted constant, which is why it costs zero
degrees of freedom.

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
    Both gates below only ever *remove* signals, so they cannot break that.
  - Gate 1 — volatility-regime (hardcoded, NOT grid-searched):
        TR       = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        atr_fast = TR.rolling(ATR_FAST_BARS).mean()
        atr_slow = TR.rolling(ATR_SLOW_BARS).mean()
        vol_ok   = atr_fast >= atr_slow
    Why a ratio rather than an absolute vol threshold: a ratio is scale-free
    (no points/percent constant to fit or to re-fit as NQ's price level
    doubles) and, because volatility clusters, the fast/slow crossing is a
    multi-day regime switch rather than a bar-by-bar chop filter.
  - Gate 2 — variance-ratio persistence, NEW this iteration (hardcoded, NOT
    grid-searched):
        r1  = log(Close).diff(1)
        rq  = log(Close).diff(VR_Q_BARS)
        VR  = rq.rolling(VR_WINDOW_BARS).var()
              / (VR_Q_BARS * r1.rolling(VR_WINDOW_BARS).var())
        vr_ok = VR >= VR_THRESHOLD
    i.e. the variance of ~6h returns measured against VR_Q_BARS times the
    variance of 1-bar returns, both over the same rolling ~4-day window.
    VR_THRESHOLD is 1.0 — the random-walk boundary — so the gate is open only
    where the recent tape has been at least as persistent as a random walk,
    and closed in the mean-reverting/choppy regimes where a breakout pays two
    cost legs to be faded.
  - A raw signal becomes an entry only where BOTH gates are True; elsewhere
    the bar contributes NaN (no signal), exactly as if the break hadn't
    happened. Both gates are applied symmetrically to long and short. Gating
    only *fresh-from-flat* entries would need the forward-filled position
    state, which the sparse entries-series contract has no room for — that's
    `apply_session_constraint()`'s job, not this module's.
  - Every window uses strict `min_periods` on purpose, so an incompletely
    warmed leg is NaN and the gate fails *closed* (no entry) rather than
    silently degrading to an always-true no-op. The VR denominator is
    additionally masked where it is non-positive, so a degenerate
    zero-variance stretch also fails closed instead of dividing to +inf and
    forcing the gate permanently open.
  - Both gates' windows are hardcoded module constants, deliberately outside
    the grid, so together they add zero degrees of freedom for the optimizer
    to overfit. Restoring `buffer_frac`'s grid does not buy either gate a
    searchable param.

Exit is a plain flip — deliberately no stop and no target:

  - The position forward-fills from the breakout bar until either the
    opposite band breaks (a reversal, 2 cost legs) or the forced flatten on
    the session's last bar (1 leg). The ffilled 0 then propagates through
    the overnight gap, so every session starts flat.
  - Explicit consequence of gating the *exit* side too: when the opposite
    break happens while either gate is closed, the position does not reverse
    — it simply carries to the session flatten. That is part of the thesis,
    not an oversight (the flip-flop reversals in chop are exactly the
    two-cost-leg whipsaws the gates exist to remove), but with two stacked
    gates this suppressed-reversal behaviour compounds and will happen
    strictly more often than in the single-gate iteration. Read the trade log
    with that in mind.
  - A repeat same-direction signal while already positioned is a true no-op
    with zero extra cost legs (it just re-writes the same 1.0/-1.0 into a
    series that is already forward-filled to that value), satisfying the
    single-position contract by construction.
  - Dropping the path-dependent stop/target walk (done in an earlier
    iteration of this family) stays dropped. On that walk a stop was
    *detected* intrabar but the exit still priced at the triggering bar's
    Close, so it never actually capped the loss on that bar — it only decided
    when to leave, and then allowed a fresh in-session re-entry that paid two
    more cost legs.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here
and out of scope for this file. The design bets on a larger average gross move
per trade, not on a cheaper toll.

Power caveat: two stacked gates cut trade count as well as trade quality. If
the OOS trade count comes in far below the previous iterations', treat the run
as under-powered rather than as a verdict on the persistence gate itself.

Warm-up: `range_lookback` bars for the rolling extremes plus the one-bar
shift. That's a genuine bar-count lookback, so it is passed as a plain `int`
from `build_grid()` and correctly sizes the engine's pre-test-window buffer;
`buffer_frac` is a fraction and is passed as a `float` so it cannot.

The gates' own warm-ups are *invisible* to `wfo_engine._max_lookback_bars()`
because they are module constants rather than grid params, so the arithmetic
has to be checked by hand:

  - vol-regime gate: ATR_SLOW_BARS = 384 bars exactly (TR's first bar degrades
    to High-Low rather than NaN, so `Close.shift(1)` costs nothing here).
  - persistence gate: VR_WINDOW_BARS + VR_Q_BARS = 384 + 24 = 408 bars, since
    the q-bar return leg is itself NaN for its first VR_Q_BARS bars before the
    rolling variance can see a full window of it. This is now the binding leg.

With the intended grid the engine buffers
buffer_bars = max((192 + 5) * 3, day_bars + 5) = 591 bars > 408, so both gates
are fully warm before the first test-window bar. Caveat for whoever changes
the grid next: the binding floor on the longest searched `range_lookback` is
now 131, not the 123 the single-gate version needed — below that, buffer_bars
drops under 408 and every test window would open with the persistence gate
closed. Keep 192 at the top of the grid, or raise the buffer (e.g. by
threading a fixed int lookback param through `build_grid()`) rather than
loosening `min_periods`.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`apply_session_constraint()`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Volatility-regime gate windows, in bars. Hardcoded on purpose: the gate adds
# no searchable degrees of freedom. Sized for 15min bars — 96 bars is ~one
# full 24h Globex day and 384 is ~four of them, i.e. a slow, multi-day regime
# switch rather than a bar-by-bar chop filter.
ATR_FAST_BARS = 96
ATR_SLOW_BARS = 384

# Variance-ratio persistence gate. Also hardcoded, and the threshold is the
# random-walk boundary itself (VR = 1 exactly for a random walk) rather than a
# fitted number, so this gate costs zero degrees of freedom too. VR_Q_BARS =
# 24 is ~6h of 15min bars — long enough to span an intraday swing, short
# enough that a 384-bar (~4 day) window still holds many overlapping draws of
# it.
VR_Q_BARS = 24
VR_WINDOW_BARS = 384
VR_THRESHOLD = 1.0


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


def variance_ratio(df: pd.DataFrame) -> pd.Series:
    """Rolling variance ratio VR(q) of log Close returns.

        VR = Var[r_q] / (q * Var[r_1])

    computed over a trailing `VR_WINDOW_BARS` window on overlapping q-bar
    returns. VR ~ 1 is a random walk, VR > 1 persistent/trending, VR < 1
    mean-reverting.

    The q-bar leg is NaN for its own first `VR_Q_BARS` bars, so with strict
    `min_periods` the first non-NaN VR lands on the
    `VR_WINDOW_BARS + VR_Q_BARS`-th bar (positional index 407) — that, not the
    384-bar ATR leg, is this
    strategy's binding warm-up (see the module docstring's warm-up
    arithmetic). Overlapping windows are used deliberately: non-overlapping
    q-bar returns would leave only 16 draws per 384-bar window, far too few to
    estimate a variance from.

    A non-positive denominator (degenerate zero-variance stretch) is masked to
    NaN rather than allowed to divide to +/-inf, so the caller's `>=` comparison
    is False and the gate fails closed like every other unwarmed leg.
    """
    log_close = np.log(df["Close"])
    r1 = log_close.diff(1)
    rq = log_close.diff(VR_Q_BARS)

    var_1 = r1.rolling(VR_WINDOW_BARS, min_periods=VR_WINDOW_BARS).var()
    var_q = rq.rolling(VR_WINDOW_BARS, min_periods=VR_WINDOW_BARS).var()

    denom = VR_Q_BARS * var_1
    denom = denom.where(denom > 0)
    return var_q / denom


def persistence_ok(df: pd.DataFrame) -> pd.Series:
    """True where the variance ratio says the tape is at least random-walk
    persistent (VR >= VR_THRESHOLD = 1.0), False in mean-reverting chop and
    False while unwarmed (NaN compares False, then `fillna(False)` makes the
    all-NaN warm-up region explicit)."""
    return (variance_ratio(df) >= VR_THRESHOLD).fillna(False)


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
    # both gates only ever remove signals, so they preserve that.
    gate_ok = vol_regime_ok(df) & persistence_ok(df)
    long_signal = (close > upper + buffer_frac * width).fillna(False) & gate_ok
    short_signal = (close < lower - buffer_frac * width).fillna(False) & gate_ok

    entries = pd.Series(index=df.index, dtype=float)  # all-NaN = "no signal"
    entries[long_signal] = 1.0
    entries[short_signal] = -1.0

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "range_lookback": 48,
    "buffer_frac": 0.05,
    "session": "New York",
}
