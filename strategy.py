"""Variance-ratio regime polarity switch — one z-score primitive, sign set by
the measured regime, sigma-scaled stop and bounded sigma-scaled target.

One deviation primitive (a rolling z-score of Close) is traded in *both*
directions depending on what a variance-ratio test says the tape is currently
doing: continuation where the series is measurably trending, fade where it is
measurably mean-reverting, and nothing at all in the random-walk band between
them. This is `regime-changes.md`'s "Approach 2: Multiple Strategies …
Regime A (Trending): Momentum strategy; Regime B (Ranging): Mean-reversion
strategy … Switch between them based on detected regime", which every prior
iteration in this log skipped: each one committed to a single polarity for the
whole sample and was then killed by the stretches where the opposite polarity
ruled.

The obvious objection is that pure fades are 0-for-3 here (iterations 1/3/13).
None of them conditioned on a *measured* reverting regime — iteration 1 had no
regime filter at all, iteration 3 was gap-specific, and iteration 13's proxy
was the London session, which iterations 15/17 later showed was itself the
binding constraint. Here the fade leg only ever fires where
`statistical-mean-reversion-tests.md`'s variance-ratio test says the series is
genuinely mean-reverting.

Rules (all computed on the full continuous frame with no session awareness of
their own; every rolling window uses strict `min_periods` so an unwarmed leg
is NaN and both the signal and the regime measure fail *closed* — no entry —
rather than silently degrading to an always-true no-op):

  - Deviation primitive, in units of the same rolling sigma the exits use:
        mean_n = Close.rolling(z_lookback).mean()
        sd_n   = Close.rolling(z_lookback).std()
        z      = (Close - mean_n) / sd_n
    `sd_n` is masked to NaN wherever it is not strictly positive, so a
    degenerate flat stretch gives no z (and hence no signal) instead of
    dividing to +/-inf and firing at any threshold. That mask is a correctness
    guard, not an economic filter, and it also guarantees every entry bar has
    a strictly positive stop distance and a target strictly on the correct
    side of the Close — both preconditions of
    `apply_session_constraint_with_stops()`.

  - Regime measure — hardcoded module constants, NOT grid-searched, so the
    polarity switch costs the optimizer zero searchable degrees of freedom.
    Structurally identical to iteration 9's already-implemented variance
    ratio (same q = 24 bars, same 384-bar window, same overlapping-draws
    construction, same non-positive-denominator mask):
        r   = Close.diff() / Close.shift(1)
        r24 = Close.diff(VR_Q_BARS) / Close.shift(VR_Q_BARS)
        VR  = r24.rolling(VR_WINDOW_BARS).var()
              / (VR_Q_BARS * r.rolling(VR_WINDOW_BARS).var())
    For a random walk the variance of q-bar returns is q times the variance
    of 1-bar returns (VR ~ 1); a trending/persistent series overshoots that
    and a mean-reverting one undershoots it. Overlapping q-bar returns are
    used deliberately: non-overlapping draws would leave only 16 per 384-bar
    window, far too few to estimate a variance from.

    Note vs iteration 9: that iteration measured both legs on *log* returns;
    this one uses simple returns, per the specified entry rule. At 15min NQ
    the two differ in the ~1e-5 range and the difference is not the point of
    this iteration — but it does mean the ratio is not *literally*
    byte-identical to iteration 9's. The windows, the overlapping-draw
    construction and the degenerate-denominator mask are.

  - Polarity, from `statistical-mean-reversion-tests.md`'s canonical 1.0
    boundary ("VR ≈ 1 random walk, VR >> 1 trending, VR << 1 mean-reverting")
    with a hardcoded +/-0.05 deadband so the sign cannot flip on boundary
    noise:
        trend_regime  = VR >= VR_TREND_THRESHOLD   (1.05)
        revert_regime = VR <= VR_REVERT_THRESHOLD  (0.95)
    Neither is a fitted constant: 1.0 is the random-walk boundary itself and
    the deadband is symmetric around it. Between the two thresholds — and
    throughout the warm-up, where VR is NaN and both comparisons are False —
    there is no entry on either side.

  - Raw signals (the polarity switch itself):
        long  = (trend_regime  & z >= +entry_z)   # continuation of an up-push
              | (revert_regime & z <= -entry_z)   # fade of a down-stretch
        short = (trend_regime  & z <= -entry_z)
              | (revert_regime & z >= +entry_z)
    The two regimes are mutually exclusive (0.95 < 1.05) and `entry_z` is
    strictly positive on the whole intended grid, so long and short are
    mutually exclusive by construction, satisfying the single-position
    contract. (Caveat for the CLI, which will accept it: at `entry_z = 0`
    exactly, both z-conditions hold on the measure-zero set where z == 0. The
    short leg's `target_price` assignment below runs second and overwrites the
    long one, leaving a target *below* the Close — so the long branch inside
    `apply_session_constraint_with_stops()` fails its `target > close` check
    and the short branch fires. Short wins, by target-assignment order rather
    than by that function's branch order. The grid never visits that point.)

  - No lookahead: `mean_n`/`sd_n`/the VR legs are all backward-looking
    rolling windows ending on the current bar, every input is knowable at
    that bar's Close, and the entry is priced at that same Close.

Exit is a path-dependent stop/target — deliberately bounded, not the
run-to-session-flatten shape of iterations 6/10/11, all three of which
`loop-review.md` showed collapsing under gate v2's leave-top-5-out hard gate
(iteration 11: top 5 trades = 85% of net OOS P&L). The design goal is a
distribution of moderate winners, not a handful of outliers:

  - `long_signal`/`short_signal`/`stop_distance`/`target_price` go to
    `session.apply_session_constraint_with_stops()` (NOT the plain flip
    variant), which owns the bar-by-bar walk, the session gating and the
    forced flatten on the session's last bar.
  - stop_distance  = stop_sigma * sd_n      (fixed at 2.0, wide enough that
    whipsaws don't punish it per `volatility.md`'s "Stop-Loss Placement").
  - target_price   = Close + target_sigma * sd_n on a long bar
                     Close - target_sigma * sd_n on a short bar
    Both legs are quoted in the same rolling-sigma unit as the z-score, so
    the risk/reward ratio is scale-invariant across `z_lookback` and across
    volatility regimes — one grid is valid for every lookback.
  - On the fade leg, `target_sigma` ≈ `entry_z` reproduces
    `mean-reversion.md`'s documented Bollinger exit ("Enter short when price
    touches the upper band … Exit at middle"); a shorter `target_sigma`
    harvests a partial reversion instead.
  - Fill-price caveat is `session.py`'s: a stop/target hit is detected off
    the bar's High/Low but flattened at that bar's Close, so the stop caps
    *when* you exit, not the realized loss on the triggering bar.

Cost note: `metrics.py` charges ~0.102% round-trip off Close, unchanged here
and out of scope for this file. The smallest searched target (1.0 sigma of
Close over >= 24 bars on 15min NQ) is ~0.3-0.5%, roughly 3-5x that toll.

Warm-up and param types:

  - `z_lookback` is a genuine bar-count lookback, so it is passed as a plain
    `int` from `build_grid()` and correctly sizes the engine's pre-test-window
    buffer. `entry_z`, `target_sigma` and `stop_sigma` are all multiples of a
    rolling sigma, not bar counts, and are passed as `float` so they cannot
    inflate that buffer.

  - The regime leg needs VR_WINDOW_BARS + VR_Q_BARS = 384 + 24 = 408 bars
    (the q-bar return leg is itself NaN for its own first 24 bars before the
    rolling variance can see a full window of it), and it is *invisible* to
    `wfo_engine._max_lookback_bars()` because both windows are module
    constants rather than grid params. That is safe for the intended grid:
    the longest searched `z_lookback` is 192, so
        buffer_bars = max((192 + 5) * 3, day_bars + 5) = 591,
    which clears both 408 and the z leg's own 192 bars. The binding
    constraint is the regime measure, so the floor on the *longest* searched
    `z_lookback` is 131 — below that, buffer_bars drops under 408 and every
    test window would open with VR still NaN, i.e. zero trades with no error
    to explain it. Keep 192 at the top of the grid, or raise the buffer (e.g.
    by threading a fixed int lookback param through `build_grid()`) rather
    than loosening `min_periods`.

This module decides only *when* the strategy wants to be long or short. All
day-trade gating and the end-of-session flatten are delegated to
`session.apply_session_constraint_with_stops()`; see its docstring for that
contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Variance-ratio regime measure. Hardcoded on purpose: the polarity switch
# adds no searchable degrees of freedom, and the thresholds bracket the
# random-walk boundary (VR = 1 exactly for a random walk) rather than being
# fitted numbers. VR_Q_BARS = 24 is ~6h of 15min bars — long enough to span an
# intraday swing, short enough that a 384-bar (~4 day) window still holds many
# overlapping draws of it. The +/-0.05 deadband keeps the polarity from
# sign-flipping bar to bar on boundary noise.
VR_Q_BARS = 24
VR_WINDOW_BARS = 384
VR_TREND_THRESHOLD = 1.05
VR_REVERT_THRESHOLD = 0.95


def variance_ratio(close: pd.Series) -> pd.Series:
    """Rolling variance ratio VR(q) of simple Close returns.

        VR = Var[r_q] / (q * Var[r_1])

    computed over a trailing `VR_WINDOW_BARS` window on overlapping q-bar
    returns. VR ~ 1 is a random walk, VR > 1 persistent/trending, VR < 1
    mean-reverting.

    `close.diff(k) / close.shift(k)` rather than `close.pct_change(k)`: the
    two are identical, but the bare `pct_change()` call emits a pandas
    FutureWarning about its `fill_method` default.

    The q-bar leg is NaN for its own first `VR_Q_BARS` bars, so with strict
    `min_periods` the first non-NaN VR lands on the
    `VR_WINDOW_BARS + VR_Q_BARS`-th bar (positional index 407) — that is this
    strategy's binding warm-up (see the module docstring's arithmetic).

    A non-positive denominator (a degenerate zero-variance stretch) is masked
    to NaN rather than allowed to divide to +/-inf, so both of the caller's
    threshold comparisons are False and the regime gate fails closed like
    every other unwarmed leg.
    """
    r1 = close.diff(1) / close.shift(1)
    rq = close.diff(VR_Q_BARS) / close.shift(VR_Q_BARS)

    var_1 = r1.rolling(VR_WINDOW_BARS, min_periods=VR_WINDOW_BARS).var()
    var_q = rq.rolling(VR_WINDOW_BARS, min_periods=VR_WINDOW_BARS).var()

    denom = VR_Q_BARS * var_1
    denom = denom.where(denom > 0)
    return var_q / denom


def rolling_sigma(close: pd.Series, z_lookback: int) -> pd.Series:
    """Trailing `z_lookback`-bar standard deviation of Close, in price units.

    Masked to NaN where it is not strictly positive, which is what makes the
    z-score finite, the stop distance positive, and the target strictly on
    the right side of the Close at every bar that can produce a signal.
    """
    sd = close.rolling(z_lookback, min_periods=z_lookback).std()
    return sd.where(sd > 0)


def zscore(close: pd.Series, z_lookback: int, sd: pd.Series) -> pd.Series:
    """(Close - rolling mean) / rolling sigma, both over `z_lookback` bars."""
    mean_n = close.rolling(z_lookback, min_periods=z_lookback).mean()
    return (close - mean_n) / sd


def generate_positions(
    df: pd.DataFrame,
    z_lookback: int,
    entry_z: float,
    target_sigma: float,
    stop_sigma: float = 2.0,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    sd_n = rolling_sigma(close, z_lookback)
    z = zscore(close, z_lookback, sd_n)

    # Mutually exclusive regimes (0.95 < 1.05), both False throughout the
    # warm-up since NaN compares False on either side of the deadband.
    vr = variance_ratio(close)
    trend_regime = (vr >= VR_TREND_THRESHOLD).fillna(False)
    revert_regime = (vr <= VR_REVERT_THRESHOLD).fillna(False)

    stretched_up = (z >= entry_z).fillna(False)
    stretched_down = (z <= -entry_z).fillna(False)

    # The polarity switch: continue the stretch in a trending regime, fade it
    # in a reverting one. Mutually exclusive for entry_z > 0 (the whole
    # intended grid) because the regimes are.
    long_signal = (trend_regime & stretched_up) | (revert_regime & stretched_down)
    short_signal = (trend_regime & stretched_down) | (revert_regime & stretched_up)

    stop_distance = stop_sigma * sd_n

    # Direction-resolved absolute target level, NaN off signal bars (which is
    # how apply_session_constraint_with_stops() reads "no valid entry here").
    target_price = pd.Series(np.nan, index=df.index)
    target_price[long_signal] = (close + target_sigma * sd_n)[long_signal]
    target_price[short_signal] = (close - target_sigma * sd_n)[short_signal]

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
    "z_lookback": 96,
    "entry_z": 1.5,
    "target_sigma": 1.5,
    "stop_sigma": 2.0,
    "session": "New York",
}
