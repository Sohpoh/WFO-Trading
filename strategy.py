"""Regression-channel reversion — OLS-residual sigma entry, exit at the fitted line.

`mean-reversion.md` lists exactly three simple mean-reversion recipes; the log
has mined only one (Bollinger bands, iteration 1). This implements the
untouched third verbatim — "Linear Regression / Threshold: Fit a line to
recent prices. When price deviates by N standard deviations from the line,
assume reversion" — with that page's documented Bollinger exit ("expecting
reversion to middle. Exit at middle") supplying a zero-parameter target.

Why this is not a seventh z-score fade. Iteration 21 concluded the rolling-
*mean* z-score is information-free on this data in either polarity (PF 0.675
IS and OOS with both signs live). But a rolling mean lags inside a drifting
window, so price sits persistently on one side of it and the fade is really
fighting local drift — `mean-reversion.md` pitfall #1, "in a strong uptrend,
shorting strength is a losing strategy". An OLS fit removes the local slope,
so the quantity being faded (the residual) is closer to stationary, and it
does so *without* a separate trend filter — which matters because iterations
8 and 9 showed added conjunctive filters cutting OOS trade count
183 -> 100 -> 74 and collapsing into an overfit gap.

It also repairs an unnamed defect in iteration 21: its `entry_z` grid started
at 1.0, so part of the grid implied a reversion target that could not clear
`metrics.py`'s ~10.2bps round-trip toll at all. Here `entry_sigma` floors at
1.5 and `reg_lookback` at 48, putting the smallest implied target at roughly
3-5x the toll — the discipline iterations 1 and 2 applied and 21 dropped.

Rules (all computed on the full continuous frame with no session awareness of
their own; every input is a strictly backward rolling window, so unwarmed
bars are NaN and the signal fails *closed*):

  - For every bar t, fit an ordinary least-squares line of Close on bar index
    over the trailing `reg_lookback` bars ending at t. Closed form off rolling
    moments only:

        slope_t     = cov(x, y) / var(x)                  (rolling, window n)
        line_t      = mean(y) + slope_t * (x_t - mean(x))
        sigma_t     = sqrt(max(var(y) - slope_t^2 * var(x), 0))
        z_t         = (Close_t - line_t) / sigma_t

    `sigma_t` is the residual standard deviation inside that same window, via
    the OLS identity SSresid = SSyy - slope^2 * SSxx. Nothing after bar t is
    touched; `line_t` is the fitted value *at* t (the window's right edge), so
    it is knowable at t's Close and the entry is priced at that same Close.

    `x` is the positional bar index of the frame handed in. Simple regression
    is invariant to the origin of a uniformly-spaced x, so the slope/line/
    sigma at bar t are identical whether the frame is the whole dataset or one
    of `wfo_engine._simulate_window()`'s buffered slices.

  - Raw entries (crossing, not level, so the signal does not re-arm on every
    bar of a sustained excursion):
        -1.0 where z_t >  entry_sigma and z_{t-1} <=  entry_sigma
        +1.0 where z_t < -entry_sigma and z_{t-1} >= -entry_sigma
         NaN elsewhere.
    Unwarmed bars leave z NaN, and NaN compares False on both sides, so no
    entry can fire before the window is full. With pandas' default
    `min_periods == window`, z first goes non-NaN at position n-1 and
    z.shift(1) is still NaN there, so the first possible entry is position n.

  - Target = `line_t`, the fitted line level captured at the entry bar —
    mean-reversion.md's "exit at middle" read literally, held as a fixed price
    level for the life of the trade rather than recomputed. By construction a
    short entry has Close > line and a long entry has Close < line, so the
    target is always on the correct side of the entry Close, which is what
    `apply_session_constraint_with_stops()` requires. Implied target distance
    is entry_sigma * sigma_t.

  - Stop = entry price displaced a further `stop_sigma * sigma_t` *away* from
    the line (below entry for longs, above for shorts). Stop distance is
    therefore independent of `entry_sigma`, so reward:risk runs 0.75:1 to
    1.5:1 across the intended grid. `volatility.md`'s stop guidance (scale the
    stop with measured volatility, wide enough that whipsaws don't punish it)
    is satisfied by quoting it in the same residual-sigma unit as the entry.

Exit is path-dependent, so this delegates to
`session.apply_session_constraint_with_stops()` (both legs supplied on every
entry bar, as that function requires). `session.py`'s forced flatten on the
session's last bar remains the backstop — no position survives the session.
The bounded line target (rather than a run-to-session-flatten) is deliberate:
it keeps P&L from being tail-driven, which is what the gate's
leave-top-5-out hard check penalizes.

Param types / warm-up:
  - `reg_lookback` is a genuine bar count, so it is passed as a plain `int`
    from `build_grid()` and correctly feeds `wfo_engine._max_lookback_bars()`'s
    pre-test-window buffer (384 -> (384+5)*3 = 1167 bars, ample).
  - `entry_sigma` and `stop_sigma` are sigma multipliers, not bar counts, so
    both are passed as `float` and are correctly ignored by that buffer sizing.

Cost note: `metrics.py`'s ~0.102% round-trip is unchanged and out of scope.

This module decides only *when* the strategy wants to be long or short and at
what levels it wants out. All day-trade gating and the end-of-session flatten
are delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd


from session import apply_session_constraint_with_stops


def regression_channel(close: pd.Series, reg_lookback: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS fitted value at the window's right edge + residual sigma.

    Returns `(line, sigma)` where `line[t]` is the fitted value at bar t of an
    OLS fit of Close on bar index over the `reg_lookback` bars *ending at* t,
    and `sigma[t]` is the standard deviation of that fit's residuals inside
    the same window. Both are NaN until the window is full (fails closed).

    Computed from rolling moments rather than a per-bar `polyfit`: pandas'
    `rolling().cov()/.var()` are numerically stable at NQ-scale price levels
    (a naive mean-of-squares would lose precision on y^2 ~ 4e8), and the OLS
    identity SSresid = SSyy - slope^2 * SSxx gives the residual sigma without
    ever materializing the residuals. `var`/`cov` are both ddof=1 here, so the
    ddof cancels in the slope and `sigma` is sqrt(SSresid / (n - 1)).
    """
    n = int(reg_lookback)
    x = pd.Series(np.arange(len(close), dtype=float), index=close.index)

    mean_x = x.rolling(n).mean()
    mean_y = close.rolling(n).mean()
    var_x = x.rolling(n).var()
    var_y = close.rolling(n).var()
    cov_xy = close.rolling(n).cov(x)

    slope = cov_xy / var_x
    line = mean_y + slope * (x - mean_x)

    # OLS identity; clipped at 0 to absorb floating-point noise on a window
    # whose residuals are (near-)degenerate.
    resid_var = (var_y - slope**2 * var_x).clip(lower=0.0)
    sigma = np.sqrt(resid_var)

    # A perfectly-fit (zero-residual) window carries no information and would
    # divide by zero — mark it unwarmed so the signal fails closed.
    sigma = sigma.where(sigma > 0)

    return line, sigma


def generate_positions(
    df: pd.DataFrame,
    reg_lookback: int,
    entry_sigma: float,
    stop_sigma: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]

    line, sigma = regression_channel(close, reg_lookback)
    z = (close - line) / sigma
    z_prev = z.shift(1)

    # Crossing (not level) so a sustained excursion arms the signal once.
    # NaN on either leg compares False, so unwarmed bars produce no entry.
    short_signal = (z > entry_sigma) & (z_prev <= entry_sigma)
    long_signal = (z < -entry_sigma) & (z_prev >= -entry_sigma)

    # Target: the fitted line itself, captured at the entry bar and held fixed
    # for the trade. Long entries sit below the line and shorts above it, so
    # the level is always on the profitable side of the entry Close.
    target_price = line

    # Stop: a further stop_sigma of residual sigma *away* from the line.
    # apply_session_constraint_with_stops() resolves the direction (below the
    # entry for longs, above for shorts) from the sign of the position.
    stop_distance = float(stop_sigma) * sigma

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
    "reg_lookback": 96,
    "entry_sigma": 2.0,
    "stop_sigma": 2.0,
    "session": "New York",
}
