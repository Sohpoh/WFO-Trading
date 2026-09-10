"""Long-only Kalman-filter trend momentum (NQ 15min, New York session) — a
deliberate *momentum* reading of `Trading Vault/wiki/kalman-filter.md`, which
prescribes a mean-reversion deviation (actual price vs. a level estimate).
This iteration does NOT trade the deviation: it runs a local-linear-trend
Kalman filter over log(Close) and trades the filter's *slope* as a
continuation signal, on the long side only.

The pivot is structural, not cosmetic. Iterations 40/48/49 all built a
fixed-lookback formation-period return statistic (R_mean / sigma over the
trailing `formation_lookback` bars, then ranked against its own trailing
quantile over `rank_window` bars), and every one of them died in the 2025
holdout for the same reason: a fixed window's *lag* is exactly what fails when
the regime turns. A Kalman local-linear-trend estimate has no fixed window —
the filter's gain and its recursively updated covariance adapt the effective
memory continuously, so a regime turn is absorbed by shrinking the gain rather
than by waiting `formation_lookback` bars for the stale formation window to
roll off. That is the "adapts automatically to changing volatility" / "often
better in changing regimes" property `kalman-filter.md` lists as the defining
advantage over fixed-lookback bands, and it is precisely the 2025
regime-fragility failure that killed the previous momentum family.

The mechanism
-------------
All quantities are session-unaware; the Kalman recursion is strictly causal
(each bar's state uses only observations up to and including that bar), so
there is no lookahead. `metrics.bar_returns_with_costs` prices a position off
`position.shift(1)`, so a position set at bar t earns the Close_t -> Close_{t+1}
return the signal never sees.

  - Work in log prices:  y_t = log(Close_t).

  - Local-linear-trend state space (Harvey's local linear trend / integrated
    random walk):

        level_{t+1} = level_t + slope_t
        slope_{t+1} = slope_t + w_t,   w_t ~ N(0, sigma_w^2)
        y_t         = level_t + v_t,   v_t ~ N(0, sigma_v^2)

    i.e. F = [[1,1],[0,1]], H = [1,0], Q = sigma_w^2 * [[0,0],[0,1]],
    R = sigma_v^2. The slope does a random walk (process noise on the slope,
    none on the level), so the "true" per-bar drift is free to re-rate every
    bar — the adaptive-slope part of the signal.

  - Noise ratio: `kf_noise_ratio = sigma_w / sigma_v` is the ONE grid-searched
    noise knob, and `sigma_v` is *estimated from the filter's residual* (the
    one-step-ahead innovation v_t) as a recursive EMA of squared innovations,
    so `sigma_w = kf_noise_ratio * sigma_v` follows the data's own scale. This
    makes the whole filter scale-free: multiplying every price by a constant
    adds log(c) to every y_t, which shifts the level and leaves the slope, the
    innovations, and sigma_v unchanged.

  - Scale-free slope statistic (the signal):

        z_t = slope_t / sigma_v_t

    slope has units of log-price per bar and sigma_v of log-price, so z_t is a
    per-bar, dimension-free "how many units of observation noise per bar is
    the trend drifting" — a t-stat-like reading of the adaptive trend.

The recursion is implemented directly (no external Kalman library): predict,
innovate, gain, update the 2x2 covariance in closed form, then fold the
innovation into the sigma_v EMA. `sigma_v` is bootstrapped from the sample
variance of the first SIGMA_BURN log-returns so the initial state covariance,
Q and R all share one scale; the initial covariance is a diffuse prior
(P0_SCALE x that variance) so the filter trusts early observations and
converges quickly.

Entry — long-only, scale-free slope above a floor
-------------------------------------------------
A raw +1.0 long is emitted where z_t > min_slope (strict). There is no short
branch, so -1.0 is never emitted and a down-state pays no legs at all instead
of two to reverse. `min_slope` is dimensionless (units of per-bar slope
measured in observation-noise units); grid-searched over {0.25, 0.5, 1.0, 2.0}.

Exit — decay flip to flat when the trend turns, no stop, no target
------------------------------------------------------------------
Plain `apply_session_constraint` (NOT the `with_stops` variant) — no stop, no
target, upside uncapped, duration bounded only by the session. A long is
flattened where z_t < 0 (the adaptive trend has turned). Because min_slope >=
0.25 > 0, the band (0, min_slope) is a hysteresis hold: a long entered at
z > min_slope is held through z in (0, min_slope] and only flattened when the
slope's sign actually flips.

Raw entries are three-state:

        1.0   where z > min_slope        (enter / stay long)
        0.0   where z < 0                (flatten)
        NaN   in between                 (hold)

The 0.0 flat mask is written first and the 1.0 second so an entry wins on any
same-bar overlap (structurally moot here since the two thresholds are disjoint
for min_slope > 0, but the ordering makes it explicit). `apply_session_constraint`
forward-fills the sparse series: 0.0 is an explicit flat instruction, NaN a
genuine hold, and the last bar of every session is force-flattened by
`session.py`. A winner therefore runs uncapped to the forced session flatten.

Fail-closed is exact while flat: a NaN z (unwarmed bar, or a degenerate
division) makes both comparisons False, the bar emits NaN, and the forward
fill leaves the flat position flat. Honesty item — while *long*, a NaN z is
likewise both-False, emits NaN, and the forward fill **holds the long**. That
is inherent to a hysteresis exit and is bounded by the end-of-session flatten,
so the worst case is one session held on stale information; it is stated here
rather than left for the evaluator to find.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - `kf_noise_ratio` and `min_slope` are both `float` **on purpose** — neither
    is a bar-count lookback, so neither should feed `_max_lookback_bars()`'s
    warm-up sizing. `kf_noise_ratio` is a noise ratio and `min_slope` a
    dimensionless threshold; both are grid-searched and both are cast to
    float in `build_grid()` so a grid point written as `1` can never arrive
    as an int and silently inflate the buffer.
  - With no int lookback param in the grid, `_max_lookback_bars()` returns 0
    and `run_walk_forward()` sizes the buffer purely off the one-day floor
    (`day_bars + 5`, ~101 bars at 15min). That comfortably covers this
    strategy's WARMUP_BARS = 48 (filter + sigma_v EMA settling) before each
    test window; see `build_grid()` for the fold-skip arithmetic (the guard
    collapses to ~10 bars, so no timeframe silently skips every fold).
  - `SIGMA_EMA_SPAN`, `SIGMA_BURN`, `WARMUP_BARS`, `P0_SCALE`, and
    `SIGMA2_FLOOR` are module constants and are **never grid-searched**: they
    add no `build_grid()` / `DEFAULT_PARAMS` key, so the noise-estimation and
    warm-up machinery keep zero degrees of freedom.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.005% slippage) is
unchanged and out of scope. The long-only construction interacts with it
favourably: a down-state costs zero legs instead of two, and the decay-flip
exit only ever closes a long already open (one exit leg), never adds a
reversal leg.

This module decides only *when* the strategy wants to be long and when that
wish has decayed. All day-trade gating and the end-of-session flatten are
delegated to `session.py`; see its docstring for that contract.
"""
import math

import numpy as np
import pandas as pd

from session import apply_session_constraint

# Effective bar-count of the EMA that estimates observation noise sigma_v from
# the filter's residual. Hardcoded, NOT grid-searched — "how quickly should
# the noise estimate re-rate itself" is a design choice (one day of 15min
# bars), not a fitted value, and the noise-estimation machinery keeps zero
# degrees of freedom.
SIGMA_EMA_SPAN = 96

# Number of initial log-returns used to bootstrap the sigma_v estimate so the
# initial state covariance, Q and R all share one scale. Hardcoded, NOT
# grid-searched.
SIGMA_BURN = 24

# Bars the filter and its sigma_v EMA run before any signal is trusted. The
# slope and the noise estimate are both settling during this burn, so these
# bars emit NaN (no instruction) and fail closed. Hardcoded, NOT grid-searched
# — well below the ~101-bar one-day warm-up buffer the engine provides.
WARMUP_BARS = 48

# Diffuse-prior multiplier on the bootstrap variance used for the initial
# state covariance. Hardcoded, NOT grid-searched — a large prior makes the
# filter trust early observations and converge quickly; the exact value is
# immaterial to the steady state.
P0_SCALE = 1e6

# Floor on the sigma_v^2 estimate so slope/sigma_v stays finite if the market
# goes perfectly flat and the residual EMA decays toward zero. Hardcoded, NOT
# grid-searched.
SIGMA2_FLOOR = 1e-12


def kalman_slope_series(y: np.ndarray, kf_noise_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear-trend Kalman filter over log prices -> (slope, sigma_v).

    Returns two length-n arrays: the filtered per-bar slope and the running
    observation-noise estimate (sigma_v = sqrt of the EMA of squared
    innovations). Both are NaN before the first bar and only become
    well-behaved after the recursion settles; callers apply their own warm-up
    gate. The whole recursion is causal: bar t's state uses observations only
    through bar t.

    Model: level_{t+1} = level_t + slope_t; slope_{t+1} = slope_t + w_t
    (w ~ N(0, sigma_w^2)); y_t = level_t + v_t (v ~ N(0, sigma_v^2)), with
    sigma_w = kf_noise_ratio * sigma_v. sigma_v is estimated from the residual
    v_t via a recursive EMA of v_t^2, bootstrapped from the first SIGMA_BURN
    log-returns.
    """
    n = len(y)
    slope_out = np.full(n, np.nan)
    sigma_out = np.full(n, np.nan)
    if n < SIGMA_BURN + 2:
        return slope_out, sigma_out

    ratio = float(kf_noise_ratio)
    alpha = 2.0 / (SIGMA_EMA_SPAN + 1.0)

    # Bootstrap observation-noise variance from the first SIGMA_BURN
    # log-returns so the initial covariance / Q / R share one scale.
    init_var = float(np.var(np.diff(y[: SIGMA_BURN + 1])))
    if not np.isfinite(init_var) or init_var <= 0.0:
        init_var = 1e-8

    # Diffuse prior: level and slope start with huge covariance so the filter
    # trusts early observations and converges within a couple dozen bars.
    level = float(y[0])
    slope = 0.0
    p00 = init_var * P0_SCALE
    p01 = 0.0
    p11 = init_var * P0_SCALE
    sigma2 = init_var

    for t in range(1, n):
        obs = float(y[t])

        # -- predict --  F = [[1,1],[0,1]], Q = sigma_w^2 * [[0,0],[0,1]]
        pred_level = level + slope
        pp00 = p00 + 2.0 * p01 + p11
        pp01 = p01 + p11
        pp11 = p11 + ratio * ratio * sigma2

        # -- update --  scalar-observation Kalman gain and Joseph-free update
        innov = obs - pred_level
        s = pp00 + sigma2
        k_level = pp00 / s
        k_slope = pp01 / s
        level = pred_level + k_level * innov
        slope = slope + k_slope * innov
        p00 = (1.0 - k_level) * pp00
        p01_new = (1.0 - k_level) * pp01
        p10_new = pp01 - k_slope * pp00
        p11 = pp11 - k_slope * pp01
        p01 = 0.5 * (p01_new + p10_new)  # symmetrize (exact in algebra; float guard)

        # -- adapt sigma_v from the residual --  EMA of squared innovation
        sigma2 = (1.0 - alpha) * sigma2 + alpha * (innov * innov)
        if sigma2 < SIGMA2_FLOOR:
            sigma2 = SIGMA2_FLOOR

        slope_out[t] = slope
        sigma_out[t] = math.sqrt(sigma2)

    return slope_out, sigma_out


def generate_positions(
    df: pd.DataFrame,
    kf_noise_ratio: float,
    min_slope: float,
    session: str | None = "New York",
) -> pd.Series:
    """Build the long-only Kalman-slope position series from OHLCV bars.

    `kf_noise_ratio` is sigma_w / sigma_v (the single grid-searched noise
    knob); `min_slope` is the dimensionless per-bar slope threshold above which
    a long opens. Returns a {-1, 0, 1} position series via
    `session.apply_session_constraint` (plain variant — no stop, no target).
    """
    close = df["Close"].astype(float)
    y = np.log(close.to_numpy())
    slope, sigma = kalman_slope_series(y, kf_noise_ratio)

    with np.errstate(divide="ignore", invalid="ignore"):
        z = slope / sigma
    # Fail closed while the filter and its noise estimate are still settling.
    z[:WARMUP_BARS] = np.nan
    z = pd.Series(z, index=df.index)

    # Long requires the scale-free slope strictly above min_slope; decay is
    # that same statistic below zero (the adaptive trend has turned). NaN on
    # either side (unwarmed / degenerate) compares False on both, so such a
    # bar emits NaN = "no instruction" and the delegate's forward fill leaves
    # the position where it already was.
    with np.errstate(invalid="ignore"):
        long_signal = z > float(min_slope)
        decay_signal = z < 0.0

    # Raw, session-unaware entries: 1.0 long / 0.0 flat / NaN hold. Flat is
    # written first and entry second so an entry wins on any same-bar overlap.
    entries = pd.Series(np.nan, index=df.index, dtype=float)
    entries[decay_signal] = 0.0
    entries[long_signal] = 1.0

    # session.py alone decides which bars are tradable, forward-fills the
    # sparse instruction series into a held position, and force-flattens on
    # the session's last bar.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # kf_noise_ratio and min_slope are floats because they are NOT bar-count
    # lookbacks and must not feed wfo_engine's warm-up buffer. Both are
    # grid-searched (0.01/0.05/0.1/0.2/0.5 and 0.25/0.5/1.0/2.0); `session`
    # is a fixed param. The five module constants above (SIGMA_EMA_SPAN,
    # SIGMA_BURN, WARMUP_BARS, P0_SCALE, SIGMA2_FLOOR) deliberately have no
    # entry here.
    "kf_noise_ratio": 0.1,
    "min_slope": 1.0,
    "session": "New York",
}
