"""Long-only KNN next-session return prediction (bar-scale warm-up fix).

Iteration 59 (the vol-normalized KNN from this same family) errored before any
trading verdict: its 60/125/250-day `norm_lookback` plus a ~40k-day neighbor
pool cannot warm inside a 12-week (~60-day) train slice, and because `k`,
`pred_threshold` and `norm_lookback` were all threaded as floats,
`wfo_engine._max_lookback_bars()` returned 0 so the test-window warm-up buffer
collapsed to ~1 day (every fold would report 0 OOS trades). This retry keeps
the same family and applies the error's own prescribed fix: re-scale the KNN
from day-scale to bar-scale so it warms within 12/3, rather than widening the
schedule.

The feature vector is the trailing H in {3,5,8} daily close-to-close returns,
each divided by a fixed 15-day rolling std; `k` is capped at {5,10,15}; and
`norm_lookback` / `feature_lookback` are threaded as int bar counts so
`_max_lookback_bars()` returns 1440 and sizes the test buffer to ~45 days.
Long-only + hold-to-flatten keeps iteration 36's only clean robust:true
geometry and iteration 33's short-leg-worse finding.

Rules (all computed on continuous, session-unaware bars; session gating is
delegated to `apply_session_constraint`):

  - Daily returns: the last Close of each UTC day vs the prior UTC day's last
    Close (simple arithmetic `pct_change` — pred_threshold is quoted in the
    same units, e.g. 0.0005 = 5bp/day).

  - Feature vector for day d:
        f_d = [r_{d-1}, r_{d-2}, ..., r_{d-H}] / sigma_d
    where sigma_d is the std of the prior `norm_days` daily returns ending at
    d-1 (no lookahead), and H = feature_lookback // bars_per_day (bars-per-day
    derived from the df's UTC-day bar counts, matching wfo_engine._bars_per_day's
    median convention — strategy.py can't import engine internals, so it derives
    its own equivalent).

  - Prediction: Euclidean distance from f_d to every prior day j <= d-2 (j's
    feature is fully determined by data <= j, and its next-day return r_{j+1}
    needs j+1 <= d-1). pred_d = mean of the k nearest neighbors' next-day
    returns r_{j+1}. Fewer than k valid neighbors => no prediction (fails
    closed).

  - Entry: raw entries = 1.0 on bars whose UTC day d has pred_d >
    pred_threshold; NaN elsewhere. pred_d is frozen at the prior UTC day's
    close, so it is constant within a session and the delegate opens exactly
    one round trip per active session. Long-only — no short leg.

  - Exit: plain flip-to-flat via `apply_session_constraint`. Position is 1
    while in-session and pred_d > pred_threshold, 0 otherwise; the session
    boundary force-flattens, and a non-positive prediction simply means the
    next session does not re-enter. Winners run uncapped to the flatten.

Warm-up: the earliest valid prediction day needs k valid neighbors, each with
its own warmed feature/sigma, so warm-up is bounded by norm_days (15) + k
(at most 15) = ~30 days — roughly half the 60-day train slice, leaving ~30
tradable sessions per fold in-sample (and the engine's ~45-day test buffer
covers the same warm-up out-of-sample). `norm_lookback` (1440) and
`feature_lookback` (grid {288,480,768}) are genuine bar-count lookbacks and are
threaded as plain `int`s on purpose, so `_max_lookback_bars()` returns 1440 and
sizes the test buffer to ~45 days. `k` (grid {5,10,15}) is a neighbor count,
NOT a lookback, so it is threaded as a `float` and must not inflate the warm-up
buffer. `pred_threshold` is a return threshold (float).

Contract notes: `generate_positions` builds a session-unaware raw `entries`
series (1.0 at long-entry bars, NaN elsewhere) and delegates all session gating
and the end-of-session flatten to `apply_session_constraint`. Repeat signals
while already long are no-ops (the delegate forward-fills a single scalar
position), preserving the {-1,0,1} single-position contract. Degenerate
features (NaN/zero std, unwarmed windows) fail closed (no entry). Costs
(`metrics.py`, cost model v2: 0.001% fee + 0.005% slippage per leg) are out of
scope for this file.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Fixed volatility-normalization window, in BARS. 1440 bars = 15 days at the
# 15min timeframe (96 bars/day). A genuine bar-count lookback, so it is a plain
# int and is threaded through build_grid() as a fixed int on purpose: that makes
# _max_lookback_bars() return 1440 and size the test-window warm-up buffer to
# ~45 days. Never grid-searched.
NORM_LOOKBACK_BARS = 1440


def _bars_per_day(df: pd.DataFrame) -> int:
    """Median bar count per UTC calendar day, matching wfo_engine._bars_per_day.

    strategy.py can't import engine internals, so it re-derives the same
    convention (median of the per-UTC-day bar counts) to convert the bar-count
    lookbacks (feature_lookback / norm_lookback) into the day-scale horizons the
    KNN actually consumes.
    """
    if df.empty:
        return 0
    counts = df.groupby(df.index.normalize()).size()
    return int(counts.median()) if len(counts) else 0


def _daily_knn_prediction(
    df: pd.DataFrame, feature_lookback: int, norm_lookback: int, k: int
) -> pd.Series:
    """Per-UTC-day next-session return prediction (NaN where unwarmed).

    Returns a Series indexed by UTC-day (midnight) timestamps; its value for
    day d is the mean next-day return of day d's k nearest neighbors in
    normalized daily-return feature space. NaN means "no valid prediction"
    (insufficient warm-up, a zero/non-finite std, or fewer than k neighbors);
    the caller treats NaN as no entry.
    """
    k = int(k)
    bars_per_day = _bars_per_day(df)
    if bars_per_day <= 0:
        bars_per_day = 1
    H = max(1, int(feature_lookback) // bars_per_day)
    norm_days = max(1, int(norm_lookback) // bars_per_day)

    # UTC-day close-to-close returns: last Close of each UTC day vs the prior
    # UTC day's last Close. daily_ret[m] = r_m; the first value is NaN.
    daily_close = df["Close"].groupby(df.index.normalize()).last()
    daily_ret = daily_close.pct_change()
    ret = daily_ret.to_numpy(dtype=float)
    n = len(ret)

    # sigma[i] = std of the norm_days returns ending at position i (over
    # [i-norm_days+1, i]). For a day-m feature/query the spec wants the std of
    # returns ending at day m-1, so it reads sigma[m-1] below.
    sigma = daily_ret.rolling(norm_days, min_periods=norm_days).std().to_numpy(dtype=float)

    # Feature matrix F[m] = [r_{m-1}, ..., r_{m-H}] / sigma_m (oldest..newest;
    # any consistent order works for Euclidean distance). Target T[m] = r_{m+1},
    # the label day m contributes when chosen as a neighbor. Rows whose warm-up
    # is incomplete (a leading-NaN return in the window, a NaN/zero std, or an
    # H-window that reaches back past the first return) stay NaN and are
    # ineligible as query or neighbor.
    F = np.full((n, H), np.nan)
    T = np.full(n, np.nan)
    for m in range(n):
        start = m - H
        if start < 1:  # excludes the leading-NaN r_0
            continue
        s = sigma[m - 1]
        if not np.isfinite(s) or s <= 0:
            continue
        window = ret[start:m]
        if np.isnan(window).any():
            continue
        F[m] = window / s
    T[:-1] = ret[1:]

    neighbor_ok = ~np.isnan(F).any(axis=1) & ~np.isnan(T)

    pred = np.full(n, np.nan)
    for d in range(n):
        if d - 2 < 0 or np.isnan(F[d]).any():
            continue
        cand = np.arange(0, d - 1)  # j in [0, d-2]; d-1 excluded (its label r_d is lookahead)
        ok = neighbor_ok[cand]
        if int(ok.sum()) < k:
            continue
        diff = F[cand][ok] - F[d]
        dist = np.sqrt((diff * diff).sum(axis=1))
        order = np.argsort(dist, kind="stable")[:k]
        pred[d] = T[cand][ok][order].mean()

    return pd.Series(pred, index=daily_ret.index)


def generate_positions(
    df: pd.DataFrame,
    k: float,
    feature_lookback: int,
    pred_threshold: float,
    session: str | None = "New York",
    norm_lookback: int = NORM_LOOKBACK_BARS,
) -> pd.Series:
    pred = _daily_knn_prediction(df, feature_lookback, norm_lookback, k)

    # Map each bar to its UTC day's prediction (frozen at the prior UTC day's
    # close), then emit a long entry on every bar whose day's prediction clears
    # the threshold. pred_d is constant within a UTC day and therefore within a
    # session, so the delegate opens exactly one round trip per active session.
    bar_days = df.index.normalize()
    pred_by_bar = pd.Series(bar_days, index=df.index).map(pred)

    signal = pred_by_bar.gt(pred_threshold)
    entries = signal.astype(float).where(signal)

    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # k is a neighbor count, NOT a bar-count lookback: threaded as a float so a
    # huge grid value would not inflate _max_lookback_bars(). Grid-searched
    # {5,10,15}. feature_lookback is a genuine bar-count lookback (the H-feature
    # window's bar length; grid {288,480,768} = 3/5/8 days at 15min) and is a
    # plain int on purpose so it feeds the warm-up buffer. pred_threshold is a
    # daily-return threshold (float; grid {0.0,0.0005,0.001}). norm_lookback is
    # a fixed plain-int lookback (1440 bars = 15 days) — never grid-searched but
    # threaded through build_grid() so _max_lookback_bars() returns 1440.
    # `session` is a fixed param. These are the concrete set the sanity checker
    # runs generate_positions() against.
    "k": 10.0,
    "feature_lookback": 480,
    "pred_threshold": 0.0,
    "norm_lookback": NORM_LOOKBACK_BARS,
    "session": "New York",
}
