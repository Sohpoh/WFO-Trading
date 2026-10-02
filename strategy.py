"""Long-only KNN next-session return prediction + ATR hard stop (no target).

Variation type (d) on iteration 60's rejected vol-normalized KNN: the entry
signal, the three grids (`k`, `feature_lookback`, `pred_threshold`),
`norm_lookback` = 1440, symbol, timeframe, session and the 12/3 schedule are
byte-identical to iteration 60, and the ONLY change is the exit — the plain
flip-to-flat `apply_session_constraint` is swapped for a path-dependent ATR
hard stop routed through `apply_session_constraint_with_stops` (mirroring
iteration 51's exact pattern).

Why: iteration 60's rejection isolated a quantified, fixable defect — its worst
trade was -3.95% (~5.1x the mean winner), removing the bottom-5 OOS trades
lifts PF 1.11 -> 1.25, and 6 of the 10 worst trades were 2022 bear-crash
sessions, i.e. the entire loss was a handful of full-session crash longs the
plain flip-to-flat exit let run uncapped to the forced session flatten. The fix
caps only the left tail with a volatility-scaled hard stop (per volatility.md's
"Stop-Loss Placement" rule: stop distance scales with measured volatility,
wide/regime-scaled for momentum); winners stay uncapped to the session flatten
because there is deliberately no profit target (iteration 27 measured a 2R cap
flipping per-trade economics, so only the left is capped).

Rules (all computed on continuous, session-unaware bars; session gating is
delegated to `apply_session_constraint_with_stops`):

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

  - Entry: long_signal True on every bar whose UTC day d has pred_d >
    pred_threshold; short_signal all-False; NaN/unwarmed/zero-std predictions
    fail closed. Long-only — no short leg. pred_d is frozen at the prior UTC
    day's close, so it is constant within a session.

  - Exit: path-dependent ATR hard stop via `apply_session_constraint_with_stops`
    (NOT the plain variant). On a long entry at Close_e the stop is
    entry - stop_atr_mult * ATR(atr_period=14, fixed, not grid-searched),
    detected on the intrabar Low and flattened at that bar's Close (fill-price
    caveat documented in session.py). target_price = +inf everywhere so the
    target side never binds and winners stay uncapped to the forced session
    flatten. Because pred_d is frozen constant within a session, long_signal
    re-arms after a stop exit, so the walk re-enters on the next in-session bar
    — the stop chunks a continuous crash into repeated stop-sized realized
    losses rather than one -3.95% hold (accepted consequence, pre-registered).

Warm-up: the earliest valid prediction day needs k valid neighbors, each with
its own warmed feature/sigma, so warm-up is bounded by norm_days (15) + k
(at most 15) = ~30 days. `norm_lookback` (1440) and `feature_lookback` (grid
{288,480,768}) are genuine bar-count lookbacks and are threaded as plain `int`s
on purpose, so `_max_lookback_bars()` returns 1440 and sizes the test buffer to
~45 days — far above the ATR(14) window (14 bars) and the ~30-day KNN warm-up.
`k` (grid {5,10,15}) is a neighbor count, NOT a lookback, so it is threaded as
a `float` and must not inflate the warm-up buffer. `stop_atr_mult` (grid
{1.0,1.5,2.0,3.0}) is an ATR multiplier, NOT a lookback, so it is threaded as a
`float`. `pred_threshold` is a return threshold (float). `atr_period` (fixed
14) is a genuine bar-count lookback (plain `int`), threaded through build_grid()
as a fixed int so it feeds the warm-up buffer (subsumed by norm_lookback=1440).

Contract notes: `generate_positions` builds session-unaware raw signals and
delegates all session gating, the stop/target bookkeeping, and the
end-of-session flatten to `apply_session_constraint_with_stops`. The delegate
returns one scalar {-1,0,1} per bar, so repeat signals while already long are
no-ops (it never re-opens on the same bar it exits, and it holds a single
position), preserving the single-position contract. Degenerate features
(NaN/zero std, unwarmed windows) fail closed (no entry). Costs (`metrics.py`,
cost model v2: 0.001% fee + 0.005% slippage per leg) are out of scope for this
file.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Fixed volatility-normalization window, in BARS. 1440 bars = 15 days at the
# 15min timeframe (96 bars/day). A genuine bar-count lookback, so it is a plain
# int and is threaded through build_grid() as a fixed int on purpose: that makes
# _max_lookback_bars() return 1440 and size the test-window warm-up buffer to
# ~45 days. Never grid-searched.
NORM_LOOKBACK_BARS = 1440

# Fixed ATR window, in BARS, for the hard-stop distance. A genuine bar-count
# lookback (the ATR smoothing span), so it is a plain int and is threaded
# through build_grid() as a fixed int on purpose (subsumed by norm_lookback=1440
# when sizing the warm-up buffer). Never grid-searched.
ATR_PERIOD = 14


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


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range over `period` bars, index-aligned to `close`.

    True range uses the prior bar's close as the reference so session/overnight
    gaps count. The Wilder average is a recursive EMA with alpha = 1/period
    (adjust=False) so it carries state rather than using a fixed trailing
    window. The first `period` bars are NaN (the recursion hasn't seen a full
    window yet), so a stop distance built from them fails closed and cannot
    open a position.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(period), adjust=False).mean()
    atr.iloc[:period] = np.nan
    return atr


def generate_positions(
    df: pd.DataFrame,
    k: float,
    feature_lookback: int,
    pred_threshold: float,
    stop_atr_mult: float,
    session: str | None = "New York",
    norm_lookback: int = NORM_LOOKBACK_BARS,
    atr_period: int = ATR_PERIOD,
) -> pd.Series:
    """Build the long-only KNN-prediction position series with an ATR hard stop.

    The entry is byte-identical to iteration 60: per-UTC-day KNN next-session
    return prediction `pred_d`, long where pred_d > pred_threshold, never short.
    The exit is a path-dependent ATR hard stop with no profit target, routed
    through `session.apply_session_constraint_with_stops`: `stop_atr_mult`
    scales the stop distance off `ATR(atr_period)`, and the target is an
    unreachable +inf constant so winners run uncapped to the session flatten.
    Returns a {-1, 0, 1} position series.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    pred = _daily_knn_prediction(df, feature_lookback, norm_lookback, k)

    # Map each bar to its UTC day's prediction (frozen at the prior UTC day's
    # close), then emit a long signal on every bar whose day's prediction clears
    # the threshold. pred_d is constant within a UTC day and therefore within a
    # session. There is no short branch, so short_signal is all-False. A NaN
    # prediction (unwarmed / degenerate) compares False, so that bar fails
    # closed.
    bar_days = df.index.normalize()
    pred_by_bar = pd.Series(bar_days, index=df.index).map(pred)
    long_signal = pred_by_bar.gt(pred_threshold)
    short_signal = pd.Series(False, index=df.index, dtype=bool)

    # ATR hard stop distance, measured from the entry Close. stop_atr_mult is a
    # float multiplier (NOT a lookback); atr_period is the bar-count lookback.
    atr = atr_series(high, low, close, atr_period)
    stop_distance = float(stop_atr_mult) * atr

    # No profit target: an unreachable +inf constant so the target side of the
    # stop/target walk never binds, while remaining a valid entry gate for a
    # long (inf > close is always true). Winners run uncapped to session.py's
    # forced flatten.
    target_price = pd.Series(np.inf, index=df.index)

    # session.py alone decides which bars are tradable, walks the stop/target
    # bookkeeping bar-by-bar, and force-flattens on the session's last bar.
    return apply_session_constraint_with_stops(
        close,
        high,
        low,
        long_signal,
        short_signal,
        stop_distance,
        target_price,
        session,
    )


DEFAULT_PARAMS = {
    # k is a neighbor count, NOT a bar-count lookback: threaded as a float so a
    # huge grid value would not inflate _max_lookback_bars(). Grid-searched
    # {5,10,15}. feature_lookback is a genuine bar-count lookback (the H-feature
    # window's bar length; grid {288,480,768} = 3/5/8 days at 15min) and is a
    # plain int on purpose so it feeds the warm-up buffer. pred_threshold is a
    # daily-return threshold (float; grid {0.0,0.0005,0.001}). stop_atr_mult is
    # an ATR multiplier for the hard-stop distance (float, NOT a lookback; grid
    # {1.0,1.5,2.0,3.0}). norm_lookback (1440 bars = 15 days) and atr_period
    # (14 bars) are fixed plain-int lookbacks — never grid-searched but threaded
    # through build_grid() so _max_lookback_bars() returns 1440. `session` is a
    # fixed param. These are the concrete set the sanity checker runs
    # generate_positions() against.
    "k": 10.0,
    "feature_lookback": 480,
    "pred_threshold": 0.0,
    "stop_atr_mult": 2.0,
    "norm_lookback": NORM_LOOKBACK_BARS,
    "atr_period": ATR_PERIOD,
    "session": "New York",
}
