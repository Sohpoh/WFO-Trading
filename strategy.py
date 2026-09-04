"""Long-only risk-adjusted momentum rank (NQ 15min, New York session).

A new family that carries forward the repo's only clean robust:true
construction — iteration 36's long-only, self-normalizing trailing-quantile
momentum rank (slow formation grid 96-384, rank_window 960, decay-flip exit,
no stop/target) — and changes exactly one thing: the statistic that gets
ranked. Iteration 36 ranked the raw formation-period cumulative return
`Close_t / Close_{t-formation_lookback} - 1` (vault eq. 267). This ranks the
vault's documented volatility-normalized return `R_mean / σ` (eq. 269)
instead, over the same trailing `formation_lookback` window.

Why: the loop's named primary open problem is regime fragility. A raw-return
rank means "top-quantile momentum" only relative to the volatility regime it
was calibrated in — iteration 40's raw-return rank collapsed exactly when the
2025 vol regime differed from its train regime. Dividing the formation-period
mean 1-bar return by its own sample σ makes "top-quantile momentum" mean the
same thing across the 2022 high-vol bear, the 2024 low-vol grind, and 2025.
It is distinct from the failed `intraday_tsmom_risk_adjusted` family (iters
7/20) on three axes: slow 96-384 horizon instead of 24-192, percentile-rank
self-normalization instead of an absolute t-stat threshold, and long-only
instead of symmetric.

The mechanism
-------------
All quantities are session-unaware; every window is strictly backward-looking,
so there is no lookahead. `metrics.bar_returns_with_costs` prices a position
off `position.shift(1)`, so a position set at bar t earns the
Close_t -> Close_{t+1} return the signal never sees.

  - One-bar simple return (NOT log):

        r_t = Close_t / Close_{t-1} - 1

  - Formation-period mean and sample volatility over the trailing L bars:

        R_mean_t = r.rolling(L, min_periods=L).mean()
        σ_t      = r.rolling(L, min_periods=L).std()      # ddof=1, vault eq. 270

  - Risk-adjusted return (vault eq. 269):

        risk_adj_t = R_mean_t / σ_t

    σ_t <= 0 (flat-vol bars) or non-finite σ_t (unwarmed bars) sets risk_adj
    to NaN, so those bars fail closed. σ is a sample std over L >= 96
    observations, so ddof=1 is never degenerate.

  - Self-referential entry threshold — the trailing `rank_pct` quantile of the
    same statistic, current bar excluded from its own distribution:

        enter_threshold_t = risk_adj.shift(1)
                                 .rolling(rank_window, min_periods=rank_window)
                                 .quantile(rank_pct)

    The strict `min_periods=rank_window` NaNs out every unwarmed bar, and the
    `.shift(1)` excludes the current bar. `rolling(...).quantile(...)` is used
    rather than `.apply(...)` on purpose: orders of magnitude faster across a
    12-combo grid x ~48 folds.

Entry — a dense long-only state
-------------------------------
A raw +1.0 long is emitted wherever `risk_adj_t > enter_threshold_t`, strict
`>` (the top (1-rank_pct) quantile of the trailing distribution). Long-only:
there is no short branch, so -1.0 is never emitted and a down-state simply
pays no legs at all instead of paying two to reverse.

Exit — decay-flip to flat, no stop, no target
---------------------------------------------
Plain `apply_session_constraint` (NOT the `with_stops` variant) — no stop, no
target, upside uncapped, duration bounded by the session. Decay is a second
hardcoded threshold on the same statistic:

        exit_threshold_t = risk_adj.shift(1)
                                 .rolling(rank_window, min_periods=rank_window)
                                 .quantile(EXIT_PCT)        # EXIT_PCT = 0.5

Raw entries are three-state:

        1.0   where risk_adj_t > enter_threshold_t   (enter / stay long)
        0.0   where risk_adj_t < exit_threshold_t    (decayed -> go flat)
        NaN   in between                             (hysteresis band -> hold)

Exit is written first and entry second, so an entry wins on any overlap; at
`rank_pct >= 0.80` against `EXIT_PCT = 0.5` the entry quantile is strictly
above the exit quantile on any non-degenerate distribution, so the two masks
are disjoint anyway, but the ordering makes that structural.
`apply_session_constraint` forward-fills the sparse series: 0.0 is an explicit
flat instruction, NaN a genuine hold, and the last bar of every session is
force-flattened by `session.py`. A long is therefore held through ordinary
noise (the entry-to-median band) and flattened on the first below-median bar;
a still-trending winner runs uncapped to the forced session flatten.

Fail-closed is exact while flat: an unwarmed/flat-vol bar leaves both
thresholds NaN, both comparisons False, the bar emits NaN, and the forward
fill leaves the flat position flat. Honesty item — while *long*, a NaN
`risk_adj` (or NaN thresholds) is likewise both-False, emits NaN, and the
forward fill therefore **holds the long**. That is inherent to a hysteresis
exit and is bounded by the end-of-session flatten, so the worst case is one
session held on stale information; it is stated here rather than left for the
evaluator to find.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - `formation_lookback` and `rank_window` are plain `int` **on purpose** —
    both are genuine bar counts and both are exactly what should size the
    pre-test-window warm-up buffer.
  - `rank_window` is fixed at 960, never grid-searched, and is the larger of
    the two: it drives `buffer_bars = max((960 + 5) * 3, day_bars + 5)` =
    2,895 bars, comfortably covering the true requirement of
    `rank_window + max(formation_lookback)` = 960 + 384 = 1,344 at the top of
    the slow-end formation grid 96/192/288/384. Do not raise it without
    re-checking that arithmetic *and* the fold-skip guard — see build_grid().
  - `rank_pct` is a quantile level in (0, 1), never a bar count, so it is
    passed as `float` and correctly ignored by the buffer sizing.
  - `EXIT_PCT` is a module constant and is **never grid-searched**: it adds no
    `build_grid()` / `DEFAULT_PARAMS` key, so the exit keeps zero degrees of
    freedom.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.005% slippage) is
unchanged and out of scope. The long-only construction still interacts with
the cost model favourably: a down-state costs zero legs instead of two.

This module decides only *when* the strategy wants to be long and when that
wish has decayed. All day-trade gating and the end-of-session flatten are
delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Quantile of the same trailing risk-adjusted-return distribution at which an
# open long is considered to have decayed and is flattened. Hardcoded, NOT
# grid-searched — "the instrument has dropped out of the upper half of its own
# risk-adjusted momentum distribution" is an economic reading, not a fitted
# value, and the exit keeps zero degrees of freedom.
EXIT_PCT = 0.5


def risk_adjusted_return(close: pd.Series, formation_lookback: int) -> pd.Series:
    """Volatility-normalized formation-period return (vault eq. 269).

    Computes the 1-bar simple return r_t = Close_t/Close_{t-1} - 1, then the
    mean and sample std of r over the trailing `formation_lookback` bars, and
    returns mean / std. Bars where std <= 0 (flat-vol) or std is non-finite
    (unwarmed) are NaN, so they fail closed downstream. std is a sample std
    (ddof=1) over L >= 96 observations, never degenerate.
    """
    n = int(formation_lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = close / close.shift(1) - 1.0

    mean = r.rolling(n, min_periods=n).mean()
    sigma = r.rolling(n, min_periods=n).std()

    with np.errstate(divide="ignore", invalid="ignore"):
        risk_adj = mean / sigma

    # sigma <= 0 (flat-vol) or non-finite sigma (unwarmed) -> NaN, so those
    # bars carry no instruction. The replace also drops any non-finite ratio
    # (mean/sigma overflow) so only genuine finite values reach the rank.
    return risk_adj.where(np.isfinite(sigma) & (sigma > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )


def trailing_rank_threshold(stat: pd.Series, rank_window: int, pct: float) -> pd.Series:
    """`pct` quantile of `stat` over the trailing `rank_window` bars, excluding
    the current bar from its own threshold.

    Strictly warm (`min_periods == rank_window`), so unwarmed bars are NaN and
    fail closed downstream. Called twice per run — once at `rank_pct` for the
    entry level, once at `EXIT_PCT` for the decay level — off the same series.
    """
    w = int(rank_window)
    return stat.shift(1).rolling(w, min_periods=w).quantile(float(pct))


def generate_positions(
    df: pd.DataFrame,
    formation_lookback: int,
    rank_pct: float,
    rank_window: int = 960,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)

    stat = risk_adjusted_return(close, formation_lookback)
    enter_threshold = trailing_rank_threshold(stat, rank_window, rank_pct)
    exit_threshold = trailing_rank_threshold(stat, rank_window, EXIT_PCT)

    # Top-quantile risk-adjusted return -> long; the same statistic below its
    # own trailing median -> momentum has decayed, go flat. Strict comparisons;
    # NaN on either side (unwarmed formation window or unwarmed rank window)
    # compares False on both, so such a bar emits NaN = "no instruction" and
    # the delegate's forward fill leaves the position where it already was.
    with np.errstate(invalid="ignore"):
        long_signal = stat > enter_threshold
        decay_signal = stat < exit_threshold

    # Raw, session-unaware entries: 1.0 long / 0.0 flat / NaN hold. Exit is
    # written first and entry second so an entry wins on any overlap; with
    # rank_pct >= 0.80 against EXIT_PCT = 0.5 the two masks are disjoint
    # anyway, but the ordering makes that structural.
    entries = pd.Series(np.nan, index=df.index, dtype=float)
    entries[decay_signal] = 0.0
    entries[long_signal] = 1.0

    # session.py alone decides which bars are tradable, forward-fills the
    # sparse instruction series into a held position, and force-flattens on
    # the session's last bar.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    # formation_lookback and rank_window are ints because they ARE bar counts
    # and are meant to size wfo_engine's warm-up buffer; rank_pct is a float
    # quantile level and must not. formation_lookback (96/192/288/384) and
    # rank_pct (0.80/0.875/0.925) are the two grid-searched axes; rank_window
    # is fixed at 960 and threaded through, never searched. `session` is a
    # fixed param. EXIT_PCT (the 0.5 decay quantile) is a module constant, so
    # it deliberately has no entry here.
    "formation_lookback": 192,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
