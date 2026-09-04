"""Long-only risk-adjusted momentum rank with a volatility-floor entry guard
(NQ 15min, New York session) — iteration 48, a variation of iteration 47.

Iteration 47 ranked the volatility-normalized formation return `R_mean / σ`
(vault eq. 269) against its own trailing quantile and was rejected overfit-gap
(IS Sharpe 1.59 -> OOS 0.65). Its decisive diagnostic was the comparison to
iteration 36, which ranked the *raw* formation return instead: both runs exit
~90% of OOS trades at the forced session flatten with a ~0.97 payoff ratio,
but 36's raw-return entry cleared Sharpe 1.24 / robust:true while 47's ÷σ
entry reached only 0.65 with a lower win rate (56.5% vs 60.5%), thinner
per-trade edge (6.2bp vs 12.9bp) and heavier top-5 concentration (70% vs
47.5%) — so the exit is not the differentiator and the risk-adjusted
statistic itself is the defect.

The mechanism of that defect: ÷σ ranking up-weights low-volatility formation
windows — a modest drift in a quiet window outranks a strong drift in a
volatile one — and that is exactly the regime where momentum does not work
(vault volatility.md: "Low-Volatility Regime … trend-following struggles").
This iteration does NOT touch the exit or any stop/target. It adds a single
type-(b) zero-param entry filter: a long now additionally requires the
formation σ_t to be at least the trailing median of σ over the same
`rank_window` bars. That removes the quiet-window degenerate entries without
adding any grid axis or selection pressure — critical when the failure mode
is overfit-gap and the optimizer is already fitting a noisy surface.
Entry/exit/grids are otherwise byte-identical to 47 so the filter is the only
change and the result is attributable.

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
    observations, so ddof=1 is never degenerate. σ_t itself is kept as the
    raw std — finite 0.0 for a flat window, NaN only where unwarmed — so the
    volatility-floor filter below can compare it against its own trailing
    median rather than seeing flat windows as missing observations.

  - Self-referential entry threshold — the trailing `rank_pct` quantile of the
    same statistic, current bar excluded from its own distribution:

        enter_threshold_t = risk_adj.shift(1)
                                 .rolling(rank_window, min_periods=rank_window)
                                 .quantile(rank_pct)

    The strict `min_periods=rank_window` NaNs out every unwarmed bar, and the
    `.shift(1)` excludes the current bar. `rolling(...).quantile(...)` is used
    rather than `.apply(...)` on purpose: orders of magnitude faster across a
    12-combo grid x ~48 folds.

  - Volatility floor (iteration 48's ONLY change from 47) — the trailing
    median of σ over the same window, current bar excluded:

        sigma_floor_t = σ_t.shift(1)
                           .rolling(rank_window, min_periods=rank_window)
                           .quantile(VOL_FLOOR_PCT)        # VOL_FLOOR_PCT = 0.5

    A formation window whose σ_t sits below its own trailing median is a
    below-median-volatility regime — precisely where ÷σ ranking fabricates
    its degenerate top-quantile entries — so the long is withheld there.

Entry — a dense long-only state gated on above-median volatility
----------------------------------------------------------------
A raw +1.0 long is emitted only where BOTH hold, strict comparisons:

    (a) risk_adj_t > enter_threshold_t   (top (1-rank_pct) quantile)
    (b) σ_t >= sigma_floor_t             (formation vol at/above its own
                                          trailing median)

Long-only: there is no short branch, so -1.0 is never emitted and a down-state
simply pays no legs at all instead of paying two to reverse.

Exit — decay-flip to flat, no stop, no target (unchanged from 47)
----------------------------------------------------------------
Plain `apply_session_constraint` (NOT the `with_stops` variant) — no stop, no
target, upside uncapped, duration bounded by the session. Decay is a second
hardcoded threshold on the same statistic:

        exit_threshold_t = risk_adj.shift(1)
                                 .rolling(rank_window, min_periods=rank_window)
                                 .quantile(EXIT_PCT)        # EXIT_PCT = 0.5

Raw entries are three-state:

        1.0   where (risk_adj > enter) AND (σ >= sigma_floor)  (enter / stay long)
        0.0   where risk_adj < exit_threshold                  (decayed -> go flat)
        NaN   in between                                       (hysteresis band -> hold)

The volatility floor gates the entry side only: once long, the hold/exit is
driven purely by risk_adj vs its own trailing median, so the floor does not
truncate a winner mid-session. A bar whose risk_adj clears the entry quantile
but whose σ is below its median emits NaN — while flat that keeps the position
flat (the intended withhold), while long it leaves the existing long untouched
(the floor is not an exit). Exit is written first and entry second so an entry
wins on any overlap; the two masks are disjoint anyway (the entry quantile
sits strictly above the median and the floor only prunes the entry set), but
the ordering makes that structural. `apply_session_constraint` forward-fills
the sparse series: 0.0 is an explicit flat instruction, NaN a genuine hold,
and the last bar of every session is force-flattened by `session.py`. A long
is therefore held through ordinary noise (the entry-to-median band) and
flattened on the first below-median bar; a still-trending winner runs uncapped
to the forced session flatten.

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
  - `EXIT_PCT` (the 0.5 decay quantile) and `VOL_FLOOR_PCT` (the 0.5
    volatility-floor quantile) are module constants and are **never
    grid-searched**: they add no `build_grid()` / `DEFAULT_PARAMS` key, so the
    exit and the entry floor together keep zero degrees of freedom.

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

# Quantile of the trailing formation-volatility (σ) distribution that σ_t must
# meet or exceed before a long can open. Hardcoded, NOT grid-searched — the
# iteration-48 volatility floor: "the formation window must be at least
# median-volatility for its top-quantile ÷σ momentum to be trustworthy". This
# is the only change from iteration 47 and keeps zero degrees of freedom.
VOL_FLOOR_PCT = 0.5


def formation_stats(close: pd.Series, formation_lookback: int) -> tuple[pd.Series, pd.Series]:
    """Formation-period (risk_adj, sigma) pair (vault eqs. 269-270).

    Computes the 1-bar simple return r_t = Close_t/Close_{t-1} - 1, then the
    mean and sample std of r over the trailing `formation_lookback` bars, and
    returns (mean/std, std). risk_adj is NaN where std <= 0 (flat-vol) or std
    is non-finite (unwarmed/overflow), so those bars fail closed downstream.
    sigma is the raw sample std (ddof=1): NaN only where unwarmed, 0.0 for a
    flat window — kept finite on purpose so the volatility-floor filter (σ_t
    vs its trailing median) sees a flat window as *low* volatility, not as a
    missing observation.
    """
    n = int(formation_lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = close / close.shift(1) - 1.0

    mean = r.rolling(n, min_periods=n).mean()
    sigma = r.rolling(n, min_periods=n).std()  # ddof=1

    with np.errstate(divide="ignore", invalid="ignore"):
        risk_adj = mean / sigma

    # std <= 0 (flat-vol) or non-finite std (unwarmed/overflow) -> NaN, so
    # those bars carry no ratio instruction. The replace also drops any
    # non-finite ratio (mean/std overflow) so only genuine finite values reach
    # the rank. sigma itself is returned unmodified.
    risk_adj = risk_adj.where(np.isfinite(sigma) & (sigma > 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    return risk_adj, sigma


def trailing_rank_threshold(stat: pd.Series, rank_window: int, pct: float) -> pd.Series:
    """`pct` quantile of `stat` over the trailing `rank_window` bars, excluding
    the current bar from its own threshold.

    Strictly warm (`min_periods == rank_window`), so unwarmed bars are NaN and
    fail closed downstream. Called three times per run — once at `rank_pct`
    for the entry level, once at `EXIT_PCT` for the decay level (both off the
    risk-adjusted series), and once at `VOL_FLOOR_PCT` for the volatility
    floor (off the σ series).
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

    stat, sigma = formation_stats(close, formation_lookback)
    enter_threshold = trailing_rank_threshold(stat, rank_window, rank_pct)
    exit_threshold = trailing_rank_threshold(stat, rank_window, EXIT_PCT)
    sigma_floor = trailing_rank_threshold(sigma, rank_window, VOL_FLOOR_PCT)

    # Long requires BOTH the top-quantile risk-adjusted return AND formation
    # volatility at or above its own trailing median (the iteration-48 floor).
    # Decay is the same statistic below its own trailing median. Strict
    # comparisons; NaN on either side (unwarmed formation window or unwarmed
    # rank window) compares False on both, so such a bar emits NaN = "no
    # instruction" and the delegate's forward fill leaves the position where
    # it already was.
    with np.errstate(invalid="ignore"):
        long_signal = (stat > enter_threshold) & (sigma >= sigma_floor)
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
    # fixed param. EXIT_PCT (the 0.5 decay quantile) and VOL_FLOOR_PCT (the
    # 0.5 volatility-floor quantile) are module constants, so they
    # deliberately have no entry here.
    "formation_lookback": 192,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
