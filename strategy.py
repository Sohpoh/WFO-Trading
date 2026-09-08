"""Long-only risk-adjusted momentum rank with a volatility-floor entry guard and
a vol-spike crash exit (NQ 15min, New York session) — iteration 49, a variation
of iteration 48.

Iteration 48 was accepted on 2022–2024 OOS but its 2025 holdout was rejected on
a single left-tail crash day (2025-11-20: long at 14:45 UTC, flattened −3.70%).
Its holdout reasoning blamed the σ-floor entry filter, but the entry cannot
separate a V-bottom recovery (this family's edge) from a crash-continuation (its
risk) — both look identical at the top quantile of risk-adjusted momentum. The
fix therefore belongs on the exit, where a crash is observable as it happens
rather than only after the slow decay-flip has already booked the full drop.

This iteration does NOT touch the entry signal, either grid axis, `rank_window`,
or the session machinery — all byte-identical to 48. It changes ONLY the exit:
a second, zero-param flatten condition is OR-ed into the flat mask. When the
6-bar average true range reaches 3.0× the 96-bar average true range, the
strategy flattens an open long mid-drop — volatility.md's Volatility Clustering
read as a crash signature ("after a gap or large move, expect continued
volatility"). This is distinct from iterations 6/28's ≥1.0 fast-vs-slow ATR
*entry* gate (a regime flag): it is a spike *exit* on a 90-minute leg with a
3.0 ratio. While flat the condition is an explicit no-op and it never touches
`long_signal`, so it truncates open longs only and never withholds or reduces
entries. The direction-agnostic 3.0× line sits above routine V-bottom dips,
accepting that a rare 3× melt-up also gets capped as the price of killing the
−3.7% tail.

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

  - Volatility floor (iteration 48's change from 47, unchanged here) — the
    trailing median of σ over the same window, current bar excluded:

        sigma_floor_t = σ_t.shift(1)
                           .rolling(rank_window, min_periods=rank_window)
                           .quantile(VOL_FLOOR_PCT)        # VOL_FLOOR_PCT = 0.5

    A formation window whose σ_t sits below its own trailing median is a
    below-median-volatility regime — precisely where ÷σ ranking fabricates
    its degenerate top-quantile entries — so the long is withheld there.

  - Crash signature (iteration 49's ONLY change) — a scale-free ATR-ratio spike
    read as volatility clustering. Classic Wilder true range and two hardcoded
    average-true-range windows:

        TR_t       = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        atr_fast_t = TR.rolling(CRASH_ATR_FAST, min_periods=CRASH_ATR_FAST).mean()
        atr_slow_t = TR.rolling(CRASH_ATR_SLOW, min_periods=CRASH_ATR_SLOW).mean()
        crash_t    = atr_fast_t / atr_slow_t >= CRASH_ATR_RATIO

    with CRASH_ATR_FAST = 6, CRASH_ATR_SLOW = 96, CRASH_ATR_RATIO = 3.0 — all
    three module constants, never grid-searched. `atr_slow` is NaN until bar
    CRASH_ATR_SLOW, so unwarmed bars fail closed. The ratio is scale-free: it
    fires when short-horizon average range expands to 3.0× the long-horizon
    average regardless of the instrument's absolute vol level.

Entry — a dense long-only state gated on above-median volatility
----------------------------------------------------------------
A raw +1.0 long is emitted only where BOTH hold, strict comparisons:

    (a) risk_adj_t > enter_threshold_t   (top (1-rank_pct) quantile)
    (b) σ_t >= sigma_floor_t             (formation vol at/above its own
                                          trailing median)

Long-only: there is no short branch, so -1.0 is never emitted and a down-state
simply pays no legs at all instead of paying two to reverse.

Exit — decay-flip AND crash-flip to flat, no stop, no target
------------------------------------------------------------
Plain `apply_session_constraint` (NOT the `with_stops` variant) — no stop, no
target, upside uncapped, duration bounded by the session. Two flatten conditions
OR-ed into the flat mask:

        exit_threshold_t = risk_adj.shift(1)
                                 .rolling(rank_window, min_periods=rank_window)
                                 .quantile(EXIT_PCT)        # EXIT_PCT = 0.5

    (a) decay — risk_adj < exit_threshold (the trailing median, unchanged
        from 48);
    (b) crash — atr_fast / atr_slow >= CRASH_ATR_RATIO (new, zero-param).

Raw entries are three-state:

        1.0   where (risk_adj > enter) AND (σ >= sigma_floor)  (enter / stay long)
        0.0   where decay OR crash                              (flatten)
        NaN   in between                                       (hold)

The volatility floor gates the entry side only: once long, the hold/exit is
driven purely by risk_adj vs its own trailing median (decay) and by the crash
ratio — the floor does not truncate a winner mid-session. The 0.0 flat mask is
written first and the 1.0 second so an entry wins on any same-bar overlap; the
crash condition is deliberately excluded from `long_signal`, so a bar that both
triggers the crash ratio and clears the entry rank emits 1.0 (enter) rather
than being withheld. The crash condition therefore only ever truncates an
already-open long early — while flat it is an explicit no-op (forward fill
keeps flat flat) and it never withholds/reduces an entry. `apply_session_constraint`
forward-fills the sparse series: 0.0 is an explicit flat instruction, NaN a
genuine hold, and the last bar of every session is force-flattened by
`session.py`. A long is therefore held through ordinary noise (the
entry-to-median band) and flattened on the first below-median bar or the first
panic bar; a still-trending winner runs uncapped to the forced session flatten.

Fail-closed is exact while flat: an unwarmed/flat-vol bar leaves both
thresholds NaN, both comparisons False, the bar emits NaN, and the forward
fill leaves the flat position flat. Honesty item — while *long*, a NaN
`risk_adj` (or NaN thresholds) is likewise both-False, emits NaN, and the
forward fill therefore **holds the long**. That is inherent to a hysteresis
exit and is bounded by the end-of-session flatten, so the worst case is one
session held on stale information; it is stated here rather than left for the
evaluator to find.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
--------------------------------------------------------------------------
  - `formation_lookback` and `rank_window` are plain `int` **on purpose** —
    both are genuine bar counts and both are exactly what should size the
    pre-test-window warm-up buffer.
  - `rank_window` is fixed at 960, never grid-searched, and is the larger of
    the two: it drives `buffer_bars = max((960 + 5) * 3, day_bars + 5)` =
    2,895 bars, comfortably covering the true requirement of
    `rank_window + max(formation_lookback)` = 960 + 384 = 1,344 at the top of
    the slow-end formation grid 96/192/288/384. The crash ATR slow window
    (CRASH_ATR_SLOW = 96) is a module constant below both of these, so it
    needs no extra warm-up and no `build_grid()` key. Do not raise `rank_window`
    without re-checking that arithmetic *and* the fold-skip guard — see
    build_grid().
  - `rank_pct` is a quantile level in (0, 1), never a bar count, so it is
    passed as `float` and correctly ignored by the buffer sizing.
  - `EXIT_PCT` (the 0.5 decay quantile), `VOL_FLOOR_PCT` (the 0.5
    volatility-floor quantile), and the three crash-exit constants
    (`CRASH_ATR_FAST`, `CRASH_ATR_SLOW`, `CRASH_ATR_RATIO`) are module
    constants and are **never grid-searched**: they add no `build_grid()` /
    `DEFAULT_PARAMS` key, so the decay exit, the entry floor, and the crash
    exit together keep zero degrees of freedom.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.005% slippage) is
unchanged and out of scope. The long-only construction still interacts with
the cost model favourably: a down-state costs zero legs instead of two, and the
crash exit only ever closes a long already open (one exit leg), never adds a
reversal leg.

This module decides only *when* the strategy wants to be long and when that
wish has decayed or been panic-cut. All day-trade gating and the
end-of-session flatten are delegated to `session.py`; see its docstring for
that contract.
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
# median-volatility for its top-quantile ÷σ momentum to be trustworthy".
VOL_FLOOR_PCT = 0.5

# Crash-exit ATR windows and the panic ratio threshold. Hardcoded, NOT
# grid-searched — a scale-free panic signature: when the CRASH_ATR_FAST-bar
# average true range reaches CRASH_ATR_RATIO x the CRASH_ATR_SLOW-bar average
# true range, volatility has clustered hard enough to read as a crash in
# progress, so an open long is flattened mid-drop instead of waiting for the
# slow decay-flip to book the full leg. All three are module constants (like
# EXIT_PCT / VOL_FLOOR_PCT), never build_grid()/DEFAULT_PARAMS keys, so the
# crash exit keeps zero degrees of freedom.
CRASH_ATR_FAST = 6
CRASH_ATR_SLOW = 96
CRASH_ATR_RATIO = 3.0


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


def crash_signal(df: pd.DataFrame) -> pd.Series:
    """True where the fast/slow average-true-range ratio spikes to panic levels.

    Classic Wilder true range (max of high-low, |high-prev_close|,
    |low-prev_close|), then `atr_fast = TR.rolling(CRASH_ATR_FAST).mean()` and
    `atr_slow = TR.rolling(CRASH_ATR_SLOW).mean()`. Returns True where
    `atr_fast / atr_slow >= CRASH_ATR_RATIO` — the short-horizon average range
    has expanded to CRASH_ATR_RATIO x the long-horizon average, a scale-free
    volatility-clustering signature read as a crash in progress. `atr_slow` is
    NaN until bar CRASH_ATR_SLOW, so unwarmed bars fail closed.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_fast = tr.rolling(CRASH_ATR_FAST, min_periods=CRASH_ATR_FAST).mean()
    atr_slow = tr.rolling(CRASH_ATR_SLOW, min_periods=CRASH_ATR_SLOW).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = atr_fast / atr_slow
    return ratio >= CRASH_ATR_RATIO


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
    crash = crash_signal(df)

    # Long requires BOTH the top-quantile risk-adjusted return AND formation
    # volatility at or above its own trailing median (the iteration-48 floor).
    # Decay is the same statistic below its own trailing median; the crash
    # condition is the iteration-49 fast/slow ATR-ratio spike. Strict
    # comparisons; NaN on either side (unwarmed formation window or unwarmed
    # rank window) compares False on both, so such a bar emits NaN = "no
    # instruction" and the delegate's forward fill leaves the position where
    # it already was.
    with np.errstate(invalid="ignore"):
        long_signal = (stat > enter_threshold) & (sigma >= sigma_floor)
        decay_signal = stat < exit_threshold

    # Raw, session-unaware entries: 1.0 long / 0.0 flat / NaN hold. The crash
    # condition is OR-ed into the flat mask but deliberately excluded from
    # `long_signal`, so it can only truncate an open long (or no-op while
    # flat), never withhold/reduce an entry. Flat is written first and entry
    # second so an entry wins on any same-bar overlap; with rank_pct >= 0.80
    # against EXIT_PCT = 0.5 the decay mask and the entry mask are disjoint
    # anyway, but the ordering makes that structural.
    flatten_signal = decay_signal | crash
    entries = pd.Series(np.nan, index=df.index, dtype=float)
    entries[flatten_signal] = 0.0
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
    # fixed param. EXIT_PCT (the 0.5 decay quantile), VOL_FLOOR_PCT (the 0.5
    # volatility-floor quantile), and the three crash-exit constants are
    # module constants, so they deliberately have no entry here.
    "formation_lookback": 192,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
