"""Long-only percentile-rank momentum with a momentum-decay exit.

Every one of the 32 iterations before this family was symmetrically
long/short. Splitting their OOS trades by direction says the same thing in
four unrelated families: the short leg is the worse half — long vs short win
rate 47.7%/41.0% (iter 32, ES 15min CMF, 441 trades), 44.8%/38.2% (iter 25,
ES 1h location, 589), 41.2%/33.5% (iter 30, NQ 1h polarity, 335), 46.6%/43.9%
(iter 23, NQ 15min tsmom, 635). ~2,000 trades, same sign every time. So this
implements momentum-strategies.md's Long-Only portfolio construction verbatim
("Buy only the winners... w_i >= 0") and **deletes the short leg entirely**.
Under `metrics.py`'s per-leg cost model that means the down-state pays nothing
at all rather than paying two legs to reverse, roughly halving round trips on
a balanced signal (momentum-strategies.md pitfall #5, high turnover).

The second thing varied from the page: its canonical momentum rank ("rank by
cumulative returns over a formation period... buy the top decile") is
*cross-sectional*, and there is only one instrument here. So the rank is taken
against that instrument's **own trailing distribution** instead. Every prior
iteration's entry level was absolute (a z-score, a CMF pressure, an ATR
multiple) and therefore churned across folds as volatility regime shifted; a
quantile of the trailing `rank_window` bars re-scales itself automatically.

Entry — unchanged, byte-for-byte, from iterations 33 and 34
-----------------------------------------------------------
All computed on the full continuous frame with no session awareness of their
own; every window is strictly backward-looking, so there is no lookahead.

  - Formation return, the momentum statistic:

        ret_t = Close_t / Close_{t-formation_lookback} - 1

    Both endpoints are completed bars. NaN for the first
    `formation_lookback` bars.

  - Self-referential entry threshold — the trailing `rank_pct` quantile of
    that same statistic:

        q_enter_t = ret.shift(1)
                       .rolling(rank_window, min_periods=rank_window)
                       .quantile(rank_pct)

    `shift(1)` is what excludes the current bar from its own threshold (a bar
    cannot be part of the distribution it is being ranked against), and the
    strict `min_periods=rank_window` NaNs out every unwarmed bar. Because
    `ret` is itself NaN for its first `formation_lookback` bars and pandas'
    `min_periods` counts only non-NaN observations, the first finite threshold
    lands at bar `formation_lookback + rank_window` — 1152 bars at the top of
    the intended grid, which the engine's warm-up buffer covers (see below).

    `rolling(...).quantile(...)` is used rather than `rolling(...).apply(...)`
    with a rank function on purpose: the latter is orders of magnitude slower
    across a 12-combo grid x ~48 folds.

  - Long entry where `ret_t > q_enter_t`, strict `>`. There is no short leg;
    -1.0 is never emitted. NaN compares False on both sides of `>`, so
    unwarmed bars fail closed with no separate validity mask.

Exit — the only thing changed from the previous iteration
---------------------------------------------------------
Iteration 33 had **no exit at all** beyond the forced end-of-session flatten;
iteration 34 bolted on a hard stop at 0.6x the trailing daily range, routed
through the path-dependent stops delegate. That stop is **retired here** and
replaced by a momentum-decay exit on the same statistic the entry uses.

Why the stop goes rather than gets retuned, and why capping winners is not
the answer, both follow from iteration 34's own measured numbers:

  - The diagnosed failure mode is an overfit gap, and the leave-top-5-out
    diagnostics are computed on the trade set *with the top 5 winners already
    removed*: total_return_ex_top5 -7.66%, profit_factor_ex_top5 0.893. No
    profit target and no R-cap can move either number by construction — a cap
    can only shrink total_return_full. So the only fixable population is the
    **body**, and it is quantified: -7.66% across the remaining ~205 trades is
    -3.74 bps/trade, against a PF_ex5 sitting 0.007 under the 0.9 floor. About
    +3.7 bps/trade on ordinary trades — roughly a third of one 10.2 bps round
    trip — is the entire requirement. (This is also why iteration 27's warning,
    that a 2R cap flipped that family's per-trade economics +0.4 bps ->
    -11.6 bps, never has to be adjudicated here: no cap is being added.)

  - Iteration 34's own side-by-side, recorded in its docstring: 43 of 321
    exits at 48/0.875 were stop-outs, ~4x the ~10 left-tail trades the stop
    was aimed at, i.e. "most stopped trades are ordinary ones cut short". The
    stop was taxing exactly the body that has to improve. Meanwhile the tail
    trades it targeted were full-session holds whose momentum had already
    collapsed — which is precisely what a fall below the trailing median rank
    fires on, and unlike a fixed price stop it does not cut trades that are
    merely noisy.

The replacement, a second hardcoded threshold on the same statistic:

        q_exit_t = ret.shift(1)
                      .rolling(rank_window, min_periods=rank_window)
                      .quantile(EXIT_PCT)          # EXIT_PCT = 0.5

  - Raw entries series, three states:

        1.0   where ret_t > q_enter_t      (enter / stay long)
        0.0   where ret_t < q_exit_t       (decay -> go flat)
        NaN   in between                   (hysteresis band -> hold)

    then `apply_session_constraint(entries, session)` — the plain non-stops
    delegate, used unmodified. It forward-fills the sparse series, so 0.0 is
    an explicit flat instruction (not "no signal"), NaN is a genuine hold, and
    the last bar of every session is force-flattened by `session.py` before
    the fill. Iteration 34's `apply_session_constraint_with_stops()` path and
    its 0.6x-daily-range stop are dropped entirely.

  - The masks are assigned exit-first, entry-second so an entry wins if they
    ever overlap. At `rank_pct >= 0.80` against `EXIT_PCT = 0.5` they cannot
    overlap (the entry quantile is strictly above the exit quantile on any
    non-degenerate distribution), but the ordering makes that structural
    rather than incidental.

Consequences, pre-registered:

  - The entry-to-median hysteresis band is wide, so a position is held through
    ordinary noise and exited only once the instrument's own momentum ranking
    has dropped out of the upper half of its trailing distribution.

  - **Upside is untouched.** A still-trending winner keeps `ret` above the
    median and therefore runs uncapped to the forced session flatten exactly
    as in iterations 33 and 34. Nothing here caps a winner.

  - Re-entry after a decay exit requires `ret` to climb all the way back above
    the `rank_pct` quantile, so churn is self-limiting — but it is possible,
    so a traded session is not guaranteed to be exactly one round trip (that
    property was already lost in iteration 34).

  - Fail-closed is exact **while flat**: an unwarmed bar leaves both
    thresholds NaN, both comparisons are False, the bar emits NaN, and the
    forward fill leaves the (flat) position flat. Honesty item — while
    *long*, a NaN `ret` (or NaN thresholds) is likewise both-False, emits NaN,
    and the forward fill therefore **holds the long**. That is inherent to a
    hysteresis exit and is bounded by the end-of-session flatten, so the worst
    case is one session held on stale information. No code change is made for
    it; it is stated here rather than left for the evaluator to find.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):

  - `formation_lookback` and `rank_window` are both plain `int` **on purpose**
    — both are genuine bar counts, so both are exactly what should size the
    pre-test-window warm-up buffer.
  - `rank_window` is fixed at 960, never grid-searched, and is the larger of
    the two: it drives `buffer_bars = max((960 + 5) * 3, day_bars + 5)` = 2895
    bars, comfortably covering the true requirement of
    `rank_window + max(formation_lookback)` = 1152. Do not raise it without
    re-checking that arithmetic *and* the fold-skip guard — see build_grid().
  - `rank_pct` is a quantile level in (0, 1), never a bar count, so it is
    passed as `float` and correctly ignored by the buffer sizing.
  - Retiring the stop removes the one warm-up leg that was invisible to
    `_max_lookback_bars()` (iteration 34's hardcoded 96/960 daily-range
    windows, which had to be hand-checked to bar 1056). The warm-up story is
    now a single leg, `rank_window + max(formation_lookback)` = 1152 bars, all
    of it engine-derived. The binding requirement is unchanged from 33/34, so
    there are no new fold skips (the guard is still `max_lookback + 10` = 970,
    since `max_lookback` is still `rank_window` = 960).

  - `EXIT_PCT` is a module constant and is **never grid-searched**: it adds no
    `build_grid()` / `DEFAULT_PARAMS` key, so the grid and param names stay
    byte-identical to iterations 33/34 and the exit is the only variable in
    this comparison. That is also deliberate given the diagnosed overfit gap —
    iteration 10 added a zero-param exit change as its only edit and passed;
    iteration 27 promoted its exit level to a searched axis and the family's
    economics inverted.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is
unchanged and out of scope. Relative to iteration 34 the decay exit adds no
new *kind* of cost leg — it substitutes one exit trigger for another — and
relative to iteration 33 it adds at most the legs of an in-session re-entry,
which the hysteresis band makes rare. The long-only construction still
interacts with the cost model favourably: a down-state costs zero legs
instead of two.

This module decides only *when* the strategy wants to be long and when that
wish has decayed. All day-trade gating and the end-of-session flatten are
delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint

# Quantile of the same trailing formation-return distribution at which an open
# long is considered to have decayed and is flattened. Hardcoded, NOT
# grid-searched — see the module docstring: the diagnosed failure is an
# overfit gap, and "the instrument has dropped out of the upper half of its
# own momentum distribution" is an economic reading, not a fitted value.
EXIT_PCT = 0.5


def formation_return(close: pd.Series, formation_lookback: int) -> pd.Series:
    """Simple return over the last `formation_lookback` completed bars.

    NaN for the first `formation_lookback` bars (no history to measure
    against) and wherever the anchor price is non-positive.
    """
    n = int(formation_lookback)
    prior = close.shift(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = close / prior - 1.0
    return ret.where(prior > 0).replace([np.inf, -np.inf], np.nan)


def trailing_rank_threshold(ret: pd.Series, rank_window: int, pct: float) -> pd.Series:
    """`pct` quantile of `ret` over the trailing `rank_window` bars, excluding
    the current bar from its own threshold.

    Strictly warm (`min_periods == rank_window`), so unwarmed bars are NaN and
    fail closed downstream. Called twice per run — once at `rank_pct` for the
    entry level, once at `EXIT_PCT` for the decay level — off the same `ret`.
    """
    w = int(rank_window)
    return ret.shift(1).rolling(w, min_periods=w).quantile(float(pct))


def generate_positions(
    df: pd.DataFrame,
    formation_lookback: int,
    rank_pct: float,
    rank_window: int = 960,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)

    ret = formation_return(close, formation_lookback)
    enter_threshold = trailing_rank_threshold(ret, rank_window, rank_pct)
    exit_threshold = trailing_rank_threshold(ret, rank_window, EXIT_PCT)

    # Top-quantile formation return -> long; formation return below its own
    # trailing median -> momentum has decayed, go flat. Strict comparisons;
    # NaN on either side (unwarmed formation window or unwarmed rank window)
    # compares False on both, so such a bar emits NaN = "no instruction" and
    # the delegate's forward fill leaves the position where it already was.
    # These are already bool Series — no .fillna(False) needed, and none
    # added, so the behaviour stays a property of the comparisons themselves.
    with np.errstate(invalid="ignore"):
        long_signal = ret > enter_threshold
        decay_signal = ret < exit_threshold

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
    "formation_lookback": 48,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
