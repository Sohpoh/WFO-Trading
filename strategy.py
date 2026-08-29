"""Long-only percentile-rank momentum, held to the session flatten.

Every one of the 32 prior iterations was symmetrically long/short. Splitting
their OOS trades by direction says the same thing in four unrelated families:
the short leg is the worse half — long vs short win rate 47.7%/41.0% (iter 32,
ES 15min CMF, 441 trades), 44.8%/38.2% (iter 25, ES 1h location, 589),
41.2%/33.5% (iter 30, NQ 1h polarity, 335), 46.6%/43.9% (iter 23, NQ 15min
tsmom, 635). ~2,000 trades, same sign every time. So this implements
momentum-strategies.md's Long-Only portfolio construction verbatim ("Buy only
the winners... w_i >= 0") and **deletes the short leg entirely**. Under
`metrics.py`'s per-leg cost model that means the down-state pays nothing at
all rather than paying two legs to reverse, roughly halving round trips on a
balanced signal (momentum-strategies.md pitfall #5, high turnover).

The second thing varied from the page: its canonical momentum rank ("rank by
cumulative returns over a formation period... buy the top decile") is
*cross-sectional*, and there is only one instrument here. So the rank is taken
against that instrument's **own trailing distribution** instead — the first
self-normalizing entry threshold in this log. Every prior iteration's entry
level was absolute (a z-score, a CMF pressure, an ATR multiple) and therefore
churned across folds as volatility regime shifted; a quantile of the trailing
`rank_window` bars re-scales itself automatically.

Rules (all computed on the full continuous frame with no session awareness of
their own; every window is strictly backward-looking, so there is no
lookahead):

  - Formation return, the momentum statistic:

        ret_t = Close_t / Close_{t-formation_lookback} - 1

    Both endpoints are completed bars. NaN for the first
    `formation_lookback` bars.

  - Self-referential rank threshold — the trailing `rank_pct` quantile of that
    same statistic:

        q_t = ret.shift(1)
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

  - Raw entries — long side only:

        ret_t > q_t  ->  LONG (1.0)
        otherwise    ->  NaN

    -1.0 is never emitted; there is no short side. 0.0 is never emitted
    either, so NaN's ffill inside `apply_session_constraint()` holds the long
    from the first qualifying in-session bar straight through to the flatten:
    exactly one round trip per traded session. NaN compares False on both
    sides of `>`, so unwarmed bars fail *closed* with no separate validity
    mask.

Exit is plain and non-path-dependent — the raw entries series goes to
`session.apply_session_constraint()` unchanged, NOT the `_with_stops` variant.
The only exit is `session.py`'s forced flatten on the session's last bar:
there is no stop, no target, and (being long-only) no opposite signal to flip
into. Winners are uncapped deliberately — iteration 27 showed hard-capping a
continuation payoff at 2R flipping per-trade economics from +0.4bps to
-11.6bps. Accepted risk, stated up front: this is the run-to-flatten shape
that gate v2's leave-top-5-out penalised in iterations 6/10/11, so if Steps 2
and 4 pass and Step 3 alone fails, the correct next move is a scale-out or
trailing exit inside this family, not a filter that trades less.

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

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is
unchanged and out of scope. The long-only construction interacts with it
favourably — a down-state costs zero legs instead of two — but nothing in the
cost model itself is touched here.

This module decides only *when* the strategy wants to be long. All day-trade
gating and the end-of-session flatten are delegated to `session.py`; see its
docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint


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


def trailing_rank_threshold(ret: pd.Series, rank_window: int, rank_pct: float) -> pd.Series:
    """`rank_pct` quantile of `ret` over the trailing `rank_window` bars,
    excluding the current bar from its own threshold.

    Strictly warm (`min_periods == rank_window`), so unwarmed bars are NaN and
    fail closed downstream.
    """
    w = int(rank_window)
    return ret.shift(1).rolling(w, min_periods=w).quantile(float(rank_pct))


def generate_positions(
    df: pd.DataFrame,
    formation_lookback: int,
    rank_pct: float,
    rank_window: int = 960,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)

    ret = formation_return(close, formation_lookback)
    threshold = trailing_rank_threshold(ret, rank_window, rank_pct)

    # Top-quantile formation return -> long. Strict `>`; NaN on either side
    # (unwarmed formation window or unwarmed rank window) compares False, so
    # the strategy stands aside rather than guessing.
    with np.errstate(invalid="ignore"):
        long_state = (ret > threshold).to_numpy()

    # Raw, session-unaware entries. Long-only: -1.0 is never emitted, and
    # neither is 0.0 — NaN means "no new signal", so the delegate's ffill holds
    # the long until the session flatten closes it. Explicit float64 so
    # ffill/fillna arithmetic downstream stays numeric.
    entries = pd.Series(np.nan, index=df.index, dtype=float)
    entries[long_state] = 1.0

    # session.py alone decides which of those bars are tradable and force-
    # flattens on the session's last bar. No stop, no target, no path
    # dependence, so apply_session_constraint_with_stops() is deliberately not
    # used.
    return apply_session_constraint(entries, session)


DEFAULT_PARAMS = {
    "formation_lookback": 48,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
