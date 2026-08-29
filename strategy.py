"""Long-only percentile-rank momentum with a daily-range-scaled hard stop.

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

  - Raw entry signal — long side only:

        long_signal_t = (ret_t > q_t)

    `short_signal` is the all-False series: this strategy never emits -1.0
    and has no short leg. NaN compares False on both sides of `>`, so
    unwarmed bars fail *closed* with no separate validity mask.

Exit — the only thing changed from the previous iteration
--------------------------------------------------------
The previous iteration of this family (iteration 33) had **no stop at all**:
its raw entries went to `session.apply_session_constraint()` and the sole
exit was the forced flatten. That was measurably the leak. 10 of its 214 OOS
trades were full-session longs held from the open to the flatten for -2.2% to
-4.3% (summing ~-0.30 gross against a full-run OOS total of -0.053), and
`insample_trades.csv` showed the identical shape at the identical rate (7 of
165, -2.2% to -3.3%), so the un-stopped run-to-flatten exit was a structural
property of the exit rule, not an OOS accident. Its own docstring
pre-registered the fix as "a scale-out or trailing exit"; a scale-out would
violate CLAUDE.md's {-1, 0, 1} position contract and a genuine trailing stop
is not expressible here (`apply_session_constraint_with_stops()` reads
`stop_distance` only at the entry bar and never updates it), so what remains
is a fixed, volatility-scaled hard stop.

Entry signal, grids, symbol, timeframe, session and schedule are all
byte-identical to iteration 33 so the exit is the only variable.

  - Stop distance, quoted as a fraction of the trailing daily range
    (wiki:volatility.md "Stop-Loss Placement": low vol -> tight stops, high
    vol -> wider stops, sized as a multiple of ATR/daily range):

        span_t       = High.rolling(96).max() - Low.rolling(96).min()
        daily_rng_t  = span.rolling(960, min_periods=960).mean().shift(1)
        stop_dist_t  = STOP_DAY_FRAC * daily_rng_t

    96 bars is 24h at 15min on this 24h continuous frame (the same 96/384
    "24h vs 96h" convention iteration 6 used), so `span` is a rolling
    one-day high-low range and `daily_rng` averages it over the last ~10
    sessions. `.shift(1)` keeps the current bar out of its own stop.
    Measured on the local NQ 15min frame the stop sits at a median 1.13% of
    price — 0.91% median through calm 2024, 1.85% through 2022 — so it
    breathes with the regime as intended: wide enough that a typical 0.3-0.5%
    session excursion never touches it, tight enough to cut the -2.2%/-4.3%
    collapses. (The idea spec's estimate was 0.8-0.9% calm / ~1.5% in 2022;
    the realised levels are ~20-25% wider, i.e. the stop is slightly more
    permissive than advertised, not tighter.)

  - Upside stays uncapped. `target_price_t = Close_t + 100 * stop_dist_t` is
    finite and strictly above the close, so the delegate's `tgt_ok` /
    `target > close` entry guard passes, but 100 stop-widths is unreachable
    inside one session — winners still run to `session.py`'s forced flatten
    exactly as in iteration 33. Capping them is deliberately NOT tried:
    iteration 27 hard-capped a continuation payoff at 2R and per-trade
    economics flipped +0.4bps -> -11.6bps.

  - `STOP_DAY_FRAC = 0.6` is a module constant, **never grid-searched**, and
    introduces no new `build_grid()` / `DEFAULT_PARAMS` key. The diagnosed
    failure mode is an overfit gap, so adding a searched axis would add
    selection pressure in exactly the diagnosed direction: iteration 10 added
    a zero-param stop as its only change and passed; iteration 27 promoted
    the stop to a searched axis and the family's economics inverted.

  - An unwarmed daily-range window leaves `stop_dist` NaN, which propagates
    into `target_price`, and the delegate refuses to open without a strictly
    positive stop distance and a valid, correctly-sided target. Fails closed
    for free, no separate validity mask.

Pre-registered consequence, named up front rather than left for the evaluator
to find: because `_with_stops` re-enters whenever it is flat, in-session and
the signal is true, a stop-out can be followed by a same-session re-entry, so
iteration 33's "exactly one round trip per traded session" property no longer
holds. The idea spec pre-registered the churn band as 240-330 OOS trades
(north of ~400 = the stop is managing trades rather than truncating them, the
run is contaminated; at or below iteration 33's 214 = suspect a NaN/zero stop
distance suppressing entries).

Measured before the fact, so the band is not re-read after the result:
re-entry is far rarer than that band assumes. Running this file and the
archived iteration-33 file side by side over the full local NQ 15min history
at four grid points, trade count rises only +1.3% to +4.4% (48/0.875:
315 -> 321; 192/0.925, the pair that dominated iteration 33's fold
selections: 206 -> 215; 24/0.80: 607 -> 619), with at most 3 round trips in
any one session. The mechanism is benign and self-limiting: a stop-out means
price has just fallen ~1.1%, which drops `ret_t` back below `q_t`, so the bar
no longer qualifies and there is nothing to re-enter on. Entries are
demonstrably not being suppressed — 5,292 in-position bars and the first
finite stop distance at bar 1055, exactly the hand-computed warm-up. So
**expect ~215-225 OOS trades, below the 240 floor**; that is the stop working
as a tail truncator with negligible churn, not a NaN bug, and the band was
simply set too high.

What actually changed, from the same side-by-side: at 48/0.875, 43 of the 321
exits are stop-outs rather than session flattens — ~4x the 10 tail trades the
thesis targeted, so most stopped trades are ordinary ones cut short, which is
where the per-trade economics will move. The tail itself shrinks in weight
more than in count: trades at or below -2.2% go 11 -> 4 and their summed
return -0.326 -> -0.102, and the worst single trade -5.04% -> -3.06%. Note
that -3.06% is still inside the -2.2%/-4.3% band the stop was specified to
eliminate — because the stop is a *when*, not a *how much* (see the fill-price
caveat below), and because a rolling 96-bar max-min straddles day boundaries
and so runs systematically wider than a true calendar-day range. The tail is
truncated, not removed.

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
  - The stop's own windows (`DAILY_RANGE_BARS = 96`,
    `DAILY_RANGE_WINDOW = 960`) are module constants, not grid params, so
    they are invisible to `_max_lookback_bars()` and have to be hand-checked
    rather than engine-derived. Doing that: `rolling(96)` first goes finite
    at bar 95, the 960-bar strict mean at 95 + 959 = 1054, and `.shift(1)`
    pushes it to 1056. The signal leg needs
    `rank_window + max(formation_lookback)` = 960 + 192 = 1152. These are
    independent legs, so the binding requirement is the **max**, 1152 bars —
    *unchanged from iteration 33*. That is the load-bearing number: the stop
    adds no new fold skips (the guard is still `max_lookback + 10` = 970,
    since `max_lookback` is still `rank_window` = 960) and no new
    suppressed-entry region at the head of the un-buffered train windows
    `_score_params()` scores on.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is
unchanged and out of scope. The long-only construction interacts with it
favourably — a down-state costs zero legs instead of two — but nothing in the
cost model itself is touched here. Note the stop *adds* cost legs relative to
iteration 33: a stopped-out session that re-enters pays two round trips where
the un-stopped version paid one, so the truncated tail has to be worth more
than that extra ~10.2bps to show up as an improvement.

Fill-price caveat inherited from the delegate: a stop hit is *detected* off
that bar's Low against the stored level but the position is flattened at that
bar's Close, because this codebase has no intrabar series and
`metrics.bar_returns_with_costs` prices every bar off Close. A bar that gaps
through the stop still books its whole close-to-close return before exiting.
The stop caps *when* you exit, not the realized loss on the triggering bar —
do not describe results as guaranteeing a max loss of `stop_distance`.

This module decides only *when* the strategy wants to be long, and how far
away its stop and (unreachable) target sit. All day-trade gating and the
end-of-session flatten are delegated to `session.py`; see its docstring for
that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Trailing daily-range windows for the stop, in bars. Hardcoded on purpose:
# the stop adds no searchable degrees of freedom. Sized for 15min bars on a
# 24h continuous frame — 96 bars is one full 24h Globex day (iteration 6's
# 96/384 convention), and averaging that span over 960 bars smooths it across
# roughly the last ten sessions so a single wild day does not set the stop.
# DAILY_RANGE_WINDOW coincides numerically with the fixed `rank_window` param
# but is independent of it; do not couple them.
DAILY_RANGE_BARS = 96
DAILY_RANGE_WINDOW = 960

# Hard stop as a fraction of that trailing daily range. Hardcoded, NOT
# grid-searched — see the module docstring: the diagnosed failure is an
# overfit gap, and 0.6 of a day's typical range is an economic reading ("more
# than half of a normal day has gone against me"), not a fitted value.
STOP_DAY_FRAC = 0.6

# Multiple of the stop distance used as the (deliberately unreachable) profit
# target. The delegate requires a non-NaN, correctly-sided target on every
# entry bar; 100 stop-widths inside one session is not attainable, so this
# reproduces iteration 33's "no target — winners run to the session flatten".
UNREACHABLE_TARGET_MULT = 100.0


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


def trailing_daily_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Average one-day high-low span over the last `DAILY_RANGE_WINDOW` bars.

    The inner `rolling(DAILY_RANGE_BARS)` extremes give a rolling 24h range;
    the outer strict-`min_periods` mean smooths it over ~10 sessions, and the
    `.shift(1)` keeps the current bar out of the range that sizes its own
    stop. Unwarmed bars stay NaN so they fail closed in the delegate.
    """
    span = (
        high.rolling(DAILY_RANGE_BARS, min_periods=DAILY_RANGE_BARS).max()
        - low.rolling(DAILY_RANGE_BARS, min_periods=DAILY_RANGE_BARS).min()
    )
    return span.rolling(DAILY_RANGE_WINDOW, min_periods=DAILY_RANGE_WINDOW).mean().shift(1)


def generate_positions(
    df: pd.DataFrame,
    formation_lookback: int,
    rank_pct: float,
    rank_window: int = 960,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    ret = formation_return(close, formation_lookback)
    threshold = trailing_rank_threshold(ret, rank_window, rank_pct)

    # Top-quantile formation return -> long. Strict `>`; NaN on either side
    # (unwarmed formation window or unwarmed rank window) compares False, so
    # the strategy stands aside rather than guessing. Already a bool Series —
    # no .fillna(False) needed, and none added, so the fail-closed behaviour
    # stays a property of the comparison itself.
    with np.errstate(invalid="ignore"):
        long_signal = ret > threshold

    # Long-only: the delegate still wants a short side, so hand it one that is
    # never True. -1.0 can therefore never be emitted.
    short_signal = pd.Series(False, index=df.index)

    # Volatility-scaled hard stop, symmetric distance from the entry Close —
    # the delegate resolves it to the long side. NaN while the daily-range
    # window is unwarmed, which makes the delegate refuse the entry.
    stop_distance = STOP_DAY_FRAC * trailing_daily_range(high, low)

    # No real target: a finite level strictly above the entry Close (so the
    # delegate's tgt_ok / target > close guard passes) but unreachable inside
    # one session, so the profit side is governed only by the session flatten
    # and winners stay uncapped. NaN stop -> NaN target -> no entry.
    target_price = close + UNREACHABLE_TARGET_MULT * stop_distance

    # session.py alone decides which bars are tradable, walks the stop/target
    # path, and force-flattens on the session's last bar.
    return apply_session_constraint_with_stops(
        close=close,
        high=high,
        low=low,
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    "formation_lookback": 48,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
