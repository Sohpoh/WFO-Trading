"""Long-only percentile-rank momentum with a volatility-scaled hard stop.

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

Entry — unchanged, byte-for-byte, from iterations 33/34/35/36
-------------------------------------------------------------
Nothing in the entry moves this iteration: the formation grid stays at the
slow end (96/192/288/384) that iteration 36 established, `rank_pct` stays at
0.80/0.875/0.925, `rank_window` stays fixed at 960. All computed on the full
continuous frame with no session awareness of their own; every window is
strictly backward-looking, so there is no lookahead.

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
    lands at bar `formation_lookback + rank_window` — 1344 bars at the top of
    the intended grid (960 + 384), which the engine's warm-up buffer covers
    (see below).

    `rolling(...).quantile(...)` is used rather than `rolling(...).apply(...)`
    with a rank function on purpose: the latter is orders of magnitude slower
    across a 12-combo grid x ~48 folds.

  - Long entry where `ret_t > q_enter_t`, strict `>`. There is no short leg;
    -1.0 is never emitted. NaN compares False on both sides of `>`, so
    unwarmed bars fail closed with no separate validity mask.

Exit — the only thing changed from the previous iteration
---------------------------------------------------------
Iteration 36's **momentum-decay flip is retired entirely** (no second
quantile, no `EXIT_PCT`, no three-state entries series) and replaced by the
volatility-scaled hard stop of iteration 34, routed through
`session.apply_session_constraint_with_stops()`. Iteration 36 was the repo's
first clean leave-top-5-out (robust:true, PF_ex5 1.2615) and missed only the
fold-consistency gate, 25/45 = 55.6% against a >60% aim — two folds. So the
family is addressable, and the addressable defect is the *left tail*, not the
winners:

  - The decay statistic is measured over 96-384 bars (1-4 days) and therefore
    barely moves inside a single session, so it fires late. In iteration 36's
    in-sample leg 13 of 165 trades lost more than 1% and summed to about -26%
    against a +15.2% IS total return, most of them decay exits that only fired
    after -1.3% to -3.2% (01-26 -2.74, 09-02 -2.70, 10-14 -2.51, 12-13 -2.58,
    01-03 -2.49, 07-27 -2.27, 08-01 -3.18). The OOS leg has the identical
    shape: 11 of 185 trades beyond -1%, summing about -20%. This is an
    in-sample-visible structural leak, not an OOS-only artifact.

  - A winner cap is affirmatively the wrong fix. Leave-top-5-out already
    passes, so trade-level tail concentration is not the defect; the
    fold-level concentration is regime-driven (folds 42/3/15 run 8/8, 4/4 and
    6/6 — strings of consistent wins, not one outsized trade); and per
    iteration 35's arithmetic a profit target cannot move
    `total_return_ex_top5` or `profit_factor_ex_top5` by construction, only
    shrink the good folds. So **upside stays uncapped** and only the loss side
    is touched.

Materiality versus rejected iteration 34, which used the same daily-range
stop, rests on the formation grid rather than the stop fraction: 34 was
rejected on the *fast* grid 24-192, and iteration 36 showed the slow end
96-384 changes the family qualitatively (+2.3% -> +26.0% OOS, robust
false -> true). The stop x slow-grid combination has never been run. The
fraction is set to 0.5 — at or below 34's 0.6 — from the loss-tail evidence:
174 of 185 OOS trades ended inside +/-1%, so the stop is meant to sit just
outside the body, and `session.py`'s own caveat (the stop caps *when* you
exit, not the realized loss) means effective truncation runs above nominal.

  MEASURED, and stated rather than left for the evaluator to find: the
  specification quoted 0.5 x daily range as "a nominal ~1.0%", but measured on
  NQ 15min the median `stop_distance / Close` is **~1.75%**, not ~1.0%. The
  gap is the same one iteration 34 documented — a rolling 96-bar max-min
  straddles day boundaries and so runs systematically wider than a true
  calendar-day range. `STOP_FRAC` is implemented at the specified 0.5 (it is
  unambiguous in the exit rule and reaching ~1.0% would mean a fraction near
  0.29, i.e. a materially different and *tighter* stop than 34's 0.6, which
  the spec explicitly rules out). The consequence to carry into the
  evaluation: this stop sits well outside the +/-1% body, so it should
  truncate the far tail and touch few ordinary trades — which is the intended
  direction, but it also means the pre-registered "folds 2 and 8 flip" figure
  was computed against a tighter stop than the one actually running.

  The exit is nonetheless demonstrably *active*, not a no-op that quietly
  reproduces iteration 33: on a 2022-only NQ 15min walk at DEFAULT_PARAMS, 14
  of 67 long exits (21%) were stop hits and the other 53 were session
  flattens. That ratio is in the same region as iteration 34's 43-of-321 and
  is consistent with truncating a tail rather than managing the body.

The mechanics:

        daily_range_t = (High.rolling(96).max() - Low.rolling(96).min())
                            .rolling(960).mean().shift(1)
        stop_distance_t = STOP_FRAC * daily_range_t          # STOP_FRAC = 0.5
        target_price_t  = Close_t * 2.0

  - `target_price` is deliberately unreachable. The delegate requires a
    non-NaN target strictly above the entry Close or it refuses the entry;
    `Close * 2.0` satisfies that for any positive price and cannot be hit
    inside one session, so the profit side is governed only by the session
    flatten and winners run uncapped exactly as in iterations 33 and 36.

  - **Fail-closed now rests on `stop_distance` alone.** This is a real
    behavioural difference from iteration 34, which wrote the target as
    `close + 100 * stop_distance` so that a NaN stop propagated into a NaN
    target and `tgt_ok` was a second, redundant gate. `Close * 2.0` is always
    finite, so `tgt_ok` is always True and the delegate's
    `stop_dist_v[i] > 0 and not isnan(...)` check is the only thing standing
    between an unwarmed daily-range window and an entry. That check is
    sufficient (`np.nan > 0` is False, so a NaN stop refuses the entry), but
    it is single-legged now — do not remove or weaken it downstream.

  - Remaining exits are exactly two: the stop, and `session.py`'s forced
    end-of-session flatten. No profit target, and no trailing stop — the
    delegate reads `stop_distance` only at the entry bar and freezes the level
    for the life of the trade.

Pre-registered, honestly: direct truncation of the tail flips folds 2 and 8,
giving 27/45 = 60.0%, which does NOT clear a >60% aim on its own — 28/45 is
the target and the remaining margin has to come from the changed exit's effect
on the ~22 currently decay-exited trades. That is a mechanism, not a
guarantee. Churn diagnostic (per iteration 34): a trade count materially north
of ~250 against iteration 36's 185 means the stop is managing ordinary trades
rather than truncating the tail, and the run should be read as contaminated.
Iteration 34 produced 210 trades on this same architecture against iteration
33's 214, so churn is not the expected outcome — after an adverse move trips
the stop, `ret_t` itself has fallen and usually no longer clears the top
quantile, which is what makes re-entry self-limiting.

Vault grounding: momentum-strategies.md pitfall #1 ("when a trend breaks
suddenly, momentum strategies take large losses — this is the main driver of
drawdowns") and volatility.md's Stop-Loss Placement / "High volatility? Use
wide-stop momentum strategies" — which is why the stop is quoted in units of
the instrument's own recent range and auto-widens with regime rather than
being a fixed percentage.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`):

  - `formation_lookback` and `rank_window` are both plain `int` **on purpose**
    — both are genuine bar counts, so both are exactly what should size the
    pre-test-window warm-up buffer.
  - `rank_window` is fixed at 960, never grid-searched, and is the larger of
    the two: it drives `buffer_bars = max((960 + 5) * 3, day_bars + 5)` = 2895
    bars, comfortably covering the true requirement below. Do not raise it
    without re-checking that arithmetic *and* the fold-skip guard — see
    build_grid().
  - `rank_pct` is a quantile level in (0, 1), never a bar count, so it is
    passed as `float` and correctly ignored by the buffer sizing.
  - Bringing the stop back **reinstates a second warm-up leg that
    `_max_lookback_bars()` cannot see**: `DAILY_RANGE_BARS = 96` and
    `DAILY_RANGE_WINDOW = 960` are module constants, not grid params, so they
    have to be hand-checked rather than engine-derived. Doing that (iteration
    34's arithmetic, unchanged): `rolling(96)` first goes finite at bar 95,
    the strict 960-bar mean at 95 + 959 = 1054, and `.shift(1)` pushes it to
    **1056**. The signal leg needs `rank_window + max(formation_lookback)` =
    960 + 384 = **1344**. These are independent legs, so the binding
    requirement is the max, 1344 bars — still far inside the 2895-bar buffer,
    and unchanged from iteration 36 because the signal leg dominates. Nothing
    the engine keys off moves either: `max_lookback` is still `rank_window` =
    960 (384 < 960), so the buffer is still 2895 and the fold-skip guard is
    still `max_lookback + 10` = 970. **No new fold skips.**

  - `STOP_FRAC` (and the two daily-range windows) are module constants and are
    **never grid-searched**: they add no `build_grid()` / `DEFAULT_PARAMS`
    key, so the grid and param names stay byte-identical to iterations
    33/34/35/36 and the exit is the only variable in this comparison. That is
    deliberate given the family's history — iteration 10 added a zero-param
    exit change as its only edit and passed; iteration 27 promoted its exit
    level to a searched axis and the family's economics inverted.

Cost note: `metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is
unchanged and out of scope. The stop *adds* cost legs relative to a pure
session hold — a stopped-out session that re-enters pays two round trips where
the un-stopped version paid one — so the truncated tail has to be worth more
than that extra ~10.2 bps to show up as an improvement. The long-only
construction still interacts with the cost model favourably: a down-state
costs zero legs instead of two.

Fill-price caveat inherited from the delegate: a stop hit is *detected* off
that bar's Low against the stored level but the position is flattened at that
bar's Close, because this codebase has no intrabar series and
`metrics.bar_returns_with_costs` prices every bar off Close. A bar that spikes
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
# grid-searched — see the module docstring: half a normal day's range against
# the position is an economic reading, and 0.5 is read off iteration 36's loss
# tail (174 of 185 OOS trades ended inside +/-1%), not fitted.
STOP_FRAC = 0.5

# Multiplier on the entry Close used as the (deliberately unreachable) profit
# target. The delegate requires a non-NaN target strictly above the Close on
# every long entry bar; 2x price inside one session is not attainable, so this
# reproduces iterations 33/36's "no target — winners run to the session
# flatten". See the docstring: unlike iteration 34's stop-anchored target,
# this one is always finite, so `stop_distance` is the sole fail-closed gate.
UNREACHABLE_TARGET_MULT = 2.0


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
    fail closed downstream. Called once per run, for the entry level — the
    second call at the decay quantile went away with iteration 36's exit.
    """
    w = int(rank_window)
    return ret.shift(1).rolling(w, min_periods=w).quantile(float(rank_pct))


def trailing_daily_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Average one-day high-low span over the last `DAILY_RANGE_WINDOW` bars.

    The inner `rolling(DAILY_RANGE_BARS)` extremes give a rolling 24h range;
    the outer strict-`min_periods` mean smooths it over ~10 sessions, and the
    `.shift(1)` keeps the current bar out of the range that sizes its own
    stop. Unwarmed bars stay NaN so they fail closed in the delegate. First
    finite value at bar 1056 — hand-checked, since these windows are module
    constants and therefore invisible to `wfo_engine._max_lookback_bars()`.
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
    # the delegate resolves it to the long side and freezes the level at
    # entry (so this is a hard stop, not a trailing one). NaN while the
    # daily-range window is unwarmed, which makes the delegate refuse the
    # entry — and, since the target below is always finite, this is the
    # *only* thing that makes it refuse.
    stop_distance = STOP_FRAC * trailing_daily_range(high, low)

    # No real target: a finite level strictly above the entry Close (so the
    # delegate's tgt_ok / target > close guard passes) but unreachable inside
    # one session, so the profit side is governed only by the session flatten
    # and winners stay uncapped.
    target_price = close * UNREACHABLE_TARGET_MULT

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
    # Same four keys, same types, same values as iteration 36 — the exit is
    # the only thing that changed this iteration, and it adds no param.
    "formation_lookback": 192,
    "rank_pct": 0.875,
    "rank_window": 960,
    "session": "New York",
}
