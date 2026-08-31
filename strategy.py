"""Long-only overnight gap-down fill buy (NQ 15min, New York session).

The previous family (long-only dual-horizon moving-average trend state) is
retired as structural, not varied: its in-sample leg came back CAGR -7.99% at
PF 0.868 over 469 IS / 436 OOS trades, which fires both no-edge triggers. This
iteration is a deliberate, evidence-backed revisit of iteration 3's rejected
overnight-gap-fade family rather than a re-run of it — two named mechanisms
from that iteration are deleted outright.

What is deleted from iteration 3
--------------------------------
1. The gap-up SHORT half. Splitting ~2,000 OOS trades by direction across four
   unrelated families said the same thing every time: the short leg was the
   worse half. Under `metrics.py`'s per-leg cost model a no-signal day now pays
   zero legs rather than the two a short entry would cost. `short_signal` is
   False on every bar and -1.0 can never be emitted.
2. The MA-reclaim confirmation gate. Iteration 17 measured reclaim-style
   confirmation on a fade at roughly +1bp gross — worse than an unfiltered
   flip — so iteration 3's entry was crippled by a mechanism this log has
   since falsified. There is no confirmation filter here; the band itself is
   the whole entry condition.

The anchor — stated honestly, not hidden
----------------------------------------
`anchor_t` is the Close of the final bar of the *previous completed UTC day*,
broadcast forward across the current UTC day. Strictly backward looking: group
by UTC date, take the last Close of each date, shift one available day, map
back onto the bar index. Bars on the first date of the frame have no prior day
and stay NaN, which fails closed everywhere downstream.

es-futures.md's "gaps down: 60% fill within same session" (NQ 57%) is measured
against the 16:00 ET RTH close. This anchor is session-unaware and lands at the
end of the UTC day (~19:45/20:45 ET), so that figure is directional evidence
for the mechanism, NOT a base rate that transfers — this run has to stand on
its own walk-forward numbers.

A second honesty note on the same anchor: the shift is over *available* days,
not calendar days, so Monday's anchor is Sunday's last close. Sunday UTC days
carry only ~8 bars at 15min (132 such days in the local NQ frame) — a ~19:45 ET
print off roughly two hours of Globex, not a post-full-session close. That is
faithful to the specified anchor and is documented rather than "fixed".

Entry — a dense band, not a crossing
------------------------------------
    dev_t      = Close_t / anchor_t - 1          (signed, negative = gap down)
    gap_dist_t = anchor_t - Close_t              (positive inside the band)

    long_signal_t = (dev_t <= -min_gap_pct) AND (dev_t >= -BAND_CAP_MULT * min_gap_pct)

The upper band cap at `BAND_CAP_MULT` = 3.0x `min_gap_pct` is a module
constant, deliberately NOT a searched axis (same treatment as earlier
iterations' STOP_FRAC). It encodes mean-reversion.md pitfall #4 — news spikes
wipe out mean-reversion positions — so beyond 3x the threshold the move is
read as news, not as the overshoot being faded. It is also what bounds the
re-arm behaviour described below.

The signal is a dense STATE, not a crossing. The gap forms overnight, outside
the New York window, so a cross-only formulation would place nearly every
arming bar outside the session and produce almost no trades.

Exit — bounded target known at entry, gap-scaled stop
-----------------------------------------------------
Positive grounding is mean-reversion.md's short-term-overreaction / forced-
selling mechanism plus its documented Bollinger exit ("expecting reversion to
the middle. Exit at middle"), which supplies a target that is bounded and known
at entry — the frequent-small-wins shape that the gate's leave-top-5-out hard
check rewards, and the exact shape the uncapped run-to-flatten families keep
failing.

    target_price_t  = Close_t + target_frac * gap_dist_t
    stop_distance_t = stop_gap_frac * gap_dist_t

`target_frac` = 1.0 is the literal full fill back at the anchor; 0.75 is a
partial fill. Inside the band `gap_dist_t > 0` by construction, so the target
is always strictly above the entry Close and the delegate's
`target_price > close` guard is satisfied without a special case. The stop is a
positive from-entry distance (the delegate resolves it below entry for a long)
measured in the same gap-distance unit as the target, so both sides of the
trade are scale-invariant in the size of the gap being faded.

Everything path-dependent is routed through
`session.apply_session_constraint_with_stops()`; `session.py` also force-
flattens any survivor on the session's last bar. No trailing stop and no
scale-out — both are foreclosed by session.py's entry-bar-only stop level and
CLAUDE.md's {-1, 0, 1} position contract.

Re-arm behaviour, stated up front
---------------------------------
Because `long_signal` is a dense state and the delegate forbids only *same-bar*
re-entry, a stopped-out trade can re-enter on a later in-session bar while
price is still inside the band. Each re-entry costs 2 extra legs. The hardcoded
3x band cap is what bounds this to a few attempts in a monotone decline
instead of unlimited averaging-down: once price falls past 3x `min_gap_pct`
below the anchor, the state goes False and the strategy stops re-arming.
Measured here rather than asserted: on the full local NQ 15min frame at
`min_gap_pct` = 0.004 (the loosest grid point, so the worst case), the band is
in-session on 506 distinct days and the strategy takes 915 trades — ~1.8
entries per armed day, at the low end of the expected 2-4.

Cost discipline, designed into the grid rather than bolted on as a filter
------------------------------------------------------------------------
`min_gap_pct` floors at 0.40% and `target_frac` floors at 0.75, so the smallest
implied target is ~30bps against `metrics.py`'s ~10.2bps round-trip toll
(~2.9x, rising to ~3.9x at `target_frac` = 1.0). NQ is chosen over ES because
ES's smaller overnight moves would starve a 0.40% threshold.

`metrics.py`'s per-leg toll (0.001% fee + 0.05% slippage) is unchanged and out
of scope.

Param types / warm-up (see CLAUDE.md and `wfo_engine._max_lookback_bars()`)
---------------------------------------------------------------------------
  - ALL THREE searched params are `float`: `min_gap_pct` is a return
    threshold, `stop_gap_frac` and `target_frac` are fractions of the gap
    distance. None is a bar count, so none should size the warm-up buffer.
  - Consequently `_max_lookback_bars()` returns 0 for this grid. This inverts
    the hazard every previous iteration's docstring warned about: the
    fold-skip guard in `run_walk_forward()` drops to `len(train_df) < 10`,
    trivially met at every timeframe, so no fold is ever silently skipped and
    the old "always pass --timeframe 15min or you get zero folds" warning does
    NOT apply here.
  - The binding warm-up is therefore `_bars_per_day(df) + 5` alone — exactly
    the day-anchored floor that function exists for. Hand-checked (it is
    invisible to `_max_lookback_bars()`): the anchor's true requirement is
    "the final bar of the previous UTC day is inside the calc slice", i.e. at
    most (bars of the current day before `test_start`) + 1 = max bars in one
    UTC day. Measured on the local NQ frame (median / max bars per UTC day,
    then buffer vs. requirement): 1min 1380/1380 -> 1385 vs 1380; 5min 276/276
    -> 281 vs 276; 15min 92/92 -> 97 vs 92; 1h 23/23 -> 28 vs 23; 4h 6/6 -> 15
    vs 6. Every timeframe clears it, the intended 15min included.
  - That margin depends on median bars/day == max bars/day (it does on this
    data, at every timeframe above — re-measure before pointing this at
    another instrument or source). If it ever failed,
    the effect is a NaN anchor on the first bars of a test window -> no entry
    -> a few missed trades. It is a fail-closed degradation, never lookahead.

This module decides only *when* the strategy wants to be long, and at what
stop/target levels. All day-trade gating and the end-of-session flatten are
delegated to `session.py`; see its docstring for that contract.
"""
import numpy as np
import pandas as pd

from session import apply_session_constraint_with_stops

# Upper edge of the entry band, as a multiple of `min_gap_pct`. A module
# constant on purpose — the spec pins it as NOT a searched axis, so the exit
# band keeps zero degrees of freedom. Beyond this multiple the overnight move
# is read as news rather than as a fadeable overshoot (mean-reversion.md
# pitfall #4), which is also what bounds re-arming after a stop-out.
BAND_CAP_MULT = 3.0


def prior_day_close(close: pd.Series) -> pd.Series:
    """Close of the final bar of the previous completed UTC day, broadcast
    forward across every bar of the current UTC day.

    Strictly backward looking: the value attached to a bar on UTC day D is
    fully determined before day D's first bar prints. Bars on the first UTC
    day of the frame get NaN (no prior day exists yet) and fail closed.

    The shift is positional over the days actually present, so a Monday's
    anchor is the previous Sunday's last close, not the previous Friday's —
    see the module docstring for why that is documented rather than changed.
    """
    day = pd.Series(close.index.normalize(), index=close.index)
    daily_last = close.groupby(day).last()
    return day.map(daily_last.shift(1))


def generate_positions(
    df: pd.DataFrame,
    min_gap_pct: float,
    stop_gap_frac: float,
    target_frac: float,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    anchor = prior_day_close(close)

    # Signed deviation from the anchor (negative = gapped down) and the
    # positive price distance back up to it. NaN anchor -> both NaN, which
    # fails closed on every branch below.
    dev = close / anchor - 1.0
    gap_dist = anchor - close

    # Dense entry band: deep enough to clear the round-trip toll, but not so
    # deep that the move is news rather than an overshoot. A state, not a
    # crossing — the gap forms outside the New York window, so a cross-only
    # form would arm almost exclusively on untradable bars. NaN on either side
    # of a comparison is False, so an unwarmed bar stands aside with no
    # separate validity mask to keep in sync.
    with np.errstate(invalid="ignore"):
        long_signal = (dev <= -float(min_gap_pct)) & (dev >= -BAND_CAP_MULT * float(min_gap_pct))

    # Long-only: the delegate still wants a short side, so hand it one that is
    # never True. -1.0 can therefore never be emitted, and a gap-up day pays
    # zero cost legs instead of the two a short entry would cost.
    short_signal = pd.Series(False, index=df.index)

    # Bounded target known at entry: a fraction of the way back to the anchor
    # (1.0 = the literal full fill). Inside the band gap_dist > 0, so this is
    # always strictly above the entry Close and the delegate's
    # `target_price > close` guard is satisfied by construction.
    target_price = close + float(target_frac) * gap_dist

    # Gap-scaled hard stop: a positive from-entry distance in the same
    # gap-distance unit as the target, which the delegate resolves below entry
    # for a long and freezes at the entry bar (hard, not trailing).
    stop_distance = float(stop_gap_frac) * gap_dist

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
    # All three are values the intended grid actually searches
    # (min_gap_pct 0.004/0.006/0.009/0.013, stop_gap_frac 0.5/0.75/1.0,
    # target_frac 0.75/1.0), so the post-edit sanity check exercises a real
    # combo. All three are floats — none is a bar count; see the module
    # docstring's warm-up note.
    "min_gap_pct": 0.004,
    "stop_gap_frac": 0.75,
    "target_frac": 0.75,
    "session": "New York",
}
