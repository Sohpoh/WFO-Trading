"""Overnight gap fade with MA-reclaim confirmation, target = prior-day anchor.

Invented variation of the worked example in strategy-development.md ("index
gaps against the prior close and mean-reverts as institutions re-balance"),
adapted to this codebase's hard constraint that `strategy.py` may have no
session awareness of its own.

Rules:

  - Anchor, computed on the full continuous df with no session awareness:
    `day = index.normalize()` (UTC calendar day), then
        anchor = Close.groupby(day).last().shift(1)   # prior UTC day's close
    broadcast back onto every bar of the current day. The `.shift(1)` is over
    the *day* series (one row per UTC day actually present), so it never leaks
    the current day's own close — using `groupby(day).transform("last")`
    instead would be straight lookahead. Weekends/holidays fall out naturally:
    Monday's anchor is the last print of UTC Sunday (the ~19:45 ET Sunday
    Globex bar), since no Saturday rows exist.
  - Displacement from that anchor:
        gap_pct = (anchor - Close) / anchor
    Positive means price sits *below* the anchor (gapped down) — the fade of
    that is a long. Negative means price sits above it — the fade is a short.
  - Confirmation is an MA *reclaim*, not a level condition:
        ma = Close.rolling(confirm_ma).mean()
        long_signal  = (gap_pct >=  min_gap_pct) & (Close > ma) & (prev Close <= prev ma)
        short_signal = (gap_pct <= -min_gap_pct) & (Close < ma) & (prev Close >= prev ma)
    The crossing form is load-bearing, exactly as in the previous iteration:
    `session.apply_session_constraint_with_stops()` skips same-bar re-entry
    after a stop (`continue`) but re-evaluates on the very next bar, so a
    persistent *level* condition (the vault's original "RSI < 30") would
    re-open the same trade bar after bar for as long as the level held. A
    crossing fires once per turn. It also means the strategy never fades into
    an active move — it waits for price to actually turn back toward the
    anchor.
  - Long and short are mutually exclusive by construction: `min_gap_pct > 0`,
    so `gap_pct` cannot be both >= +min_gap_pct and <= -min_gap_pct.
  - No flip on an opposite signal: the walk only opens when flat, so an
    opposing signal while in a trade is a no-op.

Exits are path-dependent and owned entirely by session.py's walk:

  - target_price = `anchor` itself — the literal "take profit at yesterday's
    close", and zero-param. Unlike the previous iteration's symmetric
    R-multiple target this needs no per-side `np.where`: it is already
    direction-resolved by construction, since `gap_pct >= min_gap_pct > 0`
    implies `anchor > Close` (valid long target) and the short branch
    symmetrically implies `anchor < Close`. That satisfies the walk's
    side-validity check without any explicit split.
  - stop_distance = `stop_gap_frac * (anchor - Close).abs()` — a fraction of
    *that trade's own reward leg* rather than an independent volatility
    multiple. Realized R:R is therefore exactly `1 / stop_gap_frac` on every
    trade regardless of volatility regime, which keeps the searched param
    scale-invariant and fold-to-fold comparisons interpretable.
  - Plus the forced flatten on the session's last bar.

Cost note: `metrics.py` charges ~0.102% round-trip. The reward leg equals the
entry bar's displacement to the anchor, which is `>= min_gap_pct` by
construction, so even the smallest grid value (0.35%) clears round-trip cost
by roughly 3.4x before any win-rate assumption.

Scope note on *when* this fires: the walk evaluates entries on every
in-session bar where it is flat, and `gap_pct` is recomputed off the current
bar's Close — so the qualifier is "price is currently at least `min_gap_pct`
away from the prior-day anchor", which can be true mid-session as easily as at
the open. This is therefore a persistent-displacement fade rather than a
strictly opening-gap fade. Freezing the measurement to the session's first bar
would require this module to know where 09:30 ET is, which CLAUDE.md's
Day-Trade Session Requirement forbids.

The UTC calendar day rolls at 19:00/20:00 ET, i.e. mid-Globex-evening, so by
the time the New York session opens the anchor is already the *prior* evening's
final print and has been stable for the whole overnight. Unlike the previous
iteration's day-anchored range there is no per-day dead zone: the anchor is a
scalar broadcast (fully formed on each day's first bar, not an accumulating
window) and `ma` is a continuous rolling window that does not reset at the day
boundary. The only warm-up is at the very start of the frame — the first UTC
day (NaN anchor) and the first `confirm_ma` bars — which the engine's
`buffer_bars` floor already covers.

This module only decides *when the strategy wants to enter and where its
stop/target sit*; it has no session awareness of its own. See
`session.apply_session_constraint_with_stops()`'s docstring for why the
path-dependent walk and the session-boundary flatten live there.

Caveat carried from session.py: stops/targets are *detected* off intrabar
High/Low but the exit still prices at that bar's Close (this app has no
finer-than-bar price series and `metrics.py` costs everything off Close), so
a stop does not cap the realized loss on the triggering bar — see
`apply_session_constraint_with_stops()` for details before describing
results as risk-capped.
"""
import pandas as pd

from session import apply_session_constraint_with_stops


def compute_prior_day_anchor(df: pd.DataFrame) -> pd.Series:
    """Prior UTC calendar day's last Close, broadcast to every bar of the
    current day. NaN on the first day of the frame.

    The shift is over the per-day series (never `transform("last")`, which
    would leak the current day's own close) — see module docstring.
    """
    day = df.index.normalize()
    prior_day_close = df["Close"].groupby(day).last().shift(1)
    return pd.Series(prior_day_close.reindex(day).to_numpy(), index=df.index)


def generate_positions(
    df: pd.DataFrame,
    min_gap_pct: float,
    stop_gap_frac: float,
    confirm_ma: int,
    session: str | None = "New York",
) -> pd.Series:
    close = df["Close"]
    anchor = compute_prior_day_anchor(df)

    # positive = price below the anchor (gapped down, fade is a long)
    gap_pct = (anchor - close) / anchor
    ma = close.rolling(confirm_ma, min_periods=confirm_ma).mean()

    prev_close, prev_ma = close.shift(1), ma.shift(1)
    # Crossing (reclaim), not level — one entry per turn (see module docstring).
    long_signal = (
        (gap_pct >= min_gap_pct) & (close > ma) & (prev_close <= prev_ma)
    ).fillna(False)
    short_signal = (
        (gap_pct <= -min_gap_pct) & (close < ma) & (prev_close >= prev_ma)
    ).fillna(False)

    # Reward leg = distance to the anchor; the stop is a fixed fraction of it,
    # so every trade has the same R:R (1 / stop_gap_frac) by construction.
    stop_distance = stop_gap_frac * (anchor - close).abs()

    # Already direction-resolved: anchor > close on a long signal, anchor <
    # close on a short one, so no per-side np.where is needed to satisfy the
    # walk's side-validity check.
    target_price = anchor

    return apply_session_constraint_with_stops(
        close=close,
        high=df["High"],
        low=df["Low"],
        long_signal=long_signal,
        short_signal=short_signal,
        stop_distance=stop_distance,
        target_price=target_price,
        session=session,
    )


DEFAULT_PARAMS = {
    "min_gap_pct": 0.005,
    "stop_gap_frac": 0.75,
    "confirm_ma": 10,
    "session": "New York",
}
