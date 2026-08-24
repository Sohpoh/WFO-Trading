# Loop review: fixing the accept gate (gate v2)

Written after iteration 20. Two problems were diagnosed in the iteration history:
(1) the loop spends its idea-budget too narrowly (one family — `overnight_range_breakout`
— consumed 9 of the first 20 iterations, ES was tried once, Asia/1h/4h/1d never), and
(2) the accept gate itself was passing runs that shouldn't have passed. This doc covers
the fix for (2), since it has to come first — widening the search on top of a broken
gate just produces false positives faster.

## What was wrong

The old gate's primary metric was `OOS CAGR ÷ Retail-IS CAGR` ("efficiency ratio"),
banded per `walk-forward-optimization.md` (>0.7 strong, >0.5 acceptable, <0.3 reject).
Two defects:

1. **Retail-IS is a single whole-period curve fit, not a walk-forward comparator.**
   When its CAGR is small, dividing by it explodes or flips sign. All three ever-accepted
   runs (iterations 6, 10, 11) had a ratio **above 1.0** — never the 0.5–0.9 band a real
   edge with normal OOS degradation would show. 9 of 18 non-error iterations produced
   `efficiency_ratio: null` outright (IS CAGR ≤ 0), so the "primary" metric was undefined
   half the time.
2. **No robustness check on trade concentration.** Nothing tested whether an accepted
   run's edge survived removing its best few trades.

## Retroactive check: none of the 3 accepted strategies survive gate v2

**Leave-top-5-out** (drop the 5 best OOS trades, recompute total return and profit
factor on what's left — computed directly from each iteration's own `oos_trades.csv`,
no new data needed):

| iteration | OOS trades | full total return | full PF | ex-top-5 total return | ex-top-5 PF |
|---|---|---|---|---|---|
| 6  | 183 | +18.52% | 1.362 | +2.71%  | 1.053 |
| 10 | 180 | +13.76% | 1.269 | -2.04%  | 0.960 |
| 11 | 152 | +18.50% | 1.443 | +2.69%  | 1.065 |

Every one collapses toward breakeven (or flips negative) once its best 5 trades (out
of 150-183) are removed. Under gate v2's Step 3, all three would be `rejected` with
`failure_mode: fragile-concentration`, not `accepted`.

**ES pseudo-holdout** (same accepted code, same grid, run unchanged against ES instead
of NQ — ES was used in only 1 of the first 20 iterations, so it's the closest thing to
untouched data available today without waiting for a real holdout window; imperfect
since ES and NQ are correlated, but genuinely not fit to):

| iteration | ES OOS Sharpe | ES OOS PF | ES OOS CAGR | ex-top-5 PF on ES |
|---|---|---|---|---|
| 6  | 0.564  | 1.290 | +1.99% | 0.910 |
| 10 | -0.080 | 0.965 | -0.29% | — (already failing) |
| 11 | -0.276 | 0.876 | -0.82% | — (already failing) |

Iteration 6 is the closest to holding up (positive across the board, Sharpe just under
the 0.8 aim) but still fails leave-top-5-out on ES too. Iterations 10 and 11 fail
outright. Read together with the leave-top-5-out table, this says the entire
`overnight_range_breakout` family's three "accepted" results were driven by a small
number of outsized trades on one instrument, not a repeatable process — the family that
consumed 9 of 20 iterations and both of the loop's prior pivotal decisions didn't
actually have a validated edge at any point.

**These three log entries are left unmodified** (`status: "accepted"`, no `gate_version`
field — they're the historical record of what gate v1 concluded at the time). Going
forward, entries carry `"gate_version": "v2"`, `confirmed`, and `oos_peeked` so a later
reader can tell which gate produced which verdict without re-deriving it.

## What changed (gate v2)

1. **`wfo_engine.py` / `cli.py`** — each fold now reports its own OOS trade count,
   return, and Sharpe (`fold_table.csv` gained `oos_trades`/`oos_return`/`oos_sharpe`
   columns; `oos_trades.csv` gained a `fold` column). This replaces what the evaluator
   used to do by hand — bucketing trades by `entry_time` against fold windows and
   calling it "approximate" — with an exact, free-standing statistic.
2. **`wfo-evaluator.md`** — rewritten:
   - Step 2 (fold-level consistency: % of active folds profitable, median fold OOS
     return/Sharpe) is now primary, and it's level-based — no ratio, no fragile
     denominator.
   - Step 3 (leave-top-5-out) is a new **hard gate**: fails closed regardless of how
     good every other number looks. This is specifically the check the old gate never
     ran, and specifically what would have caught all three false accepts above.
   - The efficiency ratio survives as Step 5, explicitly demoted to a reported
     diagnostic that never decides the verdict.
   - New `failure_mode: fragile-concentration` for "passes fold-consistency and the
     absolute checklist, fails leave-top-5-out" — this is exactly what iterations 6,
     10, and 11 would each get today.
3. **`iterate-strategy.md`**:
   - A **holdout convention**: every exploration-phase `cli.py` run is capped at
     `--date-to 2024-12-31`. `2025-01-01` onward is reserved.
   - A **Step 7b confirmation run**: any `status: accepted` verdict is provisional
     until re-tested, unchanged grid and params, on the 2025 holdout via a second,
     independent `wfo-evaluator` invocation. `confirmed: true` only if that second
     verdict is itself `accepted`. This is mandatory now that the log has more than 5
     entries.
   - A mechanical **`oos_peeked`** flag: `true` whenever `source` is
     `"manual_cli_sweep"` (iteration 11's case) or similar — grid/params chosen by
     comparing prior OOS results against each other rather than by fold-internal
     evidence. `oos_peeked: true` blocks `confirmed` from ever being set by the
     exploration run alone; only Step 7b can confirm it.

## What this doesn't fix (out of scope here)

- **Idea-breadth** (the original complaint this review started from — family
  concentration, symbol/session/timeframe coverage gaps) is a separate, still-open
  problem. Fixing the gate first was deliberate: widening search before this would have
  just produced more `fragile-concentration` false accepts, faster.
- **Formal multiple-testing correction.** The evaluator is deliberately memoryless (no
  log access) so it structurally can't track "this is look #23 against this dataset."
  The mandatory Step 7b holdout confirmation above 5 entries is the practical stand-in,
  not a real Bonferroni-style adjustment — there isn't one implemented here.
- **Position-sizing / ensemble methods** that could make a `fragile-concentration`
  signal useful as one component of a portfolio rather than a standalone reject. The
  engine's `{-1,0,1}` single-position contract (see `CLAUDE.md`) forecloses this;
  changing it is a bigger redesign than this review covers.

## Reference runs behind the tables above

`results/pseudo-holdout-es-iter006/`, `results/pseudo-holdout-es-iter010/`,
`results/pseudo-holdout-es-iter011/` — each iteration's exact accepted code
(`git show <that iteration's commit>:strategy.py` etc.), run unchanged against
`--symbol ES` with that iteration's own grid. Not part of the iteration log itself
(no `idea_key`, wasn't produced by the researcher/implementer loop) — kept as the
evidence backing this doc.
