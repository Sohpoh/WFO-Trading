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

| iteration | OOS trades | full total return | full PF | ex-top-5 total return | ex-top-5 PF | gate v2 `robust` |
|---|---|---|---|---|---|---|
| 6  | 183 | +18.52% | 1.362 | +2.71%  | 1.053 | borderline |
| 10 | 180 | +13.76% | 1.269 | -2.04%  | 0.960 | **false** (sign flip) |
| 11 | 152 | +18.50% | 1.443 | +2.69%  | 1.065 | borderline |

Iteration 10 fails outright (a straight sign flip once its best 5 trades are removed).
Iterations 6 and 11 don't cleanly fail the letter of the check — profit factor stays
just above 1.0 — but both collapse from a strong-looking ~1.4 down to barely-breakeven
territory, which is exactly what gate v2's `"borderline"` tier exists to name rather
than either wave through as clean passes (the original draft of this check called all
three outright failures; that overstated it — see the calibration note in
`wfo-evaluator.md` Step 3 for why a flat "PF < 1.0" line was replaced with this 3-way
verdict). None of the three gets a clean `robust: true`. Combined with Step 2/4, gate
v2 would not have accepted any of them without a lot more supporting margin than they
actually have — but only iteration 10 is a hard, unambiguous reject on this check
alone.

**ES pseudo-holdout** (same accepted code, same grid, run unchanged against ES instead
of NQ — ES was used in only 1 of the first 20 iterations, so it's the closest thing to
untouched data available today without waiting for a real holdout window; imperfect
since ES and NQ are correlated, but genuinely not fit to):

| iteration | ES OOS Sharpe | ES OOS PF | ES OOS total return | ex-top-5 total return on ES | ex-top-5 PF on ES | gate v2 `robust` on ES |
|---|---|---|---|---|---|---|
| 6  | 0.564  | 1.290 | +7.73% | -2.40% | 0.910 | **false** (sign flip) |
| 10 | -0.080 | 0.965 | -1.08% | — | — | already fails Step 4 (Sharpe/PF/CAGR all negative or sub-bar) |
| 11 | -0.276 | 0.876 | -3.08% | — | — | already fails Step 4 |

Iteration 6 is the closest to holding up on ES on the absolute numbers alone (positive
Sharpe/PF/CAGR, Sharpe just under the 0.8 aim) — but its leave-top-5-out check flips
sign on ES too (+7.73% full → -2.40% ex-top-5), which is a hard `robust: false`
regardless of the PF number sitting just above 0.9. Iterations 10 and 11 fail the
absolute checklist outright on ES, before robustness is even the deciding factor. Read
together with the NQ leave-top-5-out table above, this says the entire
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

   Viability check before trusting this: ran iteration 6's exact grid against
   `--date-from 2025-01-01` (the full holdout window, 12/3-week schedule) — 13 folds,
   43 OOS trades. Thinner than the 65-fold/150+-trade exploration runs, but clears
   rule-of-30 and Step 2's under-powered floor; the confirmation step is viable, not
   structurally inert. An idea with a much longer `train_weeks` could still eat most
   of the 52-week holdout and produce too few folds — if that happens, shortening the
   schedule for the confirmation run specifically (and saying so) is reasonable; don't
   let it silently degrade into an unrunnable check.

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

---

# Cost model v2 (written after iteration 44)

Written after a cost-sensitivity audit found that the cost model, not idea
quality, was the dominant false-negative generator in the loop's history.

## What was wrong

`metrics.py` charged `FEE_RATE = 0.001%` + `SLIPPAGE_RATE = 0.05%` per leg —
a **10.2bp round trip ≈ $102 per contract round trip on NQ**. The slippage term
alone (0.05% ≈ 10 points ≈ $50/leg) is ~10-40x realistic for NQ/ES day trading
(0.25-1 tick = $1.25-2.50/leg; realistic round trip ~$5-15). The 22 of 38
`no-edge` rejections were, on inspection, mostly "edge < the 10.2bp toll," not
"no edge": iteration 38's own logged numbers already showed an 8.6bp gross edge
fully erased by the toll.

## Null-model calibration: the gate does NOT false-accept noise

`tools/null_calibration.py` pushes a no-skill coin-flip strategy (deterministic,
param-keyed, so the argmax-Sharpe grid search genuinely *selects* among noise
realisations) through the real walk-forward pipeline and scores it with the
gate-v2 replica in `tools/gate_v2.py`.

| cost | trials | false-accept rate | noise median OOS Sharpe | noise median PF | max PF ex-top-5 |
|---|---|---|---|---|---|
| v1 (10.2bp) | 20 | **0/20 (0.0%)** | -1.86 | 0.62 | 0.83 |
| v2 (1.2bp)  | 20 | **0/20 (0.0%)** | -0.46 | 0.90 | 1.10 |

Pure noise never reaches `robust:true` (PF_ex5 ≥ 1.15), never clears 60% fold
consistency, and never passes the absolute checklist — at either cost. The
strict gate stays as-is; it is not the thing blocking success.

## Cost-sensitivity + full re-runs: the real edges were hidden

`tools/cost_sensitivity.py` reconstructs gross returns from saved trades and
re-scores gate-v2 quantities at cost multipliers; `tools/rerun_report.py` runs
the full pipeline (from each iteration's own commit via `git archive`) at the
recalibrated cost so the fold-level optimizer re-selects under realistic costs.
All four candidates clear gate v2 at cost v2 (round trip 1.2bp ≈ $12/contract):

| iteration | old verdict (cost v1) | cost v2 OOS Sharpe | PF | PF ex-top-5 | folds profitable | verdict @ cost v2 |
|---|---|---|---|---|---|---|
| 6  | accepted (retro fragile) | 1.34 | 1.54 | 1.25 | 66.0% | accepted |
| 11 | accepted (retro fragile) | 1.89 | 1.89 | 1.48 | 69.8% | accepted |
| 36 | rejected (folds 55.6%)   | 1.78 | 1.73 | 1.48 | 67.4% | accepted |
| 40 | accepted (unconfirmed)   | 1.88 | 1.80 | 1.54 | 70.5% | accepted |

**Caveats**: iterations 6/11 predate the holdout convention (their grids were
mined on 2022-2025), so they have no untouched confirmation; and iteration 40's
2025 holdout still fails at cost v2 (OOS Sharpe -0.83, PF 0.71, 46.7% folds) —
it lost in-sample on 2025, so that collapse is **regime fragility, not a cost
artifact**. Cost fixes bookkeeping; it does not fix a strategy that dies in a
different market regime.

## What changed

- `metrics.py` — `SLIPPAGE_RATE` 0.05% -> **0.005%** (cost v2); `FEE_RATE`
  unchanged. Still conservative vs real limit-order fills.
- `CLAUDE.md`, `iterate-strategy/SKILL.md`, `strategy-implementer/SKILL.md` —
  documented the recalibration and the "never edit metrics.py per iteration" rule.
- New tools: `tools/gate_v2.py` (evaluator replica), `tools/null_calibration.py`,
  `tools/cost_sensitivity.py`, `tools/rerun_report.py`.

## What this doesn't fix (still open)

- **Regime-fragility** — the 2022-24 edges do not carry into 2025 (iteration 40's
  holdout fails even at cost v2). This is now the loop's primary open problem,
  not "find an edge." A per-calendar-year profitability check and ES
  pseudo-holdouts are the two diagnostics that localize it.
- **Engine contract** — the edges found are small and tail-dependent; the
  `{-1,0,1}` single-strategy contract is the least favorable vehicle for those.
  Sizing/ensemble remains a medium-term design decision, not an iteration-level knob.
