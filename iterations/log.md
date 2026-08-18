# Strategy Iteration Log

Human-readable mirror of `log.jsonl` (the machine-readable record the
`strategy-researcher` agent reads to avoid repeating past ideas). One row per
`/iterate-strategy` run. `status: error` means the run itself was unusable (broken
strategy edit, or a grid that found nothing tradeable) — distinct from `rejected`,
which means the run was usable but didn't clear the vault's WFO acceptance bar.

| # | Date | Idea | Source | Status | OOS Sharpe | Efficiency | Trades | Commit |
|---|------|------|--------|--------|-----------|------------|--------|--------|
| 1 | 2026-08-08 | Bollinger Mid-Band Fade (ES 15min, sigma-scaled stop) | wiki:mean-reversion.md | rejected | -2.87 | n/a (IS CAGR negative) | 461 | `e4bdbba` |
| 2 | 2026-08-08 | Overnight-Range Breakout (NQ 15min, day-anchored range, ATR stop, R-multiple target) | invented_variation_of:es-futures.md / nq-futures.md | rejected | -1.25 | n/a (IS CAGR negative) | 810 | `f948984` |
| 3 | 2026-08-18 | Overnight Gap Fade with MA-Reclaim Confirmation (NQ 15min, target = prior-day anchor) | invented_variation_of:strategy-development.md | rejected | -1.52 | n/a (IS CAGR negative) | 358 | `2497ee9` |
