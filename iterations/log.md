# Strategy Iteration Log

Human-readable mirror of `log.jsonl` (the machine-readable record the
`strategy-researcher` agent reads to avoid repeating past ideas). One row per
`/iterate-strategy` run. `status: error` means the run itself was unusable (broken
strategy edit, or a grid that found nothing tradeable) — distinct from `rejected`,
which means the run was usable but didn't clear the vault's WFO acceptance bar.

| # | Date | Idea | Source | Status | OOS Sharpe | Efficiency | Trades | Commit |
|---|------|------|--------|--------|-----------|------------|--------|--------|
| 1 | 2026-08-08 | Bollinger Band Mean-Reversion Fade | wiki:mean-reversion.md | rejected | -3.55 | n/a (IS CAGR ≤ 0) | 477 | `f348dca` |
| 2 | 2026-08-08 | ATR-Normalized Momentum Thrust | wiki:momentum-formulas.md | rejected | 0.02 | 0.028 | 67 | `PENDING` |
