# Strategy Iteration Log

Human-readable mirror of `log.jsonl` (the machine-readable record the
`strategy-researcher` agent reads to avoid repeating past ideas). One row per
`/iterate-strategy` run. `status: error` means the run itself was unusable (broken
strategy edit, or a grid that found nothing tradeable) — distinct from `rejected`,
which means the run was usable but didn't clear the acceptance bar.

| # | Date | Idea | Source | Status | Robust | Confirmed | OOS Sharpe | Efficiency (diagnostic only) | Trades | Commit |
|---|------|------|--------|--------|--------|-----------|------------|-------------------------------|--------|--------|
