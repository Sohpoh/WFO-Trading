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
| 4 | 2026-08-18 | Multi-Day Range Breakout, Flip Exit (NQ 15min, no stop/target) | invented_variation_of:overnight_range_breakout\|day_anchored_range_atr_stop_r_target | error | n/a | n/a | n/a | `ad0b3ca` |
| 5 | 2026-08-18 | Rolling N-Bar Range Breakout, Flip Exit (NQ 15min, no stops) | invented_variation_of:overnight_range_breakout\|day_anchored_range_atr_stop_r_target | rejected | 0.32 | 0.4845 | 285 | `9df5124` |
| 6 | 2026-08-18 | Rolling Range Breakout + Volatility-Regime Gate (NQ 15min, zero-param ATR-ratio filter) | invented_variation_of:volatility.md | accepted | 0.86 | 1.5355 | 183 | `2ca6297` |
| 7 | 2026-08-20 | Vol-Normalized Intraday Drift Momentum (NQ 15min, t-stat entry, flip exit, iter-6 vol gate retained) | invented_variation_of:momentum-formulas.md | rejected | -0.30 | -2.5687 | 235 | `7e2200b` |
| 8 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate, Wide-Buffer Grid (NQ 15min, buffer_frac pushed above the old 0.15 ceiling) | invented_variation_of:overnight_range_breakout\|flip_exit_vol_regime_gate_atr_ratio | rejected | 0.40 | 0.6348 | 100 | `9f67e84` |
| 9 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate + Variance-Ratio Persistence Gate (NQ 15min, zero-param VR>1 chop filter) | invented_variation_of:statistical-mean-reversion-tests.md | rejected | -0.32 | -0.952 | 74 | `e16ad3c` |
| 10 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate, Failed-Breakout Stop (NQ 15min, stop = half the channel width, no target) | invented_variation_of:overnight_range_breakout\|flip_exit_vol_regime_gate_atr_ratio | accepted | 0.73 | 1.3931 | 180 | `d34726e` |
| 11 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate, Failed-Breakout Stop, Narrow-Buffer Grid (NQ 15min, buffer_frac shifted to 0.05-0.20) | manual_cli_sweep | accepted | 1.05 | 2.0276 | 152 | `956324d` |
| 12 | 2026-08-21 | With-Trend Pullback Reclaim (NQ 15min, trend-gated z-dip re-entry, zero-param sigma stop, no target) | invented_variation_of:mean-reversion.md | rejected | -1.72 | n/a (IS CAGR negative) | 444 | `d47c0d7` |
| 13 | 2026-08-21 | Quiet-Session Band Rejection Fade (NQ 5min London, Donchian wick rejection, channel-fraction target) | invented_variation_of:volatility.md | rejected | -3.12 | n/a (IS CAGR negative) | 249 | `PENDING` |
