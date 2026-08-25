# Strategy Iteration Log

Human-readable mirror of `log.jsonl` (the machine-readable record the
`strategy-researcher` agent reads to avoid repeating past ideas). One row per
`/iterate-strategy` run. `status: error` means the run itself was unusable (broken
strategy edit, or a grid that found nothing tradeable) — distinct from `rejected`,
which means the run was usable but didn't clear the vault's WFO acceptance bar.

**Gate v2 (from iteration 21 onward)**: `Efficiency` (OOS/Retail-IS CAGR ratio) is
diagnostic only now, not the accept/reject decider — see `loop-review.md` for why.
`Robust` (leave-top-5-out check) and `Confirmed` (holdout re-test) are what actually
decide a verdict going forward. Rows 1-20 predate this change and show `—`/`n/a` in
those columns except iterations 6, 10, and 11, whose `Robust` values were computed
retroactively (see `loop-review.md`) — none of the three accepted runs get a clean
pass, and `Confirmed` is `no` for all three since none has been re-tested on the 2025
holdout.

| # | Date | Idea | Source | Status | Robust (v2 retro) | Confirmed | OOS Sharpe | Efficiency (diagnostic only) | Trades | Commit |
|---|------|------|--------|--------|--------------------|-----------|-----------|-------------------------------|--------|--------|
| 1 | 2026-08-08 | Bollinger Mid-Band Fade (ES 15min, sigma-scaled stop) | wiki:mean-reversion.md | rejected | — | n/a | -2.87 | n/a (IS CAGR negative) | 461 | `e4bdbba` |
| 2 | 2026-08-08 | Overnight-Range Breakout (NQ 15min, day-anchored range, ATR stop, R-multiple target) | invented_variation_of:es-futures.md / nq-futures.md | rejected | — | n/a | -1.25 | n/a (IS CAGR negative) | 810 | `f948984` |
| 3 | 2026-08-18 | Overnight Gap Fade with MA-Reclaim Confirmation (NQ 15min, target = prior-day anchor) | invented_variation_of:strategy-development.md | rejected | — | n/a | -1.52 | n/a (IS CAGR negative) | 358 | `2497ee9` |
| 4 | 2026-08-18 | Multi-Day Range Breakout, Flip Exit (NQ 15min, no stop/target) | invented_variation_of:overnight_range_breakout\|day_anchored_range_atr_stop_r_target | error | — | n/a | n/a | n/a | n/a | `ad0b3ca` |
| 5 | 2026-08-18 | Rolling N-Bar Range Breakout, Flip Exit (NQ 15min, no stops) | invented_variation_of:overnight_range_breakout\|day_anchored_range_atr_stop_r_target | rejected | — | n/a | 0.32 | 0.4845 | 285 | `9df5124` |
| 6 | 2026-08-18 | Rolling Range Breakout + Volatility-Regime Gate (NQ 15min, zero-param ATR-ratio filter) | invented_variation_of:volatility.md | accepted | borderline | no (unconfirmed under v2) | 0.86 | 1.5355 | 183 | `2ca6297` |
| 7 | 2026-08-20 | Vol-Normalized Intraday Drift Momentum (NQ 15min, t-stat entry, flip exit, iter-6 vol gate retained) | invented_variation_of:momentum-formulas.md | rejected | — | n/a | -0.30 | -2.5687 | 235 | `7e2200b` |
| 8 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate, Wide-Buffer Grid (NQ 15min, buffer_frac pushed above the old 0.15 ceiling) | invented_variation_of:overnight_range_breakout\|flip_exit_vol_regime_gate_atr_ratio | rejected | — | n/a | 0.40 | 0.6348 | 100 | `9f67e84` |
| 9 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate + Variance-Ratio Persistence Gate (NQ 15min, zero-param VR>1 chop filter) | invented_variation_of:statistical-mean-reversion-tests.md | rejected | — | n/a | -0.32 | -0.952 | 74 | `e16ad3c` |
| 10 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate, Failed-Breakout Stop (NQ 15min, stop = half the channel width, no target) | invented_variation_of:overnight_range_breakout\|flip_exit_vol_regime_gate_atr_ratio | accepted | **false** | no (unconfirmed under v2) | 0.73 | 1.3931 | 180 | `d34726e` |
| 11 | 2026-08-20 | Rolling Range Breakout + Vol-Regime Gate, Failed-Breakout Stop, Narrow-Buffer Grid (NQ 15min, buffer_frac shifted to 0.05-0.20) | manual_cli_sweep | accepted | borderline | no (unconfirmed under v2) | 1.05 | 2.0276 | 152 | `956324d` |
| 12 | 2026-08-21 | With-Trend Pullback Reclaim (NQ 15min, trend-gated z-dip re-entry, zero-param sigma stop, no target) | invented_variation_of:mean-reversion.md | rejected | — | n/a | -1.72 | n/a (IS CAGR negative) | 444 | `d47c0d7` |
| 13 | 2026-08-21 | Quiet-Session Band Rejection Fade (NQ 5min London, Donchian wick rejection, channel-fraction target) | invented_variation_of:volatility.md | rejected | — | n/a | -3.12 | n/a (IS CAGR negative) | 249 | `483af7a` |
| 14 | 2026-08-21 | London Volatility-Ignition Continuation (NQ 5min, expansion-bar entry, trigger-bar stop, no target) | wiki:volatility.md | rejected | — | n/a | -2.19 | n/a (IS CAGR negative) | 113 | `a84e141` |
| 15 | 2026-08-21 | London Overnight-Range Breakout -- session port of accepted iteration 11 (NQ 5min, vol-regime gate, failed-breakout stop) | invented_variation_of:overnight_range_breakout\|vol_gate_failed_breakout_stop_narrow_buffer_grid | rejected | — | n/a | -2.41 | n/a (IS CAGR negative) | 65 | `dcbbb22` |
| 16 | 2026-08-21 | London Failed-Breakout Reversal (NQ 5min, Donchian reclaim entry, plain flip exit, no stop) | invented_variation_of:london_overnight_range_breakout\|vol_gate_failed_breakout_stop_session_port_5min | error | — | n/a | n/a | n/a | n/a | `5970e71` |
| 17 | 2026-08-21 | London Overnight-Drift Hold (NQ 5min, sqrt(L)-scaled drift state signal, one round trip per session, plain flip exit) | invented_variation_of:momentum-strategies.md | rejected | — | n/a | -1.40 | n/a (IS CAGR negative) | 106 | `0e23a64` |
| 18 | 2026-08-21 | Relative-Volume Surge Continuation (NQ 15min NY, ET-slot-normalized volume trigger, trailing-return direction, plain flip exit) | invented_variation_of:es-futures.md | rejected | — | n/a | -1.10 | -5.04 | 65 | `2f4fac5` |
| 19 | 2026-08-21 | Relative-Volume Surge Continuation, Fold-Centered Grid (NQ 15min NY, rvol/thrust grids re-centered where the folds pointed, sparse tight corner deleted) | invented_variation_of:rvol_surge_continuation\|et_slot_rvol_trigger_trailing_return_direction_flip_exit | rejected | — | n/a | -1.27 | -4.84 | 200 | `5036337` |
| 20 | 2026-08-23 | Skip-Period Risk-Adjusted Drift Momentum (NQ 15min NY, drift t-stat measured with a hardcoded S=L/8 skip, vol gate, flip exit) | invented_variation_of:intraday_tsmom_risk_adjusted\|drift_tstat_entry_flip_exit_vol_gate | rejected | — | n/a | -1.36 | n/a (IS CAGR negative) | 323 | `d30a4c7` |
| 21 | 2026-08-24 | Variance-Ratio Regime Polarity Switch (NQ 15min NY, one z-score primitive whose sign is set by measured VR regime, sigma-scaled stop and bounded target) | invented_variation_of:regime-changes.md | rejected | no | n/a | -2.61 | n/a (IS CAGR negative) | 757 | `a2abe9f` |
| 22 | 2026-08-24 | Failed-Breakout Reversal (NQ 15min NY, N-bar break then close back inside, channel-width stop, bounded channel-width target) | invented_variation_of:mean-reversion.md | rejected | no | n/a | -2.84 | n/a (IS CAGR negative) | 946 | `7964d8d` |
| 23 | 2026-08-24 | Swing-Horizon Time-Series Momentum, Intraday Execution (NQ 15min NY, multi-week formation + skip period, quarter-horizon confirmation, flip exit) | wiki:momentum-strategies.md | rejected | no | n/a | -1.18 | n/a (IS CAGR negative) | 635 | `33fda06` |
| 24 | 2026-08-25 | Regression-Channel Reversion (NQ 15min NY, OLS-residual sigma entry, exit at the fitted line, fixed sigma stop) | wiki:mean-reversion.md | rejected | no | n/a | -3.03 | n/a (IS CAGR negative) | 383 | `5d0a21c` |
| 25 | 2026-08-25 | Trailing-Range Location Trend-Hold (ES 1h New York, stochastic-location state signal, one round trip per session, ATR stop) | invented_variation_of:transaction-costs.md | rejected | no | n/a | -2.10 | n/a (IS CAGR negative) | 589 | `dc1f1f9` |
| 26 | 2026-08-25 | VWAP-Deviation Continuation Cross (NQ 1h New York, volume-weighted level as the direction signal, ATR stop, target grid spanning bounded-to-uncapped) | invented_variation_of:momentum-strategies.md | rejected | no | n/a | 0.04 | 0.01 | 241 | `e640934` |
| 27 | 2026-08-25 | VWAP-Deviation Continuation, Risk-Anchored Exit (NQ 1h NY, stop promoted to the searched axis, target pinned at 2R) | invented_variation_of:vwap_deviation_continuation\|rolling_vwap_atr_dev_cross_entry_atr_stop_target_grid | rejected | no | n/a | -0.16 | -0.3037 | 268 | `cc70d1b` |
