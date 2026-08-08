# WFO-Trading

## Overview
A Streamlit dashboard that walk-forward-tests a pluggable trading strategy on NQ/ES futures, to show the difference between a curve-fit "retail" backtest and honest rolling walk-forward out-of-sample performance. The strategy itself is expected to change often — see `strategy.py` below.

## Core Commands
All work happens in the `WFO` conda env (has its own Python; nothing is installed in base/pyenv).

```bash
# install/update deps
/opt/anaconda3/envs/WFO/bin/python -m pip install -r requirements.txt

# run the UI
/opt/anaconda3/envs/WFO/bin/python -m streamlit run app.py

# run headless from the command line (same engine, no browser)
/opt/anaconda3/envs/WFO/bin/python cli.py --help
/opt/anaconda3/envs/WFO/bin/python cli.py --list                       # show local symbols/timeframes
/opt/anaconda3/envs/WFO/bin/python cli.py --symbol NQ --timeframe 1h --out-dir results/run1
```
No test suite or linter is configured yet — there's nothing to run beyond the app/CLI themselves. If you add tests, wire the command in here.

## Architecture
- `data_loader.py` — loads OHLCV bars. Two sources: local CSVs under `Data/<SYMBOL>/<timeframe>/.../*.csv` (Databento-style continuous futures export, columns `ts_event,open,high,low,close,volume`), and yfinance as a fallback for arbitrary tickers. Both paths normalize to a tz-aware (UTC) DatetimeIndex with `Open/High/Low/Close/Volume` columns. Cached with `st.cache_data`.
- `strategy.py` — the trading logic, isolated on purpose since this is the file that changes most often as strategies are swapped/iterated. `generate_positions(df, **params)` is the contract: a pure function taking OHLCV + strategy params in, returning a position series (-1/0/1, short/flat/long) out. Whatever indicators/entry-exit rules define "the strategy" live entirely here — nothing elsewhere should assume specific indicators or params. Internally it builds a raw `entries` series and hands it to `session.apply_session_constraint()` — it does not implement session logic itself.
- `session.py` — day-trade/session enforcement, deliberately split out of `strategy.py` so it survives strategy rewrites untouched. See **Day-Trade Session Requirement** below.
- `metrics.py` — cost model (fees + slippage) and all performance stats (Sharpe, CAGR, drawdown, trade-level win rate/profit factor). `bar_returns_with_costs()` is the single place trading costs get applied.
- `wfo_engine.py` — the rolling walk-forward loop (fold generation, grid-search optimization on train windows, OOS stitching) plus the single-shot "retail" in-sample comparison run. `build_grid()` is where searched-vs-fixed params get assembled into the param dicts passed to `strategy.generate_positions()` — it doesn't know or care what those params mean. No Streamlit imports here — keep it UI-agnostic.
- `app.py` — Streamlit UI only: sidebar controls, calls into `wfo_engine`, renders charts/tables. Business logic doesn't belong here. Sidebar inputs for strategy params should stay generic enough to swap out when `strategy.py`'s params change.
- `cli.py` — headless argparse entry point, the non-interactive twin of `app.py`. Calls the exact same `wfo_engine`/`metrics`/`session` functions with the same param shapes, so results always match the UI for equivalent inputs — it must NOT duplicate or reimplement any business logic, only parse args and call into the shared modules. Prints a comparison table to stdout; `--out-dir` optionally writes `fold_table.csv`, `oos_trades.csv`, `insample_trades.csv`, `equity_curves.csv`, `comparison.csv`. When `app.py` gains a new sidebar control backed by a new engine param, give `cli.py` a matching flag in the same change.

Rule of thumb: new indicators/entry-exit rules → `strategy.py`; new performance metrics → `metrics.py`; changes to how folds are built/optimized/stitched → `wfo_engine.py`; anything about what's displayed or configurable → `app.py` (interactive) and `cli.py` (scriptable) together.

## Day-Trade Session Requirement (non-negotiable — lives in `session.py`, not `strategy.py`)
This app trades day trades only, gated to one configurable session, **regardless of what the strategy logic itself is**. This is why it's a separate module: `strategy.py` gets rewritten often, `session.py` should not need to change when it does.

- `SESSION_CONFIG` (dict of session name → `{tz, start, end}` local trading hours), `session_mask(index, session)` (bool Series, `True` inside that session's local hours, DST-safe via `tz_convert` — never hardcode UTC hour offsets), and `apply_session_constraint(entries, session)` all live in `session.py`.
- Any `generate_positions(df, **params)` in `strategy.py`, no matter how it's rewritten, must:
  1. Build a raw `entries` series — 1.0/-1.0 at bars where it wants to open a long/short, `NaN` elsewhere — with **no session awareness of its own**.
  2. Return `apply_session_constraint(entries, session)` as the final position. That call is what gates entries to the session and force-flattens on the session's last bar; don't reimplement that logic inline in the strategy.
  3. Keep accepting a `session: str | None` param, where `None` means no constraint — this is the bypass `app.py` uses for 1d bars (session filtering is meaningless there since there's only one price per day).
- `app.py`'s "Day-Trade Session" sidebar selectbox (options from `session.SESSION_CONFIG`) and its `timeframe == "1d"` → `session=None` bypass must keep working.
- `wfo_engine.build_grid()` must keep threading `session` through as a fixed (never grid-searched) param into every combo dict.

If a new strategy's signals aren't simple boolean entry conditions (e.g. continuous position sizing), the *shape* of the constraint still applies — extend `apply_session_constraint()` in `session.py` to handle it, rather than adding session logic to the strategy.

## Position Contract (single position only — also applies across strategy rewrites)
`generate_positions()` must return one scalar position value per bar, restricted to `{-1, 0, 1}` (short/flat/long). There is no notion of position *size* in this codebase:

- Never more than one unit long or short at a time — no pyramiding/scaling in.
- Never simultaneously long and short.
- A repeat signal in the same direction while already in that position must be a no-op (not additive exposure).

This isn't enforced by a runtime check — it's a consequence of `entries`/`position` being a single `pd.Series` of scalars, and `metrics.extract_trades()` assumes exactly this shape (it walks position *changes*, not a stack of open trades). A rewritten strategy stays compliant automatically as long as it keeps building one `entries` value per bar and returning `apply_session_constraint(entries, session)` as-is. Adding real position sizing/pyramiding would require redesigning `position` (e.g. to a signed size instead of `{-1,0,1}`) and updating `metrics.py`'s trade extraction and cost model to match — don't do this incidentally while iterating on entry/exit rules.

## Gotchas & Constraints
- **Costs are hardcoded** in `metrics.py` (`FEE_RATE = 0.001%`, `SLIPPAGE_RATE = 0.05%`) and charged per transaction *leg* — a flat→position transition is 1 leg, a long↔short reversal is 2 (close+open). If you change the cost model, update it there, not in the engine.
- **`build_grid()` in `wfo_engine.py` decides which strategy params get grid-searched vs. held fixed** — it's written for whatever the current strategy's params are, so it needs updating in lockstep whenever `strategy.py`'s param list changes (new/renamed/removed params). Don't let it silently drop a param the strategy now expects.
- **Test-window indicators get a warm-up buffer** pulled from *before* the test window (`_simulate_window` in `wfo_engine.py`) so indicators aren't cold at the start of every fold — but only bars inside `[test_start, test_end)` count toward OOS P&L. Don't "simplify" this back to computing indicators fresh on the test slice alone; it reintroduces a warm-up bias. The buffer size is derived from the strategy's params (e.g. lookback-style values) via `build_grid()`'s output — keep that derivation generic, not tied to a specific param name.
- **Local data is NQ/ES only**, 2022-01 through 2025-12, timeframes 1min–1d. Default is NQ @ 1h. Finer timeframes (1min/5min) work but multiply the grid-search cost — expect noticeably slower runs.
- **All three sessions in `SESSION_CONFIG` are quoted in Eastern Time** (`America/New_York`), not each region's own local exchange hours — "London" and "Asia" mean the ET windows when that region's activity typically shows up for a US-based NQ trader, not London/Tokyo cash-session hours. Don't "fix" this by swapping in `Europe/London`/`Asia/Tokyo` tz's; it was changed to ET deliberately. A session whose `end` is `"00:00"` (e.g. Asia) is handled as "runs to midnight, no upper bound" in `session_mask()` — extend that special case if you add another midnight-ending window.
- **Streamlit's default file watcher can miss edits to non-entrypoint modules** (`strategy.py`, `session.py`, `wfo_engine.py`, etc.) and rerun with stale cached imports — install/keep `watchdog` (already in `requirements.txt`) and if you see an `ImportError` for something you just added, fully restart the `streamlit run` process rather than assuming the edit is wrong.
- **Streamlit reruns the whole script on every widget interaction.** Results are only recomputed on the "Run Walk-Forward Analysis" button and cached in `st.session_state["results"]`; don't move expensive calls outside that gate.
- No `.git` repo currently exists in this directory.
