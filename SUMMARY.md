# WFO-Trading — Project Summary

A Streamlit dashboard that backtests a trading strategy on NQ/ES futures using **walk-forward optimization (WFO)**, so performance reflects how the strategy would actually behave live rather than how well its parameters happen to fit historical data.

## The core idea

Optimizing a strategy's parameters once on your entire dataset and trading that on the same dataset ("retail in-sample") tends to look great — because the parameters were chosen to fit exactly that data. It tells you little about future performance.

Walk-forward optimization instead splits history into repeating **train → test folds**:
1. Optimize parameters on a rolling *training* window (e.g. the trailing 12 weeks).
2. Apply those parameters, unchanged, to the following *blind test* window (e.g. the next 3 weeks) — data the optimizer never saw.
3. Slide both windows forward and repeat across the whole dataset.
4. Stitch together **only the blind-test returns** into one continuous out-of-sample equity curve.

The dashboard runs both approaches side by side (retail in-sample vs. walk-forward out-of-sample) so the gap between them — the "curve-fitting tax" — is visible directly.

## Pipeline, end to end

1. **Data** (`data_loader.py`) — loads OHLCV bars for NQ/ES from local historical CSVs, or falls back to yfinance for other tickers. Normalized to a timezone-aware DatetimeIndex.
2. **Strategy signals** (`strategy.py`) — turns price data + parameters into a raw entry signal (long / short / no signal) per bar. This is the file expected to change as strategies are iterated on — see **Structural rules** below for what must stay constant regardless.
3. **Day-trade / session constraint** (`session.py`) — takes the raw signal and restricts it to a chosen trading session, force-closing any open position by the end of that session. No strategy ever holds a position overnight.
4. **Costs & metrics** (`metrics.py`) — applies a fixed exchange fee + slippage cost on every trade, and computes performance stats (return, Sharpe, drawdown, win rate, profit factor) from the resulting returns.
5. **Walk-forward engine** (`wfo_engine.py`) — generates the rolling train/test folds, grid-searches strategy parameters on each training window, applies the winning parameters to that fold's blind test window, and stitches the out-of-sample results together. Also runs the single-shot "retail" in-sample comparison.
6. **Dashboard** (`app.py`) — sidebar controls for data source, strategy parameters, and walk-forward settings; a live progress bar while folds run; a Gantt chart of train/test windows; and side-by-side metrics + equity curves for the two approaches.

## Structural rules (independent of whatever the strategy currently is)

These are fixed properties of the app, not something any particular strategy opts into — they hold no matter how `strategy.py` is rewritten:

- **Day trades only.** Every position must open and close within a single trading session; nothing is ever held overnight. This lives entirely in `session.py`, separate from strategy logic, specifically so it isn't lost when the strategy changes.
- **One position at a time.** A strategy's output is a single position value per bar — short, flat, or long — never partial size, never pyramided, never long and short simultaneously.
- **Costs are non-negotiable.** A fee and slippage cost is deducted on every trade leg (an open, a close, or both on a reversal). No backtest result in this app is cost-free.
- **Out-of-sample means out-of-sample.** The walk-forward equity curve only ever includes returns from a fold's blind test window — a training window's own performance never leaks into the reported result.

## What's not here yet

No automated test suite or linter is configured — verification currently happens by running the app and checking the output. There's no `.git` repository in this directory.
