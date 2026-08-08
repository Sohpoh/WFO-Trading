"""Performance metrics and trade-level accounting.

Costs: every trade (an entry, an exit, or a stop-and-reverse flip counted as
close+open) pays a 0.001% exchange fee + 0.05% slippage penalty, applied to
the price at the moment of the transaction.
"""
import numpy as np
import pandas as pd

FEE_RATE = 0.00001       # 0.001%
SLIPPAGE_RATE = 0.0005   # 0.05%
COST_RATE = FEE_RATE + SLIPPAGE_RATE  # per transaction leg


def bar_returns_with_costs(close: pd.Series, position: pd.Series) -> pd.Series:
    """Bar-by-bar strategy returns net of fees/slippage.

    A position change of magnitude 1 (flat<->long, flat<->short) pays one
    transaction leg; a reversal (long<->short) pays two (close + open).
    """
    ret = close.pct_change().fillna(0.0)
    legs = position.diff().abs().fillna(0.0)
    return position.shift(1).fillna(0.0) * ret - legs * COST_RATE


def extract_trades(close: pd.Series, position: pd.Series) -> pd.DataFrame:
    """Trade-by-trade log: one row per completed (or open) position segment."""
    changes = position != position.shift(1).fillna(0.0)
    change_idx = position.index[changes]
    if len(change_idx) == 0:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "net_return"]
        )
    rows = []
    for i, start in enumerate(change_idx):
        direction = position.loc[start]
        if direction == 0:
            continue
        end = change_idx[i + 1] if i + 1 < len(change_idx) else position.index[-1]
        entry_price = close.loc[start]
        exit_price = close.loc[end]
        gross = (exit_price / entry_price - 1) * direction
        net = gross - 2 * COST_RATE
        rows.append(
            {
                "entry_time": start,
                "exit_time": end,
                "direction": "Long" if direction > 0 else "Short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "net_return": net,
            }
        )
    return pd.DataFrame(rows)


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    if returns.empty:
        return pd.Series([start], index=[pd.Timestamp.now(tz="UTC")])
    return start * (1 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    return float(eq.iloc[-1] - 1) if len(eq) else 0.0


def cagr(returns: pd.Series, ann_factor: float) -> float:
    eq = equity_curve(returns)
    n = len(returns)
    if n < 2 or eq.iloc[-1] <= 0:
        return 0.0
    years = n / ann_factor
    if years <= 0:
        return 0.0
    return float(eq.iloc[-1] ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, ann_factor: float) -> float:
    if returns.empty or returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann_factor))


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    if len(eq) == 0:
        return 0.0
    dd = eq / eq.cummax() - 1
    return float(dd.min())


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    return float((trades["net_return"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gains = trades.loc[trades["net_return"] > 0, "net_return"].sum()
    losses = -trades.loc[trades["net_return"] < 0, "net_return"].sum()
    if losses == 0:
        return float(np.inf) if gains > 0 else 0.0
    return float(gains / losses)


def summarize(returns: pd.Series, trades: pd.DataFrame, ann_factor: float) -> dict:
    return {
        "Total Return": total_return(returns),
        "CAGR": cagr(returns, ann_factor),
        "Sharpe Ratio": sharpe_ratio(returns, ann_factor),
        "Max Drawdown": max_drawdown(returns),
        "Win Rate": win_rate(trades),
        "Profit Factor": profit_factor(trades),
        "Trades": int(len(trades)),
    }
