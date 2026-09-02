"""Replica of wfo-evaluator's accept/reject logic (gate v2), as pure functions.

Used by the null-model calibration and the cost-sensitivity audit so both judge
a run with the *same* code the evaluator persona applies, instead of two
hand-rolled approximations that could drift apart.

This module intentionally mirrors `wfo-evaluator/SKILL.md` Steps 1-6:
  Step 1  error triage
  Step 2  fold-level OOS consistency (PRIMARY)
  Step 3  leave-top-5-out robustness (HARD GATE, 3-way)
  Step 4  absolute checklist (Sharpe > 0.8, PF > 1.2, trades >= 30)
  Step 6  single verdict, with the "raised bar" for borderline/inconclusive
Step 5 (efficiency ratio) is diagnostic-only and omitted here on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pf(net: np.ndarray | pd.Series) -> float:
    """Profit factor with the same edge cases the evaluator tolerates."""
    arr = np.asarray(net, dtype=float)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses == 0:
        return float(np.inf) if gains > 0 else 0.0
    return float(gains / losses)


def robustness(net_returns: np.ndarray | pd.Series, total_return_full: float) -> dict:
    """Step 3 — leave-top-5-out, 3-way verdict. `total_return_full` is the
    compounded full-sample total return (from bar returns), matching how the
    evaluator reads comparison.csv; `total_return_ex_top5` is the PLAIN
    arithmetic sum of the remaining trades, exactly as the spec dictates."""
    arr = np.asarray(net_returns, dtype=float)
    n = len(arr)
    pf_full = _pf(arr)

    if n < 15:
        return {
            "total_return_full": total_return_full,
            "total_return_ex_top5": None,
            "profit_factor_full": pf_full,
            "profit_factor_ex_top5": None,
            "robust": "inconclusive (n<15 trades)",
            "n_trades": n,
        }

    remaining = np.sort(arr)[::-1][5:]
    tr_ex5 = float(remaining.sum())
    pf_ex5 = _pf(remaining)

    sign_flip = (tr_ex5 * total_return_full) < 0
    if sign_flip or pf_ex5 < 0.9:
        robust = False
    elif pf_ex5 >= 1.15:
        robust = True
    else:
        robust = "borderline"

    return {
        "total_return_full": total_return_full,
        "total_return_ex_top5": tr_ex5,
        "profit_factor_full": pf_full,
        "profit_factor_ex_top5": pf_ex5,
        "robust": robust,
        "n_trades": n,
    }


def fold_consistency(fold_returns: np.ndarray | pd.Series, fold_sharpes: np.ndarray | pd.Series) -> dict:
    """Step 2 — % of active folds profitable + median fold return/Sharpe."""
    rets = np.asarray(fold_returns, dtype=float)
    sharpes = np.asarray(fold_sharpes, dtype=float)
    n_active = len(rets)
    if n_active == 0:
        return {"active_folds": 0, "pct_profitable": 0.0, "median_fold_return": 0.0, "median_fold_sharpe": 0.0}
    return {
        "active_folds": int(n_active),
        "pct_profitable": float((rets > 0).mean()),
        "median_fold_return": float(np.median(rets)),
        "median_fold_sharpe": float(np.median(sharpes)),
    }


def verdict(
    fold_returns, fold_sharpes, net_returns, total_return_full, oos_sharpe, oos_pf, n_trades
) -> dict:
    """Steps 1-6 combined: return a dict with status + the deciding numbers."""
    fc = fold_consistency(fold_returns, fold_sharpes)
    rob = robustness(net_returns, total_return_full)
    robust = rob["robust"]

    # Step 1 — error triage
    if n_trades == 0 or fc["active_folds"] == 0:
        return {
            "status": "error",
            "fold_consistency": fc,
            "robustness": rob,
            "oos_sharpe": oos_sharpe,
            "oos_pf": oos_pf,
            "n_trades": int(n_trades),
            "reason": "no tradeable folds (Step 1 error)",
        }

    # Step 6 — the raised bar applies when robustness isn't a clean true
    raised = robust != True  # noqa: E712  (borderline / inconclusive / False)
    sharpe_bar = 1.0 if raised else 0.8
    pf_bar = 1.35 if raised else 1.2
    trades_bar = 50 if raised else 30
    pct_bar = 0.65 if (robust == "borderline" or isinstance(robust, str)) else 0.60

    cond1 = robust != False  # noqa: E712
    cond2 = fc["pct_profitable"] >= pct_bar
    cond3 = (oos_sharpe > sharpe_bar) and (oos_pf > pf_bar) and (n_trades >= trades_bar)

    accepted = cond1 and cond2 and cond3

    return {
        "status": "accepted" if accepted else "rejected",
        "fold_consistency": fc,
        "robustness": rob,
        "oos_sharpe": float(oos_sharpe),
        "oos_pf": float(oos_pf),
        "n_trades": int(n_trades),
        "raised_bar": raised,
        "pct_bar": pct_bar,
        "sharpe_bar": sharpe_bar,
        "pf_bar": pf_bar,
        "cond1_robust_not_false": bool(cond1),
        "cond2_fold_consistency": bool(cond2),
        "cond3_absolute": bool(cond3),
    }
