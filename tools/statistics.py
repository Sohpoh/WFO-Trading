"""Statistical validation layer for walk-forward results (ported from
quantjourney-bt's ``backtester/walkforward/statistics``).

Self-contained (numpy + stdlib only — the WFO env has no scipy), so it can be
imported by ``cli.py``, ``tools/validate.py``, and the evaluator without pulling
in any optional dependency.

What it adds on top of gate v2 (``tools/gate_v2.py``):

  - **Deflated / Probabilistic Sharpe Ratio** (Bailey & López de Prado 2014).
    Gate v2's leave-top-5-out checks *concentration* (is the edge carried by 5
    trades?), but it never corrects for *selection*: after N grid combos x M
    iterations, the best backtest Sharpe is positive even under a zero-alpha
    null.  DSR tests the observed Sharpe against that expected-maximum null.

  - **Sharpe confidence interval** via stationary block bootstrap (Politis &
    Romano 1994).  Formalizes "is this OOS Sharpe even distinguishable from
    zero?" — the question a 12–81-trade holdout alone cannot answer.

  - **Sharpe decay** (slope of per-fold OOS Sharpe): a negative slope means the
    alpha is decaying over time, the signature of an edge that will not carry
    into a later holdout.

  - **Overfit ratio / efficiency** — IS/OOS transfer diagnostics.

  - **Rank stability** (rolling top-K OOS rank, per quantjourney's non-CSCV
    diagnostic): per fold, re-backtest the top-K IS trials on that fold's OOS
    window and record where the IS-selected trial ranks.  A high rank-failure
    rate means argmax-Sharpe selection is not picking parameters that
    generalize.

Thresholds mirror quantjourney's ``interpretation.py``: DSR >= 0.95 robust,
0.80–0.95 marginal, < 0.80 likely false positive.
"""
from __future__ import annotations

import math

import numpy as np

# Euler–Mascheroni constant
_GAMMA = 0.5772156649015329


# ──────────────────────────────────────────────────────────────────────────
# Normal distribution (numpy/math only — scipy is not installed in the env)
# ──────────────────────────────────────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    """Standard normal CDF via complementary error function."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, ~1e-9)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ]
    plow, phigh = 0.02425, 1.0 - 0.02425

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


# ──────────────────────────────────────────────────────────────────────────
# Return-series helpers
# ──────────────────────────────────────────────────────────────────────────

def per_period_sharpe(returns: np.ndarray) -> float:
    """Sharpe of a per-period return series (no annualization)."""
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    sd = arr.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(arr.mean() / sd)


def skewness(returns: np.ndarray) -> float:
    """Sample skewness (γ₃, Fisher–Pearson, bias-corrected)."""
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return 0.0
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    n = arr.size
    m3 = float(((arr - mu) ** 3).mean())
    return m3 / (sd ** 3) * math.sqrt(n * (n - 1)) / (n - 2)


def raw_kurtosis(returns: np.ndarray) -> float:
    """Raw kurtosis (γ₄, 3.0 = normal)."""
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return 3.0
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return 3.0
    return float((((arr - mu) / sd) ** 4).mean())


# ──────────────────────────────────────────────────────────────────────────
# Probabilistic / Deflated Sharpe (Bailey & López de Prado 2012/2014)
# ──────────────────────────────────────────────────────────────────────────

def probabilistic_sharpe_ratio(
    observed_sr: float,
    *,
    benchmark_sr: float = 0.0,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR = P[true SR > benchmark_sr].

    ``observed_sr`` and ``n_obs`` must share the SAME periodicity (e.g. per-bar
    Sharpe with the number of bars behind it).  Annualized Sharpe with bar-level
    n_obs would massively overstate significance.
    """
    if n_obs is None or n_obs < 2:
        return 0.0
    denom_sq = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    if denom_sq <= 0.0:
        denom_sq = 1.0 + 0.5 * observed_sr ** 2
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1.0) / math.sqrt(denom_sq)
    return _norm_cdf(z)


def _expected_max_sr(sr_std: float, n_trials: float) -> float:
    """E[max SR] under the null, Euler–Mascheroni approximation (Bailey 2014)."""
    if not math.isfinite(n_trials) or n_trials <= 1:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe(
    trial_sharpes,
    n_trials: int | None = None,
    *,
    effective_n_trials: float | None = None,
    observed_sr: float | None = None,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio, a probability in [0, 1].

    ``trial_sharpes``: the optimizer-trial objective values (e.g. the train
    Sharpe of every grid combo) used to estimate √V[SR] across trials.
    ``effective_n_trials``: the number of *independent* trials the best result
    was selected over.  This is the number that matters for a loop that has run
    64 iterations x a grid each — pass the total backtest count here, not just
    this run's grid size.  ``observed_sr`` and ``n_obs`` are the candidate's
    per-period Sharpe and observation count, in the same units.
    """
    arr = np.asarray(list(trial_sharpes), dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if observed_sr is None:
        if arr.size == 0:
            return 0.0
        observed_sr = float(arr.max())

    raw_n = int(n_trials) if n_trials is not None else int(arr.size)
    if effective_n_trials is None:
        effective_n = float(raw_n)
    else:
        effective_n = float(effective_n_trials)
        if effective_n < 1.0:
            raise ValueError("effective_n_trials must be >= 1")
        if raw_n >= 1 and effective_n > raw_n:
            raise ValueError("effective_n_trials must be <= n_trials")

    sr_std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0

    if effective_n <= 1.0 or sr_std == 0.0:
        sr0 = benchmark_sr
    else:
        sr0 = benchmark_sr + _expected_max_sr(sr_std, effective_n)

    return probabilistic_sharpe_ratio(
        observed_sr,
        benchmark_sr=sr0,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )


# ──────────────────────────────────────────────────────────────────────────
# Sharpe CI — stationary block bootstrap (Politis & Romano 1994)
# ──────────────────────────────────────────────────────────────────────────

def bootstrap_sharpe_ci(
    returns,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
    ann_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> tuple[float, float] | None:
    """5%/95% stationary block-bootstrap CI for the annualized Sharpe.

    Returns ``(lo, hi)``, or ``None`` when the series is too short (< 20 obs)
    or has zero variance — the exact "not enough evidence" signal.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    t = r.size
    if t < 20 or float(np.std(r, ddof=1)) == 0.0:
        return None

    rng = np.random.default_rng(seed)
    expected_block = max(1, int(round(math.sqrt(t))))
    p_restart = 1.0 / expected_block
    rfr_daily = risk_free_rate / ann_factor
    ann = math.sqrt(ann_factor)
    pos = np.arange(t)

    sharpes = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        restarts = rng.random(t) < p_restart
        restarts[0] = True
        starts = rng.integers(0, t, size=t)
        last_restart = np.maximum.accumulate(np.where(restarts, pos, -1))
        idx = (starts[last_restart] + (pos - last_restart)) % t
        sample = r[idx]
        sd = sample.std(ddof=1)
        sharpes[i] = ((sample.mean() - rfr_daily) / sd * ann) if sd > 0 else 0.0

    lo, hi = np.percentile(sharpes, [5.0, 95.0])
    return float(lo), float(hi)


# ──────────────────────────────────────────────────────────────────────────
# Overfit diagnostics
# ──────────────────────────────────────────────────────────────────────────

def overfit_ratio(is_sharpe: float, oos_sharpe: float) -> float:
    """IS Sharpe / OOS Sharpe.  > 1.5 caution, > 2.5 likely overfit."""
    if oos_sharpe <= 0:
        return float("inf") if is_sharpe > 0 else 0.0
    return is_sharpe / oos_sharpe


def efficiency(is_cagr: float, oos_cagr: float) -> float:
    """OOS CAGR / IS CAGR.  < 0.4 red flag.  0.0 if IS CAGR <= 0."""
    if is_cagr <= 0:
        return 0.0
    return oos_cagr / is_cagr


def sharpe_decay(oos_sharpes) -> float:
    """Slope of per-fold OOS Sharpe across folds.  Negative => alpha decaying."""
    y = np.asarray(oos_sharpes, dtype=np.float64)
    n = y.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


# ──────────────────────────────────────────────────────────────────────────
# Rank stability — rolling top-K OOS rank (non-CSCV diagnostic)
# ──────────────────────────────────────────────────────────────────────────

def selected_trial_rank_logit(selected_value: float, candidate_values) -> float | None:
    """Logit of the IS-selected trial's OOS rank among the top-K candidates.

    λ <= 0 means the selection landed in the bottom half out-of-sample.
    """
    vals = np.asarray(list(candidate_values), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or not np.isfinite(selected_value):
        return None
    rank = (
        float((vals < selected_value).sum())
        + (float((vals == selected_value).sum()) + 1.0) / 2.0
    )
    omega = rank / (vals.size + 1.0)
    if not (0.0 < omega < 1.0):
        return None
    return float(math.log(omega / (1.0 - omega)))


def walk_forward_top_k_rank_failure_rate(logits) -> float:
    """Fraction of folds where the IS-selected trial ranked in the bottom half
    OOS.  This is NOT canonical CSCV PBO."""
    finite = [float(x) for x in logits if x is not None and np.isfinite(x)]
    if not finite:
        return float("nan")
    return sum(1.0 for x in finite if x <= 0.0) / len(finite)


# ──────────────────────────────────────────────────────────────────────────
# Traffic-light interpretation (mirrors quantjourney interpretation.py)
# ──────────────────────────────────────────────────────────────────────────

def interpret(value: float, metric: str, *, oos_sharpe: float | None = None, n_folds: int | None = None) -> tuple[str, str]:
    """Return ``(signal, description)`` where signal ∈ {green, yellow, red}."""
    losing = oos_sharpe is not None and oos_sharpe <= 0.0

    if metric == "overfit_ratio":
        if losing:
            return "red", "OOS Sharpe <= 0 — overfit ratio not meaningful on a losing strategy"
        if value < 0.0:
            return "yellow", "ratio < 0 — IS Sharpe negative; not interpretable"
        if value < 1.5:
            return "green", "< 1.5 — robust transfer"
        if value <= 2.5:
            return "yellow", "1.5–2.5 — caution"
        return "red", "> 2.5 — likely overfit"

    if metric == "efficiency":
        if value > 0.7:
            return "green", "> 0.7 — robust transfer"
        if value >= 0.4:
            return "yellow", "0.4–0.7 — moderate degradation"
        return "red", "< 0.4 — poor transfer"

    if metric == "sharpe_decay":
        if losing:
            return "red", "OOS Sharpe <= 0 — decay verdict suppressed"
        if value > -0.01:
            return "green", "> -0.01/fold — stable"
        if value >= -0.05:
            return "yellow", "-0.01 to -0.05/fold — moderate decay"
        return "red", "< -0.05/fold — alpha decaying"

    if metric == "deflated_sharpe":
        if value >= 0.95:
            return "green", ">= 0.95 — robust vs multiple testing"
        if value >= 0.80:
            return "yellow", "0.80–0.95 — marginal"
        return "red", "< 0.80 — likely false positive"

    if metric == "rank_failure_rate":
        if value <= 0.3:
            return "green", "<= 0.3 — IS selection generalizes OOS"
        if value <= 0.5:
            return "yellow", "0.3–0.5 — selection barely generalizes"
        return "red", "> 0.5 — argmax selection not generalizing"

    return "yellow", "no threshold defined"
