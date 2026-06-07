"""
SV-ADF (Sarkar & Wells 2026) — vectorized via cumulative sums.

The model is the AR(1) with intercept: X_t = μ + δ·X_{t-1} + ε_t.
The demeaned OLS estimator on an expanding window of τ observations is

    δ̂ = Σ(X̃_{t-1}·X̃_t) / Σ(X̃_{t-1}²)

where X̃ denotes mean-subtracted values.  Via the standard identity
    Σ X̃·Ỹ = Σ XY − τ·X̄·Ȳ
this reduces to cumulative sums, keeping the recursion O(T).

This matches the reference implementation of Sarkar (2026) exactly:
    y_tilde <- y_tau - mean(y_tau)
    xlag_tilde <- xlag_tau - mean(xlag_tau)
    delta_hat <- sum(xlag_tilde * y_tilde) / sum(xlag_tilde^2)

Asymmetric analytical thresholds (Section 5.1 of the paper):
    origination : log(τ) / 10   ≈ simulated 90th pct under H₀
    screen      : log(τ)        ≈ intermediate screening step for collapse
    collapse    : log(τ) / 2    ≈ simulated 10th pct under H₁
"""
from __future__ import annotations
import numpy as np


def svadf(y, min_fraction: float = 0.10) -> dict | None:
    """
    Vectorized SV-ADF coefficient statistic (demeaned OLS, with intercept).

    Parameters
    ----------
    y            : 1-D array of prices
    min_fraction : minimum window as fraction of total pairs (default 0.10)

    Returns
    -------
    dict with:
        coef_stat  : length-T array (nan before min window), indexed by last obs
        orig_thr   : length-T array of log(τ)/10
        screen_thr : length-T array of log(τ)
        coll_thr   : length-T array of log(τ)/2
        min_window : int (minimum τ actually used)
        dates_idx  : 1-D int array of valid indices into the above arrays
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    pairs = T - 1  # number of (X_t, X_{t-1}) pairs

    if pairs < 2:
        return None

    tau_min = max(2, int(min_fraction * pairs))
    if tau_min >= pairs:
        return None

    # Pairs: X_t = y[1:], X_{t-1} = y[:-1]
    xt = y[1:]    # X_t
    xl = y[:-1]   # X_{t-1}

    # Cumulative sums (length = pairs)
    cxt  = np.cumsum(xt)        # Σ X_t,   t = 1 … τ
    cxl  = np.cumsum(xl)        # Σ X_{t-1}, t = 1 … τ
    cxtl = np.cumsum(xt * xl)   # Σ X_t·X_{t-1}
    cxl2 = np.cumsum(xl ** 2)   # Σ X_{t-1}²

    # τ grid: τ_min … pairs (inclusive)
    tau_arr = np.arange(tau_min, pairs + 1, dtype=np.float64)
    idx     = (tau_arr - 1).astype(int)   # 0-indexed into cumsum arrays

    # Demeaned OLS numerator / denominator
    #   numer = Σ(X_{t-1}·X_t) − τ · mean(X_{t-1}) · mean(X_t)
    #   denom = Σ(X_{t-1}²)    − τ · mean(X_{t-1})²
    numer = cxtl[idx] - cxl[idx] * cxt[idx] / tau_arr
    denom = cxl2[idx] - cxl[idx] ** 2 / tau_arr

    valid     = denom > 1e-14
    delta_hat = np.where(valid, numer / denom, np.nan)

    coef_vals = tau_arr * (delta_hat - 1)

    # Date indexing: for window of size τ, the last observation is prices[τ],
    # matching Sarkar's  grid_df$Date <- date_vector[tau + 1]  (1-indexed R).
    date_idx = tau_arr.astype(int)   # == τ

    coef_path   = np.full(T, np.nan)
    orig_path   = np.full(T, np.nan)
    screen_path = np.full(T, np.nan)
    coll_path   = np.full(T, np.nan)

    log_tau = np.log(tau_arr)
    coef_path  [date_idx] = coef_vals
    orig_path  [date_idx] = log_tau / 10.0
    screen_path[date_idx] = log_tau
    coll_path  [date_idx] = log_tau / 2.0

    return {
        "coef_stat":   coef_path,
        "orig_thr":    orig_path,
        "screen_thr":  screen_path,
        "coll_thr":    coll_path,
        "min_window":  int(tau_min),
        "dates_idx":   date_idx,
    }
