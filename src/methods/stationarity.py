"""ADF, Phillips-Perron, KPSS. Pre-stage diagnostics; not bubble tests."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def run_adf(series: pd.Series, max_lags="AIC") -> dict | None:
    s = np.asarray(series.dropna(), dtype=float)
    if len(s) < 30: return None
    autolag = max_lags if isinstance(max_lags, str) else None
    maxlag  = max_lags if isinstance(max_lags, int)  else None
    stat, pval, lag, _, crit, _ = adfuller(s, maxlag=maxlag, autolag=autolag, regression="c")
    return {"stat": float(stat), "pvalue": float(pval), "lag": int(lag),
            "cv_1pct": float(crit["1%"]), "cv_5pct": float(crit["5%"]), "cv_10pct": float(crit["10%"]),
            "reject_5pct": bool(stat < crit["5%"])}


def run_pp(series: pd.Series, lags="short") -> dict | None:
    s = np.asarray(series.dropna(), dtype=float)
    n = len(s)
    if n < 30: return None
    L = (int(np.floor(4  * (n / 100) ** 0.25)) if lags == "short" else
         int(np.floor(12 * (n / 100) ** 0.25)) if lags == "long"  else int(lags))
    L = max(L, 1)
    y = s[1:]; ylag = s[:-1]
    X = add_constant(ylag)
    ols = OLS(y, X).fit()
    rho, se_rho = ols.params[1], ols.bse[1]
    resid = ols.resid; T = len(resid)
    gamma0 = float(np.dot(resid, resid) / T)
    s2 = gamma0
    for k in range(1, L + 1):
        gk = float(np.dot(resid[k:], resid[:-k]) / T)
        s2 += 2 * (1 - k / (L + 1)) * gk
    s2 = max(s2, 1e-12)
    t_stat = (rho - 1) / se_rho
    Z_t = (math.sqrt(gamma0 / s2) * t_stat
           - 0.5 * (s2 - gamma0) * (T * se_rho)
             / (math.sqrt(s2) * math.sqrt(np.var(ylag, ddof=0) * T)))
    cv = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    return {"stat": float(Z_t), "lag": L,
            "cv_1pct": cv["1%"], "cv_5pct": cv["5%"], "cv_10pct": cv["10%"],
            "reject_5pct": bool(Z_t < cv["5%"])}


def run_kpss(series: pd.Series, lags="auto") -> dict | None:
    s = np.asarray(series.dropna(), dtype=float)
    if len(s) < 30: return None
    stat, pval, lag, crit = kpss(s, regression="c", nlags=lags)
    return {"stat": float(stat), "pvalue": float(pval), "lag": int(lag),
            "cv_1pct": float(crit["1%"]), "cv_5pct": float(crit["5%"]), "cv_10pct": float(crit["10%"]),
            "reject_5pct": bool(stat > crit["5%"])}
