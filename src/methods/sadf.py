"""SADF - supremum ADF (Phillips, Wu & Yu 2011), single-bubble expanding window."""
from __future__ import annotations
import numpy as np
from src.methods.adf_engine import adf_regression
from src.methods.gsadf import minimum_window


def sadf(y, min_window: int | None = None,
         max_lags: int = 0, lag_method: str = "fixed") -> dict:
    y = np.asarray(y, dtype=float)
    T = len(y)
    if min_window is None:
        min_window = minimum_window(T)
    path = np.full(T - min_window + 1, np.nan)
    for i, end in enumerate(range(min_window, T + 1)):
        res = adf_regression(y[:end], max_lags=max_lags, lag_method=lag_method)
        if res is not None:
            path[i] = res["t_stat"]
    return {"statistic": float(np.nanmax(path)),
            "path": path,
            "argmax_index": int(np.nanargmax(path)) + min_window - 1,
            "min_window": min_window}
