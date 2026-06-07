"""Run ADF, PP, KPSS for one ticker on the FIXED sample window."""
from __future__ import annotations
import pandas as pd
from config import settings
from src.methods.stationarity import run_adf, run_pp, run_kpss
from src.io.results import save_stationarity, load_stationarity


def run_one(ticker: str, series: pd.Series, force: bool = False) -> dict | None:
    """Returns the result dict, or None if skipped."""
    if not force and load_stationarity(ticker) is not None:
        return None  # already cached
    s = series.dropna().astype(float)
    if len(s) < 30:
        return None
    out = {
        "ticker":     ticker,
        "T":          int(len(s)),
        "start_date": s.index[0].isoformat()[:10],
        "end_date":   s.index[-1].isoformat()[:10],
        "adf":  run_adf(s,  max_lags=settings.ADF_MAX_LAGS),
        "pp":   run_pp(s,   lags=settings.PP_LAGS),
        "kpss": run_kpss(s, lags=settings.KPSS_LAGS),
    }
    save_stationarity(ticker, out)
    return out
