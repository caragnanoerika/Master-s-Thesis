"""Run SADF for one ticker — fixed window and arbitrary sub-windows."""
from __future__ import annotations
import pandas as pd
from config import settings
from src.methods.sadf import sadf
from src.methods.gsadf import minimum_window
from src.methods.episodes import detect_sadf_episodes
from src.io.results import (
    save_sadf, load_sadf, save_sadf_paths,
    save_sadf_window, load_sadf_window,
)

_EP_FILTERS = {
    "min_duration_days": settings.EPISODE_MIN_DURATION,
    "merge_gap_days":    settings.EPISODE_MERGE_GAP,
}


def _build_summary(ticker: str, s: pd.Series, r: dict) -> tuple[dict, pd.DataFrame]:
    """Shared computation: summary dict + paths DataFrame from a sadf() result."""
    mw       = r["min_window"]
    path_idx = s.index[mw - 1:]   # path[i] covers y[:mw+i], last date s.index[mw-1+i]
    episodes = detect_sadf_episodes(
        r["path"], settings.SADF_CV_5PCT, path_idx,
        min_duration=settings.EPISODE_MIN_DURATION,
        merge_gap=settings.EPISODE_MERGE_GAP,
    )
    # Serialise episode timestamps so they round-trip through JSON
    eps_json = [
        {k: (v.isoformat()[:10] if isinstance(v, pd.Timestamp) else v)
         for k, v in ep.items()}
        for ep in episodes
    ]
    summary = {
        "ticker":           ticker,
        "T":                int(len(s)),
        "start_date":       s.index[0].isoformat()[:10],
        "end_date":         s.index[-1].isoformat()[:10],
        "statistic":        float(r["statistic"]),
        "argmax_date":      s.index[r["argmax_index"]].isoformat()[:10],
        "cv_5pct":          float(settings.SADF_CV_5PCT),
        "signal":           bool(r["statistic"] > settings.SADF_CV_5PCT),
        "min_window":       int(mw),
        "episodes":         eps_json,
        "episode_filters":  _EP_FILTERS,
    }
    paths_df = pd.DataFrame(
        {"sadf_stat": r["path"], "cv": settings.SADF_CV_5PCT},
        index=path_idx,
    )
    return summary, paths_df


def run_one(ticker: str, series: pd.Series, force: bool = False) -> dict | None:
    """SADF on the full series (FIXED window — for standalone and GSADF comparison)."""
    if not force and load_sadf(ticker) is not None:
        return None
    s = series.dropna().astype(float)
    if len(s) < 100:
        return None
    y  = s.to_numpy()
    mw = minimum_window(len(y))
    r  = sadf(y, min_window=mw)
    summary, paths_df = _build_summary(ticker, s, r)
    save_sadf(ticker, summary)
    save_sadf_paths(ticker, paths_df)
    return summary


def run_windowed(ticker: str, series: pd.Series,
                 w_start: str, w_end: str,
                 force: bool = False) -> dict | None:
    """SADF on an arbitrary sub-window (for SV-ADF comparison)."""
    if not force and load_sadf_window(ticker, w_start, w_end) is not None:
        return None
    mask = (series.index >= pd.Timestamp(w_start)) & (series.index <= pd.Timestamp(w_end))
    s = series.loc[mask].dropna().astype(float)
    if len(s) < 100:
        return None
    y  = s.to_numpy()
    mw = minimum_window(len(y))
    r  = sadf(y, min_window=mw)
    summary, paths_df = _build_summary(ticker, s, r)
    save_sadf_window(ticker, w_start, w_end, summary, paths_df)
    return summary
