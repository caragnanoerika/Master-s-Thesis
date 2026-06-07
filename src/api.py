"""
High-level API for the project.

These functions are pure (input → output, no global state) and form a
clean interface that an external app (e.g. a Gradio space on HuggingFace)
can call. They do not run analysis — they read pre-computed results from
disk and return DataFrames or matplotlib Figures.

To run new analyses programmatically, call the pipeline functions in
src.pipelines.* directly.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from config import settings
from src.io.results import (
    load_stationarity, load_sadf, load_gsadf, load_svadf,
    list_svadf_windows, available_tickers, available_methods,
)
from src.plotting.diagnostic import (
    plot_sadf_only, plot_gsadf_only, plot_svadf_only, plot_gsadf_segment_panel,
)
from src.plotting.comparison import plot_comparison, plot_sadf_vs_svadf
from src.plotting.tables    import (
    build_comparison_table, build_episodes_table, build_segment_summary,
)


# ── Discovery ────────────────────────────────────────────────────────────────
def list_tickers() -> list[str]:
    """All tickers that have any result on disk."""
    return available_tickers() or settings.ALL_TICKERS


def list_methods_for(ticker: str) -> list[str]:
    """Which methods have results for this ticker."""
    return available_methods(ticker)


def list_svadf_windows_for(ticker: str) -> list[tuple[str, str]]:
    """All saved SV-ADF window pairs for this ticker."""
    return list_svadf_windows(ticker)


# ── Result retrieval ─────────────────────────────────────────────────────────
def get_stationarity(ticker: str) -> dict | None:
    return load_stationarity(ticker)


def get_sadf(ticker: str) -> dict | None:
    return load_sadf(ticker)


def get_gsadf(ticker: str) -> dict | None:
    return load_gsadf(ticker)


def get_svadf(ticker: str,
              window: tuple[str, str] | None = None) -> dict | None:
    if window is None:
        wins = list_svadf_windows(ticker)
        if not wins:
            return None
        window = wins[-1]
    return load_svadf(ticker, *window)


# ── Figures ──────────────────────────────────────────────────────────────────
def get_sadf_figure(ticker: str, save_path: Optional[Path] = None,
                    ) -> plt.Figure | None:
    """SADF stat path and flat CV line — no GSADF or SV-ADF data required."""
    return plot_sadf_only(ticker, save_path=save_path, show=False)


def get_sadf_vs_svadf_figure(ticker: str,
                              svadf_window:  tuple[str, str] | None = None,
                              svadf_windows: list[tuple[str, str]] | None = None,
                              save_path: Optional[Path] = None,
                              ) -> plt.Figure | None:
    """SADF path + SV-ADF coefficient paths side by side for direct comparison."""
    return plot_sadf_vs_svadf(ticker, svadf_window=svadf_window,
                               svadf_windows=svadf_windows,
                               save_path=save_path, show=False)


def get_gsadf_figure(ticker: str, save_path: Optional[Path] = None,
                     ) -> plt.Figure | None:
    return plot_gsadf_only(ticker, save_path=save_path, show=False)


def get_gsadf_segment_figure(segment: str,
                              save_path: Optional[Path] = None,
                              ) -> "plt.Figure | None":
    """Panel plot: all GSADF charts for tickers in `segment` side by side."""
    return plot_gsadf_segment_panel(segment, save_path=save_path, show=False)


def get_svadf_figure(ticker: str,
                     window:  tuple[str, str] | None = None,
                     windows: list[tuple[str, str]] | None = None,
                     save_path: Optional[Path] = None) -> plt.Figure | None:
    """Both SV-ADF windows overlaid by default (pass window= for a single window)."""
    return plot_svadf_only(ticker, window=window, windows=windows,
                           save_path=save_path, show=False)


def get_comparison_figure(ticker: str,
                          svadf_window:  tuple[str, str] | None = None,
                          svadf_windows: list[tuple[str, str]] | None = None,
                          save_path: Optional[Path] = None) -> plt.Figure | None:
    """Both SV-ADF windows overlaid by default (pass svadf_window= for a single window)."""
    return plot_comparison(ticker, svadf_window=svadf_window,
                           svadf_windows=svadf_windows,
                           save_path=save_path, show=False)


# ── Tables ───────────────────────────────────────────────────────────────────
def get_comparison_table(svadf_window: tuple[str, str] | None = None,
                         ) -> pd.DataFrame:
    return build_comparison_table(svadf_window=svadf_window)


def get_episodes_table(svadf_window: tuple[str, str] | None = None,
                       ) -> pd.DataFrame:
    return build_episodes_table(svadf_window=svadf_window)


def get_segment_summary() -> pd.DataFrame:
    return build_segment_summary()
