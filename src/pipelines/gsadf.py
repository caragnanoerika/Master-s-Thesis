"""
Run GSADF (BSADF episode dating) for one ticker on the FIXED sample window.

Uses the cached Monte Carlo critical values. Episode detection applies the
stricter min_duration / merge_gap filters defined in config to suppress
spurious micro-bubbles from MC noise.
"""
from __future__ import annotations
import pandas as pd
from config import settings
from src.methods.gsadf   import gsadf, minimum_window
from src.methods.episodes import detect_bsadf_episodes
from src.monte_carlo     import get_critical_values
from src.io.results      import save_gsadf, load_gsadf


def run_one(ticker: str, series: pd.Series,
            mc_reps: int | None = None,
            force: bool = False,
            force_mc: bool = False,
            min_duration: int | None = None,
            merge_gap: int | None = None,
            verbose: bool = True) -> dict | None:
    """
    Runs GSADF + BSADF episode dating, saves both the summary JSON and
    a parquet with the full BSADF and CV paths.

    If results already exist but the episode-filter settings differ from what
    was used at last run, only the episode-detection step is re-applied on the
    stored BSADF/CV paths — the expensive GSADF computation and Monte Carlo
    are not repeated.
    """
    # Resolve filter params first so we can compare against cached values.
    min_duration = min_duration or settings.EPISODE_MIN_DURATION
    merge_gap    = merge_gap    or settings.EPISODE_MERGE_GAP

    if not force:
        existing = load_gsadf(ticker)
        if existing is not None:
            stored = existing["summary"].get("episode_filters", {})
            if (stored.get("min_duration_days") == min_duration and
                    stored.get("merge_gap_days") == merge_gap):
                return None  # fully cached, filters match — skip

            # Filter settings changed: re-apply episode detection on stored paths.
            if existing["paths"] is not None:
                paths   = existing["paths"]
                summary = existing["summary"]
                episodes = detect_bsadf_episodes(
                    paths["bsadf_stat"].to_numpy(),
                    paths["cv_path"].to_numpy(),
                    paths.index,
                    min_duration=min_duration,
                    merge_gap=merge_gap,
                )
                summary["episodes"] = episodes
                summary["reject"]   = bool(summary["statistic"] > summary["cv_value"])
                summary["episode_filters"] = {
                    "min_duration_days": int(min_duration),
                    "merge_gap_days":    int(merge_gap),
                }
                save_gsadf(ticker, summary, paths)
                return summary

    mc_reps = mc_reps or settings.GSADF_MC_REPS

    s = series.dropna().astype(float)
    if len(s) < 100:
        return None
    y, T = s.to_numpy(), len(s)
    mw   = minimum_window(T)

    g = gsadf(y, min_window=mw)
    cv = get_critical_values(
        T=T, min_window=mw, nrep=mc_reps, seed=settings.SEED,
        quantiles=(settings.GSADF_QUANTILE,),
        force_recompute=force_mc, verbose=verbose,
    )
    cv_path = cv["bsadf_path"][settings.GSADF_QUANTILE]
    cv_val  = cv["gsadf"][settings.GSADF_QUANTILE]

    bsadf_dates = s.index[mw - 1:]
    episodes = detect_bsadf_episodes(
        g["path"], cv_path, bsadf_dates,
        min_duration=min_duration, merge_gap=merge_gap,
    )

    summary = {
        "ticker":     ticker,
        "T":          int(T),
        "start_date": s.index[0].isoformat()[:10],
        "end_date":   s.index[-1].isoformat()[:10],
        "min_window": int(mw),
        "statistic":  float(g["statistic"]),
        "cv_value":   float(cv_val),
        "cv_quantile":float(settings.GSADF_QUANTILE),
        "reject":     bool(g["statistic"] > cv_val),
        "mc_reps":    int(mc_reps),
        "mc_seed":    int(settings.SEED),
        "episode_filters": {
            "min_duration_days": int(min_duration),
            "merge_gap_days":    int(merge_gap),
        },
        "episodes":   episodes,
    }
    paths_df = pd.DataFrame({
        "bsadf_stat": g["path"],
        "cv_path":    cv_path,
    }, index=bsadf_dates)
    paths_df.index.name = "date"
    save_gsadf(ticker, summary, paths_df)
    return summary
