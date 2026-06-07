"""
Summary tables built by loading all per-ticker JSONs.
The tables are pure functions of the on-disk results.
"""
from __future__ import annotations
import pandas as pd
from config import settings
from src.io.results import (
    load_stationarity, load_sadf, load_gsadf, load_svadf, list_svadf_windows,
)


def _yn(b) -> str:
    return "✓" if b else "·"


def build_comparison_table(svadf_window: tuple[str, str] | None = None,
                           ) -> pd.DataFrame:
    """One row per ticker × columns for each method's verdict.

    Always shows both SV-ADF windows (W1 post-ChatGPT, W2 pre-ChatGPT)
    as separate column pairs.  The legacy ``svadf_window`` argument is
    kept for back-compatibility but is ignored when both settings windows
    are available.
    """
    W1 = (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)
    W2 = (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END)

    rows = []
    for tk in settings.ALL_TICKERS:
        st = load_stationarity(tk)
        sa = load_sadf(tk)
        gs = load_gsadf(tk)
        sv1 = load_svadf(tk, *W1)
        sv2 = load_svadf(tk, *W2)

        seg = settings.SEGMENT_OF.get(tk, ("?",))[0]
        row = {"Ticker": tk, "Segment": seg}
        if st:
            row["T"]            = st["T"]
            row["ADF reject"]   = _yn(st["adf"]["reject_5pct"])
            row["PP reject"]    = _yn(st["pp"]["reject_5pct"])
            row["KPSS reject"]  = _yn(st["kpss"]["reject_5pct"])
        if sa:
            row["SADF stat"]    = round(sa["statistic"], 2)
            row["SADF signal"]  = _yn(sa["signal"])
        if gs:
            g = gs["summary"]
            row["GSADF stat"]   = round(g["statistic"], 2)
            row["GSADF cv"]     = round(g["cv_value"], 2)
            row["GSADF reject"] = _yn(g["reject"])
            row["# Episodes"]   = len(g["episodes"])
        # SV-ADF W1 (post-ChatGPT)
        if sv1:
            ep1 = sv1["summary"]["episode"]
            row["SV W1 episode"] = _yn(ep1 is not None)
            if ep1:
                row["SV W1 collapse"] = ep1["collapse_type"]
                row["SV W1 start"]    = ep1["start"][:10]
        # SV-ADF W2 (pre-ChatGPT)
        if sv2:
            ep2 = sv2["summary"]["episode"]
            row["SV W2 episode"] = _yn(ep2 is not None)
            if ep2:
                row["SV W2 collapse"] = ep2["collapse_type"]
                row["SV W2 start"]    = ep2["start"][:10]
        rows.append(row)
    return pd.DataFrame(rows)


def build_episodes_table(svadf_window: tuple[str, str] | None = None,
                         ) -> pd.DataFrame:
    """Master episode list: all GSADF episodes + both SV-ADF window episodes."""
    W1 = (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END)
    W2 = (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END)

    rows = []
    for tk in settings.ALL_TICKERS:
        seg = settings.SEGMENT_OF.get(tk, ("?",))[0]
        gs = load_gsadf(tk)
        if gs is not None:
            for ep in gs["summary"]["episodes"]:
                rows.append({
                    "Ticker": tk, "Segment": seg, "Method": "GSADF",
                    "Window": f"{gs['summary']['start_date']} → {gs['summary']['end_date']}",
                    "Start":  ep["start"][:10], "End": ep["end"][:10],
                    "Duration (days)": ep["duration_days"],
                })
        for label, win in [("SV-ADF W1", W1), ("SV-ADF W2", W2)]:
            sv = load_svadf(tk, *win)
            if sv is not None:
                ep = sv["summary"]["episode"]
                if ep:
                    rows.append({
                        "Ticker": tk, "Segment": seg, "Method": label,
                        "Window": sv["summary"]["window_id"],
                        "Start":  ep["start"][:10], "End": ep["end"][:10],
                        "Duration (days)": ep["duration_days"],
                        "Collapse type":   ep["collapse_type"],
                    })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values(["Start", "Ticker"])
                               .reset_index(drop=True))


def build_segment_summary() -> pd.DataFrame:
    """Aggregate per segment: number of tickers with at least one GSADF episode."""
    rows = []
    by_seg = {}
    for tk in settings.ALL_TICKERS:
        seg = settings.SEGMENT_OF.get(tk, ("?",))[0]
        by_seg.setdefault(seg, []).append(tk)
    for seg, members in sorted(by_seg.items()):
        with_ep = 0; total_eps = 0
        for tk in members:
            gs = load_gsadf(tk)
            if gs and gs["summary"]["episodes"]:
                with_ep += 1
                total_eps += len(gs["summary"]["episodes"])
        rows.append({
            "Segment": seg,
            "N": len(members),
            "Tickers with GSADF episode": with_ep,
            "Total episodes": total_eps,
        })
    return pd.DataFrame(rows)
