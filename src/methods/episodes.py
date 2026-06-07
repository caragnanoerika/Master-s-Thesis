"""
Episode detection.

Two detectors:

  detect_bsadf_episodes
    Multi-bubble dating for GSADF/BSADF, à la PSY (2015).
    Contiguous runs of BSADF > CV, with TWO filters to address spurious
    micro-episodes from Monte Carlo noise in the CV path:
      - min_duration  : reject episodes shorter than this many trading days
      - merge_gap     : merge episodes separated by ≤ merge_gap trading days
                        (a "fine non-bubble" between two real bubbles)

  detect_sv_episode
    Sarkar–Wells single-episode detector with asymmetric thresholds and
    collapse-gating.
"""
from __future__ import annotations
from typing import Iterable, Optional
import numpy as np
import pandas as pd


def detect_bsadf_episodes(bsadf_path, cv_path, dates: Iterable,
                          min_duration: int = 21,
                          merge_gap: int = 21) -> list[dict]:
    """
    Returns a list of {start, end, duration_days} dicts.

    `min_duration` and `merge_gap` defaults are deliberately stricter than the
    PSY heuristic (~log T) to remove micro-bubbles produced by MC noise in
    the critical-value path.
    """
    bsadf_path = np.asarray(bsadf_path, dtype=float)
    cv_path    = np.asarray(cv_path,    dtype=float)
    above      = (bsadf_path > cv_path) & np.isfinite(bsadf_path)

    # Step 1 — raw contiguous runs above CV
    raw = []
    in_ep = False; s = None
    for i, a in enumerate(above):
        if a and not in_ep:
            s = i; in_ep = True
        elif (not a) and in_ep:
            raw.append((s, i - 1)); in_ep = False
    if in_ep:
        raw.append((s, len(above) - 1))
    if not raw:
        return []

    # Step 2 — merge runs separated by ≤ merge_gap
    merged = [list(raw[0])]
    for s2, e2 in raw[1:]:
        if s2 - merged[-1][1] <= merge_gap:
            merged[-1][1] = e2
        else:
            merged.append([s2, e2])

    # Step 3 — apply min_duration filter AFTER merging
    d = pd.to_datetime(pd.Index(dates))
    return [
        {"start": pd.Timestamp(d[s2]),
         "end":   pd.Timestamp(d[e2]),
         "duration_days": (pd.Timestamp(d[e2]) - pd.Timestamp(d[s2])).days,
         "start_idx": int(s2), "end_idx": int(e2)}
        for s2, e2 in merged if (e2 - s2 + 1) >= min_duration
    ]


def detect_sadf_episodes(sadf_path, cv: float, dates: Iterable,
                         min_duration: int = 21,
                         merge_gap: int = 21) -> list[dict]:
    """
    PWY (2011) date-stamping: contiguous runs where the SADF path exceeds
    the flat asymptotic CV.  Delegates to detect_bsadf_episodes with a
    constant CV array so the two detectors stay in sync.
    """
    sadf_arr = np.asarray(sadf_path, dtype=float)
    cv_arr   = np.full_like(sadf_arr, cv)
    return detect_bsadf_episodes(sadf_arr, cv_arr, dates,
                                 min_duration=min_duration,
                                 merge_gap=merge_gap)


def detect_sv_episode(stat_path, orig_thr, screen_thr, coll_thr,
                      dates=None,
                      M: int = 60,
                      R: int = 30,
                      bridge_days: int = 90,
                      ) -> Optional[dict]:
    """
    SV-ADF single-episode detector matching Sarkar's reference implementation.

    Origination
    -----------
    First index where stat > orig_thr (log(τ)/10) for M consecutive steps.

    Collapse — two phases, tested in order
    ---------------------------------------
    Bridge (within bridge_days calendar days after origination):
        First step where stat < orig_thr (log(τ)/10).

    Post-bridge (after bridge_days have elapsed):
        First step j where stat[j] < screen_thr[j] (log(τ)) AND
        stat[j:j+R] are all below coll_thr[j:j+R] (log(τ)/2).

    If neither is found: episode extends to end of sample.
    """
    stat   = np.asarray(stat_path, dtype=float)
    up     = np.asarray(orig_thr,   dtype=float)
    screen = np.asarray(screen_thr, dtype=float)
    down   = np.asarray(coll_thr,   dtype=float)

    # Broadcast scalar thresholds to full length
    if up.ndim == 0:     up     = np.broadcast_to(up,     stat.shape)
    if screen.ndim == 0: screen = np.broadcast_to(screen, stat.shape)
    if down.ndim == 0:   down   = np.broadcast_to(down,   stat.shape)

    N = len(stat)

    # ── Origination ──────────────────────────────────────────────────────────
    origin = None
    if N >= M:
        for i in range(N - M + 1):
            w = stat[i: i + M]
            u = up[i: i + M]
            if np.all(np.isfinite(w)) and np.all(w > u):
                origin = i
                break
    if origin is None:
        return None

    # ── Bridge end index ─────────────────────────────────────────────────────
    if dates is not None:
        d          = pd.to_datetime(pd.Index(dates))
        orig_date  = d[origin]
        bridge_end = orig_date + pd.Timedelta(days=bridge_days)
        bridge_end_idx = next(
            (j for j in range(origin + 1, N) if d[j] >= bridge_end), N
        )
    else:
        bridge_end_idx = min(origin + bridge_days, N)

    collapse      = None
    collapse_type = None

    # ── Bridge collapse: first step in bridge window below orig_thr ──────────
    for j in range(origin + 1, bridge_end_idx):
        if np.isfinite(stat[j]) and stat[j] < up[j]:
            collapse      = j
            collapse_type = "bridge"
            break

    # ── Post-bridge collapse: screen (log τ) + R-run below coll_thr ─────────
    if collapse is None:
        for j in range(bridge_end_idx, N):
            if not (np.isfinite(stat[j]) and stat[j] < screen[j]):
                continue
            run_end = j + R
            if run_end > N:
                break
            run = stat[j: run_end]
            thr = down[j: run_end]
            if np.all(np.isfinite(run)) and np.all(run < thr):
                collapse      = j
                collapse_type = "post_bridge"
                break

    if collapse is None:
        collapse      = N - 1
        collapse_type = "end_of_sample"

    ep = {"start_idx": int(origin), "end_idx": int(collapse),
          "collapse_type": collapse_type}
    if dates is not None:
        ep["start"]         = pd.Timestamp(d[origin])
        ep["end"]           = pd.Timestamp(d[collapse])
        ep["duration_days"] = (ep["end"] - ep["start"]).days
    return ep
