"""
Monte Carlo critical values for GSADF, with persistent CSV cache.

The cache is keyed by (T, min_window, nrep, seed). Adding a new ticker:
- If its (T, min_window) already exists in the cache, it's reused.
- If not, MC runs only for that ticker and writes a new cache entry.

Cache layout:
    data/mc_cache/
        global_critical_values.csv               master index
        bsadf_paths/T{T}_mw{mw}_nrep{n}_seed{s}.csv   one CSV per shape
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from config import settings
from src.methods.gsadf import gsadf


def _path_file(T, mw, nrep, seed) -> Path:
    return settings.MC_PATHS_DIR / f"T{T}_mw{mw}_nrep{nrep}_seed{seed}.csv"


def simulate_critical_values(T: int, min_window: int, nrep: int, seed: int,
                              quantiles: Iterable[float] = (0.90, 0.95, 0.99),
                              verbose: bool = True) -> dict:
    """Simulate `nrep` random walks and return empirical quantiles."""
    rng = np.random.default_rng(seed)
    npath = T - min_window + 1
    gs_stats = np.empty(nrep)
    bs_paths = np.empty((nrep, npath))
    if verbose:
        print(f"  MC: {nrep} reps, T={T}, mw={min_window} …", end=" ", flush=True)
    t0 = time.time()
    for r in range(nrep):
        rw  = np.cumsum(rng.standard_normal(T))
        out = gsadf(rw, min_window=min_window)
        gs_stats[r]    = out["statistic"]
        bs_paths[r, :] = out["path"]
    if verbose:
        print(f"{time.time() - t0:.1f}s")
    quantiles = list(quantiles)
    return {
        "gsadf":      {q: float(np.quantile(gs_stats, q))         for q in quantiles},
        "bsadf_path": {q: np.quantile(bs_paths, q, axis=0)        for q in quantiles},
    }


def _save_cache(T, mw, nrep, seed, cv, quantiles):
    settings.MC_PATHS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({f"q{int(q*1000):03d}": cv["bsadf_path"][q] for q in quantiles})
    df.index.name = "path_index"
    df.to_csv(_path_file(T, mw, nrep, seed))

    row = {"T": T, "min_window": mw, "nrep": nrep, "seed": seed}
    for q in quantiles:
        row[f"gsadf_q{int(q*1000):03d}"] = cv["gsadf"][q]
    idx_path = settings.MC_GLOBAL_INDEX_FILE
    if idx_path.exists():
        existing = pd.read_csv(idx_path)
        mask = ~((existing["T"] == T) & (existing["min_window"] == mw)
                 & (existing["nrep"] == nrep) & (existing["seed"] == seed))
        out = pd.concat([existing.loc[mask], pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.sort_values(["T", "min_window", "nrep", "seed"]).to_csv(idx_path, index=False)


def _load_cache(T, mw, nrep, seed, quantiles) -> dict | None:
    p = _path_file(T, mw, nrep, seed)
    idx_path = settings.MC_GLOBAL_INDEX_FILE
    if not p.exists() or not idx_path.exists():
        return None
    idx = pd.read_csv(idx_path)
    row = idx[(idx["T"] == T) & (idx["min_window"] == mw)
              & (idx["nrep"] == nrep) & (idx["seed"] == seed)]
    if row.empty: return None
    row = row.iloc[0]
    df = pd.read_csv(p, index_col="path_index")
    out_gs, out_paths = {}, {}
    for q in quantiles:
        gcol = f"gsadf_q{int(q*1000):03d}"
        pcol = f"q{int(q*1000):03d}"
        if gcol not in row.index or pcol not in df.columns:
            return None
        out_gs[q] = float(row[gcol]); out_paths[q] = df[pcol].to_numpy()
    return {"gsadf": out_gs, "bsadf_path": out_paths}


def get_critical_values(T: int, min_window: int,
                        nrep: int, seed: int = 42,
                        quantiles: Iterable[float] = (0.95,),
                        force_recompute: bool = False,
                        verbose: bool = True) -> dict:
    """
    Cache-or-compute. Cache key: (T, mw, nrep, seed).
    First call simulates and caches; subsequent calls just load.
    """
    quantiles = sorted(set(float(q) for q in quantiles))
    if not force_recompute:
        cached = _load_cache(T, min_window, nrep, seed, quantiles)
        if cached is not None:
            if verbose:
                print(f"  MC: cache hit T={T}, mw={min_window}, nrep={nrep}")
            return cached
    sim_qs = sorted(set(quantiles) | {0.90, 0.95, 0.99})
    cv = simulate_critical_values(T, min_window, nrep, seed, sim_qs, verbose=verbose)
    _save_cache(T, min_window, nrep, seed, cv, sim_qs)
    return {"gsadf":      {q: cv["gsadf"][q]      for q in quantiles},
            "bsadf_path": {q: cv["bsadf_path"][q] for q in quantiles}}


def is_cached(T: int, min_window: int, nrep: int, seed: int) -> bool:
    """Check whether MC results exist on disk for this key."""
    return _load_cache(T, min_window, nrep, seed, [0.95]) is not None
