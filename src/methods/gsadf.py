"""GSADF / BSADF (Phillips, Shi & Yu 2015), Numba-accelerated."""
from __future__ import annotations
import time
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _gsadf_inner(y: np.ndarray, min_window: int) -> np.ndarray:
    T = len(y)
    npath = T - min_window + 1
    bsadf = np.full(npath, -np.inf)
    for i in range(npath):
        t2 = i + min_window
        best = -np.inf
        for t1 in range(0, t2 - min_window + 1):
            m = t2 - t1
            if m < 5: continue
            n_r = m - 1
            sum1 = float(n_r)
            sumx = 0.0; sumx2 = 0.0; sumy = 0.0; sumxy = 0.0
            for k in range(n_r):
                x_k = y[t1 + k]
                d_k = y[t1 + k + 1] - x_k
                sumx  += x_k; sumx2 += x_k * x_k
                sumy  += d_k; sumxy += x_k * d_k
            denom = sum1 * sumx2 - sumx * sumx
            if abs(denom) < 1e-14: continue
            alpha = (sumx2 * sumy  - sumx * sumxy) / denom
            delta = (sum1  * sumxy - sumx * sumy)  / denom
            sse = 0.0
            for k in range(n_r):
                x_k = y[t1 + k]
                d_k = y[t1 + k + 1] - x_k
                r_k = d_k - alpha - delta * x_k
                sse += r_k * r_k
            if sse < 1e-14: continue
            sigma2 = sse / (n_r - 2)
            var_delta = sigma2 * sum1 / denom
            if var_delta <= 0.0: continue
            t_stat = delta / (var_delta ** 0.5)
            if t_stat > best: best = t_stat
        bsadf[i] = best
    return bsadf


def minimum_window(T: int) -> int:
    """PSY (2015) rule of thumb."""
    return max(2, int(np.floor((0.01 + 1.8 / np.sqrt(T)) * T)))


def gsadf(y, min_window: int | None = None) -> dict:
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if min_window is None:
        min_window = minimum_window(T)
    bsadf_path = _gsadf_inner(y, min_window)
    return {"statistic": float(np.nanmax(bsadf_path)),
            "path": bsadf_path, "min_window": min_window}


def warmup(verbose: bool = True) -> float:
    if verbose:
        print("Compiling Numba GSADF kernel …", end=" ", flush=True)
    t0 = time.time()
    _ = _gsadf_inner(np.cumsum(np.random.randn(50).astype(np.float64)), minimum_window(50))
    elapsed = time.time() - t0
    if verbose:
        print(f"done in {elapsed:.1f}s")
    return elapsed
