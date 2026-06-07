"""
Step 2 — Pre-compute Monte Carlo critical values for GSADF.

Skips any (T, min_window, nrep, seed) already in the cache.  Adding a new
ticker only triggers MC for that ticker; an interrupted run can be resumed
by simply running this script again — completed tickers are skipped.

Usage:
    python scripts/02_run_montecarlo.py                # uses settings.GSADF_MC_REPS
    python scripts/02_run_montecarlo.py --mc-reps 499
    python scripts/02_run_montecarlo.py --force        # re-run everything
    python scripts/02_run_montecarlo.py --tickers NVDA AMD   # only these
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.io.data         import load_or_download_prices
from src.methods.gsadf   import minimum_window, warmup
from src.monte_carlo     import get_critical_values, is_cached


def main(mc_reps: int, force: bool, tickers: list[str] | None) -> None:
    print(f"Monte Carlo precompute  |  MC reps = {mc_reps}  seed = {settings.SEED}")
    print(f"Cache dir: {settings.MC_CACHE_DIR}\n")
    settings.MC_PATHS_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_or_download_prices()
    warmup()

    universe = tickers if tickers else settings.ALL_TICKERS
    seen = set()
    t_start = time.time()
    skipped = 0; computed = 0

    for tk in universe:
        if tk not in prices.columns:
            print(f"[skip] {tk}: not in prices"); continue
        s = prices[tk].dropna()
        T = len(s)
        if T < 100:
            print(f"[skip] {tk}: too short (T={T})"); continue
        mw = minimum_window(T)
        key = (T, mw, mc_reps, settings.SEED)
        if key in seen:
            print(f"[dup ] {tk}: same (T={T}, mw={mw}) as earlier ticker"); continue
        seen.add(key)
        if not force and is_cached(T, mw, mc_reps, settings.SEED):
            print(f"[hit ] {tk}: T={T}, mw={mw} — already cached"); skipped += 1
            continue
        print(f"[run ] {tk}: T={T}, mw={mw}")
        get_critical_values(
            T=T, min_window=mw, nrep=mc_reps, seed=settings.SEED,
            quantiles=(0.90, 0.95, 0.99),
            force_recompute=force, verbose=True,
        )
        computed += 1

    elapsed = time.time() - t_start
    print(f"\nDone. Computed {computed}, skipped {skipped} (already cached). "
          f"Total time: {elapsed/60:.1f} min")
    print(f"Index file: {settings.MC_GLOBAL_INDEX_FILE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mc-reps", type=int, default=settings.GSADF_MC_REPS,
                   help=f"Monte Carlo repetitions (default: {settings.GSADF_MC_REPS})")
    p.add_argument("--force", action="store_true",
                   help="Re-simulate even if cached")
    p.add_argument("--tickers", nargs="*",
                   help="Optional subset of tickers (default: all)")
    args = p.parse_args()
    main(mc_reps=args.mc_reps, force=args.force, tickers=args.tickers)
