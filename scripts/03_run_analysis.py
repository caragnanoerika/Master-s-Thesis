"""
Step 3 — Run diagnostic methods.

Methods are independent; you can run any subset. SV-ADF uses its own window
(configurable per call); the other methods use the FIXED window from settings.

Examples
--------
Run everything:
    python scripts/03_run_analysis.py --methods all

Run only SV-ADF on a specific window:
    python scripts/03_run_analysis.py --methods svadf \\
        --svadf-start 2022-11-01 --svadf-end 2026-05-01

Run only GSADF (uses cached MC; falls back to compute-and-cache if missing):
    python scripts/03_run_analysis.py --methods gsadf

Run stationarity tests only:
    python scripts/03_run_analysis.py --methods stationarity

Run multiple but not SV-ADF:
    python scripts/03_run_analysis.py --methods stationarity sadf gsadf

Force recompute of a specific method (overrides skip-if-exists):
    python scripts/03_run_analysis.py --methods gsadf --force

Subset tickers:
    python scripts/03_run_analysis.py --methods all --tickers NVDA AMD

Override MC reps just for this run:
    python scripts/03_run_analysis.py --methods gsadf --mc-reps 499
"""
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.io.data            import load_or_download_prices
from src.methods.gsadf      import warmup
from src.pipelines.stationarity import run_one as run_stationarity
from src.pipelines.sadf         import run_one as run_sadf, run_windowed as run_sadf_windowed
from src.pipelines.gsadf        import run_one as run_gsadf
from src.pipelines.svadf        import run_one as run_svadf


METHOD_CHOICES = ["stationarity", "sadf", "gsadf", "svadf", "all"]


def main(methods: list[str], tickers: list[str] | None,
         force: bool, force_mc: bool,
         mc_reps: int | None,
         svadf_start: str | None, svadf_end: str | None) -> None:

    if "all" in methods:
        methods = ["stationarity", "sadf", "gsadf", "svadf"]
    print(f"Methods to run: {methods}")
    print(f"Force: {force}   force_mc: {force_mc}")
    print(f"MC reps: {mc_reps or settings.GSADF_MC_REPS}")
    if "svadf" in methods:
        print(f"SV-ADF window: "
              f"{svadf_start or settings.SVADF_DEFAULT_START}  →  "
              f"{svadf_end or settings.SVADF_DEFAULT_END}")

    prices = load_or_download_prices()
    if "gsadf" in methods:
        warmup()

    universe = tickers if tickers else settings.ALL_TICKERS
    t_start = time.time()

    for i, tk in enumerate(universe, 1):
        if tk not in prices.columns:
            print(f"\n[{i:2d}/{len(universe)}] {tk}: not in prices"); continue
        series = prices[tk].dropna()
        if len(series) < 100:
            print(f"\n[{i:2d}/{len(universe)}] {tk}: too short"); continue
        print(f"\n[{i:2d}/{len(universe)}] {tk}")

        if "stationarity" in methods:
            t = time.time()
            res = run_stationarity(tk, series, force=force)
            tag = f"computed ({time.time()-t:.1f}s)" if res else "cached, skipped"
            print(f"  stationarity : {tag}")

        if "sadf" in methods:
            t = time.time()
            res = run_sadf(tk, series, force=force)
            tag = f"computed ({time.time()-t:.1f}s)" if res else "cached, skipped"
            print(f"  sadf [fixed] : {tag}")
            # Windowed SADF — run on the same windows used for SV-ADF so that
            # plot_sadf_vs_svadf can compare them on a common date range.
            if svadf_start or svadf_end:
                sadf_windows = [(svadf_start or settings.SVADF_DEFAULT_START,
                                 svadf_end   or settings.SVADF_DEFAULT_END)]
            else:
                sadf_windows = [
                    (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
                    (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
                ]
            for w_start, w_end in sadf_windows:
                t = time.time()
                res = run_sadf_windowed(tk, series, w_start, w_end, force=force)
                lbl = f"sadf [{w_start}→{w_end[:7]}]"
                tag = f"computed ({time.time()-t:.1f}s)" if res else "cached, skipped"
                print(f"  {lbl}: {tag}")

        if "gsadf" in methods:
            t = time.time()
            res = run_gsadf(tk, series, mc_reps=mc_reps,
                            force=force, force_mc=force_mc, verbose=True)
            if res:
                print(f"  gsadf        : computed ({time.time()-t:.1f}s)  "
                      f"stat={res['statistic']:+.2f} cv={res['cv_value']:.2f} "
                      f"episodes={len(res['episodes'])}")
            else:
                print(f"  gsadf        : cached, skipped")

        if "svadf" in methods:
            # When a custom window is given via CLI run only that window;
            # otherwise run both the default (post-ChatGPT) and the
            # pre-ChatGPT comparison window defined in settings.
            if svadf_start or svadf_end:
                svadf_windows = [(svadf_start or settings.SVADF_DEFAULT_START,
                                  svadf_end   or settings.SVADF_DEFAULT_END)]
            else:
                svadf_windows = [
                    (settings.SVADF_DEFAULT_START, settings.SVADF_DEFAULT_END),
                    (settings.SVADF_PRE_GPT_START, settings.SVADF_PRE_GPT_END),
                ]
            for w_start, w_end in svadf_windows:
                t = time.time()
                res = run_svadf(tk, series,
                                window_start=w_start, window_end=w_end,
                                force=force)
                lbl = f"svadf [{w_start}→{w_end[:7]}]"
                if res:
                    ep  = res["episode"]
                    tag = (f"episode {ep['start']} → {ep['end']} [{ep['collapse_type']}]"
                           if ep else "no episode")
                    print(f"  {lbl}: computed ({time.time()-t:.1f}s)  {tag}")
                else:
                    print(f"  {lbl}: cached, skipped")

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["all"],
                   choices=METHOD_CHOICES,
                   help="Which methods to run")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers (default: all from settings)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a result file already exists")
    p.add_argument("--force-mc", action="store_true",
                   help="Re-run Monte Carlo simulations (slow)")
    p.add_argument("--mc-reps", type=int, default=None,
                   help=f"Override Monte Carlo repetitions "
                        f"(default: {settings.GSADF_MC_REPS})")
    p.add_argument("--svadf-start", default=None,
                   help=f"SV-ADF window start (default: {settings.SVADF_DEFAULT_START})")
    p.add_argument("--svadf-end",   default=None,
                   help=f"SV-ADF window end (default: {settings.SVADF_DEFAULT_END})")
    args = p.parse_args()
    main(methods=args.methods, tickers=args.tickers,
         force=args.force, force_mc=args.force_mc,
         mc_reps=args.mc_reps,
         svadf_start=args.svadf_start, svadf_end=args.svadf_end)
