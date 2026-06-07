"""
Pipeline runner — executes all four analysis scripts in order.

Steps
-----
  1  01_download_data.py      Download / refresh price cache
  2  02_run_montecarlo.py     Pre-compute GSADF Monte Carlo critical values
  3  03_run_analysis.py       Run all methods on the main universe
  4  04_run_validation.py     Run all methods on historical validation cases

Usage
-----
Run everything (default):
    python scripts/00_run_pipeline.py

Force re-download and re-run everything from scratch:
    python scripts/00_run_pipeline.py --force --force-mc

Skip individual steps (e.g. skip download if prices already exist):
    python scripts/00_run_pipeline.py --skip 1

Run only specific steps:
    python scripts/00_run_pipeline.py --steps 2 3

Pass extra arguments to step 3 (03_run_analysis.py):
    python scripts/00_run_pipeline.py --methods gsadf svadf
    python scripts/00_run_pipeline.py --mc-reps 499

Examples
--------
    # Full fresh run
    python scripts/00_run_pipeline.py --force --force-mc

    # Already have prices and MC; only redo analysis + validation
    python scripts/00_run_pipeline.py --skip 1 2

    # Only GSADF + SV-ADF, skip validation
    python scripts/00_run_pipeline.py --methods gsadf svadf --skip 4
"""
from __future__ import annotations
import sys
import argparse
import subprocess
import time
from pathlib import Path

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent


# ── Step definitions ──────────────────────────────────────────────────────────
# Each entry: (step_number, script_filename, description)
STEPS = [
    (1, "01_download_data.py",    "Download / refresh price cache"),
    (2, "02_run_montecarlo.py",   "Pre-compute MC critical values"),
    (3, "03_run_analysis.py",     "Run all methods on main universe"),
    (4, "04_run_validation.py",   "Run methods on historical validation cases"),
    (5, "05_generate_figures.py", "Generate and save all figures"),
]


def _build_argv(step_num: int, args: argparse.Namespace) -> list[str]:
    """Translate pipeline CLI args into per-script argv."""
    extra: list[str] = []

    if step_num == 1:
        if args.force:
            extra += ["--force"]

    elif step_num == 2:
        if args.force_mc:
            extra += ["--force"]
        if args.mc_reps:
            extra += ["--mc-reps", str(args.mc_reps)]
        if args.tickers:
            extra += ["--tickers"] + args.tickers

    elif step_num == 3:
        methods = args.methods or ["all"]
        extra += ["--methods"] + methods
        if args.force:
            extra += ["--force"]
        if args.force_mc:
            extra += ["--force-mc"]
        if args.mc_reps:
            extra += ["--mc-reps", str(args.mc_reps)]
        if args.tickers:
            extra += ["--tickers"] + args.tickers

    elif step_num == 4:
        methods = args.methods or ["all"]
        extra += ["--methods"] + methods

    elif step_num == 5:
        if args.tickers:
            extra += ["--tickers"] + args.tickers

    return extra


def _run_step(step_num: int, script: str, description: str,
              extra_argv: list[str]) -> tuple[bool, float]:
    """Run a single step; return (success, elapsed_seconds)."""
    script_path = SCRIPTS_DIR / script
    cmd = [sys.executable, str(script_path)] + extra_argv

    print(f"\n{'─'*70}")
    print(f"  STEP {step_num}  {description}")
    print(f"  cmd : {' '.join(cmd)}")
    print(f"{'─'*70}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    print(f"\n  → Step {step_num} {status}  ({elapsed:.1f}s)")
    return ok, elapsed


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the full AI-bubble-detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--steps", nargs="*", type=int, default=None,
                   metavar="N",
                   help="Which step numbers to run (default: all). "
                        "E.g. --steps 1 3")
    p.add_argument("--skip", nargs="*", type=int, default=None,
                   metavar="N",
                   help="Step numbers to skip. E.g. --skip 1 2")
    p.add_argument("--force", action="store_true",
                   help="Pass --force to steps 1 and 3 (re-download / recompute)")
    p.add_argument("--force-mc", action="store_true",
                   help="Pass --force-mc to step 3 (re-run Monte Carlo)")
    p.add_argument("--mc-reps", type=int, default=None,
                   help="Override Monte Carlo repetitions (steps 2 and 3)")
    p.add_argument("--methods", nargs="*", default=None,
                   choices=["stationarity", "sadf", "gsadf", "svadf", "all"],
                   help="Methods to run in steps 3 and 4 (default: all)")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Restrict steps 2 and 3 to these tickers")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the pipeline if any step fails")
    args = p.parse_args()

    # Determine which steps to execute
    all_nums  = [s[0] for s in STEPS]
    run_nums  = set(args.steps) if args.steps else set(all_nums)
    skip_nums = set(args.skip)  if args.skip  else set()
    to_run    = sorted(run_nums - skip_nums)

    if not to_run:
        print("No steps selected. Use --steps or remove --skip entries.")
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"  AI Bubble Detection — Pipeline Runner")
    print(f"  Steps to run : {to_run}")
    print(f"  force        : {args.force}")
    print(f"  force-mc     : {args.force_mc}")
    print(f"  mc-reps      : {args.mc_reps or 'default'}")
    print(f"  methods      : {args.methods or 'all'}")
    print(f"  stop-on-error: {args.stop_on_error}")
    print(f"{'='*70}")

    pipeline_start = time.time()
    results: list[tuple[int, str, bool, float]] = []

    for step_num, script, description in STEPS:
        if step_num not in to_run:
            print(f"\n  [skip] Step {step_num}: {description}")
            continue

        extra = _build_argv(step_num, args)
        ok, elapsed = _run_step(step_num, script, description, extra)
        results.append((step_num, description, ok, elapsed))

        if not ok and args.stop_on_error:
            print(f"\n  Pipeline aborted at step {step_num} (--stop-on-error).")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE SUMMARY   (total: {total/60:.1f} min)")
    print(f"{'='*70}")
    for step_num, desc, ok, elapsed in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  Step {step_num}  {desc:<45}  {elapsed:6.1f}s")

    n_failed = sum(1 for _, _, ok, _ in results if not ok)
    if n_failed:
        print(f"\n  {n_failed} step(s) failed.")
        sys.exit(1)
    else:
        print(f"\n  All steps completed successfully.")


if __name__ == "__main__":
    main()
