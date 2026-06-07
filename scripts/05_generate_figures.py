"""
Step 5 — Generate and save all per-ticker diagnostic and comparison figures,
plus per-segment GSADF panel figures.

Reads pre-computed results from disk and writes PNG files to outputs/figures/.
Always overwrites so figures stay in sync with the latest analysis results.
Tickers with missing results for a given figure type are skipped with a note.

Output layout
-------------
    outputs/figures/sadf/           <ticker>_sadf.png
    outputs/figures/gsadf/          <ticker>_gsadf.png
    outputs/figures/svadf/          <ticker>_svadf.png
    outputs/figures/comparison/     <ticker>_gsadf_vs_svadf.png
                                    <ticker>_sadf_vs_svadf.png
    outputs/figures/gsadf_panels/   gsadf_panel_<segment_slug>.png

Usage
-----
    python scripts/05_generate_figures.py               # all tickers, all figures
    python scripts/05_generate_figures.py --tickers NVDA AMD MSFT
    python scripts/05_generate_figures.py --types sadf gsadf comparison gsadf_panel
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required

from config import settings
from src.plotting.diagnostic import (
    plot_sadf_only, plot_gsadf_only, plot_svadf_only, plot_gsadf_segment_panel,
)
from src.plotting.comparison import plot_comparison, plot_sadf_vs_svadf

FIGURE_TYPES = ["sadf", "gsadf", "svadf", "gsadf_vs_svadf", "sadf_vs_svadf", "gsadf_panel"]

DIRS = {
    "sadf":          settings.FIGURES_DIR / "sadf",
    "gsadf":         settings.FIGURES_DIR / "gsadf",
    "svadf":         settings.FIGURES_DIR / "svadf",
    "gsadf_vs_svadf": settings.FIGURES_DIR / "comparison",
    "sadf_vs_svadf": settings.FIGURES_DIR / "comparison",
    "gsadf_panel":   settings.FIGURES_DIR / "gsadf_panels",
}


def _run(ticker: str, types: list[str]) -> None:
    if "sadf" in types:
        fig = plot_sadf_only(
            ticker,
            save_path=DIRS["sadf"] / f"{ticker}_sadf.png",
            show=False,
        )
        print(f"  sadf           : {'saved' if fig else 'skipped (no results)'}")

    if "gsadf" in types:
        fig = plot_gsadf_only(
            ticker,
            save_path=DIRS["gsadf"] / f"{ticker}_gsadf.png",
            show=False,
        )
        print(f"  gsadf          : {'saved' if fig else 'skipped (no results)'}")

    if "svadf" in types:
        fig = plot_svadf_only(
            ticker,
            save_path=DIRS["svadf"] / f"{ticker}_svadf.png",
            show=False,
        )
        print(f"  svadf          : {'saved' if fig else 'skipped (no results)'}")

    if "gsadf_vs_svadf" in types:
        fig = plot_comparison(
            ticker,
            save_path=DIRS["gsadf_vs_svadf"] / f"{ticker}_gsadf_vs_svadf.png",
            show=False,
        )
        print(f"  gsadf_vs_svadf : {'saved' if fig else 'skipped (no results)'}")

    if "sadf_vs_svadf" in types:
        fig = plot_sadf_vs_svadf(
            ticker,
            save_path=DIRS["sadf_vs_svadf"] / f"{ticker}_sadf_vs_svadf.png",
            show=False,
        )
        print(f"  sadf_vs_svadf  : {'saved' if fig else 'skipped (no results)'}")


def _run_panels(types: list[str]) -> None:
    """Generate one GSADF segment panel PNG per segment."""
    if "gsadf_panel" not in types:
        return
    panel_dir = DIRS["gsadf_panel"]
    panel_dir.mkdir(parents=True, exist_ok=True)
    all_segments = list(dict.fromkeys(
        seg for _, (seg, _) in settings.SEGMENT_OF.items()
    ))
    print(f"\nGenerating GSADF segment panels ({len(all_segments)} segments)…")
    for seg in all_segments:
        slug = "".join(c if c.isalnum() else "_" for c in seg).strip("_")
        save_path = panel_dir / f"gsadf_panel_{slug}.png"
        try:
            fig = plot_gsadf_segment_panel(seg, save_path=save_path, show=False)
            status = "saved" if fig else "skipped (no data)"
        except Exception as exc:
            status = f"error: {exc}"
        print(f"  {seg[:55]:<55} → {status}")


def main(tickers: list[str] | None, types: list[str]) -> None:
    universe = tickers or settings.ALL_TICKERS
    # Ensure output directories exist
    for d in set(DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)

    # Per-ticker figures
    per_ticker_types = [t for t in types if t != "gsadf_panel"]
    if per_ticker_types:
        print(f"Generating per-ticker figures for {len(universe)} ticker(s): {per_ticker_types}")
        for i, tk in enumerate(universe, 1):
            print(f"\n[{i:2d}/{len(universe)}] {tk}")
            _run(tk, per_ticker_types)

    # Segment panel figures (one per segment, independent of --tickers filter)
    _run_panels(types)

    print("\nDone. Figures saved to:", settings.FIGURES_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate and save all diagnostic and comparison figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subset of tickers (default: all from settings)")
    p.add_argument("--types", nargs="*", default=FIGURE_TYPES,
                   choices=FIGURE_TYPES,
                   help=f"Which figure types to generate (default: all). "
                        f"Choices: {FIGURE_TYPES}")
    args = p.parse_args()
    main(tickers=args.tickers, types=args.types)
