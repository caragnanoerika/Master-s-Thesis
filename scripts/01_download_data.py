"""
Step 1 — Download prices for the FIXED sample window (2019-01-01 → 2026-05-01).

This is the canonical price file used by stationarity, SADF, and GSADF.
SV-ADF can be run on any sub-window of this same file.

Usage:
    python scripts/01_download_data.py            # uses cache if present
    python scripts/01_download_data.py --force    # force re-download
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.io.data import load_or_download_prices

def main(force: bool = False) -> None:
    print(f"Sample window: {settings.FIXED_START} → {settings.FIXED_END}")
    print(f"Universe:      {len(settings.ALL_TICKERS)} assets")
    prices = load_or_download_prices(force=force)
    print(f"\nPrices shape: {prices.shape[0]} rows × {prices.shape[1]} tickers")
    print(prices.tail(3))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    main(force=p.parse_args().force)
