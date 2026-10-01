"""
build_prices.py — convert the Intraday market Excel into day-specific prices
===========================================================================
Reads the "Data" sheet of the Intraday export and writes day-specific
*quarter-hourly* price profiles (96 values per day) in EUR/kWh to
``optimization/intraday_prices.json``.

The Excel pools all traded Intraday contracts together, mixed by delivery
length (the ``IntervalLength`` column: 0.25 = quarter hour, 0.5 = half hour,
1.0 = hour, plus longer block products). We keep only the genuine
quarter-hour products (``IntervalLength`` ≈ 0.25) so the price signal matches
the 15-minute resolution of the simulation/optimisation.

The optimisation then uses these prices via ``optimization.price_json`` in
config.json: for each simulated step it looks up the matching calendar date and
quarter-hour slot (falling back to the same month/day of another year, then the
overall mean if a date is missing).

Run (once, or whenever the Excel is updated):
    python optimization/build_prices.py path/to/Intraday_fortlaufend_*.xlsx
"""

import os
import sys
import json

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_SCRIPT_DIR, "intraday_prices.json")

SLOTS_PER_DAY = 96      # quarter-hourly resolution (24 h * 4)
SLOT_MINUTES = 15

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python optimization/build_prices.py <Intraday_export.xlsx>")
    xlsx = sys.argv[1]
    print(f"Reading {xlsx}")
    # The 'Data' sheet has some preamble rows; locate the real header row by
    # finding the one that contains the 'WeightedAveragePrice' column label.
    raw = pd.read_excel(xlsx, sheet_name="Data", header=None)
    hdr = None
    for i in range(min(15, len(raw))):
        if (raw.iloc[i].astype(str) == "WeightedAveragePrice").any():
            hdr = i
            break
    if hdr is None:
        raise ValueError("Could not find 'WeightedAveragePrice' header in 'Data' sheet")
    df = raw.iloc[hdr + 1:].copy()
    df.columns = raw.iloc[hdr].tolist()
    df = df[["Start", "IntervalLength", "WeightedAveragePrice"]].dropna(
        subset=["Start", "WeightedAveragePrice"])
    df["Start"] = pd.to_datetime(df["Start"])

    # Keep only the quarter-hour products. IntervalLength is stored as a float
    # with rounding noise (0.24999.. / 0.25000..), so round before comparing.
    il = pd.to_numeric(df["IntervalLength"], errors="coerce").round(2)
    df = df[il == 0.25].copy()

    df["date"] = df["Start"].dt.strftime("%Y-%m-%d")
    # quarter-hour slot of the day, 0..95
    df["slot"] = df["Start"].dt.hour * 4 + df["Start"].dt.minute // SLOT_MINUTES
    # EUR/MWh -> EUR/kWh
    df["price"] = df["WeightedAveragePrice"].astype(float) / 1000.0

    prices = {}
    for date, g in df.groupby("date"):
        # several contracts can settle for the same slot (e.g. around the DST
        # switch) → average them, then fill any missing slots.
        slots = g.groupby("slot")["price"].mean().reindex(range(SLOTS_PER_DAY))
        slots = slots.interpolate().ffill().bfill()
        if slots.isna().any():
            continue
        prices[date] = [round(float(v), 5) for v in slots.values]

    all_vals = [v for arr in prices.values() for v in arr]
    meta = {
        "source": os.path.basename(xlsx),
        "value": "WeightedAveragePrice (volume-weighted), IntervalLength=0.25",
        "unit": "EUR/kWh",
        "resolution": "quarter-hourly (96 values per day)",
        "n_days": len(prices),
        "date_range": [min(prices), max(prices)] if prices else None,
        "mean_eur_per_kwh": round(float(np.mean(all_vals)), 5) if all_vals else None,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "prices": prices}, fh, indent=0)

    print(f"Wrote {len(prices)} day profiles -> {OUT_PATH}")
    print(f"  range {meta['date_range']}, mean "
          f"{meta['mean_eur_per_kwh']*100:.2f} ct/kWh")


if __name__ == "__main__":
    main()
