"""
build_pv.py — convert a renewables.ninja hourly PV CSV to quarter-hourly capacity factors
==========================================================================================
Reads the hourly capacity-factor CSV exported from renewables.ninja (request the
system with capacity = 1 kW so the electricity column equals the capacity factor
directly, 0–1) and writes day-specific quarter-hourly profiles (96 values per day)
to pv_capacity_factors.json in the same folder.

The JSON mirrors the structure of optimization/intraday_prices.json so the
simulation can look up each step by calendar date + quarter-hour slot, with the
same month/day fallback when the exact year is missing.

Run (once, or whenever the CSV is updated):
    python input/build_pv.py [path/to/hourly_pv.csv]
"""

import os
import sys
import json

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "hourly_pv.csv")
OUT_PATH = os.path.join(_SCRIPT_DIR, "pv_capacity_factors.json")

SLOTS_PER_DAY = 96


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Reading {csv_path}")

    df = pd.read_csv(csv_path, comment="#", index_col="time", parse_dates=True)
    if "electricity" not in df.columns:
        raise ValueError(
            f"Expected an 'electricity' column, got: {list(df.columns)}. "
            "Make sure to export from renewables.ninja with capacity = 1 kW.")

    # Add a dummy midnight entry one day past the last value so that the final
    # day's trailing slots (e.g. 23:15, 23:30, 23:45 on Dec 31) can be filled.
    # The value is 0 — it is always dark at that hour in December.
    midnight_next = pd.Timestamp(df.index[-1].date()) + pd.Timedelta(days=1)
    df = pd.concat([df[["electricity"]],
                    pd.DataFrame({"electricity": [0.0]}, index=[midnight_next])])

    # Resample from hourly to 15-min using linear interpolation.
    cf_15min = df["electricity"].resample("15min").interpolate("linear").clip(lower=0.0)

    # Organise by calendar date: 96 quarter-hourly values per day.
    cf_by_date = {}
    for date, group in cf_15min.groupby(cf_15min.index.date):
        vals = group.values
        if len(vals) != SLOTS_PER_DAY:
            continue                         # skip incomplete days (DST transitions)
        cf_by_date[str(date)] = [round(float(v), 5) for v in vals]

    if not cf_by_date:
        raise ValueError("No complete days found in the CSV.")

    all_vals = [v for arr in cf_by_date.values() for v in arr]
    meta = {
        "source": os.path.basename(csv_path),
        "unit": "capacity factor (0-1, normalised to 1 kW system)",
        "resolution": "quarter-hourly (96 values per day)",
        "n_days": len(cf_by_date),
        "date_range": [min(cf_by_date), max(cf_by_date)],
        "peak_capacity_factor": round(float(np.max(all_vals)), 5),
        "mean_capacity_factor": round(float(np.mean(all_vals)), 5),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "capacity_factors": cf_by_date}, fh, indent=0)

    print(f"Wrote {len(cf_by_date)} day profiles -> {OUT_PATH}")
    print(f"  date range : {meta['date_range']}")
    print(f"  peak CF    : {meta['peak_capacity_factor']:.4f}")
    print(f"  mean CF    : {meta['mean_capacity_factor']:.4f}")


if __name__ == "__main__":
    main()
