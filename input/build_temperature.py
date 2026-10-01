"""
build_temperature.py — convert a renewables.ninja hourly temperature CSV
to quarter-hourly profiles for the SyncroEffects heat pump model.
=========================================================================
Reads the hourly 2m air-temperature CSV exported from renewables.ninja
(variable t2m, in °C) and writes day-specific quarter-hourly profiles
(96 values per day) to temperature_profiles.json in the same folder.

The JSON mirrors the structure of pv_capacity_factors.json so the model
can look up each step by calendar date + quarter-hour slot with the same
month/day fallback when the exact year is missing.

Run (once, or whenever the CSV is updated):
    python input/build_temperature.py [path/to/hourly_temperature.csv]
"""

import os
import sys
import json

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, "ninja_temperature_2025.csv")
OUT_PATH = os.path.join(_SCRIPT_DIR, "temperature_profiles.json")

SLOTS_PER_DAY = 96


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    print(f"Reading {csv_path}")

    df = pd.read_csv(csv_path, comment="#", index_col="time", parse_dates=True)
    if "t2m" not in df.columns:
        raise ValueError(
            f"Expected a 't2m' column, got: {list(df.columns)}. "
            "Make sure to export from renewables.ninja with variable t2m.")

    # Add a dummy entry one day past the last value so that the final
    # day's trailing 15-min slots can be filled by interpolation.
    midnight_next = pd.Timestamp(df.index[-1].date()) + pd.Timedelta(days=1)
    df = pd.concat([df[["t2m"]],
                    pd.DataFrame({"t2m": [float(df["t2m"].iloc[-1])]},
                                 index=[midnight_next])])

    # Resample from hourly to 15-min using linear interpolation.
    t_15min = df["t2m"].resample("15min").interpolate("linear")

    # Organise by calendar date: 96 quarter-hourly values per day.
    t_by_date = {}
    for date, group in t_15min.groupby(t_15min.index.date):
        vals = group.values
        if len(vals) != SLOTS_PER_DAY:
            continue  # skip incomplete days (DST transitions)
        t_by_date[str(date)] = [round(float(v), 3) for v in vals]

    if not t_by_date:
        raise ValueError("No complete days found in the CSV.")

    all_vals = [v for arr in t_by_date.values() for v in arr]
    meta = {
        "source": os.path.basename(csv_path),
        "unit": "°C (2m air temperature)",
        "resolution": "quarter-hourly (96 values per day)",
        "n_days": len(t_by_date),
        "date_range": [min(t_by_date), max(t_by_date)],
        "t_min_celsius": round(float(np.min(all_vals)), 3),
        "t_max_celsius": round(float(np.max(all_vals)), 3),
        "t_mean_celsius": round(float(np.mean(all_vals)), 3),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({"metadata": meta, "temperatures": t_by_date}, fh, indent=0)

    print(f"Wrote {len(t_by_date)} day profiles -> {OUT_PATH}")
    print(f"  date range  : {meta['date_range']}")
    print(f"  temperature : {meta['t_min_celsius']:.1f} to {meta['t_max_celsius']:.1f} °C")
    print(f"  mean        : {meta['t_mean_celsius']:.1f} °C")


if __name__ == "__main__":
    main()
