"""
weeks_analysis.py — KPIs of the week-realization runs (run_weeks.py).

Reads output_weeks/y<year>_w<ww>_<scenario>/*_results.json and writes
output_weeks/weeks_summary.csv (one row per run) with grid KPIs and the price
KPIs of the week actually used (same ISO-week mapping as price_year in the
optimisation). Does not touch output/.

Usage:  python analysis/weeks_analysis.py [--jobs 6]
"""

import csv
import datetime as dt
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEEKS = REPO / "output_weeks"
PRICES = REPO / "optimization" / "intraday_prices.json"
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def grid_kpis(path):
    d = json.load(open(path, encoding="utf-8"))
    tr = [r["loading_percent"] or 0.0 for r in d["transformer"]]
    line = {}
    for r in d["lines"]:
        v = r.get("loading_percent")
        if v is not None:
            line[r["step"]] = max(line.get(r["step"], 0.0), v)
    vm = [r["vm_pu"] for r in d["buses"] if r.get("vm_pu") is not None]
    uv_steps = {r["step"] for r in d["buses"] if r.get("vm_pu") is not None and r["vm_pu"] < 0.9}
    i_max = max(range(len(tr)), key=lambda i: tr[i])
    return {"trafo_max_pct": max(tr), "trafo_h_above_100": sum(x > 100 for x in tr) * 0.25,
            "peak_hour": int(d["transformer"][i_max]["timestamp"][11:13]),
            "line_max_pct": max(line.values()), "vm_min_pu": min(vm),
            "h_undervoltage": len(uv_steps) * 0.25}


def price_kpis(prices, year, week):
    days = [dt.date.fromisocalendar(year, week, wd).isoformat() for wd in range(1, 8)]
    arrs = [prices[d] for d in days if d in prices and len(prices[d]) == 96]
    if len(arrs) < 7:
        # Missing price days: the optimisation fell back to another year's
        # prices for these days -> exclude the week from statistics.
        return {"price_complete": False}
    vals = [v for a in arrs for v in a]
    hmin = [min(range(len(a)), key=lambda j: a[j]) * 24 // len(a) for a in arrs]
    return {"price_complete": True,
            "price_mean_ct": 100 * sum(vals) / len(vals),
            "price_spread_ct": 100 * sorted(max(a) - min(a) for a in arrs)[len(arrs) // 2],
            "neg_share": sum(v < 0 for v in vals) / len(vals),
            "min_night_share": sum(h < 6 for h in hmin) / len(hmin),
            "min_midday_share": sum(10 <= h < 16 for h in hmin) / len(hmin)}


def one(folder):
    m = re.match(r"y(\d{4})_w(\d{2})_(.+)", folder.name)
    if not m:
        return None
    res = list(folder.glob("*_results.json"))
    if not res:
        return None
    year, week, sc = int(m.group(1)), int(m.group(2)), m.group(3)
    monday = dt.date(2024, 1, 1) + dt.timedelta(weeks=week - 1)  # sim calendar
    row = {"year": year, "week": week, "scenario": sc,
           "season": SEASON[monday.month],
           "quarter": (monday.month - 1) // 3 + 1}
    row.update(grid_kpis(res[0]))
    return row


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    folders = sorted(p for p in WEEKS.iterdir() if p.is_dir() and not p.name.startswith("_"))
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = [r for r in ex.map(one, folders, chunksize=4) if r]
    prices = json.load(open(PRICES, encoding="utf-8"))["prices"]
    for r in rows:
        r.update(price_kpis(prices, r["year"], r["week"]))
    out = WEEKS / "weeks_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["year"], r["week"], r["scenario"])))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
