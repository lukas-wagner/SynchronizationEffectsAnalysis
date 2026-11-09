"""
run_weeks.py — Robustness runs over many price weeks.
==========================================================================
Runs the reference case (EV=PV=HP=30 %) for every calendar week 1–52 and
every price year in PRICE_YEARS, in the style of independent price
realizations: one real week = one realization. Only the prices vary between
price years; PV, temperature and household load are the fixed reference
profiles of that calendar week.

Scenarios: decentral, fixedsoc_times, fixedsoc_random for all weeks;
decentral_m3 additionally for the Module-3 quarters (Q1/Q4 weeks).

Output goes to <repo>/output_weeks/y<year>_w<ww>_<scenario>/ — deliberately
OUTSIDE output/, so analysis/summarize.py on output/ and output/summary.csv
are never affected. Existing folders with a results JSON are skipped, nothing
is ever overwritten. Plots are switched off (only the results JSON is kept).

Usage:
    python run_weeks.py                 # all runs, 4 in parallel
    python run_weeks.py --only 2023:35-39 --price-json <prices.json> --suffix _rerun
    python run_weeks.py --dry-run
    python run_weeks.py --jobs 6
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_sweep

REPO = Path(__file__).resolve().parent
OUT = REPO / "output_weeks"
PRICE_YEARS = [2022, 2023, 2024, 2025]
WEEKS = range(1, 53)
REF = (0.30, 0.30, 0.30)  # ev, pv, hp
ALL_WEEK_SCENARIOS = ["decentral", "fixedsoc_times", "fixedsoc_random"]
M3_SCENARIO = "decentral_m3"
M3_WEEKS = set(range(1, 14)) | set(range(40, 53))  # Q1 and Q4 of the 2024 sim calendar


def week_start_doy(week):
    """Day of year of the Monday of ISO week `week` in the 2024 sim calendar
    (2024-01-01 is a Monday)."""
    return 1 + 7 * (week - 1)


def plan():
    runs = []
    for year in PRICE_YEARS:
        for w in WEEKS:
            if ONLY is not None and (year, w) not in ONLY:
                continue
            scen = list(ALL_WEEK_SCENARIOS)
            if w in M3_WEEKS:
                scen.append(M3_SCENARIO)
            for sc in scen:
                runs.append((year, w, sc))
    return runs


SUFFIX = ""          # e.g. "_rerun" for re-runs with another price file
PRICE_JSON = None    # None = config default (optimization/intraday_prices.json)
ONLY = None          # e.g. {(2023, 35), ...} to restrict the plan


def folder(year, w, sc):
    return OUT / f"y{year}_w{w:02d}_{sc}{SUFFIX}"


def done(path):
    return path.exists() and any(path.glob("*_results.json"))


def build_cfg(base, year, w, sc):
    ev, pv, hp = REF
    key = f"w{w:02d}"
    cfg = run_sweep._build_config(base, ev, pv, hp, "winter", sc)
    cfg = copy.deepcopy(cfg)
    cfg["time"]["season_start_day"][key] = week_start_doy(w)
    cfg["grid"]["season"] = key
    cfg.setdefault("optimization", {})["price_year"] = year
    if PRICE_JSON:
        cfg["optimization"]["price_json"] = str(PRICE_JSON)
    cfg.setdefault("output", {})["make_plots"] = False
    return cfg


def run_one(base, year, w, sc):
    target = folder(year, w, sc)
    if done(target):
        return "skip"
    cfg = build_cfg(base, year, w, sc)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as tf:
        json.dump(cfg, tf)
        tmp = tf.name
    env = os.environ.copy()
    env["SYNCRO_RUN_DIR"] = str(target)  # absolute -> outside output/
    script = run_sweep._SCENARIOS[sc]["script"]
    log = OUT / "_logs" / f"{target.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, script, tmp], cwd=str(REPO),
                               env=env, stdout=fh, stderr=subprocess.STDOUT)
        return "ok" if r.returncode == 0 and done(target) else f"ERROR rc={r.returncode}"
    finally:
        os.unlink(tmp)


def main():
    global SUFFIX, PRICE_JSON, ONLY
    dry = "--dry-run" in sys.argv
    if "--price-json" in sys.argv:
        PRICE_JSON = (REPO / sys.argv[sys.argv.index("--price-json") + 1]).resolve()
    if "--suffix" in sys.argv:
        SUFFIX = sys.argv[sys.argv.index("--suffix") + 1]
    if "--only" in sys.argv:   # "2023:35-39"
        y, ws = sys.argv[sys.argv.index("--only") + 1].split(":")
        a, b = (int(x) for x in ws.split("-"))
        ONLY = {(int(y), w) for w in range(a, b + 1)}
    jobs = 4
    if "--jobs" in sys.argv:
        jobs = int(sys.argv[sys.argv.index("--jobs") + 1])
    base = json.load(open(REPO / "config.json", encoding="utf-8"))
    runs = plan()
    todo = [r for r in runs if not done(folder(*r))]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, "
          f"{len(todo)} to run -> {OUT}")
    if dry:
        for r in todo[:10]:
            print("  ", folder(*r).name)
        return
    OUT.mkdir(exist_ok=True)
    stats = {"ok": 0, "skip": 0, "err": 0}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, base, *r): r for r in todo}
        for i, f in enumerate(futs, 1):
            res = f.result()
            r = futs[f]
            if res == "ok":
                stats["ok"] += 1
            elif res == "skip":
                stats["skip"] += 1
            else:
                stats["err"] += 1
                print(f"  {folder(*r).name}: {res}")
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done", flush=True)
    print(f"Done: {stats['ok']} ok, {stats['err']} errors, {stats['skip']} skipped")


if __name__ == "__main__":
    main()
