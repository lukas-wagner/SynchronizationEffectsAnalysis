"""
run_het.py — HP heterogeneity sensitivity.
====================================================================
Reference case (EV=PV=HP=30 %, winter, week 3, 2024 prices) with
hp.heterogeneity enabled, for several independent draws (seeds). All other
settings identical to the main runs, so the homogeneous results in
output/ev30_pv30_hp30_winter_<scenario>/ are the comparison.

Output goes to <repo>/output_het/s<seed>_<scenario>/ — outside output/, never
overwritten (existing folders with a results JSON are skipped).

Usage:
    python run_het.py [--dry-run] [--jobs 4] [--normalized]
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
OUT = REPO / "output_het"
SEEDS = [101, 102, 103, 104, 105]
SCENARIOS = ["decentral", "fixedsoc_times", "fixedsoc_random",
             "central_cost", "central_peak"]
REF = (0.30, 0.30, 0.30)
SEASON = "winter"


NORMALIZE = "--normalized" in sys.argv   # normalize factors to mean 1 within each draw
SUFFIX = "_norm" if NORMALIZE else ""


def folder(seed, sc):
    return OUT / f"s{seed}_{sc}{SUFFIX}"


def done(path):
    return path.exists() and any(path.glob("*_results.json"))


def run_one(base, seed, sc):
    target = folder(seed, sc)
    if done(target):
        return "skip"
    cfg = copy.deepcopy(run_sweep._build_config(base, *REF, SEASON, sc))
    cfg["hp"].setdefault("heterogeneity", {})
    cfg["hp"]["heterogeneity"]["enabled"] = True
    cfg["hp"]["heterogeneity"]["seed"] = seed
    cfg["hp"]["heterogeneity"]["normalize_per_draw"] = NORMALIZE
    cfg.setdefault("output", {})["make_plots"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as tf:
        json.dump(cfg, tf)
        tmp = tf.name
    env = dict(os.environ, SYNCRO_RUN_DIR=str(target))
    log = OUT / "_logs" / f"{target.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, run_sweep._SCENARIOS[sc]["script"], tmp],
                               cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT)
        return "ok" if r.returncode == 0 and done(target) else f"ERROR rc={r.returncode}"
    finally:
        os.unlink(tmp)


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    base = json.load(open(REPO / "config.json", encoding="utf-8"))
    runs = [(s, sc) for s in SEEDS for sc in SCENARIOS]
    todo = [r for r in runs if not done(folder(*r))]
    print(f"{len(runs)} runs planned, {len(todo)} to run -> {OUT}")
    if "--dry-run" in sys.argv:
        return
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for r, res in zip(todo, ex.map(lambda r: run_one(base, *r), todo)):
            if res not in ("ok", "skip"):
                print(f"  {folder(*r).name}: {res}")
    print("Done:", sum(done(folder(*r)) for r in runs), "of", len(runs), "with results")


if __name__ == "__main__":
    main()
