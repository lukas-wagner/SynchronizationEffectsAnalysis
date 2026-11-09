"""
weeks_energy.py — delivered EV/HP energy of the price-based strategies per week
(price-based strategies may charge more than the trip energy at negative prices).

Re-solves only the optimisation (no power flow) for every week realization of
run_weeks.py, same configuration, and writes output_weeks/weeks_energy.csv:
EV grid energy per EV-week, mean/max end SoC, HP grid energy, number of
negative-price quarter-hours of the week.
Fixed strategies need no re-solve: they deliver exactly trip energy / eta
(see output/*fixedsoc*/*_energy_*.json).

Usage:  python analysis/weeks_energy.py [--jobs 6]
"""

import contextlib
import csv
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "simulation"))
sys.path.insert(0, str(REPO / "optimization"))


def solve(args):
    year, week, sc = args
    import run_weeks
    import sim_core
    import optimize_charging as oc
    base = json.load(open(REPO / "config.json", encoding="utf-8"))
    cfg = run_weeks.build_cfg(base, year, week, sc)
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = sim_core.load_grid(cfg)
        evs = oc.assign_evs(cfg, ctx)
        hps = oc.assign_hps(cfg, ctx)
        p, soc, info = oc.build_and_solve(cfg, ctx, evs, hps=hps)
        prices = oc.price_vector(cfg, ctx)
    dt = 1.0 / ctx["sph"]
    php = info.get("p_hp_sol")
    return {"year": year, "week": week, "scenario": sc,
            "ev_kwh_per_ev": float(p.sum() * dt / max(1, len(evs))),
            "ev_soc_end_mean": float(soc[:, -1].mean()),
            "ev_soc_end_max": float(soc[:, -1].max()),
            "hp_kwh_total": float(php.sum() * dt) if php is not None else 0.0,
            "neg_price_quarter_hours": int((prices < 0).sum())}


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    import run_weeks
    tasks = [(y, w, sc) for (y, w, sc) in run_weeks.plan()
             if sc in ("decentral", "decentral_m3")]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(solve, tasks), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    out = REPO / "output_weeks" / "weeks_energy.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["year"], r["week"], r["scenario"])))
    print(f"{len(rows)} solves -> {out}")


if __name__ == "__main__":
    main()
