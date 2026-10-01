"""
Desynchronised charging — each EV has its own randomised start time.
===================================================================
Like fixed_times, but each EV household draws a charging start hour from a normal
distribution (config ``scenarios.fixed_but_random_times``). Same household, same
time each day, but not synchronised with the others. Shared pipeline: sim_core.

Run:
    python simulation/fixed_but_random_times.py [path/to/config.json]
"""

import sys
import numpy as np
import sim_core
import sim_io

cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
ctx = sim_core.load_grid(cfg)

ev_loads = sim_core.select_ev_loads(cfg, ctx)
sc = cfg["scenarios"]["fixed_but_random_times"]
wallbox_mw = float(cfg["ev"]["wallbox_kw"]) / 1000.0
charge_steps = int(round(float(sc["duration_h"]) * ctx["sph"]))

# One random start hour per EV (reused every day), reproducible via the seed.
rng = np.random.default_rng(int(cfg["ev"]["assignment_seed"]))
start_hours = np.clip(
    rng.normal(sc["start_hour_mean"], sc["start_hour_std"], len(ev_loads)),
    sc["start_hour_min"], sc["start_hour_max"])

ev_mw = sim_core.zero_ev_load(ctx)
for k, col in enumerate(ev_loads):
    start_slot = int(round(start_hours[k] * ctx["sph"]))
    for day in range(ctx["T"] // ctx["spd"] + 1):
        s = day * ctx["spd"] + start_slot
        e = min(s + charge_steps, ctx["T"])
        if s >= ctx["T"]:
            break
        ev_mw[s:e, col] = wallbox_mw
ev_mw += sim_core.hp_load_uncontrolled(cfg, ctx)

tag = sim_io.run_tag(cfg)
sim_core.run_and_report(cfg, ctx, ev_mw, scenario=f"fixed_but_random_{tag}",
                        label="Desynchronised charging")
