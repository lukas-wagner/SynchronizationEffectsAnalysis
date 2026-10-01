"""
Synchronised charging — all EVs start at the same fixed hour.
============================================================
A share of households (config ``ev.penetration``) get a wallbox that charges at
``scenarios.fixed_times.start_hour`` for ``duration_h`` hours — every EV at once
(worst-case synchronisation). The shared pipeline (sim_core) handles the rest.

Run:
    python simulation/fixed_times.py [path/to/config.json]
"""

import sys
import numpy as np
import sim_core
import sim_io

cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
ctx = sim_core.load_grid(cfg)

# Build the EV charging matrix: same fixed window every day, for the EV loads.
ev_loads = sim_core.select_ev_loads(cfg, ctx)
sc = cfg["scenarios"]["fixed_times"]
wallbox_mw = float(cfg["ev"]["wallbox_kw"]) / 1000.0
charge_steps = int(round(float(sc["duration_h"]) * ctx["sph"]))
start_slot = int(sc["start_hour"]) * ctx["sph"]

ev_mw = sim_core.zero_ev_load(ctx)
for day in range(ctx["T"] // ctx["spd"] + 1):
    s = day * ctx["spd"] + start_slot
    e = min(s + charge_steps, ctx["T"])
    if s >= ctx["T"]:
        break
    for col in ev_loads:
        ev_mw[s:e, col] = wallbox_mw
ev_mw += sim_core.hp_load_uncontrolled(cfg, ctx)

tag = sim_io.run_tag(cfg)
sim_core.run_and_report(cfg, ctx, ev_mw, scenario=f"fixed_times_{tag}",
                        label="Synchronised charging")
