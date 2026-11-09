"""
Energy-consistent desynchronised charging — each EV armed at its own hour.
=========================================================================
Like fixed_but_random_times (same per-EV start-hour draw, reused every day),
but each EV only charges while at home and only until its state of charge is
back at the target (see fixedsoc_common).

Run:
    python simulation/fixedsoc_random.py [path/to/config.json]
"""

import sys
import numpy as np
import sim_core
import fixedsoc_common

cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
ctx = sim_core.load_grid(cfg)

n_ev = len(sim_core.select_ev_loads(cfg, ctx))
sc = cfg["scenarios"]["fixed_but_random_times"]
# Same draw as fixed_but_random_times.py, so each EV keeps its start hour.
rng = np.random.default_rng(int(cfg["ev"]["assignment_seed"]))
start_hours = np.clip(
    rng.normal(sc["start_hour_mean"], sc["start_hour_std"], n_ev),
    sc["start_hour_min"], sc["start_hour_max"])
fixedsoc_common.run(cfg, ctx, start_hours, "fixedsoc_random",
                    "Desynchronised charging (energy-consistent)")
