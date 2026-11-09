"""
Energy-consistent synchronised charging — all EVs armed at the same hour.
========================================================================
Like fixed_times, but each EV only charges while at home and only until its
state of charge is back at the target (see fixedsoc_common). Every EV's daily
session is armed at ``scenarios.fixed_times.start_hour``.

Run:
    python simulation/fixedsoc_times.py [path/to/config.json]
"""

import sys
import numpy as np
import sim_core
import fixedsoc_common

cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
ctx = sim_core.load_grid(cfg)

n_ev = len(sim_core.select_ev_loads(cfg, ctx))
start_hours = np.full(n_ev, float(cfg["scenarios"]["fixed_times"]["start_hour"]))
fixedsoc_common.run(cfg, ctx, start_hours, "fixedsoc_times",
                    "Synchronised charging (energy-consistent)")
