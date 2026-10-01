"""
Baseline scenario — residential loads only, no EVs.
===================================================
Runs the SimBench LV feeder with just the household load profiles. Everything
(grid, season, horizon, limits) comes from the project config.json; the shared
pipeline in sim_core does the power flow, KPIs, plots and JSON export.

Run:
    python simulation/baseline_grid.py [path/to/config.json]
"""

import sys
import sim_core
import sim_io

cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
ctx = sim_core.load_grid(cfg)

ev_mw = sim_core.zero_ev_load(ctx)   # no EVs

tag = sim_io.run_tag(cfg)
sim_core.run_and_report(cfg, ctx, ev_mw, scenario=f"baseline_{tag}",
                        label="Baseline (no EV)")
