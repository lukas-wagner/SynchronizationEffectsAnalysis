"""
Optimisation-coupled charging — trigger the optimiser, simulate the result.
==========================================================================
Triggers the Gurobi charging optimisation (optimization/optimize_charging.py),
maps each EV's optimised charging power onto its grid node, and runs the full
power flow on that load through the shared pipeline (sim_core). With the default
decentralized mode the self-interested HEMS synchronise on cheap hours, and the
power flow reveals the resulting grid stress.

Run:
    python simulation/optimized_charging.py [path/to/config.json]
"""

import os
import sys

import sim_core
import sim_io

# Make the optimisation package importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "optimization"))
import optimize_charging as opt  # noqa: E402

cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
ctx = sim_core.load_grid(cfg)

# Trigger the optimisation on this grid (same ctx, so it is fully consistent).
evs = opt.assign_evs(cfg, ctx)
hps = opt.assign_hps(cfg, ctx)
p_sol, soc_sol, info = opt.build_and_solve(cfg, ctx, evs, hps=hps)
if p_sol is None:
    print("\nOptimisation produced no schedule — see messages above.")
    sys.exit(1)

# Map optimised EV and HP schedules onto the grid load matrix.
ev_mw = sim_core.zero_ev_load(ctx)
for e, ev in enumerate(evs):
    ev_mw[:, ev["load_index"]] += p_sol[e] / 1000.0        # kW → MW
p_hp_sol = info.get("p_hp_sol")
if p_hp_sol is not None and len(hps) > 0:
    for h, hp_info in enumerate(hps):
        ev_mw[:, hp_info["load_index"]] += p_hp_sol[h] / 1000.0  # kW → MW

tag = sim_io.run_tag(cfg)
opt_mode = cfg.get("optimization", {}).get("mode", "decentralized")
mode_abbr = "central" if opt_mode == "centralized" else "decentral"
scenario = f"optimized_{tag}_{mode_abbr}"
if opt_mode == "centralized":
    scenario += f"_{cfg.get('optimization', {}).get('objective', 'cost')}"
grid_fee_mode = cfg.get("optimization", {}).get("grid_fees", {}).get("mode", "none")
if grid_fee_mode != "none":
    scenario += f"_{grid_fee_mode}"

sim_core.run_and_report(
    cfg, ctx, ev_mw, scenario=scenario,
    label=f"Optimised charging ({info.get('mode')})", ev_schedule=p_sol)
