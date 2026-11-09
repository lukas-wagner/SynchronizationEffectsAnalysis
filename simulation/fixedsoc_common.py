"""
Energy-consistent fixed-time charging (shared by fixedsoc_times / fixedsoc_random).
===================================================================================
Unlike the legacy fixed_times / fixed_but_random scenarios (every EV draws full
wallbox power for a fixed 2 h window every day, regardless of presence or state
of charge), here each EV follows the SAME mobility model as the optimised
strategies (optimize_charging.build_mobility) and only recharges what it drove:

  * every day at its start time a charging session is armed
    (fixed_times: 18:00 for all EVs; fixed_random: the EV's own drawn hour),
  * while armed, at home and SoC < target: charge at min(P_wallbox, remaining),
  * an EV that is away at the start time begins on arrival; an evening trip
    pauses the session, which resumes on return (also past midnight),
  * the session ends once SoC reaches the target (= soc_target, the same
    start / end-of-week level the optimised strategies use).

SoC dynamics as in the optimisation: soc[t] = soc[t-1] + eta*p*dt - drive[t].
Heat pumps follow instantaneous thermal demand, as in the legacy fixed scenarios.
"""

import json
import os
import sys

import numpy as np

import sim_core
import sim_io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "optimization"))
import optimize_charging as opt  # noqa: E402


def soc_schedule(cfg, ctx, evs, start_hours):
    """Charging power (n_ev, T) in kW plus energy bookkeeping.

    start_hours: per-EV daily start hour (array of len(evs)).
    """
    ev = cfg["ev"]
    cap = float(ev["capacity_kwh"])
    p_max = float(ev["wallbox_kw"])
    eta = float(ev["charging_efficiency"])
    soc_min = float(ev["soc_min"]) * cap
    soc_init = float(ev["soc_initial"]) * cap
    soc_target = float(ev["soc_target"]) * cap
    T, sph, spd = ctx["T"], ctx["sph"], ctx["spd"]
    dt = 1.0 / sph

    avail, drive, _ = opt.build_mobility(cfg, ctx, evs)
    n_ev = len(evs)
    p = np.zeros((n_ev, T))
    soc_end = np.zeros(n_ev)
    unmet = np.zeros(n_ev)          # kWh the battery would have been short
    for e in range(n_ev):
        start_slot = int(round(float(start_hours[e]) * sph))
        soc = soc_init
        armed = False
        for t in range(T):
            if t % spd == start_slot:
                armed = True
            soc -= drive[e, t]
            if soc < soc_min:       # trip not coverable -> book as unmet
                unmet[e] += soc_min - soc
                soc = soc_min
            if armed and avail[e, t] > 0 and soc < soc_target - 1e-9:
                pk = min(p_max, (soc_target - soc) / (eta * dt))
                p[e, t] = pk
                soc += eta * pk * dt
            if soc >= soc_target - 1e-9:
                armed = False
        soc_end[e] = soc
    stats = {
        "n_ev": n_ev,
        "ev_grid_energy_kwh_total": float(p.sum() * dt),
        "ev_grid_energy_kwh_per_ev": float(p.sum() * dt / max(1, n_ev)),
        "trip_energy_kwh_total": float(drive.sum()),
        "unmet_trip_energy_kwh_total": float(unmet.sum()),
        "n_ev_with_unmet": int((unmet > 1e-6).sum()),
        "soc_end_min_kwh": float(soc_end.min()) if n_ev else None,
        "soc_end_mean_kwh": float(soc_end.mean()) if n_ev else None,
        "soc_target_kwh": soc_target,
    }
    return p, stats


def run(cfg, ctx, start_hours, scenario_prefix, label):
    evs = opt.assign_evs(cfg, ctx)
    ev_loads = sim_core.select_ev_loads(cfg, ctx)
    assert [e["load_index"] for e in evs] == list(ev_loads), \
        "EV assignment differs between optimiser and simulation"
    p_kw, stats = soc_schedule(cfg, ctx, evs, start_hours)

    ev_mw = sim_core.zero_ev_load(ctx)
    for e, info in enumerate(evs):
        ev_mw[:, info["load_index"]] += p_kw[e] / 1000.0
    hp_mw = sim_core.hp_load_uncontrolled(cfg, ctx)
    stats["hp_grid_energy_kwh_total"] = float(hp_mw.sum() * 1000.0 / ctx["sph"])
    ev_mw += hp_mw

    tag = sim_io.run_tag(cfg)
    scenario = f"{scenario_prefix}_{tag}"
    print("[energy] " + json.dumps(stats))
    out_dir = sim_io.ensure_output_dir()
    with open(os.path.join(out_dir, f"{scenario}_energy_{ctx['season']}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    sim_core.run_and_report(cfg, ctx, ev_mw, scenario=scenario, label=label,
                            ev_schedule=p_kw)
