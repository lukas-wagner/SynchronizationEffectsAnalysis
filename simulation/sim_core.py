"""
sim_core.py — shared simulation pipeline for SyncroEffects
=========================================================
All scenarios (baseline, fixed/random charging, optimisation-coupled) share the
same steps: load the grid, run a power flow for every 15-min step, print KPIs,
write plots and a JSON result. That common machinery lives here so each scenario
script only has to build its EV charging matrix and call ``run_and_report``.

A scenario script is therefore tiny, e.g.:

    import sys, sim_core
    cfg = sim_core.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ctx = sim_core.load_grid(cfg)
    ev_mw = sim_core.zero_ev_load(ctx)              # baseline: no EVs
    sim_core.run_and_report(cfg, ctx, ev_mw, scenario="baseline",
                            label="Baseline (no EV)")

Everything is configured through the project config.json (see the main README).
"""

import os
import json

import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sim_io
import netplot

load_config = sim_io.load_config  # re-export for scenario scripts


# ─────────────────────────────────────────────────────────────────────────────
# GRID LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _add_rooftop_pv(cfg, net, sgen_p, start, end, T, t_idx):
    """Add configurable rooftop PV systems to net and return the extended sgen profile.

    New sgen elements are appended to net.sgen so pandapower includes them in
    every power flow. Returns a (T, n_original + n_new) array — assign it to
    ctx['sgen'] and the power-flow loop (net.sgen['p_mw'] = ctx['sgen'][i])
    works unchanged.

    Capacity factors come from pv.capacity_factor_json: a day-based JSON built by
    input/build_pv.py (mirrors intraday_prices.json structure). Looks up each step
    by calendar date + quarter-hour slot with a month/day fallback. When a CF file
    is provided the SimBench synthetic profiles are zeroed out so only the
    Norderstedt-based systems (at pv.penetration) contribute. When null, SimBench
    profiles are kept and new systems output zero (placeholder).
    """
    pv = cfg.get("pv", {})
    cf_path = pv.get("capacity_factor_json")
    # Zero out SimBench profiles when a real CF file is provided.
    base = (np.zeros((T, sgen_p.shape[1])) if cf_path
            else sgen_p.iloc[start:end].values)     # (T, n_original)
    n_new = int(round(float(pv.get("penetration", 0.0)) * len(net.load)))
    if n_new <= 0:
        return base

    rng = np.random.default_rng(int(pv.get("assignment_seed", 43)))
    load_idx = sorted(rng.permutation(len(net.load))[:n_new].tolist())
    buses = [int(net.load.iloc[i]["bus"]) for i in load_idx]

    sizes_kw = np.clip(
        rng.normal(float(pv.get("size_kw_mean", 10.0)),
                   float(pv.get("size_kw_std", 3.0)), n_new),
        float(pv.get("size_kw_min", 3.0)),
        float(pv.get("size_kw_max", 20.0)))
    sizes_mw = sizes_kw / 1000.0

    for bus, s_mw in zip(buses, sizes_mw):
        pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, sn_mva=float(s_mw),
                       name=f"NordPV_bus{bus}", type="PV")

    if cf_path:
        with open(cf_path, "r", encoding="utf-8") as fh:
            cf_data = json.load(fh)
        cf_by_date = cf_data["capacity_factors"]
        # Build (T,) array by looking up each step's date + quarter-hour slot,
        # falling back to the same month/day of any available year if the exact
        # date is missing (mirrors the intraday price lookup in optimize_charging).
        cf_window = np.zeros(T)
        for i, ts in enumerate(t_idx):
            date_str = ts.strftime("%Y-%m-%d")
            slot = ts.hour * 4 + ts.minute // 15
            if date_str in cf_by_date:
                day_vals = cf_by_date[date_str]
            else:
                month_day = date_str[5:]
                day_vals = next(
                    (cf_by_date[k] for k in cf_by_date if k[5:] == month_day), None)
            cf_window[i] = day_vals[slot] if day_vals else 0.0
        new_profiles = cf_window[:, None] * sizes_mw  # (T, n_new)
    else:
        new_profiles = np.zeros((T, n_new))

    print(f"[pv]   {n_new} rooftop PV added to {n_new}/{len(net.load)} loads | "
          f"sizes {sizes_kw.min():.1f}–{sizes_kw.max():.1f} kW "
          f"(total {sizes_kw.sum():.1f} kW) | "
          + (f"profiles from {cf_path}" if cf_path else "zero output (no CF file yet)"))

    return np.hstack([base, new_profiles])


def load_grid(cfg):
    """Load the SimBench grid + profiles and slice the configured season window.

    Returns a ``ctx`` dict with everything the pipeline (and the optimisation)
    needs. All power arrays are per-load and windowed to the simulation horizon.
    """
    code = cfg["grid"]["simbench_code"]
    season = cfg["grid"]["season"]
    days = int(cfg["grid"]["simulate_days"])
    sph = int(cfg["time"]["steps_per_hour"])
    spd = 24 * sph

    print(f"[grid] Loading SimBench {code} | season {season} | {days} day(s)")
    net = sb.get_simbench_net(code)
    prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load_p = prof[("load", "p_mw")]
    load_q = prof[("load", "q_mvar")]
    sgen_p = prof[("sgen", "p_mw")]
    n_annual = load_p.shape[0]

    # Snap the start back to the Monday of the configured week, so every run is
    # a clean block beginning on a Monday (2024-01-01 is itself a Monday). With
    # the default simulate_days = 7 this gives full Mon–Sun weeks.
    day0 = cfg["time"]["season_start_day"][season] - 1
    day0 = max(0, day0 - (pd.Timestamp("2024-01-01")
                          + pd.Timedelta(days=day0)).weekday())
    start = day0 * spd
    end = min(start + days * spd, n_annual)
    T = end - start
    t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=start / sph)
    t_idx = pd.date_range(t0, periods=T, freq=f"{60 // sph}min")

    base_per_load = load_p.iloc[start:end].values * 1000.0  # kW, (T, n_loads)
    sn_mva = float(net.trafo["sn_mva"].iloc[0])
    trafo_kw = sn_mva * 1000.0 * \
        float(cfg["network_limits"]["transformer_loading_max_pct"]) / 100.0

    print(f"       {len(net.bus)} buses, {len(net.line)} lines, "
          f"{len(net.load)} loads, transformer {sn_mva*1000:.0f} kVA")
    print(f"       window {t_idx[0]:%a %d.%m.%Y %H:%M} to {t_idx[-1]:%a %d.%m %H:%M} "
          f"({T} steps); base-load peak {base_per_load.sum(axis=1).max():.1f} kW")

    return {
        "net": net, "code": code, "season": season, "days": days,
        "sph": sph, "spd": spd, "start": start, "T": T, "t_idx": t_idx,
        "load_p": load_p.iloc[start:end].values,      # MW windows
        "load_q": load_q.iloc[start:end].values,
        "sgen": _add_rooftop_pv(cfg, net, sgen_p, start, end, T, t_idx),
        "n_buses": len(net.bus), "n_lines": len(net.line),
        "n_loads": len(net.load),
        "base_per_load": base_per_load,               # kW (for the optimisation)
        "base_total": base_per_load.sum(axis=1),      # kW
        "load_bus": net.load["bus"].values,
        "load_names": list(net.load["name"]),
        "trafo_kw": trafo_kw,
        "V_UPPER": float(cfg["limits"]["v_upper_pu"]),
        "V_LOWER": float(cfg["limits"]["v_lower_pu"]),
        "LINE_MAX": float(cfg["limits"]["line_loading_max_pct"]),
        "LINE_ALERT": float(cfg["limits"]["line_loading_alert_pct"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EV-LOAD HELPERS  (used by the scenario scripts)
# ─────────────────────────────────────────────────────────────────────────────

def zero_ev_load(ctx):
    """An all-zero EV charging matrix (T × n_loads, MW) — the baseline."""
    return np.zeros((ctx["T"], ctx["n_loads"]))


def select_ev_loads(cfg, ctx):
    """Pick which load columns get an EV (by n_ev or penetration + seed)."""
    ev = cfg["ev"]
    n_loads = ctx["n_loads"]
    rng = np.random.default_rng(int(ev["assignment_seed"]))
    if ev.get("n_ev") is not None:
        n_ev = int(ev["n_ev"])
    else:
        n_ev = int(round(float(ev["penetration"]) * n_loads))
    n_ev = max(0, min(n_ev, n_loads))
    idx = sorted(rng.permutation(n_loads)[:n_ev].tolist())
    print(f"[evs]  {len(idx)} of {n_loads} loads get an EV "
          f"({len(idx)/max(1,n_loads)*100:.0f}%)")
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# HEAT PUMP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_temperature_window(cfg, t_idx):
    """Quarter-hourly outdoor temperatures (°C) for the simulation window."""
    hp = cfg.get("hp", {})
    temp_path = hp.get("temperature_json")
    if not temp_path:
        return np.full(len(t_idx), 10.0)
    if not os.path.isabs(temp_path):
        root = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir))
        candidate = os.path.join(root, temp_path)
        temp_path = candidate if os.path.exists(candidate) else temp_path
    with open(temp_path, "r", encoding="utf-8") as fh:
        t_by_date = json.load(fh)["temperatures"]
    T = len(t_idx)
    t_outdoor = np.zeros(T)
    for i, ts in enumerate(t_idx):
        date_str = ts.strftime("%Y-%m-%d")
        slot = ts.hour * 4 + ts.minute // 15
        day_vals = t_by_date.get(date_str)
        if day_vals is None:
            month_day = date_str[5:]
            day_vals = next((t_by_date[k] for k in t_by_date if k[5:] == month_day), None)
        t_outdoor[i] = day_vals[slot] if day_vals else 10.0
    return t_outdoor


def select_hp_loads(cfg, ctx):
    """Pick which load columns get a heat pump (by penetration + seed)."""
    hp = cfg.get("hp", {})
    n_loads = ctx["n_loads"]
    n_hp = int(round(float(hp.get("penetration", 0.0)) * n_loads))
    if n_hp == 0:
        return []
    rng = np.random.default_rng(int(hp.get("assignment_seed", 44)))
    idx = sorted(rng.permutation(n_loads)[:n_hp].tolist())
    print(f"[hp]   {len(idx)} of {n_loads} loads get a heat pump "
          f"({len(idx)/max(1,n_loads)*100:.0f}%)")
    return idx


def hp_load_uncontrolled(cfg, ctx):
    """Uncontrolled HP electrical load matrix (T × n_loads, MW).

    Each HP runs at whatever instantaneous power covers the thermal demand
    (Q_heat / COP) without buffer shifting. All HPs react to the same
    outdoor temperature, so they synchronise on cold periods — the
    worst-case SyncroEffect for heat pumps.
    """
    hp = cfg.get("hp", {})
    t_outdoor = _load_temperature_window(cfg, ctx["t_idx"])
    t_set = float(hp.get("t_setpoint_celsius", 20.0))
    loss = float(hp.get("heat_loss_kw_per_kelvin", 0.15))
    cop = float(hp.get("cop", 3.5))
    q_heat_kw = np.maximum(t_set - t_outdoor, 0.0) * loss  # kW thermal
    p_hp_kw = q_heat_kw / cop                               # kW electrical
    hp_loads = select_hp_loads(cfg, ctx)
    load_mw = np.zeros((ctx["T"], ctx["n_loads"]))
    for col in hp_loads:
        load_mw[:, col] = p_hp_kw / 1000.0
    return load_mw


# ─────────────────────────────────────────────────────────────────────────────
# POWER FLOW
# ─────────────────────────────────────────────────────────────────────────────

def run_powerflow(ctx, ev_mw):
    """Run a Newton-Raphson power flow for every step with base + EV load."""
    net = ctx["net"]
    T = ctx["T"]
    vm = np.zeros((T, ctx["n_buses"]))
    ll = np.zeros((T, ctx["n_lines"]))
    tr = np.zeros(T)
    pl = np.zeros((T, ctx["n_loads"]))
    conv = np.ones(T, dtype=bool)

    print(f"[pf]   Running {T} power flows …")
    for i in range(T):
        net.load["p_mw"] = ctx["load_p"][i] + ev_mw[i]
        net.load["q_mvar"] = ctx["load_q"][i]
        net.sgen["p_mw"] = ctx["sgen"][i]
        pl[i] = net.load["p_mw"].values
        try:
            pp.runpp(net, algorithm="nr", numba=False)
            vm[i] = net.res_bus["vm_pu"].values
            ll[i] = net.res_line["loading_percent"].values
            if len(net.res_trafo):
                tr[i] = net.res_trafo["loading_percent"].values[0]
        except pp.powerflow.LoadflowNotConverged:
            conv[i] = False
            vm[i] = ll[i] = tr[i] = np.nan

    n_fail = int((~conv).sum())
    print("       " + (f"WARNING: {n_fail} steps did not converge"
                       if n_fail else "all steps converged"))
    return {"vm_pu": vm, "line": ll, "trafo": tr, "p_load": pl, "converged": conv}


# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────

def _print_kpis(ctx, res, label):
    valid = res["converged"]
    vm_v = res["vm_pu"][valid]
    ll_v = res["line"][valid]
    total_kw = res["p_load"].sum(axis=1) * 1000
    peak = int(np.nanargmax(total_kw))
    t_idx = ctx["t_idx"]

    print(f"\n{'='*60}\n  KPIs — {label}\n{'='*60}")
    print(f"  Total load   peak {total_kw[peak]:.1f} kW "
          f"({t_idx[peak]:%a %d.%m %H:%M}) | mean {np.nanmean(total_kw):.1f} kW")
    print(f"  Bus voltage  {np.nanmin(vm_v):.4f}–{np.nanmax(vm_v):.4f} p.u. | "
          f"< {ctx['V_LOWER']}: {(vm_v < ctx['V_LOWER']).sum()} bus-steps, "
          f"> {ctx['V_UPPER']}: {(vm_v > ctx['V_UPPER']).sum()}")
    print(f"  Line loading max {np.nanmax(ll_v):.1f}% | "
          f"> {ctx['LINE_ALERT']:.0f}%: {(ll_v > ctx['LINE_ALERT']).sum()}, "
          f"> {ctx['LINE_MAX']:.0f}%: {(ll_v > ctx['LINE_MAX']).sum()} line-steps")
    print(f"  Transformer  max {np.nanmax(res['trafo']):.1f}% | "
          f"mean {np.nanmean(res['trafo']):.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS  (dark theme, consistent across scenarios)
# ─────────────────────────────────────────────────────────────────────────────

_BG, _PANEL, _GRID = "#0f1923", "#19262f", "#2a3a47"
_C_LOAD, _C_EV, _C_VMIN = "#4fc3f7", "#81c784", "#ffd54f"
_C_TRAFO, _C_VIOL, _C_PLAN, _C_NORM = "#ce93d8", "#ff6b6b", "#ffa726", "#ef9a9a"


def _style(ax, title=""):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors="#aab8c2", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.yaxis.label.set_color("#aab8c2")
    if title:
        ax.set_title(title, color="#e0eaf0", fontsize=9, fontweight="bold", pad=6)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.7)


def _timeseries_plot(ctx, res, ev_mw, scenario, label, out_dir):
    t_idx = ctx["t_idx"]
    base_kw = ctx["base_per_load"].sum(axis=1)
    total_kw = res["p_load"].sum(axis=1) * 1000
    peak = int(np.nanargmax(total_kw))

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.patch.set_facecolor(_BG)
    fig.suptitle(f"{label} — {ctx['code']} | {ctx['season'].capitalize()}",
                 color="#e0eaf0", fontsize=12, fontweight="bold", y=0.98)

    ax = axes[0]
    _style(ax, "Total active load")
    ax.fill_between(t_idx, 0, base_kw, color=_C_LOAD, alpha=0.30, label="base load")
    if ev_mw.any():
        ax.fill_between(t_idx, base_kw, total_kw, color=_C_EV, alpha=0.5,
                        label="EV charging")
        ax.legend(fontsize=7.5, loc="upper left", facecolor=_PANEL,
                  edgecolor=_GRID, labelcolor="#e0eaf0")
    ax.plot(t_idx, total_kw, color=_C_LOAD, lw=1.2)
    ax.set_ylabel("kW")
    ax.axvline(t_idx[peak], color=_C_VIOL, lw=1, ls="--", alpha=0.8)
    ax.annotate(f" peak {total_kw[peak]:.0f} kW", (t_idx[peak], total_kw[peak]),
                color=_C_VIOL, fontsize=7.5)

    ax = axes[1]
    _style(ax, "Bus voltage (all buses)")
    vmin = np.nanmin(res["vm_pu"], axis=1)
    vmax = np.nanmax(res["vm_pu"], axis=1)
    ax.fill_between(t_idx, vmin, vmax, alpha=0.20, color=_C_VMIN, label="min–max")
    ax.plot(t_idx, vmin, color=_C_VMIN, lw=1.3, label="min")
    ax.axhline(ctx["V_LOWER"], color=_C_NORM, lw=1.2, ls="--",
               label=f"EN 50160 {ctx['V_LOWER']} p.u.")
    ax.set_ylabel("p.u.")
    ax.legend(fontsize=7.5, loc="lower right", facecolor=_PANEL,
              edgecolor=_GRID, labelcolor="#e0eaf0")

    ax = axes[2]
    _style(ax, "Line loading")
    ax.fill_between(t_idx, np.nanmax(res["line"], axis=1), alpha=0.20, color=_C_EV)
    ax.plot(t_idx, np.nanmax(res["line"], axis=1), color=_C_EV, lw=1.3, label="max")
    ax.axhline(ctx["LINE_ALERT"], color=_C_PLAN, lw=1.2, ls="--",
               label=f"alert {ctx['LINE_ALERT']:.0f}%")
    ax.axhline(ctx["LINE_MAX"], color=_C_NORM, lw=1.2, ls="--",
               label=f"limit {ctx['LINE_MAX']:.0f}%")
    ax.set_ylabel("%")
    ax.legend(fontsize=7.5, loc="upper right", facecolor=_PANEL,
              edgecolor=_GRID, labelcolor="#e0eaf0")

    ax = axes[3]
    _style(ax, "Transformer loading")
    ax.fill_between(t_idx, res["trafo"], alpha=0.25, color=_C_TRAFO)
    ax.plot(t_idx, res["trafo"], color=_C_TRAFO, lw=1.2)
    ax.axhline(ctx["LINE_MAX"], color=_C_NORM, lw=1.2, ls="--",
               label=f"limit {ctx['LINE_MAX']:.0f}%")
    ax.set_ylabel("%")
    ax.legend(fontsize=7.5, facecolor=_PANEL, edgecolor=_GRID, labelcolor="#e0eaf0")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%H:%M"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"{scenario}_timeseries_{ctx['season']}.pdf")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return path


def _voltage_heatmap(ctx, res, scenario, out_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL)
    im = ax.imshow(res["vm_pu"].T, aspect="auto", cmap=plt.cm.RdYlGn,
                   vmin=0.88, vmax=1.05, origin="lower", interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("voltage [p.u.]", color="#aab8c2")
    cb.ax.tick_params(colors="#aab8c2")
    T = ctx["T"]
    xt = np.arange(0, T, max(1, T // 8))
    ax.set_xticks(xt)
    ax.set_xticklabels([ctx["t_idx"][i].strftime("%a\n%H:%M") for i in xt],
                       fontsize=7.5, color="#aab8c2")
    ax.set_ylabel("bus index", color="#aab8c2")
    ax.tick_params(colors="#aab8c2")
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.set_title(f"Voltage heatmap — {ctx['season'].capitalize()}",
                 color="#e0eaf0", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{scenario}_voltage_heatmap_{ctx['season']}.pdf")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return path


def _hist_style(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.tick_params(colors="#aab8c2")
    ax.xaxis.label.set_color("#aab8c2")
    ax.yaxis.label.set_color("#aab8c2")
    ax.grid(True, color=_GRID, alpha=0.5)


def _distributions_plot(ctx, res, scenario, out_dir):
    """Histograms of bus voltage and line loading + peak power per load."""
    valid = res["converged"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(f"Distribution of grid values — {ctx['season'].capitalize()}",
                 color="#e0eaf0", fontsize=11, fontweight="bold")

    # Bus voltage histogram (bars below the lower limit in red)
    ax = axes[0]
    ax.set_facecolor(_PANEL)
    vm = res["vm_pu"][valid].flatten()
    vm = vm[~np.isnan(vm)]
    _, bins, patches = ax.hist(vm, bins=40, color=_C_VMIN, edgecolor=_GRID)
    for patch, left in zip(patches, bins[:-1]):
        if left < ctx["V_LOWER"]:
            patch.set_facecolor(_C_VIOL)
    ax.axvline(ctx["V_LOWER"], color="#ff4444", lw=2.0, ls="--",
               label=f"EN 50160 {ctx['V_LOWER']} p.u.")
    ax.axvline(1.0, color="#ffffff", lw=1.6, ls="--", label="1.0 (nominal)")
    ax.set_xlabel("voltage [p.u.]")
    ax.set_ylabel("count (bus-steps)")
    ax.set_title("Bus voltage", color="#e0eaf0", fontsize=9)
    ax.legend(fontsize=7.5, facecolor=_PANEL, edgecolor=_GRID, labelcolor="#e0eaf0")
    _hist_style(ax)

    # Line loading histogram (bars above alert orange, above limit red)
    ax = axes[1]
    ax.set_facecolor(_PANEL)
    ll = res["line"][valid].flatten()
    ll = ll[~np.isnan(ll)]
    _, bins2, patches2 = ax.hist(ll, bins=40, color=_C_EV, edgecolor=_GRID)
    for patch, left in zip(patches2, bins2[:-1]):
        if left > ctx["LINE_ALERT"]:
            patch.set_facecolor(_C_PLAN)
        if left > ctx["LINE_MAX"]:
            patch.set_facecolor(_C_VIOL)
    ax.axvline(ctx["LINE_ALERT"], color=_C_PLAN, lw=2.0, ls="--",
               label=f"alert {ctx['LINE_ALERT']:.0f}%")
    ax.axvline(ctx["LINE_MAX"], color="#ff4444", lw=2.0, ls="--",
               label=f"limit {ctx['LINE_MAX']:.0f}%")
    ax.set_xlabel("loading [%]")
    ax.set_ylabel("count (line-steps)")
    ax.set_title("Line loading", color="#e0eaf0", fontsize=9)
    ax.legend(fontsize=7.5, facecolor=_PANEL, edgecolor=_GRID, labelcolor="#e0eaf0")
    _hist_style(ax)

    # Peak power per load (sorted)
    ax = axes[2]
    ax.set_facecolor(_PANEL)
    peak = res["p_load"].max(axis=0) * 1000.0
    ax.barh(range(len(peak)), sorted(peak, reverse=True), color=_C_LOAD,
            edgecolor=_BG, alpha=0.85)
    ax.set_xlabel("peak power [kW]")
    ax.set_ylabel("load index (sorted)")
    ax.set_title("Peak power per load", color="#e0eaf0", fontsize=9)
    _hist_style(ax)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{scenario}_distributions_{ctx['season']}.pdf")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_and_report(cfg, ctx, ev_mw, scenario, label, ev_schedule=None):
    """Run the power flow for ``ev_mw`` and write KPIs, plots and JSON.

    scenario     short id used for file names (e.g. "fixed_times").
    label        human-readable title shown in plots/KPIs.
    ev_schedule  optional (n_ev, T) charging-power array → adds an EV heatmap.
    """
    res = run_powerflow(ctx, ev_mw)
    _print_kpis(ctx, res, label)

    out_dir = sim_io.ensure_output_dir()
    season = ctx["season"]
    outputs = []
    if cfg.get("output", {}).get("make_plots", True):
        print("[plot] writing figures …")
        outputs.append(_timeseries_plot(ctx, res, ev_mw, scenario, label, out_dir))
        outputs.append(_voltage_heatmap(ctx, res, scenario, out_dir))
        outputs.append(_distributions_plot(ctx, res, scenario, out_dir))
        outputs.append(netplot.line_loading_heatmap(
            res["line"], ctx["t_idx"],
            os.path.join(out_dir, f"{scenario}_line_heatmap_{season}.pdf"),
            title=f"Line loading over time — {season.capitalize()} | {label}"))
        outputs.append(netplot.network_loading_plot(
            ctx["net"], res["vm_pu"], res["line"], res["trafo"], ctx["t_idx"],
            os.path.join(out_dir, f"{scenario}_network_{season}.pdf"),
            title=f"LV grid loading — {ctx['code']} | {season.capitalize()} | {label}"))
        if ev_schedule is not None and len(ev_schedule):
            outputs.append(netplot.ev_schedule_heatmap(
                ev_schedule, ctx["t_idx"],
                os.path.join(out_dir, f"{scenario}_ev_schedule_{season}.pdf"),
                title=f"EV charging schedule — {season.capitalize()}"))

    out_json = sim_io.export_results_json(
        os.path.join(out_dir, f"{scenario}_{season}_results.json"),
        scenario=scenario, simbench_code=ctx["code"], season=season,
        simulate_days=ctx["days"], t_idx=ctx["t_idx"], net=ctx["net"],
        vm_pu=res["vm_pu"], line_load_pct=res["line"],
        trafo_load_pct=res["trafo"], converged=res["converged"])
    outputs.append(out_json)

    print("\nDone. Outputs:")
    for o in outputs:
        print(f"   {o}")
    return res, outputs
