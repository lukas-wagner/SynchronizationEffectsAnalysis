"""
EV Charging Schedule Optimisation — decentralised HEMS, Gurobi
==============================================================
Builds node-aware ("knotenscharf") optimised charging schedules for electric
vehicles on a SimBench low-voltage feeder. Everything is configured through the
project-wide ``config.json``; the models are solved with Gurobi.

Two modes (``optimization.mode``):

* ``decentralized`` (default) — each EV is optimised **on its own**, as if its
  schedule were produced by a Home Energy Management System (HEMS) at its node.
  Each HEMS only minimises its own charging cost against the (dynamic intraday)
  price signal, subject to its own SoC / availability constraints. It does NOT
  know about the grid or the other EVs. Self-interested HEMS all react to the
  same cheap hours and therefore *synchronise* — the simulation step then reveals
  the resulting grid stress. This is the core "SyncroEffect".
* ``centralized`` — one coordinated model over all EVs with a transformer (and
  optional per-node) power limit, minimising the transformer peak, the squared
  total load, or the total charging cost. Useful as the coordinated counterpart.

EV presence/absence follows a configurable stochastic mobility model
(``ev.mobility``): per EV, per day, weekday/weekend trip patterns with randomised
departure/return times, randomised trip energy and multiple trips per day.

EV default values are taken from the ETFA2026_InterPhaSe project
(60 kWh battery, 11 kW wallbox, eta 0.95, target SoC 0.75, ~11.7 kWh/day).

Run:
    python optimization/optimize_charging.py [path/to/config.json]
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

import simbench as sb
import pandapower as pp

# Reuse the simulation's output-folder helper (timestamped run dir under output/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, os.pardir, "simulation"))
import sim_io  # noqa: E402

_DEFAULT_SEASON_START = {"winter": 15, "spring": 105, "summer": 196, "autumn": 288}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG + TIME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path=None):
    """Load the project config (defaults to the repo-root config.json)."""
    return sim_io.load_config(path)


def steps_per_hour(cfg):
    return int(cfg.get("time", {}).get("steps_per_hour", 4))


def steps_per_day(cfg):
    return 24 * steps_per_hour(cfg)


def dt_h(cfg):
    return 1.0 / steps_per_hour(cfg)


def season_start_day(cfg):
    return cfg.get("time", {}).get("season_start_day", _DEFAULT_SEASON_START)


def _freq(cfg):
    return f"{60 // steps_per_hour(cfg)}min"


# ─────────────────────────────────────────────────────────────────────────────
# GRID + PROFILES  (same loading path as simulation/baseline_grid.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_grid(cfg):
    code = cfg["grid"]["simbench_code"]
    season = cfg["grid"]["season"]
    days = int(cfg["grid"]["simulate_days"])
    spd = steps_per_day(cfg)

    print(f"[1/5] Loading SimBench grid: {code}")
    net = sb.get_simbench_net(code)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load_p = profiles[("load", "p_mw")]  # MW, columns aligned with net.load order
    n_annual = load_p.shape[0]

    # Snap the start back to the Monday of the configured week (see sim_core).
    day0 = season_start_day(cfg)[season] - 1
    day0 = max(0, day0 - (pd.Timestamp("2024-01-01")
                          + pd.Timedelta(days=day0)).weekday())
    start = day0 * spd
    end = min(start + days * spd, n_annual)
    T = end - start

    t0 = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=dt_h(cfg) * start)
    t_idx = pd.date_range(t0, periods=T, freq=_freq(cfg))

    base_per_load = load_p.iloc[start:end].values * 1000.0  # kW, shape (T, n_loads)
    load_bus = net.load["bus"].values                       # bus id per load column
    load_names = list(net.load["name"])

    # Transformer kW limit (approximate kVA as kW)
    sn_mva = float(net.trafo["sn_mva"].iloc[0])
    loading_max = float(cfg["network_limits"]["transformer_loading_max_pct"]) / 100.0
    trafo_kw = sn_mva * 1000.0 * loading_max

    print(f"      Buses {len(net.bus)} | Loads {len(net.load)} | "
          f"Transformer {sn_mva*1000:.0f} kVA -> limit {trafo_kw:.0f} kW")
    print(f"      Window: {t_idx[0]:%a %d.%m.%Y %H:%M} -> {t_idx[-1]:%a %d.%m.%Y %H:%M} "
          f"({T} steps, {days} day(s))")
    print(f"      Base load peak: {base_per_load.sum(axis=1).max():.1f} kW")

    return {
        "net": net, "season": season, "days": days, "code": code,
        "T": T, "t_idx": t_idx, "start": start,
        "base_per_load": base_per_load, "base_total": base_per_load.sum(axis=1),
        "load_bus": load_bus, "load_names": load_names,
        "trafo_kw": trafo_kw,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EV ASSIGNMENT  (node-aware)
# ─────────────────────────────────────────────────────────────────────────────

def assign_evs(cfg, grid):
    ev_cfg = cfg["ev"]
    n_loads = grid["base_per_load"].shape[1]
    load_bus = grid["load_bus"]
    load_names = grid["load_names"]
    rng = np.random.default_rng(int(ev_cfg["assignment_seed"]))

    if ev_cfg.get("bus_ids"):
        wanted = set(int(b) for b in ev_cfg["bus_ids"])
        load_idx = [i for i in range(n_loads) if int(load_bus[i]) in wanted]
        if not load_idx:
            raise ValueError(f"No loads found on bus_ids {sorted(wanted)}")
    else:
        if ev_cfg.get("n_ev") is not None:
            n_ev = int(ev_cfg["n_ev"])
        else:
            n_ev = int(round(float(ev_cfg["penetration"]) * n_loads))
        n_ev = max(0, min(n_ev, n_loads))
        load_idx = sorted(rng.permutation(n_loads)[:n_ev].tolist())

    evs = [{"ev": k, "load_index": int(i),
            "bus": int(load_bus[i]), "bus_name": str(load_names[i])}
           for k, i in enumerate(load_idx)]
    print(f"[2/5] EVs assigned: {len(evs)} "
          f"(at {len(set(e['bus'] for e in evs))} distinct nodes), "
          f"penetration {len(evs)/max(1,n_loads)*100:.0f}%")
    return evs


def assign_hps(cfg, grid):
    """Assign heat pump households using the same permutation logic as assign_evs."""
    hp_cfg = cfg.get("hp", {})
    n_loads = grid["base_per_load"].shape[1]
    load_bus = grid["load_bus"]
    load_names = grid["load_names"]
    n_hp = int(round(float(hp_cfg.get("penetration", 0.0)) * n_loads))
    if n_hp == 0:
        return []
    rng = np.random.default_rng(int(hp_cfg.get("assignment_seed", 44)))
    load_idx = sorted(rng.permutation(n_loads)[:n_hp].tolist())
    hps = [{"hp": k, "load_index": int(i),
             "bus": int(load_bus[i]), "bus_name": str(load_names[i])}
            for k, i in enumerate(load_idx)]
    print(f"      HPs assigned: {len(hps)} "
          f"({len(hps)/max(1,n_loads)*100:.0f}%)")
    return hps


# ─────────────────────────────────────────────────────────────────────────────
# MOBILITY MODEL  (per-EV stochastic presence / driving)
# ─────────────────────────────────────────────────────────────────────────────

def build_mobility(cfg, grid, evs):
    """Per-EV availability mask and driving draws over the horizon.

    Returns:
        avail      (n_ev, T)  1.0 = plugged in (can charge), 0.0 = away
        drive      (n_ev, T)  kWh drawn from the battery at each departure
        departures list[n_ev] of (step, trip_energy_kwh) tuples

    Driven by ``ev.mobility``: per day a weekday/weekend trip list is used, each
    trip with a probability of happening and normal-distributed departure hour,
    return hour and trip energy. ``randomize: false`` falls back to a single
    deterministic trip (ev.departure_slot / return_slot / daily_driving_kwh).
    Reproducible via ``ev.assignment_seed`` (+ mobility.seed_offset); each EV
    gets its own independent RNG stream.
    """
    ev = cfg["ev"]
    T = grid["T"]
    t_idx = grid["t_idx"]
    sph = steps_per_hour(cfg)
    spd = steps_per_day(cfg)
    cap = float(ev["capacity_kwh"])
    n_ev = len(evs)
    mob = ev.get("mobility", {})
    randomize = bool(mob.get("randomize", True))
    seed = int(ev["assignment_seed"]) + int(mob.get("seed_offset", 0))

    avail = np.ones((n_ev, T), dtype=float)
    drive = np.zeros((n_ev, T), dtype=float)
    departures = [[] for _ in range(n_ev)]

    n_days = int(np.ceil(T / spd))
    is_weekend = [t_idx[min(d * spd, T - 1)].dayofweek >= 5 for d in range(n_days)]

    def _add_trip(e, d, dep_slot, ret_slot, energy):
        ds = d * spd + dep_slot
        rs = d * spd + ret_slot
        for t in range(max(0, ds), min(rs, T)):
            avail[e, t] = 0.0
        if 0 <= ds < T:
            drive[e, ds] += energy
            departures[e].append((ds, energy))

    if not randomize:
        dep = int(ev["departure_slot"])
        ret = int(ev["return_slot"])
        energy = float(ev["daily_driving_kwh"])
        for e in range(n_ev):
            for d in range(n_days):
                _add_trip(e, d, dep, ret, energy)
        return avail, drive, departures

    hour_min = float(mob.get("hour_min", 0.0))
    hour_max = float(mob.get("hour_max", 24.0))
    for e in range(n_ev):
        rng = np.random.default_rng(seed + e)   # independent stream per EV
        for d in range(n_days):
            spec = mob.get("weekend" if is_weekend[d] else "weekday", {})
            for trip in spec.get("trips", []):
                if rng.random() > float(trip.get("probability", 1.0)):
                    continue
                dh = float(np.clip(rng.normal(*trip["departure_hour"]),
                                   hour_min, hour_max))
                rh = float(np.clip(rng.normal(*trip["return_hour"]),
                                   hour_min, hour_max))
                if rh <= dh:
                    rh = min(hour_max, dh + 1.0 / sph)
                energy = max(0.0, float(rng.normal(*trip["energy_kwh"])))
                energy = min(energy, 0.9 * cap)
                _add_trip(e, d, int(round(dh * sph)), int(round(rh * sph)), energy)
    return avail, drive, departures


# ─────────────────────────────────────────────────────────────────────────────
# PV SELF-CONSUMPTION PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def build_pv_profiles(cfg, grid, evs, hps=None):
    """Per-EV and per-HP PV output (kW) available over the simulation window.

    Replicates the assignment and sizing logic from sim_core._add_rooftop_pv
    exactly (same seed, same rng call order) so the same households carry PV in
    the optimisation as in the power-flow simulation.

    Returns:
        pv_kw       (n_ev, T) float  PV output per EV household (kW)
        feed_in     float            opportunity cost of self-consumption (EUR/kWh)
        pv_grid_kw  (T,) float       total PV generation across ALL PV households (kW)
        pv_kw_hp    (n_hp, T) float  PV output per HP household (kW); zero if no PV
    """
    pv = cfg.get("pv", {})
    cf_path = pv.get("capacity_factor_json")
    n_loads = grid["base_per_load"].shape[1]
    T = grid["T"]
    n_ev = len(evs)
    n_hp = len(hps) if hps else 0
    pv_kw = np.zeros((n_ev, T))
    pv_kw_hp = np.zeros((n_hp, T))
    feed_in = float(pv.get("feed_in_tariff_eur_per_kwh", 0.0))

    n_new = int(round(float(pv.get("penetration", 0.0)) * n_loads))
    if n_new <= 0 or not cf_path:
        return pv_kw, feed_in, np.zeros(T), pv_kw_hp

    # Mirror sim_core._add_rooftop_pv: same seed, permutation first, then normal draw
    rng = np.random.default_rng(int(pv.get("assignment_seed", 43)))
    pv_load_list = sorted(rng.permutation(n_loads)[:n_new].tolist())
    sizes_kw = np.clip(
        rng.normal(float(pv.get("size_kw_mean", 10.0)),
                   float(pv.get("size_kw_std", 3.0)), n_new),
        float(pv.get("size_kw_min", 3.0)),
        float(pv.get("size_kw_max", 20.0)))
    pv_size_by_load = dict(zip(pv_load_list, sizes_kw))

    # Resolve path relative to project root (one level above optimization/)
    if not os.path.isabs(cf_path):
        root = os.path.normpath(os.path.join(_SCRIPT_DIR, os.pardir))
        candidate = os.path.join(root, cf_path)
        cf_path = candidate if os.path.exists(candidate) else os.path.join(_SCRIPT_DIR, cf_path)
    with open(cf_path, "r", encoding="utf-8") as fh:
        cf_by_date = json.load(fh)["capacity_factors"]

    # Build CF window for the simulation horizon (same lookup as sim_core)
    t_idx = grid["t_idx"]
    cf_window = np.zeros(T)
    for i, ts in enumerate(t_idx):
        date_str = ts.strftime("%Y-%m-%d")
        slot = ts.hour * 4 + ts.minute // 15
        day_vals = cf_by_date.get(date_str)
        if day_vals is None:
            month_day = date_str[5:]
            day_vals = next((cf_by_date[k] for k in cf_by_date if k[5:] == month_day), None)
        cf_window[i] = day_vals[slot] if day_vals else 0.0

    n_pv_ev = 0
    for e, ev_info in enumerate(evs):
        load_i = ev_info["load_index"]
        if load_i in pv_size_by_load:
            pv_kw[e] = cf_window * pv_size_by_load[load_i]
            n_pv_ev += 1

    n_pv_hp = 0
    if hps:
        for h, hp_info in enumerate(hps):
            load_i = hp_info["load_index"]
            if load_i in pv_size_by_load:
                pv_kw_hp[h] = cf_window * pv_size_by_load[load_i]
                n_pv_hp += 1

    # Total PV generation across ALL PV households at each timestep.
    # All systems share the same CF (same location), scaled by individual size.
    # This matches what pandapower sees as sgen injection in the power-flow simulation.
    pv_grid_kw = cf_window * sizes_kw.sum()

    print(f"      PV: {n_new}/{n_loads} households with PV | "
          f"{n_pv_ev}/{n_ev} EVs at PV households | "
          f"{n_pv_hp}/{n_hp} HPs at PV households | "
          f"feed-in {feed_in * 100:.1f} ct/kWh")
    return pv_kw, feed_in, pv_grid_kw, pv_kw_hp


# ─────────────────────────────────────────────────────────────────────────────
# HEAT PUMP DEMAND
# ─────────────────────────────────────────────────────────────────────────────

def build_hp_demand(cfg, grid, hps):
    """Per-timestep thermal demand (kWh) for each HP household.

    All HP households share the same outdoor temperature and building
    parameters, so Q_heat is a (T,) vector broadcast to (n_hp, T).
    Returns an empty (0, T) array when no HPs are assigned.
    """
    T = grid["T"]
    n_hp = len(hps)
    if n_hp == 0:
        return np.zeros((0, T))

    hp_cfg = cfg.get("hp", {})
    temp_path = hp_cfg.get("temperature_json")
    if not temp_path:
        return np.zeros((n_hp, T))

    if not os.path.isabs(temp_path):
        root = os.path.normpath(os.path.join(_SCRIPT_DIR, os.pardir))
        candidate = os.path.join(root, temp_path)
        temp_path = candidate if os.path.exists(candidate) else os.path.join(_SCRIPT_DIR, temp_path)
    with open(temp_path, "r", encoding="utf-8") as fh:
        t_by_date = json.load(fh)["temperatures"]

    t_idx = grid["t_idx"]
    t_outdoor = np.zeros(T)
    for i, ts in enumerate(t_idx):
        date_str = ts.strftime("%Y-%m-%d")
        slot = ts.hour * 4 + ts.minute // 15
        day_vals = t_by_date.get(date_str)
        if day_vals is None:
            month_day = date_str[5:]
            day_vals = next((t_by_date[k] for k in t_by_date if k[5:] == month_day), None)
        t_outdoor[i] = day_vals[slot] if day_vals else 10.0

    t_set = float(hp_cfg.get("t_setpoint_celsius", 20.0))
    loss = float(hp_cfg.get("heat_loss_kw_per_kelvin", 0.15))
    dt = dt_h(cfg)
    q_heat_kw = np.maximum(t_set - t_outdoor, 0.0) * loss  # kW thermal per slot
    q_heat_kwh = q_heat_kw * dt                             # kWh thermal per slot

    print(f"      HP demand: {q_heat_kwh.sum():.0f} kWh thermal/week per household | "
          f"peak {q_heat_kw.max():.1f} kW thermal")
    return np.tile(q_heat_kwh, (n_hp, 1))  # (n_hp, T)


# ─────────────────────────────────────────────────────────────────────────────
# PRICE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def price_vector(cfg, grid):
    """Per-step electricity price (EUR/kWh) over the horizon.

    Resolution order:
      1. ``optimization.price_json`` — day-specific quarter-hourly prices (built
         from the Intraday Excel by build_prices.py); looked up by calendar date
         + time of day (resolution inferred from the stored profile length).
      2. ``optimization.price_csv`` — a CSV with a 'price' column.
      3. ``optimization.price_profile_eur_per_kwh`` — 24 hourly, steps_per_day,
         or T values.
      4. flat ``grid_buy_price_eur_per_kwh``.
    """
    opt = cfg["optimization"]
    T = grid["T"]
    sph = steps_per_hour(cfg)
    spd = steps_per_day(cfg)

    pj = opt.get("price_json")
    if pj:
        pj_path = pj if os.path.isabs(pj) else os.path.join(_SCRIPT_DIR, pj)
        if os.path.exists(pj_path):
            return _prices_from_json(pj, grid, opt)
        print(f"      WARNING: price_json '{pj}' not found — falling back to "
              f"price_csv / price_profile_eur_per_kwh (see README)")

    csv = opt.get("price_csv")
    if csv:
        df = pd.read_csv(csv)
        cols = [c for c in df.columns if "price" in c.lower()]
        vals = (df[cols[0]] if cols else df.iloc[:, -1]).to_numpy(dtype=float)
        return _expand_prices(vals, T, sph, spd)

    prof = opt.get("price_profile_eur_per_kwh")
    if prof:
        return _expand_prices(np.asarray(prof, dtype=float), T, sph, spd)

    return np.full(T, float(opt.get("grid_buy_price_eur_per_kwh", 0.30)))


def _prices_from_json(path, grid, opt):
    """Day-specific prices: look up each step by calendar date + time of day.

    The profile resolution is taken from the stored array length (96 =
    quarter-hourly, 24 = hourly), so the same code handles either. Within a day
    the matching slot is ``(hour*60 + minute) // (1440 / len(profile))``.

    For a given simulated day the *exact* calendar date is used first. The
    simulation calendar is anchored in 2024 (see ``load_grid``), so e.g.
    2024-10-14 resolves to that day's 2024 prices. If a date is missing, it
    falls back to the same month/day of the earliest available year, then to
    the overall mean across all days.
    """
    if not os.path.isabs(path):
        path = os.path.join(_SCRIPT_DIR, path)
    with open(path, "r", encoding="utf-8") as fh:
        pdict = json.load(fh).get("prices", {})

    by_md = {}  # "MM-DD" -> profile of the earliest year that has that day
    for d, arr in pdict.items():
        by_md.setdefault(d[5:], arr)
    all_vals = [v for arr in pdict.values() for v in arr]
    fallback = (float(np.mean(all_vals)) if all_vals
                else float(opt.get("grid_buy_price_eur_per_kwh", 0.30)))

    t_idx = grid["t_idx"]
    out = np.empty(grid["T"])
    n_miss = 0
    for t in range(grid["T"]):
        ts = t_idx[t]
        arr = pdict.get(ts.strftime("%Y-%m-%d")) or by_md.get(ts.strftime("%m-%d"))
        if arr:
            slot = int((ts.hour * 60 + ts.minute) * len(arr) // 1440)
            out[t] = arr[min(slot, len(arr) - 1)]
        else:
            out[t] = fallback
            n_miss += 1
    if n_miss:
        print(f"      price_json: {n_miss}/{grid['T']} steps had no matching "
              f"date → used mean {fallback*100:.1f} ct/kWh")
    return out


def _expand_prices(vals, T, sph, spd):
    n = len(vals)
    if n == 24:                       # hourly → repeat each hour over the day
        per_slot = np.repeat(vals, sph)
        return np.array([per_slot[t % spd] for t in range(T)])
    if n == spd:                      # per slot of day → repeat per day
        return np.array([vals[t % spd] for t in range(T)])
    if n >= T:
        return vals[:T]
    return np.array([vals[t % n] for t in range(T)])  # tile anything else


def grid_fee_vector(cfg, grid):
    """Per-step network charge (Netzentgelt) in EUR/kWh.

    ``optimization.grid_fees.mode``:
      * ``none``     → zeros (spot price only),
      * ``constant`` → a flat ``constant_eur_per_kwh``,
      * ``module3``  → the §14a EnWG Modul-3 time-variable tariff: standard /
        high (HT) / low (NT) levels, time-variable only in the configured
        quarters (default Q1 & Q4); other quarters use the standard level all day.
    """
    gf = cfg["optimization"].get("grid_fees", {})
    mode = gf.get("mode", "none")
    T, t_idx = grid["T"], grid["t_idx"]

    if not gf or mode == "none":
        return np.zeros(T)
    if mode == "constant":
        return np.full(T, float(gf.get("constant_eur_per_kwh", 0.0)))
    if mode == "module3":
        m3 = gf["module3"]
        st = float(m3["standard_eur_per_kwh"])
        hi = float(m3["high_eur_per_kwh"])
        lo = float(m3["low_eur_per_kwh"])
        hi_h = set(m3.get("high_hours", []))
        lo_h = set(m3.get("low_hours", []))
        tv_q = set(m3.get("time_variable_quarters", [1, 4]))
        out = np.empty(T)
        for t in range(T):
            ts = t_idx[t]
            quarter = (ts.month - 1) // 3 + 1
            if quarter not in tv_q:
                out[t] = st                      # flat standard outside Q1/Q4
            elif ts.hour in hi_h:
                out[t] = hi
            elif ts.hour in lo_h:
                out[t] = lo
            else:
                out[t] = st
        return out
    raise ValueError(f"Unknown grid_fees.mode '{mode}'")


def cost_price_vector(cfg, grid):
    """Effective price the EV is billed: spot price + network charge (EUR/kWh)."""
    return price_vector(cfg, grid) + grid_fee_vector(cfg, grid)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def _ev_params(cfg):
    ev = cfg["ev"]
    cap = float(ev["capacity_kwh"])
    return {
        "cap": cap,
        "p_max": float(ev["wallbox_kw"]),
        "eta": float(ev["charging_efficiency"]),
        "soc_min": float(ev["soc_min"]) * cap,
        "soc_max": float(ev["soc_max"]) * cap,
        "soc_init": float(ev["soc_initial"]) * cap,
        "soc_target": float(ev["soc_target"]) * cap,
    }


def _hp_params(cfg):
    hp = cfg.get("hp", {})
    return {
        "p_max_kw":          float(hp.get("p_max_kw", 8.0)),
        "cop":               float(hp.get("cop", 3.5)),
        "buffer_kwh":        float(hp.get("buffer_kwh", 10.0)),
        "buffer_min_kwh":    float(hp.get("buffer_min_kwh", 2.0)),
        "buffer_initial_kwh":float(hp.get("buffer_initial_kwh", 5.0)),
        "buffer_target_kwh": float(hp.get("buffer_target_kwh", 5.0)),
    }


def _add_hp_constraints(m, p, soc, T, Q_heat_h, pr_hp, dt):
    """Thermal SoC dynamics and end-of-horizon target for one heat pump."""
    for t in range(T):
        prev = pr_hp["buffer_initial_kwh"] if t == 0 else soc[t - 1]
        m.addConstr(soc[t] == prev + pr_hp["cop"] * p[t] * dt - Q_heat_h[t])
    m.addConstr(soc[T - 1] >= pr_hp["buffer_target_kwh"])


def build_and_solve(cfg, grid, evs, hps=None):
    """Dispatch to the decentralised or centralised solver (optimization.mode)."""
    if hps is None:
        hps = []
    if len(evs) == 0 and len(hps) == 0:
        print("[3/5] No EVs or HPs to schedule — skipping optimisation.")
        return np.zeros((0, grid["T"])), np.zeros((0, grid["T"])), {
            "mode": "none", "objective": "none", "status": "no_ev"}

    mode = cfg["optimization"].get("mode", "decentralized")
    avail, drive, departures = build_mobility(cfg, grid, evs)
    pv_kw, feed_in, pv_grid_kw, pv_kw_hp = build_pv_profiles(cfg, grid, evs, hps)
    Q_heat = build_hp_demand(cfg, grid, hps)
    if mode == "centralized":
        return _solve_centralized(cfg, grid, evs, avail, drive, departures,
                                  pv_kw, feed_in, pv_grid_kw, hps, Q_heat, pv_kw_hp)
    return _solve_decentralized(cfg, grid, evs, avail, drive, departures,
                                pv_kw, feed_in, hps, Q_heat, pv_kw_hp)


def _add_ev_constraints(m, p, soc, T, avail_e, drive_e, departures_e, pr, dt):
    """Add availability, SoC balance and target constraints for one EV."""
    for t in range(T):
        p[t].UB = pr["p_max"] * avail_e[t]
    for t in range(T):
        prev = pr["soc_init"] if t == 0 else soc[t - 1]
        m.addConstr(soc[t] == prev + pr["eta"] * p[t] * dt - drive_e[t])
    for (ds, energy) in departures_e:
        if ds - 1 >= 0:
            req = min(pr["soc_max"], max(pr["soc_target"], pr["soc_min"] + energy))
            m.addConstr(soc[ds - 1] >= req)
    m.addConstr(soc[T - 1] >= pr["soc_target"])


def _solve_decentralized(cfg, grid, evs, avail, drive, departures, pv_kw, feed_in,
                         hps, Q_heat, pv_kw_hp):
    """One independent cost-minimising HEMS optimisation per EV and per HP.

    For EVs at PV households a p_pv[t] variable captures how much of the charging
    comes from local solar rather than the grid. The HEMS sees the opportunity cost
    of self-consumption as feed_in (EUR/kWh) instead of the full grid price, so it
    always prefers to charge from the roof first and only draws from the grid for
    the remainder. Heat pumps at PV households follow the same logic, but with an
    availability mask: slots where a co-located EV is home (avail=1) are excluded
    from HP PV self-consumption so the two devices never claim the same kWh.
    """
    T = grid["T"]
    n_ev = len(evs)
    pr = _ev_params(cfg)
    dt = dt_h(cfg)
    prices = cost_price_vector(cfg, grid)

    print(f"[3/5] Decentralised HEMS optimisation: {n_ev} independent EVs "
          f"(cost vs. price signal) …")

    p_sol = np.zeros((n_ev, T))
    soc_sol = np.zeros((n_ev, T))
    n_infeasible = 0
    for e in range(n_ev):
        m = gp.Model(f"hems_{e}")
        m.Params.OutputFlag = 0
        p = m.addVars(T, lb=0.0, name="p")
        soc = m.addVars(T, lb=pr["soc_min"], ub=pr["soc_max"], name="soc")
        _add_ev_constraints(m, p, soc, T, avail[e], drive[e], departures[e],
                            pr, dt)

        has_pv = bool(pv_kw[e].any())
        if has_pv:
            # p_pv[t]: kW drawn from local PV rather than the grid
            p_pv = m.addVars(T, lb=0.0, name="p_pv")
            for t in range(T):
                p_pv[t].UB = float(pv_kw[e, t])   # can't exceed available PV output
                m.addConstr(p_pv[t] <= p[t])        # can't use more PV than charging
            # Cost = grid cost + opportunity cost of PV self-use (foregone feed-in)
            # Since prices[t] > feed_in, the optimizer maximises p_pv (charges from PV first)
            m.setObjective(
                gp.quicksum(
                    (prices[t] * p[t] - (prices[t] - feed_in) * p_pv[t]) * dt
                    for t in range(T)),
                GRB.MINIMIZE)
        else:
            m.setObjective(gp.quicksum(prices[t] * dt * p[t] for t in range(T)),
                           GRB.MINIMIZE)

        m.optimize()
        if m.SolCount == 0:
            n_infeasible += 1
            continue
        p_sol[e] = [p[t].X for t in range(T)]
        soc_sol[e] = [soc[t].X for t in range(T)]

    if n_infeasible:
        print(f"      WARNING: {n_infeasible}/{n_ev} EVs infeasible "
              f"(left at zero charging) — check mobility / SoC targets")
    energy = p_sol.sum() * dt
    print(f"      Done: {n_ev - n_infeasible} EVs scheduled, "
          f"{energy:.0f} kWh charged")

    # ── Heat pumps: one independent LP per HP household ──
    n_hp = len(hps)
    p_hp_sol = np.zeros((n_hp, T))
    soc_hp_sol = np.zeros((n_hp, T))
    if n_hp > 0:
        pr_hp = _hp_params(cfg)
        ev_by_load = {evs[e]["load_index"]: e for e in range(n_ev)}
        n_hp_infeasible = 0
        print(f"[3b/5] Decentralised HP optimisation: {n_hp} independent HPs …")
        for h in range(n_hp):
            m = gp.Model(f"hp_{h}")
            m.Params.OutputFlag = 0
            p = m.addVars(T, lb=0.0, ub=pr_hp["p_max_kw"], name="p")
            soc = m.addVars(T, lb=pr_hp["buffer_min_kwh"],
                            ub=pr_hp["buffer_kwh"], name="soc")
            _add_hp_constraints(m, p, soc, T, Q_heat[h], pr_hp, dt)
            # PV self-consumption: mask slots where a co-located EV is home
            load_i = hps[h]["load_index"]
            if pv_kw_hp[h].any():
                if load_i in ev_by_load:
                    pv_hp = pv_kw_hp[h] * (1.0 - avail[ev_by_load[load_i]])
                else:
                    pv_hp = pv_kw_hp[h]
            else:
                pv_hp = None
            if pv_hp is not None and pv_hp.any():
                p_pv = m.addVars(T, lb=0.0, name="p_pv")
                for t in range(T):
                    p_pv[t].UB = float(pv_hp[t])
                    m.addConstr(p_pv[t] <= p[t])
                m.setObjective(
                    gp.quicksum(
                        (prices[t] * p[t] - (prices[t] - feed_in) * p_pv[t]) * dt
                        for t in range(T)),
                    GRB.MINIMIZE)
            else:
                m.setObjective(
                    gp.quicksum(prices[t] * dt * p[t] for t in range(T)),
                    GRB.MINIMIZE)
            m.optimize()
            if m.SolCount == 0:
                n_hp_infeasible += 1
            else:
                p_hp_sol[h] = [p[t].X for t in range(T)]
                soc_hp_sol[h] = [soc[t].X for t in range(T)]
        if n_hp_infeasible:
            print(f"      WARNING: {n_hp_infeasible}/{n_hp} HPs infeasible "
                  f"— buffer too small or demand too high for this week")
        print(f"      Done: {n_hp - n_hp_infeasible} HPs scheduled, "
              f"{p_hp_sol.sum() * dt:.0f} kWh el. consumed")

    return p_sol, soc_sol, {
        "mode": "decentralized", "objective": "cost",
        "status": "optimal" if n_infeasible == 0 else "partial",
        "n_infeasible": n_infeasible,
        "avg_price_eur_per_kwh": float(np.mean(prices)),
        "p_hp_sol": p_hp_sol,
        "soc_hp_sol": soc_hp_sol,
    }


def _solve_centralized(cfg, grid, evs, avail, drive, departures, pv_kw, feed_in,
                       pv_grid_kw, hps, Q_heat, pv_kw_hp):
    """One coordinated model over all EVs and HPs with grid limits.

    PV self-consumption is modelled per EV: p_pv[e,t] is kW charged from local
    solar. The transformer constraint and all objectives use the corrected grid
    load: base household consumption minus total PV generation (pv_grid_kw, all
    PV households) plus net EV draw plus total HP draw. This matches what
    pandapower sees.
    """
    T = grid["T"]
    n_ev = len(evs)
    n_hp = len(hps)
    pr = _ev_params(cfg)
    pr_hp = _hp_params(cfg)
    dt = dt_h(cfg)
    opt = cfg["optimization"]
    objective = opt["objective"]
    base_total = grid["base_total"]

    print(f"[3/5] Centralised optimisation: {n_ev} EVs + {n_hp} HPs x {T} steps, "
          f"objective '{objective}' …")

    m = gp.Model("ev_charging_central")
    m.Params.OutputFlag = 0
    m.Params.MIPGap = float(opt.get("mip_gap", 0.0))
    if opt.get("time_limit_s"):
        m.Params.TimeLimit = float(opt["time_limit_s"])

    p = m.addVars(n_ev, T, lb=0.0, name="p")
    soc = m.addVars(n_ev, T, lb=pr["soc_min"], ub=pr["soc_max"], name="soc")
    for e in range(n_ev):
        _add_ev_constraints(m, {t: p[e, t] for t in range(T)},
                            {t: soc[e, t] for t in range(T)},
                            T, avail[e], drive[e], departures[e], pr, dt)

    # PV self-consumption variables: UB=0 for non-PV EVs (effectively fixed at 0)
    p_pv = m.addVars(n_ev, T, lb=0.0, ub=0.0, name="p_pv")
    for e in range(n_ev):
        if pv_kw[e].any():
            for t in range(T):
                p_pv[e, t].UB = float(pv_kw[e, t])
                m.addConstr(p_pv[e, t] <= p[e, t])

    ev_load = [gp.quicksum(p[e, t] for e in range(n_ev)) for t in range(T)]

    # Heat pump variables
    p_pv_hp = None
    if n_hp > 0:
        p_hp = m.addVars(n_hp, T, lb=0.0, ub=pr_hp["p_max_kw"], name="p_hp")
        soc_hp = m.addVars(n_hp, T, lb=pr_hp["buffer_min_kwh"],
                           ub=pr_hp["buffer_kwh"], name="soc_hp")
        for h in range(n_hp):
            _add_hp_constraints(m, {t: p_hp[h, t] for t in range(T)},
                                {t: soc_hp[h, t] for t in range(T)},
                                T, Q_heat[h], pr_hp, dt)
        hp_total = [gp.quicksum(p_hp[h, t] for h in range(n_hp)) for t in range(T)]
        # HP PV self-consumption: UB=0 for non-PV households (fixed at zero by LP)
        ev_by_load = {ev["load_index"]: e for e, ev in enumerate(evs)}
        p_pv_hp = m.addVars(n_hp, T, lb=0.0, ub=0.0, name="p_pv_hp")
        for h in range(n_hp):
            if not pv_kw_hp[h].any():
                continue
            load_i = hps[h]["load_index"]
            for t in range(T):
                p_pv_hp[h, t].UB = float(pv_kw_hp[h, t])
                m.addConstr(p_pv_hp[h, t] <= p_hp[h, t])
            # Co-located EV and HP share the same roof PV — joint cap
            if load_i in ev_by_load:
                e_co = ev_by_load[load_i]
                for t in range(T):
                    m.addConstr(p_pv[e_co, t] + p_pv_hp[h, t]
                                <= float(pv_kw_hp[h, t]))
    else:
        hp_total = [0.0] * T

    nl = cfg["network_limits"]
    if nl.get("use_transformer_limit", True):
        for t in range(T):
            # Actual pandapower transformer load = base + ev_load + hp_total - pv_grid_kw.
            # ev_net/hp_net would double-subtract PV self-consumption (already in pv_grid_kw).
            m.addConstr(base_total[t] - pv_grid_kw[t] + ev_load[t] + hp_total[t]
                        <= grid["trafo_kw"])

    node_limit = nl.get("node_connection_limit_kw")
    if node_limit:
        node_limit = float(node_limit)
        evs_by_bus = {}
        for e, info in enumerate(evs):
            evs_by_bus.setdefault(info["bus"], []).append(e)
        hps_by_bus = {}
        for h, info in enumerate(hps):
            hps_by_bus.setdefault(info["bus"], []).append(h)
        base_per_bus = {}
        for j in range(grid["base_per_load"].shape[1]):
            b = int(grid["load_bus"][j])
            base_per_bus.setdefault(b, np.zeros(T))
            base_per_bus[b] += grid["base_per_load"][:, j]
        for b in set(evs_by_bus) | set(hps_by_bus):
            for t in range(T):
                m.addConstr(base_per_bus[b][t]
                            + gp.quicksum(p[e, t] - p_pv[e, t]
                                          for e in evs_by_bus.get(b, []))
                            + gp.quicksum(p_hp[h, t] - p_pv_hp[h, t]
                                          for h in hps_by_bus.get(b, []))
                            <= node_limit)

    if objective == "peak_shaving":
        peak = m.addVar(lb=0.0)
        for t in range(T):
            m.addConstr(peak >= base_total[t] - pv_grid_kw[t] + ev_load[t] + hp_total[t])
        m.setObjective(peak, GRB.MINIMIZE)
    elif objective == "flatten":
        total = [base_total[t] - pv_grid_kw[t] + ev_load[t] + hp_total[t]
                 for t in range(T)]
        m.setObjective(gp.quicksum(total[t] * total[t] for t in range(T)),
                       GRB.MINIMIZE)
    elif objective == "cost":
        prices = cost_price_vector(cfg, grid)
        ev_cost = gp.quicksum(
            (prices[t] * p[e, t] - (prices[t] - feed_in) * p_pv[e, t]) * dt
            for e in range(n_ev) for t in range(T))
        hp_cost = (gp.quicksum(
                       (prices[t] * p_hp[h, t]
                        - (prices[t] - feed_in) * p_pv_hp[h, t]) * dt
                       for h in range(n_hp) for t in range(T))
                   if n_hp > 0 else 0.0)
        m.setObjective(ev_cost + hp_cost, GRB.MINIMIZE)
    else:
        raise ValueError(f"Unknown objective '{objective}'")

    m.optimize()
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
              GRB.INFEASIBLE: "infeasible",
              GRB.INF_OR_UNBD: "inf_or_unbounded"}.get(m.Status, str(m.Status))
    if m.SolCount == 0:
        print(f"      INFEASIBLE ({status}): EVs/HPs cannot be scheduled within the "
              f"transformer limit. Raise transformer_loading_max_pct, lower "
              f"penetration, or extend simulate_days.")
        return None, None, {"mode": "centralized", "objective": objective,
                            "status": status}

    p_sol = np.array([[p[e, t].X for t in range(T)] for e in range(n_ev)])
    soc_sol = np.array([[soc[e, t].X for t in range(T)] for e in range(n_ev)])
    p_hp_sol = (np.array([[p_hp[h, t].X for t in range(T)] for h in range(n_hp)])
                if n_hp > 0 else np.zeros((0, T)))
    soc_hp_sol = (np.array([[soc_hp[h, t].X for t in range(T)] for h in range(n_hp)])
                  if n_hp > 0 else np.zeros((0, T)))
    print(f"      Done: status={status}, objective={m.ObjVal:.2f}, {m.Runtime:.2f}s")
    return p_sol, soc_sol, {
        "mode": "centralized", "objective": objective, "status": status,
        "objective_value": float(m.ObjVal), "solve_time_s": float(m.Runtime),
        "p_hp_sol": p_hp_sol, "soc_hp_sol": soc_hp_sol,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POWER-FLOW VERIFICATION  (optional)
# ─────────────────────────────────────────────────────────────────────────────

def powerflow_check(cfg, grid, evs, p_sol):
    """Re-run a pandapower power flow on the optimised schedule.

    The optimisation does not model voltages (and, in decentralised mode, not
    even the transformer). This plays the optimised EV power back into the
    network and reports the resulting bus voltages and line/transformer loading.
    """
    net = grid["net"]
    T = grid["T"]
    spd = steps_per_day(cfg)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load_p = profiles[("load", "p_mw")]
    load_q = profiles[("load", "q_mvar")]
    sgen_p = profiles[("sgen", "p_mw")]
    start = grid["start"]

    ev_mw = np.zeros((T, len(net.load)))
    for e, info in enumerate(evs):
        ev_mw[:, info["load_index"]] += p_sol[e] / 1000.0

    print(f"[4/5] Power-flow verification: {T} runs …")
    vmin = np.full(T, np.nan); vmax = np.full(T, np.nan)
    ll_max = np.full(T, np.nan); trafo = np.full(T, np.nan)
    n_fail = 0
    for i in range(T):
        step = start + i
        net.load["p_mw"] = load_p.iloc[step].values + ev_mw[i]
        net.load["q_mvar"] = load_q.iloc[step].values
        net.sgen["p_mw"] = sgen_p.iloc[step].values
        try:
            pp.runpp(net, algorithm="nr", numba=False)
            vmin[i] = net.res_bus["vm_pu"].min()
            vmax[i] = net.res_bus["vm_pu"].max()
            ll_max[i] = net.res_line["loading_percent"].max()
            if len(net.res_trafo):
                trafo[i] = net.res_trafo["loading_percent"].values[0]
        except pp.powerflow.LoadflowNotConverged:
            n_fail += 1

    res = {
        "v_min_pu": float(np.nanmin(vmin)), "v_max_pu": float(np.nanmax(vmax)),
        "line_loading_max_pct": float(np.nanmax(ll_max)),
        "transformer_loading_max_pct": float(np.nanmax(trafo)),
        "non_converged_steps": int(n_fail),
        "vmin_series": vmin, "vmax_series": vmax,
        "ll_max_series": ll_max, "trafo_series": trafo,
    }
    print(f"      Voltage: {res['v_min_pu']:.4f}–{res['v_max_pu']:.4f} p.u. | "
          f"line max {res['line_loading_max_pct']:.1f}% | "
          f"trafo max {res['transformer_loading_max_pct']:.1f}%"
          + (f" | {n_fail} not converged" if n_fail else ""))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT + PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def export(cfg, grid, evs, p_sol, soc_sol, info, pf_res, out_dir):
    season = grid["season"]
    T = grid["T"]
    t_idx = grid["t_idx"]
    base_total = grid["base_total"]
    cap = float(cfg["ev"]["capacity_kwh"])

    ev_total = p_sol.sum(axis=0) if p_sol is not None and len(p_sol) else np.zeros(T)
    total = base_total + ev_total
    timestamps = [t.isoformat() for t in t_idx]

    schedule = []
    if p_sol is not None:
        for e, ginfo in enumerate(evs):
            for t in range(T):
                schedule.append({
                    "ev": ginfo["ev"], "bus": ginfo["bus"],
                    "bus_name": ginfo["bus_name"],
                    "step": t, "timestamp": timestamps[t],
                    "p_charge_kw": round(float(p_sol[e, t]), 4),
                    "soc_kwh": round(float(soc_sol[e, t]), 4),
                    "soc_pct": round(float(soc_sol[e, t]) / cap * 100.0, 2),
                })

    aggregate = []
    for t in range(T):
        rec = {
            "step": t, "timestamp": timestamps[t],
            "base_load_kw": round(float(base_total[t]), 4),
            "ev_load_kw": round(float(ev_total[t]), 4),
            "total_kw": round(float(total[t]), 4),
            "transformer_limit_kw": round(float(grid["trafo_kw"]), 2),
        }
        if pf_res:
            rec["v_min_pu"] = _safe(pf_res["vmin_series"][t])
            rec["v_max_pu"] = _safe(pf_res["vmax_series"][t])
            rec["line_loading_max_pct"] = _safe(pf_res["ll_max_series"][t])
            rec["transformer_loading_pct"] = _safe(pf_res["trafo_series"][t])
        aggregate.append(rec)

    peak_simul = round(float(base_total.max() + len(evs) * float(cfg["ev"]["wallbox_kw"])), 2)
    data = {
        "metadata": {
            "simbench_code": grid["code"], "season": season,
            "simulate_days": grid["days"], "n_steps": T,
            "n_ev": len(evs), "n_nodes_with_ev": len(set(e["bus"] for e in evs)),
            "transformer_kw": round(float(grid["trafo_kw"]), 2),
            "mode": info.get("mode"), "objective": info.get("objective"),
            "status": info.get("status"),
            "n_infeasible": info.get("n_infeasible"),
            "avg_price_eur_per_kwh": info.get("avg_price_eur_per_kwh"),
            "peak_base_kw": round(float(base_total.max()), 2),
            "peak_total_optimised_kw": round(float(total.max()), 2),
            "peak_all_simultaneous_kw": peak_simul,
        },
        "evs": evs,
        "schedule": schedule,
        "aggregate": aggregate,
    }
    if pf_res:
        data["metadata"]["powerflow"] = {
            k: pf_res[k] for k in
            ("v_min_pu", "v_max_pu", "line_loading_max_pct",
             "transformer_loading_max_pct", "non_converged_steps")
        }

    json_path = os.path.join(out_dir, f"charging_schedule_{season}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"[5/5] Schedule JSON -> {json_path}")

    outputs = [json_path]
    if cfg.get("output", {}).get("make_plots", True) and p_sol is not None:
        outputs += _plots(cfg, grid, evs, p_sol, soc_sol, ev_total, total,
                          peak_simul, pf_res, out_dir)
    return data, outputs


def _safe(x):
    x = float(x)
    return None if (np.isnan(x) or np.isinf(x)) else round(x, 4)


def _plots(cfg, grid, evs, p_sol, soc_sol, ev_total, total, peak_simul, pf_res,
           out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    season = grid["season"]
    t_idx = grid["t_idx"]
    base_total = grid["base_total"]
    cap = float(cfg["ev"]["capacity_kwh"])
    mode = cfg["optimization"].get("mode", "decentralized")
    prices = cost_price_vector(cfg, grid)
    outs = []

    # Figure 1: aggregate load + price signal
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(t_idx, 0, base_total, color="#4fc3f7", alpha=0.6, label="base load")
    ax.fill_between(t_idx, base_total, total, color="#81c784", alpha=0.7,
                    label="EV charging (optimised)")
    ax.plot(t_idx, total, color="#2e7d32", lw=1.2)
    ax.axhline(grid["trafo_kw"], color="#ef5350", ls="--", lw=1.3,
               label=f"transformer limit {grid['trafo_kw']:.0f} kW")
    ax.set_ylabel("kW")
    ax2 = ax.twinx()
    fee_on = cfg["optimization"].get("grid_fees", {}).get("mode", "none") != "none"
    price_lbl = "price + grid fee" if fee_on else "price"
    ax2.plot(t_idx, prices * 100.0, color="#8e44ad", lw=1.0, ls=":", label=price_lbl)
    ax2.set_ylabel(f"{price_lbl} [ct/kWh]", color="#8e44ad")
    ax2.tick_params(axis="y", colors="#8e44ad")
    ax.set_title(f"Optimised EV charging ({mode}) — {grid['code']} | {season} | "
                 f"{len(evs)} EVs")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    p1 = os.path.join(out_dir, f"charging_aggregate_{season}.pdf")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outs.append(p1)

    # Figure 2: EV charging heatmap (EV x time)
    fig, ax = plt.subplots(figsize=(12, max(3, len(evs) * 0.18)))
    im = ax.imshow(p_sol, aspect="auto", cmap="YlOrRd", origin="lower",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.01, label="charging power [kW]")
    ax.set_xlabel("time step")
    ax.set_ylabel("EV index")
    ax.set_title(f"EV charging schedule heatmap ({mode}) — {season}")
    p2 = os.path.join(out_dir, f"charging_heatmap_{season}.pdf")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outs.append(p2)

    # Figure 3: SoC trajectories
    fig, ax = plt.subplots(figsize=(12, 5))
    for e in range(len(evs)):
        ax.plot(t_idx, soc_sol[e] / cap * 100.0, lw=0.7, alpha=0.5)
    ax.axhline(float(cfg["ev"]["soc_target"]) * 100.0, color="#ef5350", ls="--",
               lw=1.2, label=f"target SoC {float(cfg['ev']['soc_target'])*100:.0f}%")
    ax.set_ylabel("SoC [%]")
    ax.set_title(f"EV state of charge — {season}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    p3 = os.path.join(out_dir, f"charging_soc_{season}.pdf")
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    outs.append(p3)

    if pf_res:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(t_idx, pf_res["vmin_series"], pf_res["vmax_series"],
                        color="#ffd54f", alpha=0.3, label="min–max voltage")
        ax.plot(t_idx, pf_res["vmin_series"], color="#f9a825", lw=1.0)
        ax.axhline(0.90, color="#ef5350", ls="--", lw=1.2,
                   label="EN 50160 lower 0.90 p.u.")
        ax.set_ylabel("p.u.")
        ax.set_title(f"Bus voltage under optimised schedule — {season}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        p4 = os.path.join(out_dir, f"charging_voltage_{season}.pdf")
        fig.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outs.append(p4)

    for o in outs:
        print(f"      Plot -> {o}")
    return outs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Optimise EV charging (Gurobi).")
    ap.add_argument("config", nargs="?", default=None,
                    help="path to config.json (default: project config.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("=" * 64)
    print("  EV Charging Optimisation")
    print("=" * 64)

    grid = load_grid(cfg)
    evs = assign_evs(cfg, grid)
    p_sol, soc_sol, info = build_and_solve(cfg, grid, evs)

    if p_sol is None:
        print("\nNo schedule produced — see messages above.")
        return 1

    pf_res = None
    if cfg.get("powerflow_check", False) and len(evs):
        pf_res = powerflow_check(cfg, grid, evs, p_sol)

    out_dir = sim_io.ensure_output_dir()
    data, outputs = export(cfg, grid, evs, p_sol, soc_sol, info, pf_res, out_dir)

    md = data["metadata"]
    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  Mode                      : {md['mode']} ({md['objective']})")
    print(f"  EVs                       : {md['n_ev']} at {md['n_nodes_with_ev']} nodes")
    print(f"  Base load peak            : {md['peak_base_kw']:.1f} kW")
    print(f"  Optimised peak            : {md['peak_total_optimised_kw']:.1f} kW")
    print(f"  Transformer limit         : {md['transformer_kw']:.1f} kW")
    if pf_res:
        print(f"  Voltage range (PF check)  : "
              f"{pf_res['v_min_pu']:.4f}–{pf_res['v_max_pu']:.4f} p.u.")
        print(f"  Transformer max (PF)      : "
              f"{pf_res['transformer_loading_max_pct']:.1f}%")
    print("\nAll done. Outputs:")
    for o in outputs:
        print(f"   {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
