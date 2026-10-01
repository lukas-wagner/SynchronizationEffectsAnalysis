"""
sim_io.py — shared output helpers for the SyncroEffects simulations.

* ``ensure_output_dir()`` returns (and creates) the repo-level ``output``
  folder, independent of the current working directory.
* ``export_results_json()`` writes the full power-flow time series (bus
  voltages, line loadings, transformer loading) to a JSON file, organised
  by node and time step, together with the run metadata.
"""

import os
import json
import math
from datetime import datetime

import numpy as np  # noqa: F401  (kept for type clarity / future use)


def repo_root():
    """Absolute path to the repository root (the parent of simulation/)."""
    return os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir))


def default_config_path():
    """Path to the project-wide config.json at the repository root."""
    return os.path.join(repo_root(), "config.json")


def load_config(path=None):
    """Load the project config.

    Resolution order: explicit ``path`` argument > ``SYNCRO_CONFIG`` env var >
    the project-wide ``config.json`` at the repository root.
    """
    if not path:
        path = os.environ.get("SYNCRO_CONFIG") or default_config_path()
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_tag(cfg):
    """Short parameter tag for file names, e.g. 'ev50_pv40_hp30'.

    Encodes EV, PV and HP penetration from the config so output files are
    self-describing without opening them.
    """
    ev_pct = int(round(float(cfg.get("ev", {}).get("penetration", 0)) * 100))
    pv_pct = int(round(float(cfg.get("pv", {}).get("penetration", 0)) * 100))
    hp_pct = int(round(float(cfg.get("hp", {}).get("penetration", 0)) * 100))
    return f"ev{ev_pct}_pv{pv_pct}_hp{hp_pct}"


def ensure_output_dir():
    """Return the absolute path to a per-run output folder, creating it if needed.

    Results are placed in a timestamped sub-folder of ``<repo>/output``, e.g.
    ``output/2026-06-23_14-39-24/``. Set the environment variable
    ``SYNCRO_RUN_DIR`` to use a fixed sub-folder name instead — handy for
    writing several scenarios of one experiment into the same folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(os.path.join(script_dir, os.pardir, "output"))
    sub = os.environ.get("SYNCRO_RUN_DIR") or datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, sub)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _num(x):
    """JSON-safe float: NaN / inf become ``None`` (valid JSON null)."""
    if x is None:
        return None
    xf = float(x)
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


def export_results_json(path, *, scenario, simbench_code, season, simulate_days,
                        t_idx, net, vm_pu, line_load_pct, trafo_load_pct,
                        converged):
    """Write the simulation results to ``path`` as JSON, indexed by node and time.

    The file contains run metadata plus three long-form record lists, each row
    carrying both its node identifier and its timestamp:

    * ``buses``       — one record per (time step, bus): ``vm_pu``
    * ``lines``       — one record per (time step, line): ``loading_percent``
    * ``transformer`` — one record per time step: ``loading_percent``
    """
    n_steps = len(t_idx)
    timestamps = [t.isoformat() for t in t_idx]
    conv = [bool(c) for c in converged]

    bus_ids = [int(b) for b in net.bus.index]
    bus_names = [str(n) for n in net.bus["name"]]
    line_ids = [int(l) for l in net.line.index]
    line_names = [str(n) for n in net.line["name"]]

    buses = []
    for i in range(n_steps):
        ts = timestamps[i]
        for j, (bid, bname) in enumerate(zip(bus_ids, bus_names)):
            buses.append({
                "step": i,
                "timestamp": ts,
                "bus": bid,
                "bus_name": bname,
                "vm_pu": _num(vm_pu[i, j]),
                "converged": conv[i],
            })

    lines = []
    for i in range(n_steps):
        ts = timestamps[i]
        for j, (lid, lname) in enumerate(zip(line_ids, line_names)):
            lines.append({
                "step": i,
                "timestamp": ts,
                "line": lid,
                "line_name": lname,
                "loading_percent": _num(line_load_pct[i, j]),
            })

    transformer = [
        {"step": i, "timestamp": timestamps[i],
         "loading_percent": _num(trafo_load_pct[i])}
        for i in range(n_steps)
    ]

    data = {
        "metadata": {
            "scenario": scenario,
            "simbench_code": simbench_code,
            "season": season,
            "simulate_days": simulate_days,
            "n_steps": n_steps,
            "n_buses": len(bus_ids),
            "n_lines": len(line_ids),
            "start": timestamps[0] if timestamps else None,
            "end": timestamps[-1] if timestamps else None,
            "created": datetime.now().isoformat(timespec="seconds"),
        },
        "buses": buses,
        "lines": lines,
        "transformer": transformer,
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return path
