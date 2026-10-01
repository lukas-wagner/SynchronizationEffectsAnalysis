"""
analysis/summarize.py — Aggregate simulation JSON outputs into a KPI summary CSV.

Scans output/ recursively for power-flow result JSONs (those that contain
transformer, buses and lines time series).  For each file it parses the
scenario metadata, computes grid-stress KPIs, and writes one row per run to
output/summary.csv.

KPIs computed
─────────────
Transformer  : max / mean / 95th-pct loading, peak-to-average ratio,
               hours above 80 % and 100 %
Bus voltage  : min / mean / max vm_pu, number of (bus, step) pairs below 0.9
               or above 1.1 p.u., hours with at least one undervoltage event
Line loading : max / mean / 95th-pct loading, hours above 80 %,
               number of overloaded (line, step) pairs
Convergence  : share of converged power-flow steps, count of diverged steps

Usage
─────
    python analysis/summarize.py                    # scans output/ recursively
    python analysis/summarize.py output/my_run/     # specific subfolder only
    python analysis/summarize.py output/ out.csv    # custom output path
"""

import os
import sys
import json
import re
import csv
from pathlib import Path

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output"

# EN 50160 voltage band for low-voltage networks
VOLTAGE_LOWER_PU = 0.90
VOLTAGE_UPPER_PU = 1.10

# Quarter-hourly resolution → each step = 0.25 h
DT_H = 0.25


# ─── Scenario name parser ─────────────────────────────────────────────────────

def _parse_scenario(scenario_name: str, run_dir_name: str) -> dict:
    """Extract structured fields from the scenario name and folder name.

    Handles both the legacy naming ('optimized_charging', 'fixed_times', …)
    and the current naming that embeds penetration tags
    ('optimized_ev30_pv30_hp30_decentral', …).
    For legacy EV-penetration sweep folders (pen30_spot / pen60_module3)
    the folder name is used as a fallback.
    """
    r = {
        "scenario_type": None,
        "ev_pct":        None,
        "pv_pct":        None,
        "hp_pct":        None,
        "opt_mode":      None,
        "objective":     None,
        "grid_fee":      None,
    }
    name = scenario_name

    # ── Penetration tag ev30_pv30_hp30 in scenario name (new format) ──
    m = re.search(r"ev(\d+)_pv(\d+)_hp(\d+)", name)
    if m:
        r["ev_pct"] = int(m.group(1))
        r["pv_pct"] = int(m.group(2))
        r["hp_pct"] = int(m.group(3))

    # ── EV penetration + grid-fee from legacy folder name (pen30_spot …) ──
    m2 = re.match(r"pen(\d+)_(spot|module3)", run_dir_name)
    if m2:
        if r["ev_pct"] is None:
            r["ev_pct"] = int(m2.group(1))
        r["grid_fee"] = m2.group(2)

    # ── Grid-fee hint inside scenario name ──
    if r["grid_fee"] is None:
        if "module3" in name:
            r["grid_fee"] = "module3"
        elif "spot" in name:
            r["grid_fee"] = "spot"
    # ── Fallback: infer from folder name (new sweep naming includes scenario suffix) ──
    if r["grid_fee"] is None and ("module3" in run_dir_name or "decentral_m3" in run_dir_name):
        r["grid_fee"] = "module3"

    # ── Scenario type ──
    if name.startswith("baseline"):
        r["scenario_type"] = "baseline"
    elif "fixed_but_random" in name:
        r["scenario_type"] = "fixed_but_random"
    elif "fixed_times" in name:
        r["scenario_type"] = "fixed_times"
    elif "optimized" in name:
        r["scenario_type"] = "optimized"
        if "central_" in name:
            r["opt_mode"] = "centralized"
            m3 = re.search(r"central_(\w+)$", name)
            if m3:
                r["objective"] = m3.group(1)
        elif "decentral" in name:
            r["opt_mode"] = "decentralized"
            r["objective"] = "cost"
        else:
            # Legacy name "optimized_charging" — assumed decentralized cost
            r["opt_mode"] = "decentralized"
            r["objective"] = "cost"

    return r


# ─── KPI computation ──────────────────────────────────────────────────────────

def _kpis(data: dict) -> dict:
    """Compute all grid-stress KPIs from a loaded result JSON."""
    k = {}

    # ── Transformer loading ──────────────────────────────────────────────────
    trafo_vals = [r["loading_percent"]
                  for r in data.get("transformer", [])
                  if r.get("loading_percent") is not None]
    if trafo_vals:
        a = np.array(trafo_vals, dtype=float)
        k["trafo_max_pct"]        = round(float(np.max(a)), 2)
        k["trafo_mean_pct"]       = round(float(np.mean(a)), 2)
        k["trafo_p95_pct"]        = round(float(np.percentile(a, 95)), 2)
        mean = float(np.mean(a))
        k["trafo_peak_to_avg"]    = round(float(np.max(a)) / mean, 3) if mean > 0 else None
        k["trafo_h_above_100pct"] = round(float(np.sum(a > 100)) * DT_H, 2)
        k["trafo_h_above_80pct"]  = round(float(np.sum(a > 80)) * DT_H, 2)
    else:
        for key in ("trafo_max_pct", "trafo_mean_pct", "trafo_p95_pct",
                    "trafo_peak_to_avg", "trafo_h_above_100pct", "trafo_h_above_80pct"):
            k[key] = None

    # ── Bus voltages (only converged steps) ──────────────────────────────────
    conv_bus = [r for r in data.get("buses", [])
                if r.get("vm_pu") is not None and r.get("converged", True)]
    if conv_bus:
        vm = np.array([r["vm_pu"] for r in conv_bus], dtype=float)
        k["vm_min_pu"]               = round(float(np.min(vm)), 4)
        k["vm_mean_pu"]              = round(float(np.mean(vm)), 4)
        k["vm_max_pu"]               = round(float(np.max(vm)), 4)
        k["n_undervoltage_bus_steps"] = int(np.sum(vm < VOLTAGE_LOWER_PU))
        k["n_overvoltage_bus_steps"]  = int(np.sum(vm > VOLTAGE_UPPER_PU))
        steps_uv = len({r["step"] for r in conv_bus
                        if r["vm_pu"] < VOLTAGE_LOWER_PU})
        steps_ov = len({r["step"] for r in conv_bus
                        if r["vm_pu"] > VOLTAGE_UPPER_PU})
        k["steps_with_undervoltage"] = steps_uv
        k["steps_with_overvoltage"]  = steps_ov
        k["h_with_undervoltage"]     = round(steps_uv * DT_H, 2)
        k["h_with_overvoltage"]      = round(steps_ov * DT_H, 2)
    else:
        for key in ("vm_min_pu", "vm_mean_pu", "vm_max_pu",
                    "n_undervoltage_bus_steps", "n_overvoltage_bus_steps",
                    "steps_with_undervoltage", "steps_with_overvoltage",
                    "h_with_undervoltage", "h_with_overvoltage"):
            k[key] = None

    # ── Line loadings ─────────────────────────────────────────────────────────
    line_vals = [r["loading_percent"]
                 for r in data.get("lines", [])
                 if r.get("loading_percent") is not None]
    if line_vals:
        a = np.array(line_vals, dtype=float)
        k["line_max_pct"]          = round(float(np.max(a)), 2)
        k["line_mean_pct"]         = round(float(np.mean(a)), 2)
        k["line_p95_pct"]          = round(float(np.percentile(a, 95)), 2)
        k["line_h_above_80pct"]    = round(float(np.sum(a > 80)) * DT_H, 2)
        k["n_line_overload_steps"] = int(np.sum(a > 100))
    else:
        for key in ("line_max_pct", "line_mean_pct", "line_p95_pct",
                    "line_h_above_80pct", "n_line_overload_steps"):
            k[key] = None

    # ── Power-flow convergence ────────────────────────────────────────────────
    step_conv = {}
    for r in data.get("buses", []):
        if "converged" in r:
            step_conv[r["step"]] = r["converged"]
    if step_conv:
        n_total = len(step_conv)
        n_ok = sum(step_conv.values())
        k["converged_pct"]    = round(n_ok / n_total * 100, 1)
        k["n_diverged_steps"] = n_total - n_ok
    else:
        k["converged_pct"]    = None
        k["n_diverged_steps"] = None

    return k


# ─── Main ─────────────────────────────────────────────────────────────────────

def summarize(search_dir=None, out_csv=None):
    search_dir = Path(search_dir) if search_dir else _OUTPUT_DIR
    out_csv    = Path(out_csv)    if out_csv    else _OUTPUT_DIR / "summary.csv"

    # Collect only power-flow result JSONs (must have transformer + buses + lines)
    print(f"Scanning {search_dir} …")
    result_files = []
    for fpath in sorted(search_dir.rglob("*.json")):
        if "input" in fpath.parts:
            continue
        if fpath.stat().st_size < 5_000:
            continue
        if fpath == out_csv:
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if "transformer" in data and "buses" in data and "lines" in data:
                result_files.append((fpath, data))
        except Exception:
            pass

    if not result_files:
        print(f"  No power-flow result JSONs found in {search_dir}.")
        return None

    print(f"  Found {len(result_files)} result files — computing KPIs …\n")

    rows = []
    for fpath, data in result_files:
        meta     = data.get("metadata", {})
        scenario = meta.get("scenario", fpath.stem)
        run_dir  = fpath.parent.name
        parsed   = _parse_scenario(scenario, run_dir)
        kpis     = _kpis(data)

        row = {
            "run_dir":       run_dir,
            "file":          fpath.name,
            "scenario":      scenario,
            "season":        meta.get("season", ""),
            "simulate_days": meta.get("simulate_days", ""),
            "n_steps":       meta.get("n_steps", ""),
            **parsed,
            **kpis,
        }
        rows.append(row)

        print(f"  {run_dir:30s}  {scenario:45s}  {meta.get('season',''):6s}"
              f"  trafo_max={kpis.get('trafo_max_pct','?'):6}%"
              f"  vm_min={kpis.get('vm_min_pu','?')} pu"
              f"  UV_h={kpis.get('h_with_undervoltage','?')}")

    if not rows:
        print("No valid rows produced.")
        return None

    fieldnames = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {out_csv}  ({len(rows)} rows)")
    return out_csv


if __name__ == "__main__":
    _search = sys.argv[1] if len(sys.argv) > 1 else None
    _out    = sys.argv[2] if len(sys.argv) > 2 else None
    summarize(_search, _out)
