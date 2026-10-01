"""
run_sweep.py — Run simulation sweeps with systematic, self-describing output folders.

Output folder naming:
    output/ev{pct}_pv{pct}_hp{pct}_{season}_{scenario}/

Sweep blocks are defined in configs/sweep_blocks.json. Each block specifies
lists of ev/pv/hp penetrations, seasons and scenarios; this script computes
the cartesian product and deduplicates across blocks.

Scenario abbreviations (see configs/sweep_blocks.json for descriptions):
  baseline, fixed_times, fixed_random, decentral, decentral_m3,
  central_cost, central_peak, central_flat

Usage
─────
  python run_sweep.py                       # run all blocks
  python run_sweep.py --dry-run             # print plan, don't execute
  python run_sweep.py Block1_core           # run one named block
  python run_sweep.py Block1_core Block3_pv_sweep   # run multiple blocks
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
_REPO      = Path(__file__).resolve().parent
_BASE_CFG  = _REPO / "config.json"
_SWEEP_CFG = _REPO / "configs" / "sweep_blocks.json"
_PYTHON    = _REPO / ".venv" / "Scripts" / "python.exe"

# ─── Scenario → script + config overrides ─────────────────────────────────────
_SCENARIOS = {
    "baseline": {
        "script": "simulation/baseline_grid.py",
        "overrides": {},
    },
    "fixed_times": {
        "script": "simulation/fixed_times.py",
        "overrides": {},
    },
    "fixed_random": {
        "script": "simulation/fixed_but_random_times.py",
        "overrides": {},
    },
    "decentral": {
        "script": "simulation/optimized_charging.py",
        "overrides": {"optimization": {"mode": "decentralized",
                                       "grid_fees": {"mode": "none"}}},
    },
    "decentral_m3": {
        "script": "simulation/optimized_charging.py",
        "overrides": {"optimization": {"mode": "decentralized",
                                       "grid_fees": {"mode": "module3"}}},
    },
    "central_cost": {
        "script": "simulation/optimized_charging.py",
        "overrides": {"optimization": {"mode": "centralized", "objective": "cost",
                                       "grid_fees": {"mode": "none"}}},
    },
    "central_peak": {
        "script": "simulation/optimized_charging.py",
        "overrides": {"optimization": {"mode": "centralized", "objective": "peak_shaving",
                                       "grid_fees": {"mode": "none"}}},
    },
    "central_flat": {
        "script": "simulation/optimized_charging.py",
        "overrides": {"optimization": {"mode": "centralized", "objective": "flatten",
                                       "grid_fees": {"mode": "none"}}},
    },
}


def _load_blocks():
    """Load sweep blocks from configs/sweep_blocks.json and expand cartesian products."""
    with open(_SWEEP_CFG, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    blocks = {}
    for name, blk in data["blocks"].items():
        runs = []
        for ev, pv, hp, season, scenario in product(
                blk["ev"], blk["pv"], blk["hp"], blk["seasons"], blk["scenarios"]):
            runs.append((ev, pv, hp, season, scenario))
        blocks[name] = runs
    return blocks

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _folder_name(ev, pv, hp, season, scenario):
    ev_pct  = int(round(ev * 100))
    pv_pct  = int(round(pv * 100))
    hp_pct  = int(round(hp * 100))
    return f"ev{ev_pct}_pv{pv_pct}_hp{hp_pct}_{season}_{scenario}"


def _build_config(base_cfg, ev, pv, hp, season, scenario):
    overrides = _SCENARIOS[scenario]["overrides"]
    cfg = _deep_merge(base_cfg, overrides)
    cfg["grid"]["season"]    = season
    cfg["ev"]["penetration"] = ev
    cfg["pv"]["penetration"] = pv
    cfg["hp"]["penetration"] = hp
    return cfg


def _already_done(folder_name):
    """Return True if the output folder already contains at least one result JSON."""
    out_dir = _REPO / "output" / folder_name
    if not out_dir.exists():
        return False
    return any(f.suffix == ".json" and f.stat().st_size > 5000
               for f in out_dir.iterdir())


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all(blocks_to_run=None, dry_run=False):
    with open(_BASE_CFG, "r", encoding="utf-8") as fh:
        base_cfg = json.load(fh)

    all_blocks = _load_blocks()
    selected = {k: v for k, v in all_blocks.items()
                if blocks_to_run is None or k in blocks_to_run}

    if blocks_to_run:
        missing = [b for b in blocks_to_run if b not in all_blocks]
        if missing:
            print(f"Unknown blocks: {missing}")
            print(f"Available: {list(all_blocks.keys())}")
            return

    # Flatten and deduplicate (same (ev,pv,hp,season,scenario) can appear in multiple blocks)
    seen = set()
    runs = []
    for block_name, entries in selected.items():
        for entry in entries:
            if entry not in seen:
                seen.add(entry)
                runs.append((block_name, entry))

    total = len(runs)
    skipped = sum(1 for _, (ev, pv, hp, season, sc) in runs
                  if _already_done(_folder_name(ev, pv, hp, season, sc)))
    print(f"Sweep plan: {total} runs  ({skipped} already done, "
          f"{total - skipped} to execute)\n")

    n_done = 0
    n_err  = 0
    for i, (block_name, (ev, pv, hp, season, scenario)) in enumerate(runs, 1):
        folder = _folder_name(ev, pv, hp, season, scenario)
        prefix = f"[{i:3d}/{total}] {folder}"

        if _already_done(folder):
            print(f"{prefix}  SKIP (already done)")
            continue

        script = _SCENARIOS[scenario]["script"]
        print(f"{prefix}  ({block_name})")

        if dry_run:
            print(f"         DRY-RUN: would run {script}")
            continue

        cfg = _build_config(base_cfg, ev, pv, hp, season, scenario)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False, encoding="utf-8") as tf:
            json.dump(cfg, tf, indent=2)
            tmp_path = tf.name

        env = os.environ.copy()
        env["SYNCRO_RUN_DIR"] = folder

        try:
            result = subprocess.run(
                [str(_PYTHON), script, tmp_path],
                cwd=str(_REPO),
                env=env,
                capture_output=False,
            )
            if result.returncode != 0:
                print(f"         ERROR: script exited with code {result.returncode}")
                n_err += 1
            else:
                n_done += 1
        except Exception as exc:
            print(f"         EXCEPTION: {exc}")
            n_err += 1
        finally:
            os.unlink(tmp_path)

    print(f"\nDone: {n_done} succeeded, {n_err} errors, "
          f"{skipped} skipped (already done)")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    dry  = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]
    blocks = args if args else None
    run_all(blocks_to_run=blocks, dry_run=dry)
