"""
run.py — one entry point for SyncroEffects
==========================================
A small launcher so you don't have to remember script paths, and so you can run
several parameter combinations at once.

Usage:
    python run.py                          # list the available scenarios
    python run.py <scenario> [config.json] # run one scenario
    python run.py all [config.json]        # run all scenarios into one folder
    python run.py sweep <sweep.json>       # run many parameter combinations

Scenarios: baseline, fixed_times, fixed_but_random, optimized.

A sweep file lists runs, each with its own config overrides (dotted keys) and the
scenarios to run. Results of each run land in output/<run name>/. Example
(see configs/sweep.example.json):

    {
      "runs": [
        {"name": "summer_low",  "scenarios": ["baseline", "optimized"],
         "overrides": {"grid.season": "summer", "ev.penetration": 0.3}},
        {"name": "winter_high", "scenarios": ["all"],
         "overrides": {"grid.season": "winter", "ev.penetration": 0.8}}
      ]
    }
"""

import os
import sys
import json
import copy
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = {
    "baseline": "simulation/baseline_grid.py",
    "fixed_times": "simulation/fixed_times.py",
    "fixed_but_random": "simulation/fixed_but_random_times.py",
    "optimized": "simulation/optimized_charging.py",
}


def _run_script(scenario, config_path, run_dir):
    """Run one scenario script with a config and a fixed output sub-folder."""
    script = SCENARIOS[scenario]
    env = dict(os.environ, SYNCRO_RUN_DIR=run_dir)
    print(f"\n=== {scenario}  (config: {config_path}, output: output/{run_dir}) ===")
    subprocess.run([sys.executable, script, config_path], cwd=ROOT, env=env,
                   check=True)


def _set_dotted(cfg, dotted, value):
    """Set a nested key like 'ev.penetration' to value (creating dicts)."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _scenario_list(names):
    if "all" in names:
        return list(SCENARIOS)
    return names


def cmd_single(scenario, config_path):
    config_path = config_path or os.path.join(ROOT, "config.json")
    run_dir = datetime.now().strftime(f"{scenario}_%Y-%m-%d_%H-%M-%S")
    _run_script(scenario, config_path, run_dir)


def cmd_all(config_path):
    config_path = config_path or os.path.join(ROOT, "config.json")
    run_dir = datetime.now().strftime("all_%Y-%m-%d_%H-%M-%S")
    for scenario in SCENARIOS:
        _run_script(scenario, config_path, run_dir)
    print(f"\nAll scenarios written to output/{run_dir}/")


def cmd_sweep(sweep_path):
    with open(sweep_path, "r", encoding="utf-8") as fh:
        sweep = json.load(fh)
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as fh:
        base_cfg = json.load(fh)

    for run in sweep["runs"]:
        name = run["name"]
        scenarios = _scenario_list(run.get("scenarios", ["all"]))
        cfg = copy.deepcopy(base_cfg)
        for dotted, value in run.get("overrides", {}).items():
            _set_dotted(cfg, dotted, value)

        # Write the resolved config into the run's output folder for traceability.
        out_dir = os.path.join(ROOT, "output", name)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, "config_used.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

        print(f"\n########## sweep run '{name}'  "
              f"overrides={run.get('overrides', {})} ##########")
        for scenario in scenarios:
            _run_script(scenario, cfg_path, name)
    print("\nSweep done.")


def main(argv):
    if not argv:
        print("Scenarios:", ", ".join(SCENARIOS))
        print("Usage: python run.py <scenario|all|sweep> [config.json | sweep.json]")
        return 0
    cmd = argv[0]
    arg = argv[1] if len(argv) > 1 else None
    if cmd == "sweep":
        if not arg:
            print("Usage: python run.py sweep <sweep.json>")
            return 1
        cmd_sweep(arg)
    elif cmd == "all":
        cmd_all(arg)
    elif cmd in SCENARIOS:
        cmd_single(cmd, arg)
    else:
        print(f"Unknown scenario '{cmd}'. Available: {', '.join(SCENARIOS)}, all, sweep")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
