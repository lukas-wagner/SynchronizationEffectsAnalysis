# EV Charging Optimisation

Builds a **node-aware** ("knotenscharf") optimised charging schedule for electric
vehicles on the same SimBench low-voltage feeder used by the simulation in
[`../simulation/`](../simulation). The model is configured entirely through the
project-wide [`../config.json`](../config.json) and solved with **Gurobi**.

There are two modes (`optimization.mode`):

- **`decentralized`** (default) — each EV is optimised **on its own**, as if its
  schedule came from a Home Energy Management System (HEMS) at its node. Each
  HEMS only minimises its own charging cost against the (day-specific intraday)
  price signal, with no knowledge of the grid or the other EVs. Self-interested
  HEMS all chase the same cheap hours and therefore **synchronise** — the power
  flow then reveals the resulting grid stress. This is the core "SyncroEffect".
- **`centralized`** — one coordinated model over all EVs with a transformer (and
  optional per-node) limit, minimising the transformer peak, the squared total
  load, or total cost. The coordinated counterpart.

## What it does

1. Loads the SimBench grid + 15-min residential load profiles and slices a
   season window (same loading path as `baseline_grid.py`).
2. Assigns EVs to grid nodes — by penetration + seed, by an explicit number, or
   pinned to explicit bus IDs.
3. Builds a per-EV stochastic mobility model (`ev.mobility`) — weekday/weekend
   trips with random departure/return times, random trip energy and possibly
   several trips per day.
4. Solves the optimisation (one independent problem per EV in decentralized
   mode, one coordinated model in centralized mode).
5. Optionally re-runs a pandapower power flow on the schedule to verify bus
   voltages and line/transformer loading (the optimisation models at most the
   transformer power limit, never voltages).
6. Writes a node-and-time-indexed JSON schedule and plots into a timestamped
   sub-folder of the repo-level `output/`.

## Run

```bash
pip install -r ../requirements.txt          # needs a Gurobi licence
python optimization/optimize_charging.py                 # uses the project config.json
python optimization/optimize_charging.py path/to/other_config.json
```

A free [academic Gurobi licence](https://www.gurobi.com/academia/) is enough.

## The model

For each EV *e* and time step *t*:

- **Variables**: `p[e,t]` charging power (kW, ≥ 0), `soc[e,t]` state of charge (kWh).
- **Availability**: `p[e,t] = 0` while the EV is away (from the mobility model).
- **SoC balance**: `soc[e,t] = soc[e,t-1] + η·p[e,t]·Δt − driving[e,t]`, where the
  trip energy is drawn at each departure.
- **Target**: before every departure `soc ≥ max(soc_target, soc_min + trip
  energy)` (and at the end of the horizon); `soc_min ≤ soc ≤ soc_max` throughout.

**Decentralized mode**: one independent problem per EV, objective = its own cost
`Σ price[t]·Δt·p[e,t]`. No grid constraints (the HEMS doesn't know the grid).

**Centralized mode**: one model over all EVs with, per step, base load + total EV
power ≤ transformer limit (and optionally per-node ≤ `node_connection_limit_kw`);
objective (`optimization.objective`): `peak_shaving` (LP), `flatten` (QP) or
`cost` (LP).

### Prices

The price signal comes from `optimization.price_json` →
`intraday_prices.json`, built once from a continuous-intraday market export with
`build_prices.py` (day-specific, volume-weighted quarter-hourly EUR/kWh). The
market data are licensed and not part of the repository; if the file is missing,
the code falls back to `price_csv`, then `price_profile_eur_per_kwh`, then a flat price.

### Mobility

`ev.mobility` defines a per-EV stochastic presence model: per day a weekday or
weekend trip list, each trip with a probability and normal-distributed departure
hour, return hour and energy; multiple list entries = multiple trips per day.
`randomize=false` reverts to a single deterministic trip
(`departure_slot`/`return_slot`/`daily_driving_kwh`). Reproducible via
`assignment_seed`.

## Configuration (project `../config.json`)

This package reads the same project-wide `config.json` as the simulations.
Relevant sections:

| Section | Key | Meaning |
| --- | --- | --- |
| `grid` | `simbench_code`, `season`, `simulate_days` | which feeder, season week and horizon |
| `ev` | `n_ev` / `penetration` / `bus_ids` | how many EVs and **at which nodes** (`bus_ids` pins them; else penetration × loads, chosen by `assignment_seed`) |
| `ev` | `capacity_kwh`, `wallbox_kw`, `charging_efficiency` | wallbox / battery (defaults: 60 kWh, 11 kW, 0.95) |
| `ev` | `soc_min/max/initial/target`, `daily_driving_kwh` | SoC band and daily energy need (defaults from ETFA2026_InterPhaSe) |
| `ev` | `departure_slot`, `return_slot` | away window in 15-min slots (28 = 07:00, 68 = 17:00) |
| `ev` | `mobility` | stochastic presence model (weekday/weekend trips, random times/energy, multiple trips per day) |
| `network_limits` | `use_transformer_limit`, `transformer_loading_max_pct`, `node_connection_limit_kw` | grid constraints (centralized mode) |
| `optimization` | `mode` | `decentralized` (HEMS, default) or `centralized` |
| `optimization` | `objective` | centralized only: `cost` / `peak_shaving` / `flatten` |
| `optimization` | `price_json` / `price_csv` / `price_profile_eur_per_kwh` / `grid_buy_price_eur_per_kwh` | price signal (resolved in this order) |
| `optimization` | `mip_gap`, `time_limit_s` | Gurobi solver settings (centralized) |
| `powerflow_check` | | run a pandapower verification on the result |
| `output` | `make_plots` | also write PDF plots |

EV defaults are taken from the **ETFA2026_InterPhaSe** project. See the main
[README](../README.md#configuration) for the full per-key table.

## Output

One timestamped folder under `output/` containing:

- `charging_schedule_<season>.json` — metadata, EV→node mapping, and:
  - `schedule`: one record per (EV, time step) with `p_charge_kw`, `soc_kwh`, `soc_pct`,
  - `aggregate`: per time step base/EV/total load and (if checked) voltages & loading.
- Plots: aggregate load vs. limits, charging heatmap, SoC trajectories, and the
  power-flow voltage band.

## Example result

Default config (`1-LV-rural2`, autumn, ~50 EVs, 250 kVA transformer, intraday
prices):

| Mode | Peak load | Grid |
| --- | --- | --- |
| Base load only | ~59 kW | — |
| **Decentralized** (HEMS, cost) | **~490 kW** | transformer ~208 %, voltage down to 0.89 p.u. — synchronisation on cheap hours |
| **Centralized** (peak_shaving) | **~59 kW** | within limits, fully coordinated |

The gap between the two is the synchronisation effect that uncoordinated,
price-driven HEMS create.
