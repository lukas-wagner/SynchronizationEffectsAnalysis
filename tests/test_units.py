"""Unit tests for the pure logic (no network / no solver)."""

import copy
import json

import numpy as np
import pandas as pd

import optimize_charging as oc
import sim_core
import sim_io
import netplot


# ── prices ──────────────────────────────────────────────────────────────────

def test_expand_prices_hourly():
    out = oc._expand_prices(np.arange(24.0), 96, 4, 96)
    assert len(out) == 96
    assert (out[0:4] == 0).all() and (out[4:8] == 1).all()


def test_expand_prices_perslot_and_long():
    spd = np.arange(96.0)
    assert (oc._expand_prices(spd, 96, 4, 96) == spd).all()
    assert len(oc._expand_prices(np.arange(200.0), 96, 4, 96)) == 96


def test_price_vector_profile(cfg):
    cfg["optimization"] = {"price_profile_eur_per_kwh": [0.1] * 24}
    grid = {"T": 96, "t_idx": pd.date_range("2024-07-15", periods=96, freq="15min")}
    out = oc.price_vector(cfg, grid)
    assert len(out) == 96 and np.allclose(out, 0.1)


def test_grid_fee_modes(cfg):
    grid = {"T": 96, "t_idx": pd.date_range("2024-01-15", periods=96, freq="15min")}
    cfg["optimization"]["grid_fees"]["mode"] = "none"
    assert (oc.grid_fee_vector(cfg, grid) == 0).all()
    cfg["optimization"]["grid_fees"]["mode"] = "constant"
    assert np.allclose(oc.grid_fee_vector(cfg, grid),
                       cfg["optimization"]["grid_fees"]["constant_eur_per_kwh"])


def test_grid_fee_module3_windows(cfg):
    cfg["optimization"]["grid_fees"]["mode"] = "module3"
    m3 = cfg["optimization"]["grid_fees"]["module3"]
    # Q1 (January) is time-variable: NT 00-06, ST 06-18, HT 18-22, ST 22-24
    q1 = {"T": 96, "t_idx": pd.date_range("2024-01-15", periods=96, freq="15min")}
    fee = oc.grid_fee_vector(cfg, q1)
    assert fee[12] == m3["low_eur_per_kwh"]       # 03:00 → NT
    assert fee[48] == m3["standard_eur_per_kwh"]  # 12:00 → ST
    assert fee[76] == m3["high_eur_per_kwh"]      # 19:00 → HT
    # Q3 (July) is flat standard all day
    q3 = {"T": 96, "t_idx": pd.date_range("2024-07-15", periods=96, freq="15min")}
    assert (oc.grid_fee_vector(cfg, q3) == m3["standard_eur_per_kwh"]).all()


def test_cost_price_adds_fee(cfg):
    cfg["optimization"] = {"price_profile_eur_per_kwh": [0.1] * 24,
                           "grid_fees": {"mode": "constant",
                                         "constant_eur_per_kwh": 0.04}}
    grid = {"T": 96, "t_idx": pd.date_range("2024-01-15", periods=96, freq="15min")}
    assert np.allclose(oc.cost_price_vector(cfg, grid), 0.14)


def test_price_vector_json(tmp_path, cfg):
    p = tmp_path / "px.json"
    json.dump({"prices": {"2024-07-15": [round(0.01 * h, 4) for h in range(24)]}},
              open(p, "w"))
    cfg["optimization"] = {"price_json": str(p)}
    grid = {"T": 8, "t_idx": pd.date_range("2024-07-15 00:00", periods=8, freq="15min")}
    out = oc.price_vector(cfg, grid)
    assert out[0] == 0.0 and out[4] == 0.01   # hour 0 then hour 1


# ── mobility ────────────────────────────────────────────────────────────────

def _grid96():
    return {"T": 96, "t_idx": pd.date_range("2024-07-15", periods=96, freq="15min")}


def test_mobility_deterministic_window(cfg):
    cfg["ev"]["mobility"]["randomize"] = False
    evs = [{"ev": 0, "load_index": 0, "bus": 0, "bus_name": "x"}]
    avail, drive, deps = oc.build_mobility(cfg, _grid96(), evs)
    dep, ret = cfg["ev"]["departure_slot"], cfg["ev"]["return_slot"]
    assert avail.shape == (1, 96)
    assert avail[0, dep:ret].sum() == 0           # away during the day
    assert avail[0, :dep].all() and avail[0, ret:].all()
    assert abs(drive[0, dep] - cfg["ev"]["daily_driving_kwh"]) < 1e-9
    assert deps[0][0][0] == dep


def test_mobility_reproducible(cfg):
    evs = [{"ev": i, "load_index": i, "bus": i, "bus_name": str(i)} for i in range(5)]
    a1, d1, _ = oc.build_mobility(cfg, _grid96(), evs)
    a2, d2, _ = oc.build_mobility(cfg, _grid96(), evs)
    assert (a1 == a2).all() and (d1 == d2).all()


def test_mobility_no_charge_while_away(cfg):
    # randomised: charging power must be impossible (avail=0) during every trip
    evs = [{"ev": i, "load_index": i, "bus": i, "bus_name": str(i)} for i in range(8)]
    avail, drive, deps = oc.build_mobility(cfg, _grid96(), evs)
    for e in range(8):
        for (ds, energy) in deps[e]:
            assert avail[e, ds] == 0.0            # away at departure
            assert energy >= 0.0


# ── EV assignment ───────────────────────────────────────────────────────────

def _fake_grid(n_loads=6):
    return {"base_per_load": np.zeros((10, n_loads)),
            "load_bus": np.array([10, 11, 12, 13, 14, 15][:n_loads]),
            "load_names": [f"L{i}" for i in range(n_loads)]}


def test_assign_evs_mapping(cfg):
    cfg["ev"].update(penetration=0.5, n_ev=None, bus_ids=None)
    evs = oc.assign_evs(cfg, _fake_grid())
    assert len(evs) == 3
    for e in evs:
        assert e["bus"] == int(_fake_grid()["load_bus"][e["load_index"]])


def test_assign_evs_bus_ids(cfg):
    cfg["ev"]["bus_ids"] = [11, 13]
    evs = oc.assign_evs(cfg, _fake_grid())
    assert sorted(e["bus"] for e in evs) == [11, 13]


def test_select_ev_loads_reproducible(cfg):
    ctx = {"n_loads": 100}
    a = sim_core.select_ev_loads(cfg, ctx)
    b = sim_core.select_ev_loads(cfg, ctx)
    assert a == b
    assert len(a) == round(cfg["ev"]["penetration"] * 100)


# ── netplot colour scales ───────────────────────────────────────────────────

def test_line_colour_bands():
    assert netplot._line_color(10) == netplot._LINE_COLORS[0]    # < 30 %
    assert netplot._line_color(85) == netplot._LINE_COLORS[3]    # 70–100 %
    assert netplot._line_color(150) == netplot._LINE_COLORS[-1]  # > 100 %


def test_vm_colour_bands():
    assert netplot._vm_color(0.85) == netplot._VM_COLORS[0]      # < 0.90
    assert netplot._vm_color(1.00) == "#27ae60"                  # nominal green
    assert netplot._vm_color(1.20) == netplot._VM_COLORS[-1]     # > 1.10


def test_bus_coords_from_geojson():
    class _Net:
        bus = pd.DataFrame({"geo": [
            '{"type":"Point","coordinates":[11.4,53.6]}',
            '{"type":"Point","coordinates":[11.5,53.7]}']})
    coords = netplot.bus_coords(_Net())
    assert coords[0] == (11.4, 53.6) and coords[1] == (11.5, 53.7)


# ── JSON export ─────────────────────────────────────────────────────────────

def test_export_json_structure_and_nan(tmp_path):
    import pandapower as pp
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=0.4, name="Bus 0")
    b1 = pp.create_bus(net, vn_kv=0.4, name="Bus 1")
    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1, length_km=0.1,
                                   r_ohm_per_km=0.1, x_ohm_per_km=0.1,
                                   c_nf_per_km=0.0, max_i_ka=0.4, name="Line 0")
    vm = np.array([[1.0, 0.95], [np.nan, np.nan]])
    ll = np.array([[10.0], [np.nan]])
    tr = np.array([5.0, np.nan])
    conv = np.array([True, False])
    t_idx = pd.date_range("2024-07-15", periods=2, freq="15min")
    out = tmp_path / "r.json"
    sim_io.export_results_json(str(out), scenario="t", simbench_code="x",
                               season="summer", simulate_days=1, t_idx=t_idx,
                               net=net, vm_pu=vm, line_load_pct=ll,
                               trafo_load_pct=tr, converged=conv)
    d = json.load(open(out))
    assert d["metadata"]["n_buses"] == 2 and d["metadata"]["n_lines"] == 1
    assert len(d["buses"]) == 4 and len(d["lines"]) == 2 and len(d["transformer"]) == 2
    nan_rec = [r for r in d["buses"] if r["step"] == 1][0]
    assert nan_rec["vm_pu"] is None           # NaN serialised as JSON null


def test_load_config_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    json.dump({"grid": {"season": "winter"}}, open(p, "w"))
    assert sim_io.load_config(str(p))["grid"]["season"] == "winter"
