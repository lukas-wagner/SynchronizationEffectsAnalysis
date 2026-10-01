"""End-to-end checks on a real SimBench grid (needs internet for the first load).

These verify that the optimisation result is injected at the correct node — the
linkage between EV → load column → bus in the power flow.
"""

import numpy as np
import pytest

import sim_core
import optimize_charging as oc


@pytest.fixture(scope="module")
def ctx():
    cfg = sim_core.load_config()
    cfg["grid"]["simulate_days"] = 1
    try:
        return cfg, sim_core.load_grid(cfg)
    except Exception as exc:                      # offline / SimBench unavailable
        pytest.skip(f"could not load SimBench grid: {exc}")


def test_profile_columns_aligned(ctx):
    _, c = ctx
    # base load array columns must line up with net.load row order
    assert c["base_per_load"].shape[1] == c["n_loads"]
    assert list(c["load_bus"]) == list(c["net"].load["bus"].values)


def test_ev_bus_mapping(ctx):
    cfg, c = ctx
    evs = oc.assign_evs(cfg, c)
    for e in evs:
        assert e["bus"] == int(c["load_bus"][e["load_index"]])


def test_injection_lands_at_right_bus(ctx):
    import pandapower as pp
    cfg, c = ctx
    evs = oc.assign_evs(cfg, c)
    j = evs[0]["load_index"]
    b = evs[0]["bus"]
    net = c["net"]
    net.load["p_mw"] = c["load_p"][0].copy()
    net.load["q_mvar"] = c["load_q"][0]
    net.sgen["p_mw"] = c["sgen"][0]
    pp.runpp(net, numba=False)
    base = net.res_bus.at[b, "p_mw"]
    net.load.at[net.load.index[j], "p_mw"] += 0.011   # +11 kW at this EV's load
    pp.runpp(net, numba=False)
    new = net.res_bus.at[b, "p_mw"]
    assert abs((new - base) - 0.011) < 1e-6           # appears exactly at bus b
