"""Optimisation invariants — decentralized HEMS solve (needs Gurobi, no network)."""

import pandas as pd
import numpy as np

import optimize_charging as oc


def test_decentralized_solve_invariants(cfg):
    cfg["optimization"]["mode"] = "decentralized"
    cfg["optimization"]["price_json"] = None
    cfg["optimization"]["price_profile_eur_per_kwh"] = [
        0.05 if h == 3 else 0.30 for h in range(24)]   # cheapest at 03:00
    cfg["ev"]["mobility"]["randomize"] = False

    grid = {"T": 96, "t_idx": pd.date_range("2024-07-15", periods=96, freq="15min"),
            "base_per_load": np.zeros((96, 99))}
    evs = [{"ev": 0, "load_index": 0, "bus": 0, "bus_name": "x"}]
    p_sol, soc, info = oc.build_and_solve(cfg, grid, evs)

    assert info["mode"] == "decentralized"
    assert info["status"] in ("optimal", "partial")
    assert p_sol.shape == (1, 96)
    assert (p_sol >= -1e-6).all()

    cap = cfg["ev"]["capacity_kwh"]
    assert soc.min() >= cfg["ev"]["soc_min"] * cap - 1e-6
    assert soc.max() <= cfg["ev"]["soc_max"] * cap + 1e-6

    # No charging while the EV is away (deterministic window).
    dep, ret = cfg["ev"]["departure_slot"], cfg["ev"]["return_slot"]
    assert np.allclose(p_sol[0, dep:ret], 0.0)

    # Target SoC must be met before departure (soc is a (n_ev, T) array).
    assert soc[0, dep - 1] >= cfg["ev"]["soc_target"] * cap - 1e-6


def test_cheap_hours_preferred(cfg):
    """A single very cheap hour should attract most of the charging energy."""
    cfg["optimization"].update(mode="decentralized", price_json=None,
                               price_profile_eur_per_kwh=[
                                   0.02 if h == 2 else 0.40 for h in range(24)])
    cfg["ev"]["mobility"]["randomize"] = False
    grid = {"T": 96, "t_idx": pd.date_range("2024-07-15", periods=96, freq="15min"),
            "base_per_load": np.zeros((96, 99))}
    evs = [{"ev": 0, "load_index": 0, "bus": 0, "bus_name": "x"}]
    p_sol, soc, info = oc.build_and_solve(cfg, grid, evs)
    # hour 2 = steps 8..11
    cheap = p_sol[0, 8:12].sum()
    assert cheap > 0.5 * p_sol[0].sum()   # most energy in the cheap window
