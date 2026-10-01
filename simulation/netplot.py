"""
netplot.py — shared network + loading visualisations for SyncroEffects
=====================================================================
Two reusable plots, used by all simulation scenarios:

* ``network_loading_plot`` — a geographic network plot (real SimBench bus
  coordinates) with lines coloured by loading and buses coloured by voltage,
  using discrete EN-50160-style colour bands (inspired by DISEGO).
* ``line_loading_heatmap`` — an improved line-loading-over-time heatmap: lines
  sorted by peak loading, a real date/time axis and discrete threshold colours.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

# Discrete colour bands (line loading % and bus voltage p.u.)
_LINE_BOUNDS = [0, 30, 50, 70, 100]
_LINE_COLORS = ["#27ae60", "#82e0aa", "#f9e231", "#e67e22", "#c0392b"]
_LINE_LABELS = ["< 30 %", "30–50 %", "50–70 %", "70–100 %", "> 100 %"]

_VM_BOUNDS = [0.90, 0.93, 0.95, 0.97, 1.03, 1.05, 1.07, 1.10]
_VM_COLORS = ["#c0392b", "#e67e22", "#f9e231", "#82e0aa", "#27ae60",
              "#82e0aa", "#f9e231", "#e67e22", "#c0392b"]


def _line_color(pct):
    for i, b in enumerate(_LINE_BOUNDS[1:], 1):
        if pct < b:
            return _LINE_COLORS[i - 1]
    return _LINE_COLORS[-1]


def _vm_color(vm):
    for i, b in enumerate(_VM_BOUNDS, 0):
        if vm < b:
            return _VM_COLORS[i]
    return _VM_COLORS[-1]


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATES
# ─────────────────────────────────────────────────────────────────────────────

def bus_coords(net):
    """Bus coordinates as {bus_id: (x, y)} from net.bus['geo'] (GeoJSON).

    Falls back to a NetworkX spring layout for any buses without coordinates.
    """
    coords = {}
    if "geo" in net.bus.columns:
        for bid, g in net.bus["geo"].items():
            if isinstance(g, str) and g.strip():
                try:
                    c = json.loads(g)["coordinates"]
                    coords[int(bid)] = (float(c[0]), float(c[1]))
                except Exception:
                    pass
    if len(coords) < len(net.bus):
        coords = _fallback_layout(net, coords)
    return coords


def _fallback_layout(net, coords):
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(int(b) for b in net.bus.index)
        for _, r in net.line.iterrows():
            g.add_edge(int(r["from_bus"]), int(r["to_bus"]))
        for _, r in net.trafo.iterrows():
            g.add_edge(int(r["hv_bus"]), int(r["lv_bus"]))
        pos = nx.spring_layout(g, seed=42)
        return {int(n): (float(p[0]), float(p[1])) for n, p in pos.items()}
    except Exception:
        # last resort: line them up
        return {int(b): (float(i), 0.0) for i, b in enumerate(net.bus.index)}


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK PLOT
# ─────────────────────────────────────────────────────────────────────────────

def network_loading_plot(net, vm_pu, line_load_pct, trafo_load_pct, t_idx,
                         out_path, title="", step=None):
    """Geographic network plot coloured by loading at the most-stressed step.

    vm_pu / line_load_pct are (T, n) arrays (as produced by the simulations).
    If ``step`` is None, the step with the highest maximum line loading is used.
    """
    line_max = np.nanmax(line_load_pct, axis=1)
    if step is None:
        step = int(np.nanargmax(np.where(np.isnan(line_max), -1, line_max)))

    coords = bus_coords(net)
    vm_by_bus = dict(zip([int(b) for b in net.bus.index], vm_pu[step]))
    load_by_line = dict(zip([int(l) for l in net.line.index], line_load_pct[step]))
    hv = int(net.trafo["hv_bus"].iloc[0])
    lv = int(net.trafo["lv_bus"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 8))

    for lid, row in net.line.iterrows():
        fb, tb = int(row["from_bus"]), int(row["to_bus"])
        if fb in coords and tb in coords:
            (x1, y1), (x2, y2) = coords[fb], coords[tb]
            pct = float(load_by_line.get(int(lid), 0.0))
            ax.plot([x1, x2], [y1, y2], color=_line_color(pct),
                    lw=3.0 if pct >= 100 else 1.8, solid_capstyle="round", zorder=1)

    if hv in coords and lv in coords:
        (x1, y1), (x2, y2) = coords[hv], coords[lv]
        ax.plot([x1, x2], [y1, y2], color="#555555", lw=2.5, zorder=2)

    for bid, (x, y) in coords.items():
        if bid == hv:
            ax.plot(x, y, "s", color="#f0c000", ms=11, mec="#333", mew=1.2,
                    zorder=4)
        else:
            ax.plot(x, y, "o", color=_vm_color(float(vm_by_bus.get(bid, 1.0))),
                    ms=6, mec="#222", mew=0.4, zorder=3)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ts = t_idx[step].strftime("%a %d.%m. %H:%M")
    ax.set_title(f"{title}\nworst line-loading step: {ts} "
                 f"(max {line_max[step]:.0f} %)", fontsize=11, fontweight="bold")

    line_handles = [Line2D([0], [0], color=c, lw=3) for c in _LINE_COLORS]
    leg1 = ax.legend(line_handles, _LINE_LABELS, title="line loading",
                     fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.add_artist(leg1)

    # Bus-voltage legend — the colour bands are symmetric around the nominal
    # band, so list each unique colour with its low/high p.u. ranges.
    vm_legend = [
        ("#27ae60", "0.97–1.03 (nominal)"),
        ("#82e0aa", "0.95–0.97 / 1.03–1.05"),
        ("#f9e231", "0.93–0.95 / 1.05–1.07"),
        ("#e67e22", "0.90–0.93 / 1.07–1.10"),
        ("#c0392b", "< 0.90 / > 1.10 (EN 50160)"),
    ]
    bus_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="#222", markersize=8)
                   for c, _ in vm_legend]
    bus_handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor="#f0c000", markeredgecolor="#333",
                              markersize=9))
    bus_labels = [lbl for _, lbl in vm_legend] + ["transformer (HV bus)"]
    ax.legend(bus_handles, bus_labels, title="bus voltage [p.u.]",
              fontsize=7.5, loc="lower left", framealpha=0.9)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def ev_schedule_heatmap(p_sol, t_idx, out_path, title="EV charging schedule"):
    """EVs (rows) × time (cols) heatmap of charging power [kW]."""
    T = p_sol.shape[1]
    fig, ax = plt.subplots(figsize=(13, max(3.2, p_sol.shape[0] * 0.16)))
    im = ax.imshow(p_sol, aspect="auto", origin="lower", cmap="YlGnBu",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, pad=0.01, label="charging power [kW]")
    step = max(1, T // 8)
    xt = np.arange(0, T, step)
    ax.set_xticks(xt)
    ax.set_xticklabels([t_idx[i].strftime("%a\n%H:%M") for i in xt], fontsize=7.5)
    ax.set_xlabel("time")
    ax.set_ylabel("EV index (per node)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# LINE-LOADING HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def line_loading_heatmap(line_load_pct, t_idx, out_path, title="Line loading"):
    """Lines (rows, sorted by peak loading) × time (cols) heatmap.

    Uses discrete EN-50160-style threshold colours so overloads (> 100 %) stand
    out, with a real date/time axis.
    """
    T, n_lines = line_load_pct.shape
    peak = np.nanmax(line_load_pct, axis=0)
    order = np.argsort(np.where(np.isnan(peak), -1, peak))[::-1]
    mat = line_load_pct[:, order].T  # (n_lines, T), busiest line on top row

    vmax = max(120.0, float(np.nanmax(line_load_pct)) if np.isfinite(
        np.nanmax(line_load_pct)) else 120.0)
    bounds = _LINE_BOUNDS + [vmax]
    cmap = ListedColormap(_LINE_COLORS)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(13, max(3.2, n_lines * 0.16)))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap=cmap, norm=norm,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=bounds, extend="neither")
    cb.set_label("line loading [%]")

    step = max(1, T // 8)
    xt = np.arange(0, T, step)
    ax.set_xticks(xt)
    ax.set_xticklabels([t_idx[i].strftime("%a\n%H:%M") for i in xt], fontsize=7.5)
    ax.set_xlabel("time")
    ax.set_ylabel("line (sorted by peak loading)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
