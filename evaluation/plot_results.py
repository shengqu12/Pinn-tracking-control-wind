"""
Academic-Style Result Visualisation (TurtleBot PINN)
=====================================================
Generates publication-quality figures in PDF and PNG formats.

All figures use a consistent colour scheme and are designed for
inclusion in an academic report.

Functions
---------
plot_trajectory_comparison  — annotated XY trajectory overlay
plot_tracking_error         — lateral + heading error time series
plot_generalization_curve   — ATE vs friction (key contribution figure)
plot_generalization_bar     — grouped bar chart: train vs val ATE per method
plot_summary_figure         — 2×2 publication summary figure
plot_results_table          — Markdown-formatted numeric table
plot_training_history       — training / validation loss curves
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import Dict, List, Optional

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "lines.linewidth":   2.0,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

# ── Method colour / style map ─────────────────────────────────────────────────
# 3 methods: Open_loop (red), LQR_physics (blue), PINN_LQR (green)
METHOD_STYLE = {
    "Reference":   {"color": "#333333", "ls": (0,(4,2)),  "lw": 1.5,  "label": "Reference",      "marker": None,  "zorder": 1},
    "Open_loop":   {"color": "#D62728", "ls": "-.",        "lw": 1.8,  "label": "Open-loop",       "marker": None,  "zorder": 2},
    "LQR_physics": {"color": "#1F77B4", "ls": "--",        "lw": 1.8,  "label": "LQR (physics)",   "marker": None,  "zorder": 3},
    "PINN_LQR":    {"color": "#2CA02C", "ls": "-",         "lw": 2.3,  "label": "PINN-LQR (ours)", "marker": None,  "zorder": 4},
}

TRAIN_COLOR = "#AEC6E8"   # light blue shade for training region
VAL_COLOR   = "#FFBDBD"   # light red shade for validation region


def _save(fig, path: str):
    """Save figure as PNG and PDF."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
    print(f"  Saved → {path}.png / .pdf")
    plt.close(fig)


def _legend_handles():
    """Return Line2D legend handles for the 3 methods + reference."""
    handles = []
    for key in ["Reference", "Open_loop", "LQR_physics", "PINN_LQR"]:
        s = METHOD_STYLE[key]
        handles.append(
            Line2D([0], [0], color=s["color"], ls=s["ls"] if isinstance(s["ls"], str) else "-",
                   lw=s["lw"], label=s["label"])
        )
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# 1. Trajectory comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(
    results_dict: Dict[str, np.ndarray],
    save_path:    str  = "results/trajectory_comparison",
    title:        str  = "",
    mu:           float = None,
    is_val:       bool  = False,
):
    """
    Annotated XY trajectory comparison.

    Parameters
    ----------
    results_dict : { method_name: (T, 5) state array }
    save_path    : output path without extension
    title        : figure title override
    mu           : friction value for annotation
    is_val       : mark with (VAL) if True
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # ── draw trajectories ─────────────────────────────────────────────────────
    order = ["Reference", "Open_loop", "LQR_physics", "PINN_LQR"]
    for method in order:
        if method not in results_dict:
            continue
        states = results_dict[method]
        s = METHOD_STYLE[method]
        ax.plot(states[:, 0], states[:, 1],
                color=s["color"], ls=s["ls"], lw=s["lw"],
                label=s["label"], zorder=s["zorder"])

    # ── start / end markers ───────────────────────────────────────────────────
    if "Reference" in results_dict:
        ref = results_dict["Reference"]
        ax.scatter(ref[0, 0], ref[0, 1], s=60, c="black", zorder=10,
                   marker="o", label="Start")
        # direction arrow at start
        ax.annotate("",
            xy=(ref[5, 0], ref[5, 1]), xytext=(ref[0, 0], ref[0, 1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # ── friction annotation ───────────────────────────────────────────────────
    tag = ""
    if mu is not None:
        split = "VAL" if is_val else "TRAIN"
        tag = f"μ = {mu:.2f}  ({split})"
    if not title:
        title = f"Trajectory Comparison  {tag}"
    ax.set_title(title, pad=8)

    ax.set_xlabel("X  [m]")
    ax.set_ylabel("Y  [m]")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", framealpha=0.85)

    # Add textbox with friction value inside plot
    if mu is not None:
        ax.text(0.03, 0.97, tag,
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tracking error (time series)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracking_error(
    errors_dict:  Dict[str, np.ndarray],
    time_array:   np.ndarray,
    save_path:    str  = "results/tracking_error",
    title_suffix: str  = "",
):
    """
    Two-panel tracking error over time.
      (a) Lateral position error
      (b) Heading angle error

    Heading errors should be keyed as "heading_{method_name}".
    """
    lateral = {k: v for k, v in errors_dict.items() if not k.startswith("heading_")}
    heading = {k.replace("heading_", ""): v
               for k, v in errors_dict.items() if k.startswith("heading_")}
    if not heading:
        heading = lateral

    sfx = f"  {title_suffix}" if title_suffix else ""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.subplots_adjust(hspace=0.10)

    for method, err in lateral.items():
        s = METHOD_STYLE.get(method, {"color": "gray", "ls": "-", "lw": 1.8, "label": method})
        ax1.plot(time_array[:len(err)], err,
                 color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

    ax1.set_ylabel("Position Error  [m]")
    ax1.set_title(f"(a) Lateral Tracking Error{sfx}")
    ax1.legend(loc="upper right")
    # shade the first 2 s as "transient"
    ax1.axvspan(0, min(2.0, time_array[-1]), alpha=0.07, color="gray",
                label="Transient")

    for method, err in heading.items():
        s = METHOD_STYLE.get(method, {"color": "gray", "ls": "-", "lw": 1.8, "label": method})
        ax2.plot(time_array[:len(err)], err,
                 color=s["color"], ls=s["ls"], lw=s["lw"], label=s["label"])

    ax2.set_ylabel("Heading Error  [rad]")
    ax2.set_xlabel("Time  [s]")
    ax2.set_title(f"(b) Heading Angle Error{sfx}")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generalisation curve (ATE vs friction)
# ─────────────────────────────────────────────────────────────────────────────

def plot_generalization_curve(
    friction_values:  List[float],
    ate_dict:         Dict[str, List[float]],
    train_friction:   Optional[List[float]] = None,
    save_path:        str = "results/generalization_curve",
):
    """
    ATE vs friction coefficient — the key contribution figure.

    Training friction values are shown as filled markers; validation
    (unseen) values as hollow markers, separated by a shaded region.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    fv = np.array(friction_values)
    train_set = set(train_friction) if train_friction else set()

    # ── shaded regions ────────────────────────────────────────────────────────
    if train_friction:
        max_t = max(train_friction)
        ax.axvspan(min(fv) - 0.02, max_t + 0.025,
                   alpha=1.0, color=TRAIN_COLOR, zorder=0, label="Training region")
        ax.axvspan(max_t + 0.025, max(fv) + 0.02,
                   alpha=1.0, color=VAL_COLOR,   zorder=0, label="Validation (unseen)")
        ax.axvline(x=max_t + 0.025, color="#888888", ls="--", lw=1.2, zorder=1)
        ymax = max(v for lst in ate_dict.values() for v in lst) * 1.15
        ax.text(max_t + 0.03, ymax * 0.97, "← Train  |  Val →",
                fontsize=9, color="#555555", va="top")

    # ── per-method lines ──────────────────────────────────────────────────────
    markers_train = {"Open_loop": "o",   "LQR_physics": "s",   "PINN_LQR": "D"}
    markers_val   = {"Open_loop": "o",   "LQR_physics": "s",   "PINN_LQR": "D"}

    for method, ate_list in ate_dict.items():
        s  = METHOD_STYLE.get(method, {"color": "gray", "ls": "-", "lw": 2.0, "label": method})
        fv_clipped = fv[:len(ate_list)]
        ate_arr    = np.array(ate_list)

        # draw the line
        ax.plot(fv_clipped, ate_arr,
                color=s["color"], ls=s["ls"], lw=s["lw"],
                zorder=s["zorder"], label=s["label"])

        # filled markers = training, hollow = validation
        for i, (mu, ate) in enumerate(zip(fv_clipped, ate_arr)):
            mk = markers_train.get(method, "o")
            if mu in train_set:
                ax.scatter(mu, ate, color=s["color"], marker=mk,
                           s=55, zorder=s["zorder"] + 1, edgecolors="white", lw=1)
            else:
                ax.scatter(mu, ate, color="white", marker=mk,
                           s=55, zorder=s["zorder"] + 1,
                           edgecolors=s["color"], lw=1.5)

    # ── annotations: show the generalisation gap for PINN_LQR vs LQR_physics ─
    if "PINN_LQR" in ate_dict and "LQR_physics" in ate_dict and train_friction:
        # find a validation point to annotate
        val_mus = [mu for mu in fv if mu not in train_set]
        if val_mus:
            mu_ann = val_mus[-1]
            idx_ann = list(fv).index(mu_ann)
            pinn_ate = ate_dict["PINN_LQR"][idx_ann]
            lqr_ate  = ate_dict["LQR_physics"][idx_ann]
            gap = lqr_ate - pinn_ate
            if gap > 0.001:
                ax.annotate(
                    f"Δ = {gap:.3f} m",
                    xy=(mu_ann, (pinn_ate + lqr_ate) / 2),
                    xytext=(mu_ann + 0.015, (pinn_ate + lqr_ate) / 2),
                    fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=1.0),
                )
                ax.annotate(
                    "", xy=(mu_ann - 0.005, pinn_ate),
                    xytext=(mu_ann - 0.005, lqr_ate),
                    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.0),
                )

    # ── legend with custom train/val marker explanation ───────────────────────
    legend_extra = [
        Line2D([0], [0], marker="o", color="gray", ls="None",
               markersize=7, markerfacecolor="gray",  label="Train μ (seen)"),
        Line2D([0], [0], marker="o", color="gray", ls="None",
               markersize=7, markerfacecolor="white", markeredgecolor="gray",
               markeredgewidth=1.5, label="Val μ (unseen)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    # exclude shaded-region labels from auto-legend (they are too verbose)
    filtered = [(h, l) for h, l in zip(handles, labels)
                if l not in ("Training region", "Validation (unseen)")]
    h, l = zip(*filtered) if filtered else ([], [])
    ax.legend(list(h) + legend_extra, list(l) + [e.get_label() for e in legend_extra],
              loc="upper left", framealpha=0.9)

    ax.set_xlabel("Surface Friction Coefficient  (μ)")
    ax.set_ylabel("Average Tracking Error (ATE)  [m]")
    ax.set_title("Generalisation Across Friction Coefficients")
    ax.set_xlim(min(fv) - 0.02, max(fv) + 0.02)

    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grouped bar chart: train ATE vs val ATE
# ─────────────────────────────────────────────────────────────────────────────

def plot_generalization_bar(
    friction_values: List[float],
    ate_dict:        Dict[str, List[float]],
    train_friction:  List[float],
    save_path:       str = "results/generalization_bar",
):
    """
    Grouped bar chart showing mean Train ATE and mean Val ATE per method.
    Visually emphasises which method generalises best.
    """
    methods = list(ate_dict.keys())
    train_set = set(train_friction)
    fv = np.array(friction_values)

    train_ate = []
    val_ate   = []
    for method in methods:
        arr = np.array(ate_dict[method])
        t = [arr[i] for i, mu in enumerate(fv[:len(arr)]) if mu in train_set]
        v = [arr[i] for i, mu in enumerate(fv[:len(arr)]) if mu not in train_set]
        train_ate.append(np.mean(t) if t else 0.0)
        val_ate.append(np.mean(v)   if v else 0.0)

    x     = np.arange(len(methods))
    width = 0.32
    colors_train = [METHOD_STYLE.get(m, {}).get("color", "gray") for m in methods]
    colors_val   = [matplotlib.colors.to_rgba(c, alpha=0.45) for c in colors_train]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_t = ax.bar(x - width / 2, train_ate, width, label="Train (seen)",
                    color=colors_train, edgecolor="white", linewidth=0.7)
    bars_v = ax.bar(x + width / 2, val_ate,   width, label="Val (unseen)",
                    color=colors_val, edgecolor=colors_train, linewidth=1.2,
                    linestyle="--")

    # value labels on bars
    for bar in list(bars_t) + list(bars_v):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # draw generalisation gap arrows for each method
    for i, method in enumerate(methods):
        gap = val_ate[i] - train_ate[i]
        if abs(gap) > 0.001:
            mid = (train_ate[i] + val_ate[i]) / 2
            ax.annotate(
                f"+{gap:.3f}" if gap > 0 else f"{gap:.3f}",
                xy=(x[i] + width / 2 + 0.03, mid),
                fontsize=7.5, color="#444444", va="center",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_STYLE.get(m, {}).get("label", m) for m in methods],
                       fontsize=10)
    ax.set_ylabel("Mean ATE  [m]")
    ax.set_title("Train vs. Validation ATE  (Generalisation Gap)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(train_ate), max(val_ate)) * 1.30)

    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 2×2 Summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_figure(
    sim_cache:      dict,
    ate_dict:       Dict[str, List[float]],
    friction_list:  List[float],
    train_friction: List[float],
    reference:      np.ndarray,
    save_path:      str = "results/summary_figure",
    dt:             float = 0.02,
):
    """
    Publication-quality 2×2 summary figure.

    Layout
    ------
    (a) top-left    : XY trajectory at training friction
    (b) top-right   : XY trajectory at validation friction (shows degradation)
    (c) bottom-left : ATE vs friction generalisation curve
    (d) bottom-right: Train/Val ATE bar chart (generalisation gap)
    """
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])   # trajectory – training
    ax_b = fig.add_subplot(gs[0, 1])   # trajectory – validation
    ax_c = fig.add_subplot(gs[1, 0])   # generalisation curve
    ax_d = fig.add_subplot(gs[1, 1])   # bar chart

    mus_sorted = sorted(sim_cache.keys())
    mu_train   = mus_sorted[0]    # training example
    mu_val     = mus_sorted[-1]   # validation example
    train_set  = set(train_friction)
    fv         = np.array(friction_list)

    # ── (a) & (b) : trajectory panels ────────────────────────────────────────
    for ax, mu, panel in [(ax_a, mu_train, "a"), (ax_b, mu_val, "b")]:
        sim_res  = sim_cache[mu]
        is_val   = mu not in train_set
        split    = "VAL" if is_val else "TRAIN"
        bg_color = VAL_COLOR if is_val else TRAIN_COLOR

        ax.set_facecolor(bg_color + "55")   # very light tint

        # reference
        s = METHOD_STYLE["Reference"]
        ax.plot(reference[:, 0], reference[:, 1],
                color=s["color"], ls="-", lw=s["lw"],
                label=s["label"], zorder=1, alpha=0.7)

        # methods
        for method in ["Open_loop", "LQR_physics", "PINN_LQR"]:
            if method not in sim_res:
                continue
            states = sim_res[method]
            ms = METHOD_STYLE[method]
            ax.plot(states[:, 0], states[:, 1],
                    color=ms["color"], ls=ms["ls"], lw=ms["lw"],
                    label=ms["label"], zorder=ms["zorder"])

        # start marker
        ax.scatter(reference[0, 0], reference[0, 1],
                   s=50, c="black", zorder=8, marker="o")

        ax.set_title(f"({panel})  μ = {mu:.2f}  [{split}]", fontsize=12)
        ax.set_xlabel("X  [m]", fontsize=10)
        ax.set_ylabel("Y  [m]", fontsize=10)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    # ── (c) : generalisation curve (compact version) ──────────────────────────
    if train_friction:
        max_t = max(train_friction)
        ax_c.axvspan(min(fv) - 0.02, max_t + 0.025,
                     alpha=1.0, color=TRAIN_COLOR, zorder=0)
        ax_c.axvspan(max_t + 0.025, max(fv) + 0.02,
                     alpha=1.0, color=VAL_COLOR,   zorder=0)
        ax_c.axvline(x=max_t + 0.025, color="#888888", ls="--", lw=1.1, zorder=1)

    for method, ate_list in ate_dict.items():
        s = METHOD_STYLE.get(method, {"color": "gray", "ls": "-", "lw": 2.0, "label": method})
        fv_c = fv[:len(ate_list)]
        ax_c.plot(fv_c, ate_list,
                  color=s["color"], ls=s["ls"], lw=s["lw"],
                  label=s["label"], zorder=s["zorder"])
        for i, (mu, ate) in enumerate(zip(fv_c, ate_list)):
            mk = "D" if method == "PINN_LQR" else ("s" if method == "LQR_physics" else "o")
            fc = s["color"] if mu in train_set else "white"
            ax_c.scatter(mu, ate, color=fc, marker=mk,
                         s=40, zorder=s["zorder"] + 1,
                         edgecolors=s["color"], lw=1.3)

    ax_c.set_xlabel("Surface Friction Coefficient  (μ)", fontsize=10)
    ax_c.set_ylabel("ATE  [m]", fontsize=10)
    ax_c.set_title("(c)  Generalisation Curve", fontsize=12)
    ax_c.legend(loc="upper left", fontsize=7.5)
    ax_c.set_xlim(min(fv) - 0.02, max(fv) + 0.02)

    # ── (d) : bar chart (generalisation gap) ──────────────────────────────────
    methods = list(ate_dict.keys())
    t_ate, v_ate = [], []
    for method in methods:
        arr = np.array(ate_dict[method])
        t = [arr[i] for i, mu in enumerate(fv[:len(arr)]) if mu in train_set]
        v = [arr[i] for i, mu in enumerate(fv[:len(arr)]) if mu not in train_set]
        t_ate.append(np.mean(t) if t else 0.0)
        v_ate.append(np.mean(v)   if v else 0.0)

    x_pos = np.arange(len(methods))
    w     = 0.30
    c_list = [METHOD_STYLE.get(m, {}).get("color", "gray") for m in methods]
    c_pale = [matplotlib.colors.to_rgba(c, alpha=0.40) for c in c_list]

    bars_t = ax_d.bar(x_pos - w / 2, t_ate, w, color=c_list,
                      edgecolor="white", lw=0.8, label="Train (seen)")
    bars_v = ax_d.bar(x_pos + w / 2, v_ate, w, color=c_pale,
                      edgecolor=c_list, lw=1.5, linestyle="--",
                      label="Val (unseen)")

    for bar in list(bars_t) + list(bars_v):
        h = bar.get_height()
        ax_d.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                  f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([METHOD_STYLE.get(m, {}).get("label", m) for m in methods],
                         fontsize=8.5, rotation=10, ha="right")
    ax_d.set_ylabel("Mean ATE  [m]", fontsize=10)
    ax_d.set_title("(d)  Train vs. Val ATE  (Generalisation Gap)", fontsize=12)
    ax_d.legend(loc="upper right", fontsize=8)
    ax_d.set_ylim(0, max(max(t_ate), max(v_ate)) * 1.35 + 0.01)

    # ── global title ──────────────────────────────────────────────────────────
    fig.suptitle(
        "PINN-LQR: Friction Generalisation Study\n"
        "TurtleBot3 Circular Trajectory  (r = 2 m,  v = 0.2 m/s)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Results table (terminal)
# ─────────────────────────────────────────────────────────────────────────────

def plot_results_table(results_dict: Dict[str, Dict[str, float]]):
    """Print a Markdown-formatted results table."""
    header  = "\n| Method             | ATE [m]  | MTE [m]  | AVE [m/s] | MVE [m/s] |"
    divider =   "|--------------------|----------|----------|-----------|-----------|"
    print(header)
    print(divider)
    for method, m in results_dict.items():
        label = METHOD_STYLE.get(method, {}).get("label", method)
        print(
            f"| {label:<18s} | {m.get('ATE',0):8.4f} | {m.get('MTE',0):8.4f} | "
            f"{m.get('AVE',0):9.4f} | {m.get('MVE',0):9.4f} |"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Training loss curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history:   Dict[str, list],
    save_path: str = "results/training_loss",
):
    """Plot training and validation loss curves."""
    ep = np.arange(1, len(history.get("train_total", [])) + 1)
    if len(ep) == 0:
        print("  [plot_training_history] No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(ep, history["train_total"], label="Train total", color="#1F77B4", lw=2)
    if "val_total" in history and history["val_total"]:
        axes[0].semilogy(ep, history["val_total"],  label="Val total",   color="#FF7F0E", ls="--", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    comp_map = {
        "train_data":        ("#1F77B4", "Data"),
        "train_physics_vel": ("#D62728", "Physics vel"),
        "train_physics_ang": ("#2CA02C", "Physics ang"),
        "train_ic":          ("#9467BD", "IC"),
    }
    for key, (color, lbl) in comp_map.items():
        if key in history and history[key]:
            axes[1].semilogy(ep, history[key], label=lbl, color=color, lw=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (log scale)")
    axes[1].set_title("Loss Components")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    _save(fig, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — plot from saved JSON
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", type=str, default="results/exp1_results.json")
    parser.add_argument("--history_json", type=str, default="results/training_history.json")
    args = parser.parse_args()

    if os.path.exists(args.history_json):
        with open(args.history_json) as f:
            history = json.load(f)
        plot_training_history(history)
        print("Training history plotted.")

    if os.path.exists(args.results_json):
        with open(args.results_json) as f:
            exp_res = json.load(f)

        friction_values = exp_res.get("friction_values", [])
        results_by_mu   = exp_res.get("results", {})

        # Reconstruct ate_dict: method → [ATE_per_friction]
        ate_dict: Dict[str, List[float]] = {}
        for mu_str in [str(m) for m in friction_values]:
            if mu_str not in results_by_mu:
                continue
            for method, metrics in results_by_mu[mu_str].items():
                ate_dict.setdefault(method, []).append(metrics.get("ATE", 0.0))

        train_friction = [0.1, 0.2, 0.3, 0.4, 0.5]
        plot_generalization_curve(
            friction_values, ate_dict,
            train_friction=train_friction,
            save_path="results/generalization_curve",
        )
        plot_generalization_bar(
            friction_values, ate_dict,
            train_friction=train_friction,
            save_path="results/generalization_bar",
        )
        print("Generalization figures plotted.")
