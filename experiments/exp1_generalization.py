"""
Experiment 1 — Friction Generalisation Study
=============================================
Compares three controllers across training and unseen validation friction
coefficients to demonstrate PINN's generalisation advantage:

  Open_loop    : constant trim, no feedback (lower-bound baseline)
  LQR_physics  : classical LQR with analytical linearization at nominal friction
                 (best possible without learned dynamics)
  PINN_LQR     : LQR using PINN-learned dynamics (proposed method)

Key narrative:
  - LQR_physics degrades when friction deviates from its nominal assumption.
  - PINN_LQR generalises to unseen friction because its Jacobians reflect
    the learned friction-dependent dynamics.

Robot: TurtleBot3 Burger (differential drive)
Task:  Circular trajectory tracking (radius=2m, speed=0.2 m/s)

Usage
-----
python experiments/exp1_generalization.py --friction 0.3
python experiments/exp1_generalization.py --friction all --save_results
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.turtlebot_physics     import cruise_trim, DEFAULT_PARAMS
from models.pinn_dynamics          import PINNDynamics
from controllers.ilqr_controller   import (
    LQRController, AnalyticalLQRController, OpenLoopController, NX, NU
)
from simulation.python_sim         import run_simulation, circular_trajectory
from evaluation.metrics            import (
    compute_metrics, compute_lateral_error, compute_heading_error, print_metrics_table
)
from evaluation.plot_results       import (
    plot_trajectory_comparison,
    plot_tracking_error,
    plot_generalization_curve,
    plot_generalization_bar,
    plot_results_table,
    plot_summary_figure,
)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FRICTION = [0.1, 0.2, 0.3, 0.4, 0.5]
VAL_FRICTION   = [0.15, 0.25, 0.35, 0.45, 0.55]
ALL_FRICTION   = sorted(set(TRAIN_FRICTION) | set(VAL_FRICTION))

DT          = 0.02
T_TOTAL     = 20.0
RADIUS      = 2.0
SPEED       = 0.2
N_HORIZON   = 20
NOMINAL_MU  = 0.2   # friction value assumed by LQR_physics

# Cost matrices: x = [px, py, theta, v, omega]
_Q = np.diag([10.0, 10.0, 5.0, 1.0, 0.5])
_R = np.diag([0.1, 0.1])


# ─────────────────────────────────────────────────────────────────────────────
# Controller factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_controllers(pinn_model: PINNDynamics, traj: np.ndarray):
    """Build all three controllers for a single experiment run."""
    # Operating point: first waypoint on the circle
    x_op = traj[0].copy()                                      # (5,)
    u_op = cruise_trim(v_target=SPEED, omega_target=SPEED/RADIUS)

    # 1. Open-loop
    open_loop = OpenLoopController(
        v_target=SPEED, omega_target=SPEED/RADIUS, nominal_mu=NOMINAL_MU
    )

    # 2. LQR with analytical physics (fixed nominal friction)
    lqr_phys = AnalyticalLQRController(_Q, _R, dt=DT, nominal_mu=NOMINAL_MU)
    try:
        lqr_phys.linearise(x_op, u_op)
    except Exception as e:
        print(f"  [LQR_physics] Linearise failed: {e}")

    # 3. PINN_LQR (LQR using PINN-learned Jacobians)
    pinn_lqr = LQRController(pinn_model, _Q, _R)
    try:
        pinn_lqr.linearise(x_op, u_op)
    except Exception as e:
        print(f"  [PINN_LQR] Linearise failed: {e}")

    return {
        "Open_loop":   open_loop,
        "LQR_physics": lqr_phys,
        "PINN_LQR":    pinn_lqr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-friction run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_single(
    mu:           float,
    pinn_model:   PINNDynamics,
    save_results: bool = True,
    show_table:   bool = True,
) -> dict:
    """
    Run all three controllers at a single friction coefficient.
    Returns metric dict: { method: {ATE, MTE, AVE, MVE} }
    """
    print(f"\n{'='*60}")
    print(f"  mu_lin = {mu:.2f}  ({'TRAIN' if mu in TRAIN_FRICTION else 'VAL'})")
    print(f"{'='*60}")

    friction = {"mu_lin": mu, "mu_ang": mu * 0.5}
    traj     = circular_trajectory(radius=RADIUS, speed=SPEED, dt=DT, t_total=T_TOTAL)
    x0       = np.array([RADIUS, 0.0, np.pi/2, SPEED, SPEED/RADIUS])

    controllers = _build_controllers(pinn_model, traj)

    # ── Simulate all methods ──────────────────────────────────────────────────
    sim_results = {"Reference": traj}
    for name, ctrl in controllers.items():
        print(f"  [{name}] simulating...")
        try:
            res = run_simulation(
                ctrl, traj, friction_params=friction,
                x0=x0, dt=DT, ctrl_type="lqr"
            )
            sim_results[name] = res["states"]
            sim_results[f"_res_{name}"] = res   # store full result for error plots
        except Exception as e:
            print(f"    Warning: {name} failed ({e}), using zeros")
            T = len(traj)
            sim_results[name] = np.zeros((T, NX))

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {}
    for method in controllers:
        metrics[method] = compute_metrics(sim_results[method], traj)

    if show_table:
        print(f"\nResults at mu_lin = {mu:.2f}:")
        plot_results_table(metrics)
        print_metrics_table(metrics)

    # ── Save plots ────────────────────────────────────────────────────────────
    if save_results:
        os.makedirs("results", exist_ok=True)
        tag = f"mu{mu:.2f}"

        # trajectory comparison (position only)
        traj_data = {k: v for k, v in sim_results.items() if not k.startswith("_res_")}
        plot_trajectory_comparison(
            traj_data,
            save_path=f"results/{tag}_trajectory",
            mu=mu,
            is_val=(mu in VAL_FRICTION),
        )

        # tracking error over time
        t_arr  = np.arange(len(traj)) * DT
        errors = {}
        for method in controllers:
            errors[method]              = compute_lateral_error(sim_results[method], traj)
            errors[f"heading_{method}"] = compute_heading_error(sim_results[method], traj)
        plot_tracking_error(errors, t_arr, save_path=f"results/{tag}_tracking_error",
                            title_suffix=f"(μ = {mu:.2f})")

    return metrics, sim_results


# ─────────────────────────────────────────────────────────────────────────────
# Full multi-friction experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_all(
    pinn_model:    PINNDynamics,
    friction_list: list = ALL_FRICTION,
    save_results:  bool = True,
) -> dict:
    """Run across all friction coefficients and aggregate results."""
    all_results   = {"friction_values": friction_list, "results": {}}
    ate_dict      = {}
    sim_cache     = {}   # store sim_results at selected friction values for summary figure

    selected_show = [TRAIN_FRICTION[1], VAL_FRICTION[-1]]   # one train, one val example

    for mu in friction_list:
        metrics, sim_res = run_experiment_single(
            mu, pinn_model, save_results=save_results, show_table=True
        )
        all_results["results"][str(mu)] = {k: dict(v) for k, v in metrics.items()}

        for method, m in metrics.items():
            ate_dict.setdefault(method, []).append(m["ATE"])

        if mu in selected_show:
            sim_cache[mu] = sim_res

    # ── Generalisation curve ──────────────────────────────────────────────────
    if save_results:
        plot_generalization_curve(
            friction_values=friction_list,
            ate_dict=ate_dict,
            train_friction=TRAIN_FRICTION,
            save_path="results/generalization_curve",
        )
        plot_generalization_bar(
            friction_values=friction_list,
            ate_dict=ate_dict,
            train_friction=TRAIN_FRICTION,
            save_path="results/generalization_bar",
        )

        # Summary 2×2 figure (two trajectory examples + generalisation curve + error)
        if len(sim_cache) >= 2:
            traj_ref = circular_trajectory(
                radius=RADIUS, speed=SPEED, dt=DT, t_total=T_TOTAL
            )
            plot_summary_figure(
                sim_cache=sim_cache,
                ate_dict=ate_dict,
                friction_list=friction_list,
                train_friction=TRAIN_FRICTION,
                reference=traj_ref,
                save_path="results/summary_figure",
            )

        with open("results/exp1_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nAll results saved to results/exp1_results.json")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Generalisation Summary ===")
    for method in ate_dict:
        arr   = ate_dict[method]
        t_ate = [arr[i] for i, mu in enumerate(friction_list) if mu in TRAIN_FRICTION]
        v_ate = [arr[i] for i, mu in enumerate(friction_list) if mu in VAL_FRICTION]
        print(f"  {method:<14s}  Train ATE: {np.mean(t_ate):.4f} m  |  "
              f"Val ATE: {np.mean(v_ate):.4f} m  "
              f"(gap: {abs(np.mean(v_ate)-np.mean(t_ate)):.4f} m)")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TurtleBot friction generalisation experiment"
    )
    parser.add_argument("--friction",     type=str, default="0.3",
                        help="Friction coefficient or 'all' for full sweep")
    parser.add_argument("--model_path",   type=str, default=None,
                        help="Path to trained PINN checkpoint (.pt)")
    parser.add_argument("--save_results", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(42);  np.random.seed(42)

    pinn = PINNDynamics().double()
    if args.model_path and os.path.exists(args.model_path):
        pinn.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        print(f"Loaded PINN from {args.model_path}")
    else:
        print("No checkpoint — using randomly initialised PINN (run run_demo.py first!)")
    pinn.eval()

    if args.friction.lower() == "all":
        run_experiment_all(pinn, friction_list=ALL_FRICTION,
                           save_results=args.save_results)
    else:
        mu = float(args.friction)
        run_experiment_single(mu, pinn, save_results=args.save_results)
