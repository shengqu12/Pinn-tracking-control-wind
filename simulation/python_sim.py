"""
Pure-Python TurtleBot Simulator
================================
Lightweight 2D simulation using scipy.integrate.solve_ivp for rapid
controller validation without Gazebo.

Usage
-----
from simulation.python_sim import run_simulation
from controllers.lqr_controller import LQRController
import numpy as np

results = run_simulation(controller, trajectory, friction_params={"mu_lin": 0.3})
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Dict, Optional, Callable

from models.turtlebot_physics import dynamics, cruise_trim, DEFAULT_PARAMS



# ─────────────────────────────────────────────────────────────────────────────
# Trajectory generators (2D, for differential drive robot)
# ─────────────────────────────────────────────────────────────────────────────

def circular_trajectory(
    radius:  float = 2.0,
    speed:   float = 0.2,
    dt:      float = 0.02,
    t_total: float = 20.0,
) -> np.ndarray:
    """
    Generate a circular reference trajectory in the XY-plane for TurtleBot.

    The robot travels at constant linear speed `speed` around a circle of
    given `radius`.  The heading theta is always tangent to the circle.

    Returns
    -------
    ref : (T, 5) reference states [px, py, theta, v, omega]
    """
    t      = np.arange(0, t_total, dt)
    T      = len(t)
    omega  = speed / radius            # angular velocity [rad/s]
    ref    = np.zeros((T, 5))

    ref[:, 0] = radius * np.cos(omega * t)          # px
    ref[:, 1] = radius * np.sin(omega * t)          # py
    ref[:, 2] = omega  * t + np.pi / 2              # theta (tangent direction)
    ref[:, 3] = speed                                # v (constant)
    ref[:, 4] = omega                                # omega (constant)
    return ref


def figure_eight_trajectory(
    scale:   float = 2.0,
    speed:   float = 0.2,
    dt:      float = 0.02,
    t_total: float = 20.0,
) -> np.ndarray:
    """
    Generate a figure-eight (lemniscate) reference trajectory.

    Returns
    -------
    ref : (T, 5) reference states [px, py, theta, v, omega]
    """
    t   = np.arange(0, t_total, dt)
    T   = len(t)
    w   = 2 * np.pi / t_total   # one full figure-eight per t_total
    ref = np.zeros((T, 5))

    ref[:, 0] = scale * np.sin(w * t)               # px
    ref[:, 1] = scale * np.sin(2 * w * t) / 2       # py
    # heading: numerical derivative
    dx = scale * w * np.cos(w * t)
    dy = scale * w * np.cos(2 * w * t)
    ref[:, 2] = np.arctan2(dy, dx)
    ref[:, 3] = speed
    return ref


def stationary_trajectory(
    position: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    dt:       float = 0.02,
    t_total:  float = 10.0,
) -> np.ndarray:
    """Constant setpoint trajectory."""
    T   = int(t_total / dt)
    ref = np.zeros((T, 5))
    ref[:, :len(position)] = position
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    controller,
    trajectory:      np.ndarray,
    friction_params: dict = None,
    x0:              Optional[np.ndarray] = None,
    dt:              float = 0.02,
    params:          dict  = None,
    ctrl_type:       str   = "lqr",    # "lqr" | default
) -> Dict[str, np.ndarray]:
    """
    Simulate TurtleBot under the given controller and surface friction.

    Parameters
    ----------
    controller      : controller object with .control() (LQR)
    trajectory      : (T, 5) reference state sequence
    friction_params : {'mu_lin', 'mu_ang'} surface friction (the disturbance)
    x0              : (5,) initial state (uses trajectory[0] if None)
    dt              : simulation time step [s]
    params          : physics parameters dict
    ctrl_type       : controller API type

    Returns
    -------
    dict with keys: 'states', 'controls', 'time', 'reference'
    """
    if params is None:
        params = DEFAULT_PARAMS
    if friction_params is None:
        friction_params = {"mu_lin": params["mu_lin"], "mu_ang": params["mu_ang"]}
    if x0 is None:
        x0 = trajectory[0].copy()

    T   = len(trajectory)
    t   = np.arange(T) * dt
    X   = np.zeros((T, 5))
    U   = np.zeros((T, 2))
    X[0] = x0

    u_trim = cruise_trim(v_target=0.0, omega_target=0.0, params=params)

    for i in range(T - 1):
        x_curr = X[i]
        x_ref  = trajectory[i]

        # ── compute control ───────────────────────────────────────────────────
        try:
            if ctrl_type == "lqr":
                u_curr = controller.control(x_curr, x_ref)
            else:
                u_curr = u_trim

        except Exception as e:
            print(f"  [Sim] Controller error at step {i}: {e}  — using trim")
            u_curr = u_trim

        # Clip accelerations to reasonable range
        u_curr[0] = np.clip(u_curr[0], -2.0, 2.0)    # a_lin [m/s²]
        u_curr[1] = np.clip(u_curr[1], -5.0, 5.0)    # alpha_ang [rad/s²]
        U[i] = u_curr

        # ── integrate dynamics ────────────────────────────────────────────────
        sol = solve_ivp(
            fun=lambda tt, x: dynamics(x, u_curr, friction_params, params),
            t_span=(0.0, dt),
            y0=x_curr,
            method="RK45",
            max_step=dt / 4,
            rtol=1e-6,
            atol=1e-8,
        )
        X[i + 1] = sol.y[:, -1]

    U[-1] = U[-2]

    return {
        "states":    X,
        "controls":  U,
        "time":      t,
        "reference": trajectory,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d_trajectory(results: Dict, title: str = "TurtleBot 2D Trajectory", save_path: str = None):
    """Plot the 2D (XY) trajectory of the TurtleBot."""
    X   = results["states"]
    ref = results["reference"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(X[:, 0],   X[:, 1],   "b-",  lw=2,   label="Actual")
    ax.plot(ref[:, 0], ref[:, 1], "r--", lw=1.5, label="Reference")
    ax.scatter(*X[0, :2],   color="green",  s=80, zorder=5, label="Start")
    ax.scatter(*X[-1, :2],  color="orange", s=80, zorder=5, label="End")

    # Draw heading arrows at regular intervals
    step = max(1, len(X) // 20)
    for i in range(0, len(X), step):
        ax.annotate("",
            xy=(X[i, 0] + 0.1*np.cos(X[i, 2]),
                X[i, 1] + 0.1*np.sin(X[i, 2])),
            xytext=(X[i, 0], X[i, 1]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=1),
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved 2D trajectory to {save_path}")
    return fig


def plot_tracking_error(results: Dict, save_path: str = None):
    """Plot position and heading tracking error over time."""
    X   = results["states"]
    ref = results["reference"]
    t   = results["time"]
    pos_err = np.linalg.norm(X[:, :2] - ref[:, :2], axis=1)
    hdg_err = np.abs((X[:, 2] - ref[:, 2] + np.pi) % (2*np.pi) - np.pi)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, pos_err, "b-", label="Position error")
    axes[0].set_ylabel("Position Error [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, hdg_err, "r-", label="Heading error")
    axes[1].set_ylabel("Heading Error [rad]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Tracking Performance")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: open-loop straight-line simulation
# ─────────────────────────────────────────────────────────────────────────────

def open_loop_cruise_sim(
    t_total:         float = 10.0,
    dt:              float = 0.02,
    friction_params: dict  = None,
    v_target:        float = 0.2,
) -> Dict:
    """
    Cruise at constant speed with friction disturbance — useful as a baseline.
    No closed-loop controller needed.
    """
    T       = int(t_total / dt)
    t_arr   = np.arange(T) * dt
    X       = np.zeros((T, 5))
    U       = np.zeros((T, 2))

    x0      = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u_t     = np.array([0.5, 0.0])   # constant acceleration command
    X[0]    = x0

    if friction_params is None:
        friction_params = {"mu_lin": 0.2, "mu_ang": 0.1}

    for i in range(T - 1):
        U[i] = u_t
        sol  = solve_ivp(
            fun=lambda tt, x: dynamics(x, u_t, friction_params),
            t_span=(0.0, dt),
            y0=X[i],
            method="RK45",
            max_step=dt / 4,
        )
        X[i + 1] = sol.y[:, -1]

    U[-1] = u_t
    ref   = np.tile(x0, (T, 1))
    ref[:, 0] = v_target * t_arr   # straight-line reference
    ref[:, 3] = v_target
    return {"states": X, "controls": U, "time": t_arr, "reference": ref}


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running open-loop cruise simulation (friction=0.3)...")

    friction = {"mu_lin": 0.3, "mu_ang": 0.15}
    res      = open_loop_cruise_sim(t_total=10.0, dt=0.02, friction_params=friction)

    X   = res["states"]
    err = np.linalg.norm(X[:, :2] - res["reference"][:, :2], axis=1)
    print(f"  Max position error: {err.max():.3f} m")
    print(f"  Final position:     {X[-1, :2]}")
    print(f"  Final speed:        {X[-1, 3]:.3f} m/s")

    os.makedirs("results", exist_ok=True)
    plot_2d_trajectory(res, title="Open-loop Cruise (friction=0.3)", save_path="results/cruise_2d.png")
    plot_tracking_error(res, save_path="results/cruise_error.png")
    print("Plots saved to results/")
