"""
TurtleBot Differential Drive Dynamics
======================================
State vector x = [px, py, theta, v, omega]  (5-dim)
  px, py   : position in world frame [m]
  theta    : heading angle [rad]
  v        : linear velocity [m/s]
  omega    : angular velocity [rad/s]

Control input u = [a_lin, alpha_ang]  (2-dim)
  a_lin    : linear acceleration command [m/s²]
  alpha_ang: angular acceleration command [rad/s²]

Disturbance: surface friction (analogous to wind for UAV)
  mu_lin   : linear friction coefficient  (dimensionless drag, [1/m])
  mu_ang   : angular friction coefficient (dimensionless drag, [1/rad])

Dynamics:
  px_dot    = v * cos(theta)
  py_dot    = v * sin(theta)
  theta_dot = omega
  v_dot     = a_lin - mu_lin * v * |v|
  omega_dot = alpha_ang - mu_ang * omega * |omega|

No ROS dependency — pure numpy.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Default physical parameters (TurtleBot3 Burger approximate values)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    "m":      1.8,    # mass [kg]
    "mu_lin": 0.2,    # linear friction coefficient [s/m]
    "mu_ang": 0.1,    # angular friction coefficient [s/rad]
    "v_max":  0.22,   # maximum linear speed [m/s]
    "omega_max": 2.84,  # maximum angular speed [rad/s]
}


# ─────────────────────────────────────────────────────────────────────────────
# 5-state Differential Drive Dynamics
# ─────────────────────────────────────────────────────────────────────────────

def dynamics(
    x: np.ndarray,
    u: np.ndarray,
    friction_params: dict = None,
    params: dict = None,
) -> np.ndarray:
    """
    Compute the time derivative  dx/dt  of the 5-dimensional TurtleBot state.

    Kinematic equations (world frame):
        px_dot    = v * cos(theta)
        py_dot    = v * sin(theta)
        theta_dot = omega

    Dynamic equations (with friction disturbance):
        v_dot     = a_lin    - mu_lin * v * |v|
        omega_dot = alpha_ang - mu_ang * omega * |omega|

    Parameters
    ----------
    x               : (5,) state  [px, py, theta, v, omega]
    u               : (2,)  control [a_lin, alpha_ang]
    friction_params : dict with 'mu_lin' and 'mu_ang' overrides
    params          : dict of physical parameters (uses DEFAULT_PARAMS if None)

    Returns
    -------
    dxdt : (5,) state derivative
    """
    if params is None:
        params = DEFAULT_PARAMS

    mu_lin = params["mu_lin"]
    mu_ang = params["mu_ang"]

    # Override with friction_params if provided (the 'disturbance')
    if friction_params is not None:
        mu_lin = friction_params.get("mu_lin", mu_lin)
        mu_ang = friction_params.get("mu_ang", mu_ang)

    # ── unpack state ──────────────────────────────────────────────────────────
    px, py, theta, v, omega = x[0], x[1], x[2], x[3], x[4]

    # ── unpack control ────────────────────────────────────────────────────────
    a_lin     = u[0]   # linear acceleration [m/s²]
    alpha_ang = u[1]   # angular acceleration [rad/s²]

    # ── kinematic derivatives ─────────────────────────────────────────────────
    px_dot    = v * np.cos(theta)
    py_dot    = v * np.sin(theta)
    theta_dot = omega

    # ── dynamic derivatives (friction model) ──────────────────────────────────
    v_dot     = a_lin     - mu_lin * v     * np.abs(v)
    omega_dot = alpha_ang - mu_ang * omega * np.abs(omega)

    return np.array([px_dot, py_dot, theta_dot, v_dot, omega_dot])


# ─────────────────────────────────────────────────────────────────────────────
# Utility: trim control for steady-state cruise
# ─────────────────────────────────────────────────────────────────────────────

def cruise_trim(
    v_target: float = 0.0,
    omega_target: float = 0.0,
    params: dict = None,
) -> np.ndarray:
    """
    Return trim control [a_lin, alpha_ang] to maintain (v_target, omega_target)
    at steady state (v_dot = 0, omega_dot = 0).

    a_lin_trim    = mu_lin * v_target * |v_target|
    alpha_ang_trim = mu_ang * omega_target * |omega_target|
    """
    if params is None:
        params = DEFAULT_PARAMS
    a_lin_trim    = params["mu_lin"] * v_target * abs(v_target)
    alpha_ang_trim = params["mu_ang"] * omega_target * abs(omega_target)
    return np.array([a_lin_trim, alpha_ang_trim])


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Analytical discrete-time linearization
# ─────────────────────────────────────────────────────────────────────────────

def compute_jacobians(
    x_op: np.ndarray,
    u_op: np.ndarray,
    params: dict = None,
    dt: float = 0.02,
) -> tuple:
    """
    Compute analytical discrete-time linearization (A_d, B_d) at (x_op, u_op).

    Continuous-time Jacobians (derived from the dynamics equations):
        A_c = ∂f/∂x,  B_c = ∂f/∂u

    Euler discretization:
        A_d = I + dt * A_c
        B_d = dt * B_c

    Parameters
    ----------
    x_op   : (5,) operating-point state [px, py, theta, v, omega]
    u_op   : (2,) operating-point control [a_lin, alpha_ang]
    params : physics parameter dict (uses DEFAULT_PARAMS if None)
    dt     : sampling period [s]

    Returns
    -------
    A_d : (5, 5) discrete state matrix
    B_d : (5, 2) discrete control matrix
    """
    if params is None:
        params = DEFAULT_PARAMS

    mu_lin = params["mu_lin"]
    mu_ang = params["mu_ang"]

    theta = x_op[2]
    v     = x_op[3]
    omega = x_op[4]

    # ── Continuous-time A = ∂f/∂x ────────────────────────────────────────────
    # f = [v*cos(theta), v*sin(theta), omega, a_lin - mu*v|v|, alpha - mu_r*omega|omega|]
    A_c = np.zeros((5, 5))
    A_c[0, 2] = -v * np.sin(theta)          # ∂(px_dot)/∂theta
    A_c[0, 3] =  np.cos(theta)              # ∂(px_dot)/∂v
    A_c[1, 2] =  v * np.cos(theta)          # ∂(py_dot)/∂theta
    A_c[1, 3] =  np.sin(theta)              # ∂(py_dot)/∂v
    A_c[2, 4] =  1.0                        # ∂(theta_dot)/∂omega
    # d(v|v|)/dv = 2|v|  →  ∂(v_dot)/∂v = -mu_lin * 2|v|
    A_c[3, 3] = -mu_lin * 2.0 * abs(v)
    # d(omega|omega|)/domega = 2|omega|
    A_c[4, 4] = -mu_ang * 2.0 * abs(omega)

    # ── Continuous-time B = ∂f/∂u ────────────────────────────────────────────
    B_c = np.zeros((5, 2))
    B_c[3, 0] = 1.0   # ∂(v_dot)/∂a_lin = 1
    B_c[4, 1] = 1.0   # ∂(omega_dot)/∂alpha_ang = 1

    # ── Euler discretization ─────────────────────────────────────────────────
    A_d = np.eye(5) + dt * A_c
    B_d = dt * B_c

    return A_d, B_d


if __name__ == "__main__":
    print("=== TurtleBot Physics Model Self-Test ===\n")

    # Test 1: stationary trim (v=0, omega=0) → all derivatives should be 0
    x0 = np.zeros(5)
    u0 = cruise_trim(v_target=0.0, omega_target=0.0)
    dxdt = dynamics(x0, u0)
    print(f"Stationary trim control: {u0}")
    print(f"dx/dt at rest (should be ~0):\n  {dxdt}")
    assert np.allclose(dxdt, 0, atol=1e-10), "Stationary equilibrium check FAILED"
    print("[PASS] Stationary equilibrium test passed.\n")

    # Test 2: cruise trim at v=0.2 m/s → v_dot should be ~0
    x_cruise = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
    u_cruise  = cruise_trim(v_target=0.2, omega_target=0.0)
    dxdt2     = dynamics(x_cruise, u_cruise)
    print(f"Cruise trim at v=0.2: u={u_cruise}")
    print(f"dx/dt at cruise (v_dot should be ~0):\n  {dxdt2}")
    assert abs(dxdt2[3]) < 1e-10, "Cruise equilibrium check FAILED"
    print("[PASS] Cruise equilibrium test passed.\n")

    # Test 3: kinematic consistency — moving at heading pi/4
    x_diag = np.array([0.0, 0.0, np.pi/4, 1.0, 0.0])
    u_zero  = np.zeros(2)
    dxdt3   = dynamics(x_diag, u_zero, friction_params={"mu_lin": 0.0, "mu_ang": 0.0})
    expected_vx = np.cos(np.pi/4)
    expected_vy = np.sin(np.pi/4)
    assert abs(dxdt3[0] - expected_vx) < 1e-10, "Kinematic x-velocity check FAILED"
    assert abs(dxdt3[1] - expected_vy) < 1e-10, "Kinematic y-velocity check FAILED"
    print(f"[PASS] Kinematic consistency test passed (vx={dxdt3[0]:.4f}, vy={dxdt3[1]:.4f}).\n")

    # Test 4: friction decelerates the robot
    x_moving = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    u_zero   = np.zeros(2)
    dxdt4    = dynamics(x_moving, u_zero)
    assert dxdt4[3] < 0, "Friction should decelerate positive linear velocity"
    print(f"[PASS] Friction deceleration test passed (v_dot={dxdt4[3]:.4f}).\n")

    print("All tests passed.")
