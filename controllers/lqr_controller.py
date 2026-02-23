"""
LQR Controller (TurtleBot)
==========================
Implements three controllers:
  LQRController          — standard LQR linearised at an operating point via PINN Jacobians.
  AnalyticalLQRController — classical LQR using analytical physics Jacobians (fixed nominal friction).
  OpenLoopController     — constant trim feed-forward (no state feedback, lower-bound baseline).

State:   x = [px, py, theta, v, omega]   (5-dim)
Control: u = [a_lin, alpha_ang]           (2-dim)

No ROS dependency — pure numpy / scipy.
"""

import numpy as np
import scipy.linalg
import torch
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Dimensions
# ─────────────────────────────────────────────────────────────────────────────
NX = 5   # state dimension: [px, py, theta, v, omega]
NU = 2   # control dimension: [a_lin, alpha_ang]


# ─────────────────────────────────────────────────────────────────────────────
# LQRController
# ─────────────────────────────────────────────────────────────────────────────

class LQRController:
    """
    Standard (infinite-horizon) discrete LQR linearised around a trim point.

    Solves the discrete algebraic Riccati equation (DARE) using scipy.

    Parameters
    ----------
    pinn_model : PINNDynamics
    Q          : (NX, NX) state cost
    R          : (NU, NU) control cost
    """

    def __init__(self, pinn_model, Q: np.ndarray, R: np.ndarray):
        self.model  = pinn_model
        self.Q      = Q
        self.R      = R
        self.K_lqr: Optional[np.ndarray] = None
        self.x_op:  Optional[np.ndarray] = None
        self.u_op:  Optional[np.ndarray] = None

    def linearise(self, x_op: np.ndarray, u_op: np.ndarray):
        """
        Linearise dynamics at operating point and solve DARE.

        Parameters
        ----------
        x_op : (NX,) operating-point state  (e.g. stationary or cruise)
        u_op : (NU,) operating-point control
        """
        self.x_op = x_op.copy()
        self.u_op = u_op.copy()

        x_t = torch.tensor(x_op, dtype=torch.float64)
        u_t = torch.tensor(u_op, dtype=torch.float64)
        A, B = self.model.get_jacobian(x_t, u_t)   # (NX,NX), (NX,NU)

        try:
            P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
            self.K_lqr = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        except Exception as e:
            print(f"[LQR] DARE failed: {e}  — falling back to pseudo-inverse")
            self.K_lqr = np.linalg.pinv(B) @ (A - np.eye(NX))

        print(f"[LQR] Linearised at operating point. K_lqr shape: {self.K_lqr.shape}")

    def control(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """
        Compute LQR control u = u_op - K * (x - x_ref).

        Parameters
        ----------
        x     : (NX,) current state
        x_ref : (NX,) reference state

        Returns
        -------
        u : (NU,) control input
        """
        if self.K_lqr is None:
            raise RuntimeError("Call linearise() before control().")
        dx = x - x_ref
        return self.u_op - self.K_lqr @ dx


# ─────────────────────────────────────────────────────────────────────────────
# AnalyticalLQRController
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticalLQRController:
    """
    Classical LQR using analytical physics linearization.

    Linearizes the TurtleBot dynamics at the operating point using the known
    physics equations with a FIXED nominal friction coefficient.  This
    represents what you would use without a learned dynamics model — the
    best you can do with a calibrated-but-fixed physics model.

    Limitation: when actual surface friction deviates from the nominal value,
    the Jacobians become inaccurate and tracking performance degrades.

    Parameters
    ----------
    Q          : (NX, NX) state cost matrix
    R          : (NU, NU) control cost matrix
    dt         : time step [s]
    nominal_mu : friction coefficient assumed during linearization
    """

    def __init__(self, Q: np.ndarray, R: np.ndarray,
                 dt: float = 0.02, nominal_mu: float = 0.2):
        from models.turtlebot_physics import DEFAULT_PARAMS, compute_jacobians
        self._compute_jacobians = compute_jacobians
        self.Q  = Q
        self.R  = R
        self.dt = dt
        self.nominal_params = dict(DEFAULT_PARAMS)
        self.nominal_params["mu_lin"] = nominal_mu
        self.nominal_params["mu_ang"] = nominal_mu * 0.5

        self.K_lqr: Optional[np.ndarray] = None
        self.x_op:  Optional[np.ndarray] = None
        self.u_op:  Optional[np.ndarray] = None

    def linearise(self, x_op: np.ndarray, u_op: np.ndarray):
        """Linearise at (x_op, u_op) using analytical Jacobians."""
        self.x_op = x_op.copy()
        self.u_op = u_op.copy()

        A, B = self._compute_jacobians(x_op, u_op, self.nominal_params, self.dt)

        try:
            P = scipy.linalg.solve_discrete_are(A, B, self.Q, self.R)
            self.K_lqr = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        except Exception as e:
            print(f"[AnalyticalLQR] DARE failed: {e}  — falling back to pseudo-inverse")
            self.K_lqr = np.linalg.pinv(B) @ (A - np.eye(NX))

        mu_nom = self.nominal_params["mu_lin"]
        print(f"[AnalyticalLQR] Linearised at nominal mu={mu_nom:.2f}. K shape: {self.K_lqr.shape}")

    def control(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """u = u_op - K * (x - x_ref)"""
        if self.K_lqr is None:
            raise RuntimeError("Call linearise() before control().")
        return self.u_op - self.K_lqr @ (x - x_ref)


# ─────────────────────────────────────────────────────────────────────────────
# OpenLoopController
# ─────────────────────────────────────────────────────────────────────────────

class OpenLoopController:
    """
    Open-loop (feed-forward only) controller.

    Applies a constant trim control computed for circular motion at the
    NOMINAL friction.  No state feedback — serves as a lower-bound baseline
    that shows how much the PINN / LQR feedback helps.

    Parameters
    ----------
    v_target   : desired linear speed [m/s]
    omega_target : desired angular rate [rad/s]  (= v_target / radius)
    nominal_mu : friction assumed when computing trim
    """

    def __init__(self, v_target: float = 0.2, omega_target: float = 0.1,
                 nominal_mu: float = 0.2):
        mu_lin = nominal_mu
        mu_ang = nominal_mu * 0.5
        a_trim     = mu_lin * v_target * abs(v_target)
        alpha_trim = mu_ang * omega_target * abs(omega_target)
        self.u_trim = np.array([a_trim, alpha_trim])

    def control(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Ignore state — return constant trim."""
        return self.u_trim.copy()

    def solve(self, x0: np.ndarray, x_ref: np.ndarray,
              U_init: Optional[np.ndarray] = None):
        N = len(x_ref) - 1 if hasattr(x_ref, "__len__") else 20
        U = np.tile(self.u_trim, (N, 1))
        X = np.zeros((N + 1, NX))
        X[0] = x0
        return U, X, {"converged": True}


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.pinn_dynamics import PINNDynamics
    from models.turtlebot_physics import cruise_trim

    torch.manual_seed(0)
    model = PINNDynamics().double()

    # Diagonal cost: penalise position / heading error more than velocity
    Q = np.diag([10.0, 10.0, 5.0, 1.0, 0.5])
    R = np.diag([0.1, 0.1])

    # ── LQR (PINN-based Jacobians) ────────────────────────────────────────────
    x0    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([2.0, 1.0, 0.0, 0.0, 0.0])
    u_op  = cruise_trim(v_target=0.0)

    lqr = LQRController(model, Q, R)
    lqr.linearise(x0, u_op)
    u_lqr = lqr.control(x0 + np.random.randn(NX) * 0.05, x_ref)
    print(f"PINN LQR control: {u_lqr}")

    # ── AnalyticalLQR ─────────────────────────────────────────────────────────
    alqr = AnalyticalLQRController(Q, R, nominal_mu=0.2)
    alqr.linearise(x0, u_op)
    u_alqr = alqr.control(x0 + np.random.randn(NX) * 0.05, x_ref)
    print(f"Analytical LQR control: {u_alqr}")

    print("Controller self-test complete.")
