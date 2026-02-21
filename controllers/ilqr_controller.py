"""
iLQR Controller + LQR baseline (TurtleBot)
===========================================
Implements:
  ILQRController  — iterative Linear Quadratic Regulator with Levenberg-Marquardt
                    regularisation and backtracking line search.
  LQRController   — standard LQR linearised at a single operating point.

State:   x = [px, py, theta, v, omega]   (5-dim)
Control: u = [a_lin, alpha_ang]           (2-dim)

Reference: Cao et al. (2025), Algorithm 1
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
# ILQRController
# ─────────────────────────────────────────────────────────────────────────────

class ILQRController:
    """
    iterative Linear Quadratic Regulator (iLQR) with:
      - Levenberg-Marquardt (LM) regularisation for numerical stability
      - Backtracking line search with Armijo condition

    The dynamics model is queried via pinn_model.get_jacobian(x, u).

    Parameters
    ----------
    pinn_model : PINNDynamics  — differentiable TurtleBot dynamics model
    Q          : (NX, NX)     — state cost matrix (positive semi-definite)
    R          : (NU, NU)     — control cost matrix (positive definite)
    N          : int          — prediction horizon (time steps)
    dt         : float        — time step [s]
    """

    def __init__(self, pinn_model, Q: np.ndarray, R: np.ndarray, N: int, dt: float = 0.02):
        self.model = pinn_model
        self.Q  = Q
        self.R  = R
        self.Qf = Q * 10
        self.N  = N
        self.dt = dt

        # LM hyper-parameters
        self.lambda_lm      = 1.0
        self.lambda_min     = 1e-6
        self.lambda_max     = 1e10
        self.lambda_factor  = 10.0

        # Line search
        self.alpha_list = [1.0, 0.5, 0.25, 0.125, 0.0625]
        self.armijo_c   = 1e-4

        # Convergence
        self.max_iter = 50
        self.tol      = 1e-6

    # ── cost functions ────────────────────────────────────────────────────────

    def _running_cost(self, x: np.ndarray, u: np.ndarray, x_ref: np.ndarray) -> float:
        dx = x - x_ref
        return float(dx @ self.Q @ dx + u @ self.R @ u)

    def _terminal_cost(self, x: np.ndarray, x_ref: np.ndarray) -> float:
        dx = x - x_ref
        return float(dx @ self.Qf @ dx)

    def _total_cost(self, X: np.ndarray, U: np.ndarray, X_ref: np.ndarray) -> float:
        J = sum(self._running_cost(X[t], U[t], X_ref[t]) for t in range(self.N))
        J += self._terminal_cost(X[-1], X_ref[-1])
        return J

    # ── dynamics rollout ──────────────────────────────────────────────────────

    def _rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        X    = np.zeros((self.N + 1, NX))
        X[0] = x0

        x_t = torch.tensor(x0, dtype=torch.float64)
        for t in range(self.N):
            u_t  = torch.tensor(U[t], dtype=torch.float64)
            with torch.no_grad():
                x_next = self.model(
                    x_t.unsqueeze(0), u_t.unsqueeze(0)
                ).squeeze(0)
            X[t + 1] = x_next.numpy()
            x_t = x_next

        return X

    # ── backward pass ─────────────────────────────────────────────────────────

    def backward_pass(
        self,
        X:         np.ndarray,
        U:         np.ndarray,
        X_ref:     np.ndarray,
        lambda_lm: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Riccati recursion with LM regularisation.

        Returns
        -------
        K   : (N, NU, NX)  feedback gain matrices
        k   : (N, NU)      feed-forward increments
        dJ  : expected cost improvement
        """
        dx_N = X[-1] - X_ref[-1]
        Vx   = 2 * self.Qf @ dx_N
        Vxx  = 2 * self.Qf

        K  = np.zeros((self.N, NU, NX))
        k  = np.zeros((self.N, NU))
        dJ = 0.0

        for t in reversed(range(self.N)):
            x_t = X[t]
            u_t = U[t]

            x_torch = torch.tensor(x_t, dtype=torch.float64)
            u_torch = torch.tensor(u_t, dtype=torch.float64)
            fx, fu  = self.model.get_jacobian(x_torch, u_torch)   # (NX,NX), (NX,NU)

            dx_t = x_t - X_ref[t]
            lx   = 2 * self.Q  @ dx_t
            lu   = 2 * self.R  @ u_t
            lxx  = 2 * self.Q
            luu  = 2 * self.R
            lxu  = np.zeros((NX, NU))

            Qx  = lx  + fx.T @ Vx
            Qu  = lu  + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lxu.T + fu.T @ Vxx @ fx

            Quu_reg = Quu + lambda_lm * np.eye(NU)

            try:
                Quu_inv = np.linalg.inv(Quu_reg)
            except np.linalg.LinAlgError:
                Quu_inv = np.linalg.pinv(Quu_reg)

            K[t] = -Quu_inv @ Qux
            k[t] = -Quu_inv @ Qu

            dJ += float(Qu @ k[t] + 0.5 * k[t] @ Quu @ k[t])

            Vx  = Qx  + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]

        return K, k, dJ

    # ── forward pass ─────────────────────────────────────────────────────────

    def forward_pass(
        self,
        X:     np.ndarray,
        U:     np.ndarray,
        K:     np.ndarray,
        k:     np.ndarray,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_new    = np.zeros_like(X)
        U_new    = np.zeros_like(U)
        X_new[0] = X[0]

        x_t = torch.tensor(X[0], dtype=torch.float64)
        for t in range(self.N):
            dx      = X_new[t] - X[t]
            u_new   = U[t] + alpha * k[t] + K[t] @ dx
            U_new[t] = u_new

            u_torch  = torch.tensor(u_new, dtype=torch.float64)
            with torch.no_grad():
                x_next = self.model(
                    x_t.unsqueeze(0), u_torch.unsqueeze(0)
                ).squeeze(0)
            X_new[t + 1] = x_next.numpy()
            x_t = x_next

        return X_new, U_new

    # ── main solve ────────────────────────────────────────────────────────────

    def solve(
        self,
        x0:     np.ndarray,
        x_ref:  np.ndarray,
        U_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Full iLQR solver (Algorithm 1 from Cao et al.).

        Parameters
        ----------
        x0     : (NX,)     initial state
        x_ref  : (N+1, NX) or (NX,) reference trajectory / setpoint
        U_init : (N, NU)   initial control guess (zeros if None)

        Returns
        -------
        U_opt : (N, NU)   optimal control sequence
        X_opt : (N+1, NX) optimal state trajectory
        info  : convergence information dict
        """
        if x_ref.ndim == 1:
            X_ref = np.tile(x_ref, (self.N + 1, 1))
        else:
            X_ref = x_ref

        U = np.zeros((self.N, NU)) if U_init is None else U_init.copy()
        X = self._rollout(x0, U)

        J_prev    = self._total_cost(X, U, X_ref)
        lambda_lm = self.lambda_lm
        info      = {"iterations": 0, "converged": False, "cost_history": [J_prev]}

        for itr in range(self.max_iter):
            try:
                K, k, dJ = self.backward_pass(X, U, X_ref, lambda_lm)
            except Exception as e:
                print(f"  [iLQR] Backward pass failed at iter {itr}: {e}")
                lambda_lm = min(lambda_lm * self.lambda_factor, self.lambda_max)
                continue

            success = False
            for alpha in self.alpha_list:
                X_new, U_new = self.forward_pass(X, U, K, k, alpha)
                J_new = self._total_cost(X_new, U_new, X_ref)
                if J_new < J_prev + self.armijo_c * alpha * dJ:
                    success = True
                    break

            if not success:
                lambda_lm = min(lambda_lm * self.lambda_factor, self.lambda_max)
                print(f"  [iLQR] Iter {itr}: line search failed, λ → {lambda_lm:.2e}")
                continue

            dJ_actual = J_prev - J_new
            X, U      = X_new, U_new
            J_prev    = J_new
            lambda_lm = max(lambda_lm / self.lambda_factor, self.lambda_min)

            info["iterations"]    += 1
            info["cost_history"].append(J_new)

            if itr % 5 == 0:
                print(f"  [iLQR] Iter {itr:3d} | J={J_new:.4e} | dJ={dJ_actual:.4e} | λ={lambda_lm:.2e}")

            if dJ_actual < self.tol:
                info["converged"] = True
                print(f"  [iLQR] Converged at iter {itr}.")
                break

        return U, X, info


# ─────────────────────────────────────────────────────────────────────────────
# LQR baseline
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
    N = 20

    # ── iLQR ──────────────────────────────────────────────────────────────────
    ctrl = ILQRController(model, Q, R, N, dt=0.02)
    x0   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # start at origin, stationary
    x_ref = np.array([2.0, 1.0, 0.0, 0.0, 0.0])  # target: (2, 1) m

    print("Running iLQR...")
    U_opt, X_opt, info = ctrl.solve(x0, x_ref)
    print(f"  Converged: {info['converged']} | Iterations: {info['iterations']}")
    print(f"  Final state: {X_opt[-1]}")

    # ── LQR ───────────────────────────────────────────────────────────────────
    lqr   = LQRController(model, Q, R)
    u_op  = cruise_trim(v_target=0.0)
    lqr.linearise(x0, u_op)
    u_lqr = lqr.control(x0 + np.random.randn(NX) * 0.05, x_ref)
    print(f"LQR control: {u_lqr}")
    print("Controller self-test complete.")
