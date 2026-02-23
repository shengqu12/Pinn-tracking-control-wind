"""
PINN Dynamics Model (TurtleBot)
================================
Architecture:
  VelocityNet     — predicts linear velocity residual  [dv]
  AngularRateNet  — predicts angular velocity residual [domega]
  PINNDynamics    — combines both, provides predict() and get_jacobian()

Input: historical state-control pairs
    H_t = [x_t, u_t, x_{t-1}, u_{t-1}, x_{t-2}, u_{t-2}]
    dim  = 3 * (5 + 2) = 21

State layout (5-dim): [px, py, theta, v, omega]
Control layout (2-dim): [a_lin, alpha_ang]

Reference: Cao et al. (2025), Section 3.2 — adapted for differential drive
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Network hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
STATE_DIM   = 5    # |x| = [px, py, theta, v, omega]
CONTROL_DIM = 2    # |u| = [a_lin, alpha_ang]
HISTORY     = 3    # number of (x, u) pairs in the history window
INPUT_DIM   = HISTORY * (STATE_DIM + CONTROL_DIM)   # 21
HIDDEN_DIM  = 32
N_LAYERS    = 4


# ─────────────────────────────────────────────────────────────────────────────
# Shared MLP builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(in_dim: int, out_dim: int, hidden: int = HIDDEN_DIM, layers: int = N_LAYERS) -> nn.Sequential:
    """Build a fully-connected network with ReLU activations."""
    blocks = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers - 2):
        blocks += [nn.Linear(hidden, hidden), nn.ReLU()]
    blocks.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*blocks)


# ─────────────────────────────────────────────────────────────────────────────
# VelocityNet
# ─────────────────────────────────────────────────────────────────────────────

class VelocityNet(nn.Module):
    """
    Predicts linear velocity residual  [dv]  (friction-induced correction).

    Input  : H_t  (B, INPUT_DIM) — flattened history of (state, control) pairs
    Output : dv   (B, 1)         — predicted linear velocity correction [m/s]
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = _build_mlp(input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H : (B, INPUT_DIM) history feature vector

        Returns
        -------
        dv : (B, 1) linear velocity residual
        """
        return self.net(H)


# ─────────────────────────────────────────────────────────────────────────────
# AngularRateNet
# ─────────────────────────────────────────────────────────────────────────────

class AngularRateNet(nn.Module):
    """
    Predicts angular velocity residual  [domega]  (friction-induced correction).

    Input  : H_t  (B, INPUT_DIM)
    Output : domega (B, 1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = _build_mlp(input_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.net(H)


# ─────────────────────────────────────────────────────────────────────────────
# PINNDynamics
# ─────────────────────────────────────────────────────────────────────────────

class PINNDynamics(nn.Module):
    """
    Combined PINN dynamics model for differential drive robot.

    Given the current (and recent) state-control history, predicts the full
    next state.  Position kinematics are integrated analytically; velocity
    residuals are predicted by the sub-networks (capturing unknown friction).

    Usage
    -----
    model = PINNDynamics()
    x_next = model(x_curr, u_curr,
                   x_prev=x_prev, u_prev=u_prev,
                   x_prev2=x_prev2, u_prev2=u_prev2,
                   dt=0.02)
    """

    def __init__(self, input_dim: int = INPUT_DIM, dt: float = 0.02):
        super().__init__()
        self.velocity_net     = VelocityNet(input_dim)
        self.angular_rate_net = AngularRateNet(input_dim)
        self.dt = dt

    # ── internal helper: build history feature vector ─────────────────────────
    @staticmethod
    def _build_history(
        x_t:  torch.Tensor, u_t:  torch.Tensor,
        x_t1: torch.Tensor, u_t1: torch.Tensor,
        x_t2: torch.Tensor, u_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate 3-step history into a single feature vector (B, 21)."""
        return torch.cat([x_t, u_t, x_t1, u_t1, x_t2, u_t2], dim=-1)

    # ── forward pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        x_curr: torch.Tensor,
        u_curr: torch.Tensor,
        x_prev:  Optional[torch.Tensor] = None,
        u_prev:  Optional[torch.Tensor] = None,
        x_prev2: Optional[torch.Tensor] = None,
        u_prev2: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Predict next state using:
          - Nonlinear kinematic integration (exact) for [px, py, theta]
          - Network residuals for [v, omega]

        Parameters
        ----------
        x_curr  : (B, 5) current state  [px, py, theta, v, omega]
        u_curr  : (B, 2) current control [a_lin, alpha_ang]
        x_prev  : (B, 5) state one step ago  (zeros if None)
        u_prev  : (B, 2) control one step ago
        x_prev2 : (B, 5) state two steps ago
        u_prev2 : (B, 2) control two steps ago
        dt      : integration step [s]; uses self.dt if None

        Returns
        -------
        x_next  : (B, 5) predicted next state
        """
        if dt is None:
            dt = self.dt
        B      = x_curr.shape[0]
        device = x_curr.device
        dtype  = x_curr.dtype

        def _zeros_xu(dim):
            return torch.zeros(B, dim, device=device, dtype=dtype)

        if x_prev  is None: x_prev  = _zeros_xu(STATE_DIM)
        if u_prev  is None: u_prev  = _zeros_xu(CONTROL_DIM)
        if x_prev2 is None: x_prev2 = _zeros_xu(STATE_DIM)
        if u_prev2 is None: u_prev2 = _zeros_xu(CONTROL_DIM)

        H = self._build_history(x_curr, u_curr, x_prev, u_prev, x_prev2, u_prev2)

        # ── velocity residuals from networks ──────────────────────────────────
        dv     = self.velocity_net(H)       # (B, 1)
        domega = self.angular_rate_net(H)   # (B, 1)

        # ── unpack current state ──────────────────────────────────────────────
        px    = x_curr[:, 0:1]
        py    = x_curr[:, 1:2]
        theta = x_curr[:, 2:3]
        v     = x_curr[:, 3:4]
        omega = x_curr[:, 4:5]

        # ── nonlinear kinematic integration ───────────────────────────────────
        px_next    = px    + dt * v     * torch.cos(theta)
        py_next    = py    + dt * v     * torch.sin(theta)
        theta_next = theta + dt * omega

        # ── velocity dynamics (network corrects for friction) ─────────────────
        v_next     = v     + dv
        omega_next = omega + domega

        x_next = torch.cat([px_next, py_next, theta_next, v_next, omega_next], dim=-1)
        return x_next

    # ── predict convenience wrapper ───────────────────────────────────────────
    def predict(
        self,
        H_t: torch.Tensor,
        dt:  float = None,
    ) -> torch.Tensor:
        """
        Predict next state from a pre-built history tensor H_t (B, 21).

        Parameters
        ----------
        H_t : (B, 21) history feature vector
        dt  : integration step [s]

        Returns
        -------
        x_next : (B, 5)
        """
        if dt is None:
            dt = self.dt

        su = STATE_DIM + CONTROL_DIM   # 7
        x_t   = H_t[:, 0:STATE_DIM]
        u_t   = H_t[:, STATE_DIM:su]
        x_t1  = H_t[:, su:su+STATE_DIM]
        u_t1  = H_t[:, su+STATE_DIM:2*su]
        x_t2  = H_t[:, 2*su:2*su+STATE_DIM]
        u_t2  = H_t[:, 2*su+STATE_DIM:3*su]

        return self.forward(x_t, u_t, x_t1, u_t1, x_t2, u_t2, dt=dt)

    # ── Jacobians for LQR linearization ──────────────────────────────────────
    def get_jacobian(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute linearised system matrices  fx = ∂f/∂x,  fu = ∂f/∂u
        at the operating point (x, u) using torch.autograd.functional.jacobian.

        Parameters
        ----------
        x : (5,) or (1, 5) operating-point state
        u : (2,) or (1, 2) operating-point control

        Returns
        -------
        fx : (5, 5) numpy array  — state Jacobian
        fu : (5, 2) numpy array  — control Jacobian
        """
        x_ = x.reshape(1, STATE_DIM).double().requires_grad_(True)
        u_ = u.reshape(1, CONTROL_DIM).double().requires_grad_(True)

        def f_x(x_in):
            return self.forward(x_in.double(), u_.double()).squeeze(0)

        def f_u(u_in):
            return self.forward(x_.double(), u_in.double()).squeeze(0)

        fx = torch.autograd.functional.jacobian(f_x, x_).squeeze()   # (5, 5)
        fu = torch.autograd.functional.jacobian(f_u, u_).squeeze()   # (5, 2)

        return fx.detach().numpy(), fu.detach().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    model = PINNDynamics()
    model = model.double()

    B = 8
    x = torch.randn(B, STATE_DIM, dtype=torch.float64)
    u = torch.randn(B, CONTROL_DIM, dtype=torch.float64)

    x_next = model(x, u)
    print(f"Forward pass: x.shape={x.shape} → x_next.shape={x_next.shape}")
    assert x_next.shape == (B, STATE_DIM), "Output shape mismatch!"

    # Jacobian at a single point
    fx, fu = model.get_jacobian(x[0], u[0])
    print(f"Jacobian fx.shape={fx.shape}, fu.shape={fu.shape}")
    assert fx.shape == (STATE_DIM, STATE_DIM)
    assert fu.shape == (STATE_DIM, CONTROL_DIM)

    print("All PINNDynamics tests passed.")
