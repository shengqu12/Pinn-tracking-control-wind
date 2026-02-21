"""
PINN Loss Functions (TurtleBot)
================================
Physics-informed loss terms for differential drive robot dynamics learning.

State layout (batch_size × 5):
  [:, 0:2]  → position  [px, py]
  [:, 2]    → heading   [theta]
  [:, 3]    → linear velocity [v]
  [:, 4]    → angular velocity [omega]

Control layout (batch_size × 2):
  [:, 0]   → linear acceleration [a_lin]  (m/s²)
  [:, 1]   → angular acceleration [alpha_ang]  (rad/s²)

Disturbance: surface friction
  mu_lin   : linear friction coefficient
  mu_ang   : angular friction coefficient
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# L_data — supervised data-fit loss
# ─────────────────────────────────────────────────────────────────────────────

def L_data(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """
    Mean-squared-error between predicted and true next states.

    Parameters
    ----------
    x_pred : (B, 5) predicted next state
    x_true : (B, 5) ground-truth next state

    Returns
    -------
    loss : scalar tensor
    """
    return nn.functional.mse_loss(x_pred, x_true)


# ─────────────────────────────────────────────────────────────────────────────
# L_physics_velocity — linear velocity residual
# ─────────────────────────────────────────────────────────────────────────────

def L_physics_velocity(
    model:           nn.Module,
    x_batch:         torch.Tensor,
    u_batch:         torch.Tensor,
    friction_params: Dict[str, float],
) -> torch.Tensor:
    """
    Physics residual for linear velocity dynamics.

    Governing equation:
        v_dot = a_lin - mu_lin * v * |v|

    Residual = predicted dv  −  physics-based dv

    Physical meaning: ensures the network's velocity predictions are
    consistent with the friction-drag model.  mu_lin captures the known
    (training-time) surface friction; the network learns any remaining
    discrepancy.

    Parameters
    ----------
    model           : PINNDynamics
    x_batch         : (B, 5) current states with requires_grad=True
    u_batch         : (B, 2) control inputs
    friction_params : dict with key 'mu_lin' [1/m]

    Returns
    -------
    loss : scalar residual loss
    """
    mu_lin = friction_params.get("mu_lin", 0.2)
    dtype  = x_batch.dtype
    device = x_batch.device

    v     = x_batch[:, 3]          # (B,) linear velocity
    a_lin = u_batch[:, 0]          # (B,) linear acceleration command

    # Physics-based velocity increment: dv = a_lin - mu_lin * v * |v|
    dv_physics = a_lin - mu_lin * v * torch.abs(v)

    # NN-predicted velocity increment
    x_next_pred = model(x_batch.detach(), u_batch)
    dv_pred     = x_next_pred[:, 3] - x_batch[:, 3].detach()

    residual = dv_pred - dv_physics
    return (residual ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# L_physics_angular — angular velocity residual
# ─────────────────────────────────────────────────────────────────────────────

def L_physics_angular(
    model:           nn.Module,
    x_batch:         torch.Tensor,
    u_batch:         torch.Tensor,
    friction_params: Dict[str, float],
) -> torch.Tensor:
    """
    Physics residual for angular velocity dynamics.

    Governing equation:
        omega_dot = alpha_ang - mu_ang * omega * |omega|

    Residual = predicted domega  −  physics-based domega

    Physical meaning: ensures gyroscopic/rotational friction effects are
    respected — critical for accurate turning behaviour.

    Parameters
    ----------
    model           : PINNDynamics
    x_batch         : (B, 5) current states
    u_batch         : (B, 2) control inputs
    friction_params : dict with key 'mu_ang' [1/rad]

    Returns
    -------
    loss : scalar residual loss
    """
    mu_ang    = friction_params.get("mu_ang", 0.1)

    omega     = x_batch[:, 4]      # (B,) angular velocity
    alpha_ang = u_batch[:, 1]      # (B,) angular acceleration command

    # Physics-based angular increment
    domega_physics = alpha_ang - mu_ang * omega * torch.abs(omega)

    # NN-predicted angular increment
    x_next_pred  = model(x_batch.detach(), u_batch)
    domega_pred  = x_next_pred[:, 4] - x_batch[:, 4].detach()

    residual = domega_pred - domega_physics
    return (residual ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# L_initial_condition — initial condition loss
# ─────────────────────────────────────────────────────────────────────────────

def L_initial_condition(
    x_pred_t0: torch.Tensor,
    x0:        torch.Tensor,
) -> torch.Tensor:
    """
    Enforce initial conditions: predicted trajectory at t=0 must match x0.

    Parameters
    ----------
    x_pred_t0 : (B, 5) model output at the first time step
    x0        : (B, 5) known initial states

    Returns
    -------
    loss : scalar tensor
    """
    return nn.functional.mse_loss(x_pred_t0, x0)


# ─────────────────────────────────────────────────────────────────────────────
# total_loss — weighted combination
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "data":           1.0,
    "physics_vel":    0.1,
    "physics_ang":    0.1,
    "ic":             0.01,
}


def total_loss(
    model:           nn.Module,
    batch:           Dict[str, torch.Tensor],
    friction_params: Dict[str, float],
    weights:         Dict[str, float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the total PINN loss as a weighted sum of all components.

    Total loss:
        L = w_data * L_data
          + w_vel   * L_physics_velocity
          + w_ang   * L_physics_angular
          + w_ic    * L_ic

    Parameters
    ----------
    model           : PINNDynamics model (callable)
    batch           : dict with keys
                        'x_curr'  (B, 5)  current state
                        'u_curr'  (B, 2)  current control
                        'x_next'  (B, 5)  ground-truth next state
                        'x0'      (B, 5)  initial state (for IC loss)
    friction_params : {'mu_lin', 'mu_ang'} surface friction coefficients
    weights         : loss weight dict (uses DEFAULT_WEIGHTS if None)

    Returns
    -------
    loss_total : scalar tensor
    loss_dict  : dict of individual loss values (for logging)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    x_curr = batch["x_curr"]
    u_curr = batch["u_curr"]
    x_next = batch["x_next"]
    x0     = batch["x0"]

    # Forward pass
    x_pred = model(x_curr, u_curr)   # (B, 5)

    # Individual losses
    ld   = L_data(x_pred, x_next)
    lpv  = L_physics_velocity(model, x_curr, u_curr, friction_params)
    lpa  = L_physics_angular(model, x_curr, u_curr, friction_params)
    lic  = L_initial_condition(x_pred[:x0.shape[0]], x0)

    loss_total = (
        weights["data"]         * ld
        + weights["physics_vel"] * lpv
        + weights["physics_ang"] * lpa
        + weights["ic"]          * lic
    )

    loss_dict = {
        "total":        loss_total.item(),
        "data":         ld.item(),
        "physics_vel":  lpv.item(),
        "physics_ang":  lpa.item(),
        "ic":           lic.item(),
    }
    return loss_total, loss_dict
