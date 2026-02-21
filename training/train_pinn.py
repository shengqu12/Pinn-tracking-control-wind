"""
Sequential PINN Training (TurtleBot)
======================================
Implements the alternating (sequential) training strategy from Cao et al. (2025):
  Step A: fix AngularRateNet, train VelocityNet    (50 epochs)
  Step B: fix VelocityNet,    train AngularRateNet (50 epochs)
  Repeat for max_iterations or until L_total < tol.

The disturbance is surface friction (mu_lin, mu_ang).
Generalization is tested on friction values unseen during training.

Usage
-----
python training/train_pinn.py --friction_values 0.1 0.2 0.3 0.4 0.5 --epochs 50 --lr 0.001
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pinn_dynamics import PINNDynamics, STATE_DIM, CONTROL_DIM
from training.loss_functions import total_loss, DEFAULT_WEIGHTS


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TurtlebotDynamicsDataset(Dataset):
    """
    Dataset for TurtleBot state-control trajectories.

    Each sample is a dict:
      x_curr  (5,) — current state   [px, py, theta, v, omega]
      u_curr  (2,) — current control [a_lin, alpha_ang]
      x_next  (5,) — ground-truth next state
      x0      (5,) — initial state of the trajectory (for IC loss)
      x_prev  (5,) — state one step earlier
      u_prev  (2,) — control one step earlier
      x_prev2 (5,) — state two steps earlier
      u_prev2 (2,) — control two steps earlier
    """

    def __init__(
        self,
        states:   np.ndarray,   # (N, 5)
        controls: np.ndarray,   # (N, 2)
        dt:       float = 0.02,
    ):
        assert states.shape[0] == controls.shape[0]
        assert states.shape[1] == STATE_DIM,   f"Expected STATE_DIM={STATE_DIM}, got {states.shape[1]}"
        assert controls.shape[1] == CONTROL_DIM, f"Expected CONTROL_DIM={CONTROL_DIM}, got {controls.shape[1]}"

        N = states.shape[0]
        self.x_curr  = torch.tensor(states[2:N-1],   dtype=torch.float64)
        self.u_curr  = torch.tensor(controls[2:N-1], dtype=torch.float64)
        self.x_next  = torch.tensor(states[3:N],     dtype=torch.float64)
        self.x_prev  = torch.tensor(states[1:N-2],   dtype=torch.float64)
        self.u_prev  = torch.tensor(controls[1:N-2], dtype=torch.float64)
        self.x_prev2 = torch.tensor(states[0:N-3],   dtype=torch.float64)
        self.u_prev2 = torch.tensor(controls[0:N-3], dtype=torch.float64)
        self.x0      = torch.tensor(states[0], dtype=torch.float64).unsqueeze(0).expand(len(self.x_curr), -1)
        self.dt = dt

    def __len__(self) -> int:
        return len(self.x_curr)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_curr":  self.x_curr[idx],
            "u_curr":  self.u_curr[idx],
            "x_next":  self.x_next[idx],
            "x0":      self.x0[idx],
            "x_prev":  self.x_prev[idx],
            "u_prev":  self.u_prev[idx],
            "x_prev2": self.x_prev2[idx],
            "u_prev2": self.u_prev2[idx],
        }

    @classmethod
    def from_files(cls, state_file: str, control_file: str, dt: float = 0.02) -> "TurtlebotDynamicsDataset":
        """Load from .npy files."""
        states   = np.load(state_file)
        controls = np.load(control_file)
        return cls(states, controls, dt)

    @classmethod
    def generate_synthetic(
        cls,
        n_steps:    int   = 2000,
        mu_lin:     float = 0.2,
        mu_ang:     float = 0.1,
        dt:         float = 0.02,
        seed:       int   = 0,
    ) -> "TurtlebotDynamicsDataset":
        """
        Generate synthetic training data using the TurtleBot physics model.
        The robot follows a noisy figure-eight / random-walk trajectory.
        Useful for development without Gazebo data.
        """
        from models.turtlebot_physics import dynamics as tb_dynamics, cruise_trim
        from scipy.integrate import solve_ivp

        rng = np.random.default_rng(seed)
        friction = {"mu_lin": mu_lin, "mu_ang": mu_ang}

        states   = np.zeros((n_steps, STATE_DIM))
        controls = np.zeros((n_steps, CONTROL_DIM))

        # Start with a gentle initial motion
        states[0] = np.array([0.0, 0.0, 0.0, 0.1, 0.0])

        for i in range(n_steps - 1):
            # Smooth random accelerations (exploration policy)
            a_lin     = rng.uniform(-0.5, 0.5)
            alpha_ang = rng.uniform(-1.0, 1.0)
            u = np.array([a_lin, alpha_ang])
            controls[i] = u

            sol = solve_ivp(
                fun=lambda t, x: tb_dynamics(x, u, friction),
                t_span=(0.0, dt),
                y0=states[i],
                method="RK45",
                max_step=dt / 4,
            )
            states[i + 1] = sol.y[:, -1]

        controls[-1] = controls[-2]
        return cls(states, controls, dt)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: freeze / unfreeze sub-networks
# ─────────────────────────────────────────────────────────────────────────────

def _set_requires_grad(module: nn.Module, value: bool):
    for p in module.parameters():
        p.requires_grad = value


# ─────────────────────────────────────────────────────────────────────────────
# Sequential training
# ─────────────────────────────────────────────────────────────────────────────

def sequential_training(
    model:         PINNDynamics,
    train_dataset: TurtlebotDynamicsDataset,
    val_dataset:   TurtlebotDynamicsDataset,
    config:        Dict,
) -> Dict:
    """
    Alternating sequential training strategy (Cao et al. eq. 15–17).

    Step A: Train VelocityNet     (AngularRateNet frozen)
    Step B: Train AngularRateNet  (VelocityNet frozen)
    Repeat up to config['max_iterations'] rounds or until L_total < config['tol'].

    Parameters
    ----------
    model         : PINNDynamics (double precision)
    train_dataset : training data
    val_dataset   : validation data
    config        : {
        'epochs_per_step':  50,
        'lr':               1e-3,
        'batch_size':       64,
        'max_iterations':   10,
        'tol':              1e-5,
        'save_path':        'checkpoints/',
        'friction_values':  [0.2],    # list of (mu_lin, mu_ang) or scalar mu_lin
        'weights':          None,
        'patience':         10,
    }

    Returns
    -------
    history : dict of training metrics
    """
    device    = next(model.parameters()).device
    epochs    = config.get("epochs_per_step",  50)
    lr        = config.get("lr",               1e-3)
    bs        = config.get("batch_size",        64)
    max_iters = config.get("max_iterations",    10)
    tol       = config.get("tol",              1e-5)
    save_dir  = config.get("save_path",    "checkpoints")
    fric_vals = config.get("friction_values",  [0.2])
    weights   = config.get("weights",           None)
    patience  = config.get("patience",           10)

    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=False)

    history = {
        "train_total": [], "train_data": [], "train_physics_vel": [],
        "train_physics_ang": [], "train_ic": [],
        "val_total": [], "val_data": [],
    }

    best_val_loss = float("inf")
    no_improve    = 0

    for iteration in range(max_iters):
        for step_name, freeze_net, train_net in [
            ("A (VelocityNet)",     model.angular_rate_net, model.velocity_net),
            ("B (AngularRateNet)",  model.velocity_net,     model.angular_rate_net),
        ]:
            print(f"\n[Iter {iteration+1}/{max_iters}] Step {step_name}")

            _set_requires_grad(freeze_net, False)
            _set_requires_grad(train_net,  True)

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=lr
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, min_lr=1e-6
            )

            for epoch in range(1, epochs + 1):
                model.train()
                epoch_losses = {k: 0.0 for k in ["total", "data", "physics_vel", "physics_ang", "ic"]}

                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # Sample a random friction value from training set
                    mu_lin = float(np.random.choice(fric_vals))
                    friction_params = {"mu_lin": mu_lin, "mu_ang": mu_lin * 0.5}

                    optimizer.zero_grad()
                    loss, loss_dict = total_loss(model, batch, friction_params, weights)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    for k in epoch_losses:
                        epoch_losses[k] += loss_dict.get(k, 0.0)

                n = len(train_loader)
                for k in epoch_losses:
                    epoch_losses[k] /= n

                # ── validation ────────────────────────────────────────────────
                model.eval()
                val_loss_total = 0.0
                val_loss_data  = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        mu_mean = float(np.mean(fric_vals))
                        fp = {"mu_lin": mu_mean, "mu_ang": mu_mean * 0.5}
                        _, vld = total_loss(model, batch, fp, weights)
                        val_loss_total += vld["total"]
                        val_loss_data  += vld["data"]

                val_loss_total /= len(val_loader)
                val_loss_data  /= len(val_loader)
                scheduler.step(val_loss_total)

                # ── logging ───────────────────────────────────────────────────
                history["train_total"].append(epoch_losses["total"])
                history["train_data"].append(epoch_losses["data"])
                history["train_physics_vel"].append(epoch_losses["physics_vel"])
                history["train_physics_ang"].append(epoch_losses["physics_ang"])
                history["train_ic"].append(epoch_losses["ic"])
                history["val_total"].append(val_loss_total)
                history["val_data"].append(val_loss_data)

                if epoch % 10 == 0 or epoch == 1:
                    print(
                        f"  Epoch {epoch:3d}/{epochs} | "
                        f"train_total={epoch_losses['total']:.4e} | "
                        f"data={epoch_losses['data']:.4e} | "
                        f"phys_v={epoch_losses['physics_vel']:.4e} | "
                        f"phys_a={epoch_losses['physics_ang']:.4e} | "
                        f"ic={epoch_losses['ic']:.4e} | "
                        f"val={val_loss_total:.4e}"
                    )

                # ── early stopping ────────────────────────────────────────────
                if val_loss_total < best_val_loss:
                    best_val_loss = val_loss_total
                    no_improve    = 0
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                        break

        latest_total = history["train_total"][-1] if history["train_total"] else float("inf")
        print(f"\n[Iter {iteration+1}] L_total = {latest_total:.4e}  (tol={tol:.1e})")
        if latest_total < tol:
            print("Convergence achieved!")
            break

    best_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"\nLoaded best model from {best_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PINN dynamics model for TurtleBot")
    parser.add_argument("--friction_values", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Linear friction coefficients for training data")
    parser.add_argument("--epochs",       type=int,   default=50,     help="Epochs per sub-step")
    parser.add_argument("--lr",           type=float, default=1e-3,   help="Learning rate")
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--max_iters",    type=int,   default=10,     help="Max sequential iterations")
    parser.add_argument("--save_path",    type=str,   default="checkpoints/")
    parser.add_argument("--data_dir",     type=str,   default=None,
                        help="Directory with state/control .npy files; synthesizes if None")
    parser.add_argument("--val_split",    type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load or synthesise data ────────────────────────────────────────────────
    all_datasets = []
    if args.data_dir is None:
        print("No data directory provided — generating synthetic data...")
        for mu in args.friction_values:
            ds = TurtlebotDynamicsDataset.generate_synthetic(
                n_steps=3000, mu_lin=mu, mu_ang=mu * 0.5, seed=args.seed
            )
            all_datasets.append(ds)
    else:
        for mu in args.friction_values:
            sf = os.path.join(args.data_dir, f"states_mu{mu:.2f}.npy")
            cf = os.path.join(args.data_dir, f"controls_mu{mu:.2f}.npy")
            if os.path.exists(sf) and os.path.exists(cf):
                all_datasets.append(TurtlebotDynamicsDataset.from_files(sf, cf))
                print(f"  Loaded data for mu={mu:.2f}")
            else:
                print(f"  Warning: data files for mu={mu:.2f} not found, skipping")

    if not all_datasets:
        raise RuntimeError("No datasets available for training.")

    from torch.utils.data import ConcatDataset
    full_dataset = ConcatDataset(all_datasets)
    n_val   = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {n_train} samples | Val: {n_val} samples")

    # ── model ──────────────────────────────────────────────────────────────────
    model = PINNDynamics().double().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── train ──────────────────────────────────────────────────────────────────
    config = {
        "epochs_per_step":  args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "max_iterations":   args.max_iters,
        "tol":              1e-5,
        "save_path":        args.save_path,
        "friction_values":  args.friction_values,
    }
    history = sequential_training(model, train_ds, val_ds, config)

    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete. History saved to results/training_history.json")
