#!/usr/bin/env python3
"""
run_demo.py — End-to-End Demo
==============================
One-command pipeline that:
  1. Generates synthetic TurtleBot training data
  2. Trains the PINN dynamics model  (quick: ~30 epochs × 2 sub-nets)
  3. Runs the 3-method friction generalisation experiment
  4. Saves all plots to  results/

Usage
-----
# Quick smoke-test (2 iterations × 15 epochs, ~1 min on CPU)
python run_demo.py

# Full run (saves richer results)
python run_demo.py --epochs 50 --max_iters 5 --friction all

# Use a pre-trained checkpoint (skip training)
python run_demo.py --model_path checkpoints/best_model.pt --skip_train
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

# ── make sure project root is on the path ────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from models.pinn_dynamics          import PINNDynamics
from training.train_pinn           import TurtlebotDynamicsDataset, sequential_training
from experiments.exp1_generalization import (
    run_experiment_single, run_experiment_all,
    ALL_FRICTION, TRAIN_FRICTION,
)
from evaluation.plot_results import plot_training_history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TurtleBot PINN demo")
    p.add_argument("--epochs",      type=int,   default=15,
                   help="Epochs per sequential sub-step (default 15 → fast demo)")
    p.add_argument("--max_iters",   type=int,   default=2,
                   help="Number of A/B alternating iterations")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--n_steps",     type=int,   default=2000,
                   help="Synthetic trajectory length per friction value")
    p.add_argument("--model_path",  type=str,   default="checkpoints/best_model.pt",
                   help="Path to save/load PINN checkpoint")
    p.add_argument("--skip_train",  action="store_true",
                   help="Skip training; load model_path directly")
    p.add_argument("--friction",    type=str,   default="all",
                   help="Friction for experiment: 'all' or a float, e.g. 0.3")
    p.add_argument("--save_results",action="store_true", default=True)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Synthetic data generation
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(friction_values, n_steps, seed):
    print("\n" + "="*60)
    print("  STEP 1 — Generating synthetic training data")
    print("="*60)
    datasets = []
    for mu in friction_values:
        print(f"  mu_lin={mu:.2f} ...", end=" ", flush=True)
        ds = TurtlebotDynamicsDataset.generate_synthetic(
            n_steps=n_steps, mu_lin=mu, mu_ang=mu * 0.5, seed=seed
        )
        datasets.append(ds)
        print(f"{len(ds)} samples")
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train PINN
# ─────────────────────────────────────────────────────────────────────────────

def train(datasets, args):
    print("\n" + "="*60)
    print("  STEP 2 — Training PINN dynamics model")
    print("="*60)

    from torch.utils.data import ConcatDataset

    full_ds = ConcatDataset(datasets)
    n_val   = max(1, int(len(full_ds) * 0.15))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {n_train} | Val: {n_val}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = PINNDynamics().double().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    config = {
        "epochs_per_step": args.epochs,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "max_iterations":  args.max_iters,
        "tol":             1e-5,
        "save_path":       os.path.dirname(args.model_path) or "checkpoints",
        "friction_values": TRAIN_FRICTION,
        "patience":        args.epochs,   # disable early stopping for demo
    }

    t0 = time.time()
    history = sequential_training(model, train_ds, val_ds, config)
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f} s")

    # save history
    os.makedirs("results", exist_ok=True)
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, save_path="results/training_loss")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(pinn, args):
    print("\n" + "="*60)
    print("  STEP 3 — Friction generalisation experiment")
    print("="*60)

    if args.friction.lower() == "all":
        run_experiment_all(
            pinn,
            friction_list=ALL_FRICTION,
            save_results=args.save_results,
        )
    else:
        mu = float(args.friction)
        run_experiment_single(mu, pinn, save_results=args.save_results)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "="*60)
    print("  TurtleBot PINN-LQR — End-to-End Demo")
    print("="*60)
    print(f"  epochs/step={args.epochs}  iters={args.max_iters}  "
          f"lr={args.lr}  friction={args.friction}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results",     exist_ok=True)

    # ── build / load model ────────────────────────────────────────────────────
    pinn = PINNDynamics().double()

    if args.skip_train and os.path.exists(args.model_path):
        pinn.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        print(f"\n  Loaded checkpoint from {args.model_path}")
    else:
        if args.skip_train:
            print(f"\n  Warning: --skip_train set but {args.model_path} not found — training anyway.")

        datasets = build_datasets(TRAIN_FRICTION, args.n_steps, args.seed)
        pinn     = train(datasets, args)

    pinn.eval()

    # ── run experiment ────────────────────────────────────────────────────────
    run_experiment(pinn, args)

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DONE — results saved to  results/")
    print("="*60)
    result_files = sorted(f for f in os.listdir("results")
                          if f.endswith(".png") or f.endswith(".json"))
    for f in result_files:
        print(f"    results/{f}")


if __name__ == "__main__":
    main()
