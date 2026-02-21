"""
Evaluation Metrics (TurtleBot)
================================
ATE  — Average Tracking Error    [m]
MTE  — Maximum Tracking Error    [m]
AVE  — Average Velocity Error    [m/s]
MVE  — Maximum Velocity Error    [m/s]

State layout (5-dim): [px, py, theta, v, omega]

Reference: Cao et al. (2025), Table 5 / Table 6 format
"""

import numpy as np
from typing import Dict


def compute_metrics(
    states:    np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all trajectory tracking metrics for a 2D ground robot.

    Parameters
    ----------
    states    : (T, 5) actual state trajectory   [px, py, theta, v, omega]
    reference : (T, 5) reference trajectory

    Returns
    -------
    metrics : dict with keys 'ATE', 'MTE', 'AVE', 'MVE'
    """
    pos_err = np.linalg.norm(states[:, :2] - reference[:, :2], axis=1)   # (T,)  2D position
    vel_err = np.abs(states[:, 3] - reference[:, 3])                      # (T,)  linear speed

    return {
        "ATE": float(np.mean(pos_err)),   # Average Tracking Error [m]
        "MTE": float(np.max(pos_err)),    # Maximum Tracking Error [m]
        "AVE": float(np.mean(vel_err)),   # Average Velocity Error [m/s]
        "MVE": float(np.max(vel_err)),    # Maximum Velocity Error [m/s]
    }


def compute_lateral_error(
    states:    np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Lateral (cross-track) error — 2D Euclidean distance from reference path.

    Parameters
    ----------
    states    : (T, 5)
    reference : (T, 5)

    Returns
    -------
    lateral_err : (T,) array [m]
    """
    return np.linalg.norm(states[:, :2] - reference[:, :2], axis=1)


def compute_heading_error(
    states:    np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Heading (theta) angle error wrapped to [-pi, pi].

    Parameters
    ----------
    states    : (T, 5)
    reference : (T, 5)

    Returns
    -------
    heading_err : (T,) array [rad]
    """
    raw = states[:, 2] - reference[:, 2]   # theta difference
    return (raw + np.pi) % (2 * np.pi) - np.pi


def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """
    Print a Markdown-formatted comparison table.

    Parameters
    ----------
    results : { method_name: {ATE, MTE, AVE, MVE} }
    """
    header  = "| Method         | ATE [m] | MTE [m] | AVE [m/s] | MVE [m/s] |"
    divider = "|----------------|---------|---------|-----------|-----------|"
    print(header)
    print(divider)
    for method, m in results.items():
        print(
            f"| {method:<14s} | {m['ATE']:7.4f} | {m['MTE']:7.4f} | "
            f"{m['AVE']:9.4f} | {m['MVE']:9.4f} |"
        )
