#!/usr/bin/env python3
"""
ROS 2 Data Collector Node — TurtleBot3
========================================
Subscribes to TurtleBot3 odometry and cmd_vel, then saves training data
in the format expected by TurtlebotDynamicsDataset (training/train_pinn.py).

Subscribed Topics
-----------------
/odom       (nav_msgs/Odometry)       — 2D position + velocity
/cmd_vel    (geometry_msgs/Twist)     — linear and angular velocity commands

Published Data Format
---------------------
State vector  x  = [px, py, theta, v, omega]    (5,)
Control vector u  = [a_lin, alpha_ang]            (2,)

  a_lin and alpha_ang are derived by numerical differentiation of cmd_vel.

Saved as .npy files in:
  data/training/mu{friction:.2f}_{timestamp}_states.npy
  data/training/mu{friction:.2f}_{timestamp}_controls.npy

Dependencies
------------
  ros-humble-geometry-msgs
  ros-humble-nav-msgs
  pip install numpy
"""

import os
import time
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ── State / control dimensions ────────────────────────────────────────────────
STATE_DIM   = 5   # [px, py, theta, v, omega]
CONTROL_DIM = 2   # [a_lin, alpha_ang]


def _quaternion_to_yaw(qx, qy, qz, qw) -> float:
    """Extract yaw (heading) from a quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class DataCollectorNode(Node):
    """
    Collects TurtleBot3 state-control data from Gazebo simulation.
    Saves data as .npy files for PINN training.
    """

    def __init__(self):
        super().__init__("data_collector")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("friction",       0.2)
        self.declare_parameter("sample_rate_hz", 50)
        self.declare_parameter("output_dir",     "data/training")
        self.declare_parameter("max_samples",    5000)

        self.friction    = self.get_parameter("friction").value
        self.sample_rate = self.get_parameter("sample_rate_hz").value
        self.output_dir  = self.get_parameter("output_dir").value
        self.max_samples = self.get_parameter("max_samples").value

        os.makedirs(self.output_dir, exist_ok=True)

        # ── State / control buffers ───────────────────────────────────────────
        self._states:   list = []
        self._controls: list = []
        self._lock = threading.Lock()

        # Latest sensor readings
        self._latest_x: np.ndarray = np.zeros(STATE_DIM)
        self._latest_u: np.ndarray = np.zeros(CONTROL_DIM)

        # For numerical differentiation of cmd_vel → accelerations
        self._prev_v:     float = 0.0
        self._prev_omega: float = 0.0
        self._prev_t:     float = self.get_clock().now().nanoseconds * 1e-9

        # ── QoS ──────────────────────────────────────────────────────────────
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        self._sub_odom = self.create_subscription(
            Odometry, "/odom",
            self._odom_callback, qos_sensor
        )
        self._sub_cmd = self.create_subscription(
            Twist, "/cmd_vel",
            self._cmd_callback, qos_cmd
        )

        # ── Sampling timer ────────────────────────────────────────────────────
        period = 1.0 / self.sample_rate
        self._timer = self.create_timer(period, self._sample_callback)

        self.get_logger().info(
            f"DataCollector started | friction={self.friction:.2f} | "
            f"rate={self.sample_rate} Hz | save→{self.output_dir}"
        )

    # ── Callback: odometry ────────────────────────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        t = msg.twist.twist.linear

        yaw = _quaternion_to_yaw(q.x, q.y, q.z, q.w)

        # TurtleBot3 reports body-frame velocity in odom.twist
        v_body = t.x      # forward linear velocity [m/s]
        omega  = msg.twist.twist.angular.z  # yaw rate [rad/s]

        with self._lock:
            self._latest_x[0] = p.x       # px [m]
            self._latest_x[1] = p.y       # py [m]
            self._latest_x[2] = yaw       # theta [rad]
            self._latest_x[3] = v_body    # v [m/s]
            self._latest_x[4] = omega     # omega [rad/s]

    # ── Callback: cmd_vel → derive acceleration commands ─────────────────────

    def _cmd_callback(self, msg: Twist):
        v_cmd     = msg.linear.x
        omega_cmd = msg.angular.z
        now       = self.get_clock().now().nanoseconds * 1e-9

        with self._lock:
            dt = now - self._prev_t
            if dt > 0.001:
                # Numerical derivative: approximate acceleration commands
                a_lin     = (v_cmd     - self._prev_v)     / dt
                alpha_ang = (omega_cmd - self._prev_omega) / dt
            else:
                a_lin, alpha_ang = 0.0, 0.0

            self._latest_u[0] = a_lin
            self._latest_u[1] = alpha_ang

            self._prev_v     = v_cmd
            self._prev_omega = omega_cmd
            self._prev_t     = now

    # ── Sampling timer callback ───────────────────────────────────────────────

    def _sample_callback(self):
        with self._lock:
            x_copy = self._latest_x.copy()
            u_copy = self._latest_u.copy()
            n      = len(self._states)

        if n >= self.max_samples:
            self._save_and_reset()
            return

        with self._lock:
            self._states.append(x_copy)
            self._controls.append(u_copy)

        if len(self._states) % 500 == 0:
            self.get_logger().info(f"Collected {len(self._states)} samples")

    # ── Save to disk ──────────────────────────────────────────────────────────

    def _save_and_reset(self):
        with self._lock:
            if not self._states:
                return
            states   = np.array(self._states,   dtype=np.float64)
            controls = np.array(self._controls, dtype=np.float64)
            self._states.clear()
            self._controls.clear()

        ts  = int(time.time())
        mu  = self.friction
        sf  = os.path.join(self.output_dir, f"states_mu{mu:.2f}_{ts}.npy")
        cf  = os.path.join(self.output_dir, f"controls_mu{mu:.2f}_{ts}.npy")

        np.save(sf, states)
        np.save(cf, controls)
        self.get_logger().info(
            f"Saved {len(states)} samples → {sf}"
        )

    def destroy_node(self):
        """Save remaining data before shutdown."""
        self._save_and_reset()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
