# TurtleBot PINN Dynamics & LQR Control

Physics-Informed Neural Network (PINN) for **TurtleBot3 differential drive**
dynamics learning under surface friction disturbance, combined with a
Linear Quadratic Regulator (LQR) controller.

Disturbance model: **surface friction coefficient** (mu_lin) — analogous to
wind speed in the original Cao et al. (2025) UAV formulation.

Based on: *Cao et al. (2025) — Physics-Informed Neural Network for Vehicle Dynamics*

---

## Robot Model

**State**:    `x = [px, py, theta, v, omega]`  (5-dim)

| Component | Description | Unit |
|-----------|-------------|------|
| px, py   | 2D position in world frame | m |
| theta    | heading angle | rad |
| v        | linear velocity | m/s |
| omega    | angular velocity | rad/s |

**Control**: `u = [a_lin, alpha_ang]`  (2-dim)

| Component | Description | Unit |
|-----------|-------------|------|
| a_lin    | linear acceleration command | m/s² |
| alpha_ang | angular acceleration command | rad/s² |

**Dynamics** (with friction disturbance):
```
px_dot    = v * cos(theta)
py_dot    = v * sin(theta)
theta_dot = omega
v_dot     = a_lin    - mu_lin * v * |v|
omega_dot = alpha_ang - mu_ang * omega * |omega|
```

---

## Project Structure

```
project/
├── models/
│   ├── turtlebot_physics.py   # Differential drive dynamics (pure numpy, no ROS)
│   └── pinn_dynamics.py       # VelocityNet + AngularRateNet + PINNDynamics
├── training/
│   ├── loss_functions.py      # PINN loss (data + physics_vel + physics_ang + IC)
│   └── train_pinn.py          # Sequential training (Step A / Step B alternating)
├── controllers/
│   └── lqr_controller.py      # LQR (PINN-based) + AnalyticalLQR + OpenLoop baseline
├── simulation/
│   ├── python_sim.py          # Pure-Python 2D simulation (no Gazebo required)
│   └── gazebo_env/            # ROS 2 + Gazebo TurtleBot3 simulation package
│       ├── worlds/
│       │   └── wind_template.world   # Ground plane with configurable friction
│       ├── launch/
│       │   └── uav_sim.launch.py     # TurtleBot3 Gazebo launch file
│       ├── scripts/
│       │   └── data_collector_node.py  # /odom + /cmd_vel data logger
│       ├── package.xml                 # ROS 2 package: turtlebot_sim
│       └── CMakeLists.txt
├── evaluation/
│   ├── metrics.py             # ATE, MTE, AVE, MVE (2D position/velocity)
│   └── plot_results.py        # Publication-quality figures
├── experiments/
│   └── exp1_generalization.py # Multi-friction generalisation experiment
├── data/training/             # .npy training data (gitignored)
├── results/                   # Plots & JSON (partial gitignore)
├── requirements.txt
└── .gitignore
```

---

## Quick Start (No Gazebo Required)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify physics model

```bash
python models/turtlebot_physics.py
```

### 3. Train PINN (synthetic data)

```bash
# Generates synthetic data on-the-fly, no Gazebo needed
python training/train_pinn.py \
    --friction_values 0.1 0.2 0.3 0.4 0.5 \
    --epochs 50 \
    --lr 0.001 \
    --save_path checkpoints/
```

### 4. Run open-loop simulation

```bash
python simulation/python_sim.py
# → saves results/cruise_2d.png, results/cruise_error.png
```

### 5. Run generalisation experiment

```bash
# Single friction coefficient (quick test)
python experiments/exp1_generalization.py --friction 0.3

# Full sweep: train + val friction values
python experiments/exp1_generalization.py --friction all --save_results

# With trained model
python experiments/exp1_generalization.py \
    --friction all \
    --model_path checkpoints/best_model.pt
```

### 6. Plot results

```bash
python evaluation/plot_results.py \
    --results_json results/exp1_results.json \
    --history_json results/training_history.json
```

---

## Gazebo + ROS 2 Setup (Linux)

### Prerequisites

```bash
# ROS 2 Humble (Ubuntu 22.04)
sudo apt install ros-humble-desktop ros-humble-gazebo-ros-pkgs \
                 ros-humble-turtlebot3 ros-humble-turtlebot3-gazebo

# Set TurtleBot3 model
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
```

### Build the ROS 2 package

```bash
cp -r simulation/gazebo_env ~/ros2_ws/src/turtlebot_sim

cd ~/ros2_ws
colcon build --packages-select turtlebot_sim
source install/setup.bash
```

### Collect training data

```bash
# Launch Gazebo with friction=0.2 and start data collection
ros2 launch turtlebot_sim uav_sim.launch.py friction:=0.2

# Repeat for all training friction values
for mu in 0.1 0.2 0.3 0.4 0.5; do
  ros2 launch turtlebot_sim uav_sim.launch.py friction:=${mu}
done
```

Data is saved to `data/training/states_mu{mu:.2f}_{timestamp}.npy`.

### Train with real Gazebo data

```bash
python training/train_pinn.py \
    --friction_values 0.1 0.2 0.3 0.4 0.5 \
    --data_dir data/training/ \
    --epochs 100
```

---

## Key Methods

| Module | Description |
|--------|-------------|
| `turtlebot_physics.py` | Differential drive dynamics + friction (no ROS) |
| `loss_functions.py` | `L_data + L_physics_vel + L_physics_ang + L_IC` |
| `pinn_dynamics.py` | History-conditioned MLP, Jacobian via autograd |
| `train_pinn.py` | Alternating Step A/B training (Cao et al. eq. 15-17) |
| `lqr_controller.py` | LQR (PINN Jacobians) + AnalyticalLQR + OpenLoop |
| `exp1_generalization.py` | Open_loop / LQR_physics / PINN_LQR comparison |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ATE | Average Tracking Error [m] (2D position) |
| MTE | Maximum Tracking Error [m] |
| AVE | Average Velocity Error [m/s] (linear speed) |
| MVE | Maximum Velocity Error [m/s] |

---

## Disturbance Generalisation

| Split | Friction values (mu_lin) |
|-------|--------------------------|
| Training | 0.1, 0.2, 0.3, 0.4, 0.5 |
| Validation (unseen) | 0.15, 0.25, 0.35, 0.45, 0.55 |

The angular friction is set as `mu_ang = 0.5 * mu_lin` throughout.

---

## Notes

- **Training sequence**: Start with small physics weights (`0.01`) and increase gradually
- **Gazebo TurtleBot3**: Uses `TURTLEBOT3_MODEL=burger`; tested with ROS 2 Humble
- **Data topics**: `/odom` for state, `/cmd_vel` for control commands
