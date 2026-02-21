"""
ROS 2 Launch File — TurtleBot3 Friction Simulation
====================================================
Starts Gazebo with a configurable-friction world, spawns TurtleBot3 Burger,
and launches the data-collector node for logging PINN training data.

Usage
-----
# Set TurtleBot3 model (required for turtlebot3_gazebo)
export TURTLEBOT3_MODEL=burger

ros2 launch turtlebot_sim turtlebot_sim.launch.py friction:=0.3
ros2 launch turtlebot_sim turtlebot_sim.launch.py friction:=0.3 use_rviz:=true

Parameters
----------
friction     : float   Surface friction coefficient (mu_lin).  Default 0.2.
use_rviz     : bool    Launch RViz2 for 2D visualisation. Default false.
record_data  : bool    Launch the data-collector node.  Default true.
world_file   : str     Path to custom world SDF. Default friction_template.world.

Dependencies
------------
  - ros-humble-gazebo-ros-pkgs  (or equivalent for your distro)
  - ros-humble-turtlebot3-gazebo
  - rviz2 (optional)
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# ── Helper: resolve package paths ────────────────────────────────────────────
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORLD_FILE  = os.path.join(_PACKAGE_DIR, "worlds", "friction_template.world")


def generate_launch_description():
    # ── Declare arguments ─────────────────────────────────────────────────────
    declare_friction = DeclareLaunchArgument(
        "friction",
        default_value="0.2",
        description="Surface friction coefficient (mu_lin)",
    )
    declare_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Launch RViz2 for visualisation",
    )
    declare_record = DeclareLaunchArgument(
        "record_data",
        default_value="true",
        description="Launch the data-collector node",
    )
    declare_world = DeclareLaunchArgument(
        "world_file",
        default_value=_WORLD_FILE,
        description="Path to the Gazebo world file",
    )

    friction    = LaunchConfiguration("friction")
    use_rviz    = LaunchConfiguration("use_rviz")
    record_data = LaunchConfiguration("record_data")
    world_file  = LaunchConfiguration("world_file")

    # ── Set TurtleBot3 model env variable ─────────────────────────────────────
    set_tb3_model = SetEnvironmentVariable(
        name="TURTLEBOT3_MODEL",
        value="burger",
    )

    # ── Gazebo server ─────────────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=[
            "gazebo",
            "--verbose",
            "-s", "libgazebo_ros_factory.so",
            "-s", "libgazebo_ros_init.so",
            world_file,
        ],
        output="screen",
        additional_env={"SURFACE_FRICTION": friction},
    )

    # ── Spawn TurtleBot3 ──────────────────────────────────────────────────────
    try:
        tb3_description_dir = get_package_share_directory("turtlebot3_description")
        tb3_urdf = os.path.join(tb3_description_dir, "urdf", "turtlebot3_burger.urdf")
    except Exception:
        tb3_urdf = ""

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_turtlebot3",
        output="screen",
        arguments=[
            "-entity", "turtlebot3_burger",
            "-file", tb3_urdf,
            "-x", "0.0", "-y", "0.0", "-z", "0.01",
        ] if tb3_urdf else ["-help"],
    )

    # ── Robot State Publisher ─────────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── Data Collector Node ────────────────────────────────────────────────────
    data_collector = Node(
        package="turtlebot_sim",
        executable="data_collector_node",
        name="data_collector",
        output="screen",
        condition=IfCondition(record_data),
        parameters=[{
            "friction":         friction,
            "sample_rate_hz":   50,
            "output_dir":       os.path.join(_PACKAGE_DIR, "..", "..", "data", "training"),
            "use_sim_time":     True,
        }],
    )

    # ── RViz2 (optional) ──────────────────────────────────────────────────────
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        parameters=[{"use_sim_time": True}],
    )

    # ── Static transform (world → map → odom) ─────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_map",
        arguments=["0", "0", "0", "0", "0", "0", "world", "map"],
    )

    log_info = LogInfo(msg=["Launching TurtleBot3 simulation with friction = ", friction])

    return LaunchDescription([
        declare_friction,
        declare_rviz,
        declare_record,
        declare_world,
        set_tb3_model,
        log_info,
        gazebo,
        robot_state_publisher,
        static_tf,
        spawn_robot,
        data_collector,
        rviz2,
    ])
