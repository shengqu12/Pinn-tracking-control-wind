# ── Base: ROS 2 Humble Desktop on Ubuntu 22.04 ───────────────────────────────
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Gazebo 11 (classic) + ROS 2 bridges
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    # TurtleBot3 full stack
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-turtlebot3-description \
    ros-humble-turtlebot3-bringup \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    # Build tools
    python3-colcon-common-extensions \
    python3-pip \
    # GUI / VNC / noVNC
    xvfb \
    x11vnc \
    novnc \
    websockify \
    # Software GL (needed inside Docker on Mac)
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    mesa-utils \
    # Misc
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY . /workspace/

# ── Build ROS 2 package ───────────────────────────────────────────────────────
RUN mkdir -p /ros2_ws/src && \
    cp -r /workspace/simulation/gazebo_env /ros2_ws/src/turtlebot_sim

WORKDIR /ros2_ws
RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    colcon build --packages-select turtlebot_sim 2>&1"

# ── Entrypoint ────────────────────────────────────────────────────────────────
RUN cp /workspace/docker/entrypoint.sh /entrypoint.sh && \
    chmod +x /entrypoint.sh

# ── Default environment ───────────────────────────────────────────────────────
ENV TURTLEBOT3_MODEL=burger
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=softpipe
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330

EXPOSE 6080 5900

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
