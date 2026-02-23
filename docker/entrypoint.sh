#!/bin/bash
# entrypoint.sh — 컨테이너 시작 시 자동으로 Gazebo + VNC 실행
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   TurtleBot3 PINN-LQR   ·  Gazebo Simulation         ║"
echo "║   표면 마찰 환경에서의 궤적 추적 시뮬레이션         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. 가상 디스플레이(Xvfb) ──────────────────────────────────────────────────
echo "[1/4] 가상 디스플레이 시작 (Xvfb :99)..."
rm -f /tmp/.X99-lock 2>/dev/null || true
Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
sleep 2

# ── 2. VNC 서버 ───────────────────────────────────────────────────────────────
echo "[2/4] VNC 서버 시작 (포트 5900, 비밀번호 없음)..."
x11vnc -display :99 -nopw -listen 0.0.0.0 -xkb -forever -quiet &
sleep 1

# ── 3. noVNC 웹 서버 ──────────────────────────────────────────────────────────
echo "[3/4] noVNC 웹 서버 시작 (포트 6080)..."
# noVNC web 파일 위치 자동 탐색
NOVNC_WEB=""
for candidate in /usr/share/novnc /usr/share/noVNC; do
    if [ -d "$candidate" ]; then
        NOVNC_WEB="$candidate"
        break
    fi
done
if [ -z "$NOVNC_WEB" ]; then
    echo "  경고: noVNC 파일을 찾지 못했습니다. websockify만 실행합니다."
    websockify 6080 localhost:5900 &
else
    websockify --web "$NOVNC_WEB" 6080 localhost:5900 &
fi
sleep 1

echo ""
echo "┌──────────────────────────────────────────────────────┐"
echo "│  브라우저에서 아래 주소로 접속하세요:               │"
echo "│                                                      │"
echo "│  ➜  http://localhost:6080/vnc.html                  │"
echo "│     [Connect] 클릭 → Gazebo 창이 표시됩니다         │"
echo "│                                                      │"
echo "│  VNC 클라이언트: localhost:5900  (비밀번호 없음)    │"
echo "└──────────────────────────────────────────────────────┘"
echo ""

# ── 4. ROS 2 환경 및 Gazebo 실행 ─────────────────────────────────────────────
echo "[4/4] ROS 2 환경 로드 및 Gazebo 실행..."

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

export TURTLEBOT3_MODEL=burger
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=softpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# TurtleBot3 Gazebo 모델 경로 설정
TB3_SHARE=$(ros2 pkg prefix turtlebot3_gazebo 2>/dev/null || echo "")/share/turtlebot3_gazebo
if [ -d "${TB3_SHARE}/models" ]; then
    export GAZEBO_MODEL_PATH="${TB3_SHARE}/models${GAZEBO_MODEL_PATH:+:${GAZEBO_MODEL_PATH}}"
    echo "  GAZEBO_MODEL_PATH: ${GAZEBO_MODEL_PATH}"
fi

# 설치된 world 파일 경로
WORLD_FILE=/ros2_ws/install/turtlebot_sim/share/turtlebot_sim/worlds/wind_template.world
if [ ! -f "$WORLD_FILE" ]; then
    echo "  경고: 설치된 world 파일을 찾을 수 없습니다. 원본 경로를 사용합니다."
    WORLD_FILE=/workspace/simulation/gazebo_env/worlds/wind_template.world
fi

echo ""
echo "  시뮬레이션 파라미터:"
echo "    - 마찰 계수 (friction): 0.3"
echo "    - TurtleBot3 모델: burger"
echo "    - World: $(basename $WORLD_FILE)"
echo ""
echo "  Gazebo 창이 뜨는 데 15-30초 정도 걸릴 수 있습니다..."
echo ""

exec ros2 launch turtlebot_sim uav_sim.launch.py \
    friction:=0.3 \
    record_data:=false \
    world_file:="${WORLD_FILE}"
