#!/bin/bash
# run_gazebo.sh ── TurtleBot3 PINN Gazebo 시뮬레이션을 Docker로 실행

set -e

echo ""
echo "══════════════════════════════════════════════════════════"
echo "   TurtleBot3 PINN-LQR   ·  Gazebo Docker 시뮬레이터"
echo "══════════════════════════════════════════════════════════"
echo ""

# ── Docker 실행 여부 확인 ──────────────────────────────────────────────────────
if ! docker info >/dev/null 2>&1; then
    echo "오류: Docker가 실행되지 않고 있습니다."
    echo "  → Docker Desktop을 시작한 후 다시 실행하세요."
    exit 1
fi

# ── docker compose 명령어 탐색 ────────────────────────────────────────────────
if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    COMPOSE="docker compose"
else
    echo "오류: docker-compose 또는 'docker compose'를 찾을 수 없습니다."
    echo "  → Docker Desktop을 설치/업데이트하세요."
    exit 1
fi

# ── 파라미터 처리 ─────────────────────────────────────────────────────────────
FRICTION=${1:-0.3}   # 기본 마찰 계수 0.3 (인자로 변경 가능: ./run_gazebo.sh 0.5)

echo "마찰 계수 (friction): $FRICTION"
echo ""

# ── 빌드 ──────────────────────────────────────────────────────────────────────
echo "── Step 1/2: Docker 이미지 빌드 ──────────────────────────────────────────"
echo "  (처음 빌드 시 10-20분 소요됩니다. 이후 재실행은 빠릅니다)"
echo ""
$COMPOSE build
echo ""

# ── 실행 ──────────────────────────────────────────────────────────────────────
echo "── Step 2/2: 시뮬레이션 컨테이너 시작 ───────────────────────────────────"
echo ""
echo "  잠시 후 브라우저에서 아래 주소를 열어주세요:"
echo ""
echo "    http://localhost:6080/vnc.html"
echo ""
echo "  [Connect] 버튼을 클릭하면 Gazebo 창이 표시됩니다."
echo ""
echo "  종료하려면: Ctrl+C"
echo ""
echo "──────────────────────────────────────────────────────────────────────────"

$COMPOSE up
