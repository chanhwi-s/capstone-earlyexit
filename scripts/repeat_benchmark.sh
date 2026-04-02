#!/usr/bin/env bash
# ============================================================
#  repeat_benchmark.sh  —  SKIP_BUILD=1 벤치마크를 N번 반복 실행
#
#  사용법:
#    bash scripts/repeat_benchmark.sh <N>
#
#  예시:
#    bash scripts/repeat_benchmark.sh 10          # 10번 반복
#    bash scripts/repeat_benchmark.sh 30          # 30번 반복
#    THRESHOLD=0.85 bash scripts/repeat_benchmark.sh 20   # threshold 지정
#
#  백그라운드 실행 (SSH 끊겨도 유지):
#    nohup bash scripts/repeat_benchmark.sh 30 > repeat_benchmark.log 2>&1 &
#    echo "PID: $!"
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 인자 확인 ────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "[ERROR] 반복 횟수(N)를 인자로 전달하세요."
    echo "  사용법: bash scripts/repeat_benchmark.sh <N>"
    echo "  예시:   bash scripts/repeat_benchmark.sh 30"
    exit 1
fi

N="$1"

# 숫자인지 검증
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "[ERROR] N은 1 이상의 정수여야 합니다. (입력값: $N)"
    exit 1
fi

# ── 환경변수 기본값 설정 ─────────────────────────────────────
THRESHOLD="${THRESHOLD:-0.80}"
N_SAMPLES="${N_SAMPLES:-1000}"

echo "================================================"
echo "  반복 벤치마크 시작"
echo "  반복 횟수  : $N"
echo "  Threshold  : $THRESHOLD"
echo "  Samples    : $N_SAMPLES"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

PASS=0
FAIL=0

for i in $(seq 1 "$N"); do
    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Run $i / $N   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────"

    if SKIP_BUILD=1 THRESHOLD="$THRESHOLD" N_SAMPLES="$N_SAMPLES" \
        bash "$SCRIPT_DIR/orin_pipeline.sh"; then
        PASS=$((PASS + 1))
        echo "  [OK] Run $i 완료"
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] Run $i 실패 (계속 진행)"
    fi
done

echo ""
echo "================================================"
echo "  반복 벤치마크 완료"
echo "  성공: $PASS / $N   실패: $FAIL / $N"
echo "  종료 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
