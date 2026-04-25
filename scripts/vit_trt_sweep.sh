#!/usr/bin/env bash
# ============================================================
#  vit_trt_sweep.sh  —  SelectiveExitViT TRT Threshold Sweep (Orin)
#
#  사용법:
#    bash scripts/vit_trt_sweep.sh <exit_blocks...> <N>
#
#  예시:
#    bash scripts/vit_trt_sweep.sh 8 12 10          # 2-exit, N=10회
#    bash scripts/vit_trt_sweep.sh 6 9 12 10        # 3-exit, N=10회
#    bash scripts/vit_trt_sweep.sh 8 12 5 --latency-only
#
#  백그라운드:
#    nohup bash scripts/vit_trt_sweep.sh 8 12 10 > vit_trt_2exit_sweep.log 2>&1 &
#    nohup bash scripts/vit_trt_sweep.sh 6 9 12 10 > vit_trt_3exit_sweep.log 2>&1 &
#
#  환경 변수:
#    N_SAMPLES=1000       샘플 수
#    DATA_ROOT=<path>     ImageNet val 경로 (없으면 latency-only 자동 전환)
#    THRESHOLDS="..."     threshold 목록 (기본: 0.1~0.99)
#    WARMUP=20            warmup 샘플 수
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 인자 파싱: 마지막 숫자 인자를 N으로, 나머지를 exit_blocks로 ──
ARGS=("$@")
N_ARGS=${#ARGS[@]}

if [[ $N_ARGS -lt 2 ]]; then
    echo "사용법: bash scripts/vit_trt_sweep.sh <exit_blocks...> <N> [추가옵션]"
    echo "  예) bash scripts/vit_trt_sweep.sh 8 12 10"
    echo "  예) bash scripts/vit_trt_sweep.sh 6 9 12 10"
    exit 1
fi

# 숫자인 인자 vs 옵션(--로 시작) 분리
POSITIONAL=()
EXTRA_OPTS=()
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == --* ]]; then
        EXTRA_OPTS+=("$arg")
    else
        POSITIONAL+=("$arg")
    fi
done

# POSITIONAL의 마지막이 N, 나머지가 exit_blocks
N_POS=${#POSITIONAL[@]}
N="${POSITIONAL[$((N_POS-1))]}"
EXIT_BLOCKS=("${POSITIONAL[@]:0:$((N_POS-1))}")

echo "============================================"
echo "  ViT TRT Threshold Sweep  (Jetson AGX Orin)"
echo "  Exit blocks : ${EXIT_BLOCKS[*]}"
echo "  N           : $N"
echo "  N_SAMPLES   : ${N_SAMPLES:-1000}"
echo "============================================"

N_EXIT=${#EXIT_BLOCKS[@]}

cd "$SRC_DIR"

EXTRA_ARGS=()
[[ -n "${DATA_ROOT:-}"   ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "${THRESHOLDS:-}"  ]] && EXTRA_ARGS+=(--thresholds $THRESHOLDS)
[[ -n "${WARMUP:-}"      ]] && EXTRA_ARGS+=(--warmup "$WARMUP")

python benchmark/run_vit_trt_sweep.py \
    --exit-blocks "${EXIT_BLOCKS[@]}" \
    --n "$N" \
    --num-samples "${N_SAMPLES:-1000}" \
    "${EXTRA_ARGS[@]}" \
    "${EXTRA_OPTS[@]}"
