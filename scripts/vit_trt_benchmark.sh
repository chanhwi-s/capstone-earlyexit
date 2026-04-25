#!/usr/bin/env bash
# ============================================================
#  vit_trt_benchmark.sh  —  PlainViT vs 2-exit vs 3-exit 3-way 벤치마크 (Orin)
#
#  sweep 결과에서 최적 threshold 결정 후 실행하는 최종 벤치마크.
#
#  사용법:
#    bash scripts/vit_trt_benchmark.sh <thr_2exit> <thr_3exit> <N>
#
#  예시:
#    bash scripts/vit_trt_benchmark.sh 0.80 0.75 30
#    bash scripts/vit_trt_benchmark.sh 0.80 0.80 30
#
#  백그라운드:
#    nohup bash scripts/vit_trt_benchmark.sh 0.80 0.75 30 > vit_trt_benchmark.log 2>&1 &
#
#  환경 변수:
#    N_SAMPLES=1000
#    DATA_ROOT=<path>
#    WARMUP=20
#    SKIP_2EXIT=1        2-exit 건너뜀
#    SKIP_3EXIT=1        3-exit 건너뜀
#    LATENCY_ONLY=1      ImageNet 없을 때 랜덤 노이즈로 latency만 측정
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

THR_2EXIT="${1:-0.80}"
THR_3EXIT="${2:-0.80}"
N="${3:-30}"

echo "============================================"
echo "  ViT TRT 3-way Benchmark  (Jetson AGX Orin)"
echo "  thr_2exit : $THR_2EXIT"
echo "  thr_3exit : $THR_3EXIT"
echo "  N         : $N"
echo "  N_SAMPLES : ${N_SAMPLES:-1000}"
echo "============================================"

cd "$SRC_DIR"

EXTRA_ARGS=()
[[ -n "${DATA_ROOT:-}"       ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "${WARMUP:-}"          ]] && EXTRA_ARGS+=(--warmup "$WARMUP")
[[ "${SKIP_2EXIT:-0}"   == "1" ]] && EXTRA_ARGS+=(--skip-2exit)
[[ "${SKIP_3EXIT:-0}"   == "1" ]] && EXTRA_ARGS+=(--skip-3exit)
[[ "${LATENCY_ONLY:-0}" == "1" ]] && EXTRA_ARGS+=(--latency-only)

python benchmark/benchmark_trt_vit.py \
    --thr-2exit "$THR_2EXIT" \
    --thr-3exit "$THR_3EXIT" \
    --n "$N" \
    --num-samples "${N_SAMPLES:-1000}" \
    "${EXTRA_ARGS[@]}"
