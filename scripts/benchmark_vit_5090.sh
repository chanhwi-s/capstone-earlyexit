#!/usr/bin/env bash
# ============================================================
#  benchmark_vit_5090.sh  —  PlainViT vs 2-exit vs 3-exit 벤치마크 (RTX 5090)
#
#  build_vit_trt_5090.sh 로 TRT 엔진 빌드 완료 후 실행.
#  ImageNet val 경로가 있으면 accuracy + exit rate 까지 측정.
#  없으면 자동으로 latency-only 모드.
#
#  사용법:
#    bash scripts/benchmark_vit_5090.sh <thr_2exit> <thr_3exit> <N>
#
#  예시:
#    bash scripts/benchmark_vit_5090.sh 0.80 0.80 30
#
#  백그라운드:
#    nohup bash scripts/benchmark_vit_5090.sh 0.80 0.80 30 \
#        > benchmark_vit_5090.log 2>&1 &
#
#  환경 변수:
#    DATA_ROOT=<path>   ImageNet 루트 (data_root/imagenet/val 구조)
#                       기본: ../data (configs/train.yaml 과 동일)
#    N_SAMPLES=1000
#    WARMUP=20
#    SKIP_2EXIT=1       2-exit 제외
#    SKIP_3EXIT=1       3-exit 제외
#    LATENCY_ONLY=1     accuracy 생략, latency만 측정
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

THR_2EXIT="${1:-0.80}"
THR_3EXIT="${2:-0.80}"
N="${3:-30}"

echo "============================================"
echo "  ViT TRT Benchmark  (RTX 5090)"
echo "  thr_2exit : $THR_2EXIT"
echo "  thr_3exit : $THR_3EXIT"
echo "  N         : $N"
echo "  N_SAMPLES : ${N_SAMPLES:-1000}"
echo "  DATA_ROOT : ${DATA_ROOT:-'(configs/train.yaml 자동)'}"
echo "============================================"

cd "$SRC_DIR"

EXTRA_ARGS=(--device-label "RTX 5090")
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
