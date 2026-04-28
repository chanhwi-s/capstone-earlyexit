#!/usr/bin/env bash
# ============================================================
#  benchmark_hybrid_2exit_realrun_5090.sh
#  LUT 시뮬레이션이 아닌 실제 GPU 실행 기반 2-exit 하이브리드 벤치마크.
#  ImageNet val에서 N개 랜덤 샘플링 → grid search.
#
#  사용법:
#    bash scripts/benchmark_hybrid_2exit_realrun_5090.sh --threshold 0.80
#    N_SAMPLES=2000 bash scripts/benchmark_hybrid_2exit_realrun_5090.sh --threshold 0.80
#    BATCH_SIZES="1 4 8 16" TIMEOUT_MS="1 2 5 10" \
#      bash scripts/benchmark_hybrid_2exit_realrun_5090.sh --threshold 0.80
#
#  환경 변수:
#    DATA_ROOT      ImageNet 루트 (기본: /home2/imagenet)
#    N_SAMPLES      랜덤 샘플 수 (기본: 1000)
#    SEED           랜덤 시드 (기본: 42)
#    BATCH_SIZES    (기본: 1 4 8 16 32)
#    TIMEOUT_MS     ms (기본: 1 2 5 10 20)
#    WARMUP         웜업 샘플 수 (기본: 50)
#    SKIP_PLAIN=1   PlainViT 기준선 측정 스킵
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

DATA_ROOT="${DATA_ROOT:-/home2/imagenet}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-42}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8 16 32}"
TIMEOUT_MS="${TIMEOUT_MS:-1 2 5 10 20}"
WARMUP="${WARMUP:-50}"

SKIP_ARGS=""
[[ "${SKIP_PLAIN:-0}" == "1" ]] && SKIP_ARGS="--skip-plain"

echo "============================================"
echo "  2-exit Real-Run Benchmark  (RTX 5090)"
echo "  DATA_ROOT   : $DATA_ROOT"
echo "  N_SAMPLES   : $N_SAMPLES  (seed=$SEED)"
echo "  BATCH_SIZES : $BATCH_SIZES"
echo "  TIMEOUT_MS  : $TIMEOUT_MS"
echo "  WARMUP      : $WARMUP"
echo "============================================"
echo ""

cd "$SRC_DIR"
python benchmark/hybrid_vit_2exit_realrun.py \
    --data-root   "$DATA_ROOT" \
    --n-samples   "$N_SAMPLES" \
    --seed        "$SEED" \
    --batch-sizes $BATCH_SIZES \
    --timeout-ms  $TIMEOUT_MS \
    --warmup      "$WARMUP" \
    --device-label "RTX 5090" \
    $SKIP_ARGS \
    "$@"
