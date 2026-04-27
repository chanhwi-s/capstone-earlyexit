#!/usr/bin/env bash
# ============================================================
#  benchmark_hybrid_2exit_5090.sh — 2-exit Hybrid Runtime 벤치마크 (RTX 5090)
#
#  batch_size × timeout_ms grid search → avg/p99 개선율 heatmap 출력.
#
#  사용법:
#    bash scripts/benchmark_hybrid_2exit_5090.sh --threshold 0.80
#    BATCH_SIZES="1 4 8 16" TIMEOUT_MS="1 2 5 10" \
#      bash scripts/benchmark_hybrid_2exit_5090.sh --threshold 0.80
#    EXP_DIR=experiments/exp_20260414_212957 \
#      bash scripts/benchmark_hybrid_2exit_5090.sh --threshold 0.75
#
#  환경 변수:
#    DATA_ROOT      ImageNet 루트 (기본: /home2/imagenet)
#    BATCH_SIZES    grid search batch_size 목록 (기본: 1 4 8 16 32)
#    TIMEOUT_MS     grid search timeout 목록 ms (기본: 1 2 5 10 20)
#    EXP_DIR        실험 디렉토리 직접 지정 (기본: 최신 exp_* 자동)
#    SKIP_PLAIN=1   PlainViT 기준선 측정 스킵
#
#  --threshold 는 필수 인자
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

DATA_ROOT="${DATA_ROOT:-/home2/imagenet}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8 16 32}"
TIMEOUT_MS="${TIMEOUT_MS:-1 2 5 10 20}"

if [[ -n "${EXP_DIR:-}" ]]; then
    [[ "$EXP_DIR" != /* ]] && EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    export EXP_DIR="$(realpath "$EXP_DIR")"
fi

SKIP_ARGS=""
[[ "${SKIP_PLAIN:-0}" == "1" ]] && SKIP_ARGS="$SKIP_ARGS --skip-plain"

echo "============================================"
echo "  2-exit Hybrid Benchmark  (RTX 5090)"
echo "  DATA_ROOT   : $DATA_ROOT"
echo "  BATCH_SIZES : $BATCH_SIZES"
echo "  TIMEOUT_MS  : $TIMEOUT_MS"
[[ -n "${EXP_DIR:-}" ]] && echo "  EXP_DIR     : $EXP_DIR"
echo "============================================"
echo ""

cd "$SRC_DIR"
python benchmark/hybrid_vit_2exit.py \
    --data-root   "$DATA_ROOT" \
    --batch-sizes $BATCH_SIZES \
    --timeout-ms  $TIMEOUT_MS \
    --device-label "RTX 5090" \
    $SKIP_ARGS \
    "$@"
