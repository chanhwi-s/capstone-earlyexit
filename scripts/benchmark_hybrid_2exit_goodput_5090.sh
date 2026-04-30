#!/usr/bin/env bash
# ============================================================
#  benchmark_hybrid_2exit_goodput_5090.sh
#  static seg2 ONNX + SLO 기반 goodput benchmark
#
#  실행 순서:
#    1. seg2 static export (최초 1회):
#       cd src && python export/export_onnx_seg2_static.py --batch-sizes 8 16 32 64
#
#    2. 벤치마크 실행:
#       bash scripts/benchmark_hybrid_2exit_goodput_5090.sh --threshold 0.80
#
#  사용법:
#    bash scripts/benchmark_hybrid_2exit_goodput_5090.sh --threshold 0.80
#    N_SAMPLES=2000 BATCH_SIZES="8 16 32" SLO_VALUES="20 50 100" \
#      bash scripts/benchmark_hybrid_2exit_goodput_5090.sh --threshold 0.80
#
#  환경 변수:
#    DATA_ROOT            ImageNet 루트 (기본: /home2/imagenet)
#    N_SAMPLES            평가 샘플 수 (기본: 5000, bs1 배수로 자동 조정)
#    SEED                 랜덤 시드 (기본: 42)
#    BASELINE_BATCH_SIZE  seg1/plain 고정 batch 크기 (기본: 8)
#    BATCH_SIZES          seg2 static batch 목록 (기본: 8 16 32 64)
#    SLO_VALUES           SLO 임계값 ms (기본: 10 20 30 50 75 100 150 200 300 500)
#    WARMUP               웜업 seg1 배치 수 (기본: 50)
#    SEG1_ONNX            seg1.onnx 경로 직접 지정 (미지정 시 자동 탐색)
#    PLAIN_ONNX           plain_vit.onnx 경로 직접 지정
#    ONNX_DIR             seg2_bs{N}.onnx 디렉토리 직접 지정
#    SKIP_PLAIN=1         PlainViT 기준선 측정 스킵
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

DATA_ROOT="${DATA_ROOT:-/home2/imagenet}"
N_SAMPLES="${N_SAMPLES:-5000}"
SEED="${SEED:-42}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-8}"
BATCH_SIZES="${BATCH_SIZES:-8 16 32 64}"
SLO_VALUES="${SLO_VALUES:-10 20 30 50 75 100 150 200 300 500}"
WARMUP="${WARMUP:-50}"

EXTRA_ARGS=""
[[ -n "${SEG1_ONNX:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --seg1-onnx  $SEG1_ONNX"
[[ -n "${PLAIN_ONNX:-}" ]] && EXTRA_ARGS="$EXTRA_ARGS --plain-onnx $PLAIN_ONNX"
[[ -n "${ONNX_DIR:-}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --onnx-dir   $ONNX_DIR"
[[ "${SKIP_PLAIN:-0}" == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --skip-plain"

echo "============================================"
echo "  2-exit Goodput Benchmark  (RTX 5090)"
echo "  DATA_ROOT           : $DATA_ROOT"
echo "  N_SAMPLES           : $N_SAMPLES  (seed=$SEED)"
echo "  BASELINE_BATCH_SIZE : $BASELINE_BATCH_SIZE  (seg1/plain)"
echo "  BATCH_SIZES (seg2)  : $BATCH_SIZES"
echo "  SLO_VALUES          : $SLO_VALUES ms"
echo "  WARMUP              : $WARMUP seg1 batches"
echo "============================================"
echo ""

cd "$SRC_DIR"
python benchmark/hybrid_vit_2exit_goodput.py \
    --data-root             "$DATA_ROOT" \
    --n-samples             "$N_SAMPLES" \
    --seed                  "$SEED" \
    --baseline-batch-size   "$BASELINE_BATCH_SIZE" \
    --batch-sizes           $BATCH_SIZES \
    --slo-values            $SLO_VALUES \
    --warmup                "$WARMUP" \
    --device-label          "RTX 5090" \
    $EXTRA_ARGS \
    "$@"
