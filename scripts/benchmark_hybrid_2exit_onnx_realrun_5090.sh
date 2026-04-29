#!/usr/bin/env bash
# ============================================================
#  benchmark_hybrid_2exit_onnx_realrun_5090.sh
#  ONNX Runtime (CUDA EP) 기반 2-exit 하이브리드 벤치마크.
#
#  PyTorch .pth 버전과의 차이:
#    - seg1 / plain_vit: static batch=BASELINE_BATCH_SIZE 로 export된 ONNX 사용
#    - seg2: dynamic batch (BATCH_SIZES 목록) ONNX 사용
#    - IOBinding으로 GPU 메모리 유지 (seg1→seg2 feature 전달)
#
#  실행 전 필수:
#    bash scripts/export_vit_5090.sh all  (또는 2exit+plain)
#    ※ 기존 seg2.onnx / plain_vit.onnx 가 batch=1 고정으로 export 됐으면
#      반드시 재export 필요 (--baseline-batch-size 8 로 export 해야 함)
#
#  사용법:
#    bash scripts/benchmark_hybrid_2exit_onnx_realrun_5090.sh --threshold 0.80
#    N_SAMPLES=2000 bash scripts/benchmark_hybrid_2exit_onnx_realrun_5090.sh --threshold 0.80
#    BATCH_SIZES="8 16 32 64" TIMEOUT_MS="1 2 5 10" \
#      bash scripts/benchmark_hybrid_2exit_onnx_realrun_5090.sh --threshold 0.80
#
#  환경 변수:
#    DATA_ROOT             ImageNet 루트 (기본: /home2/imagenet)
#    N_SAMPLES             평가 샘플 수 (기본: 5000, bs1 배수로 자동 보정)
#    SEED                  랜덤 시드 (기본: 42)
#    BASELINE_BATCH_SIZE   seg1/plain 고정 배치 크기 (기본: 8)
#    BATCH_SIZES           seg2 grid 탐색 크기 (기본: 8 16 32 64 128 256 512)
#    TIMEOUT_MS            ms (기본: 1 2 5 10 20 50 100 200 500)
#    WARMUP                웜업 seg1 배치 수 (기본: 50)
#    SEG1_ONNX             seg1.onnx 경로 직접 지정 (미지정 시 자동 탐색)
#    SEG2_ONNX             seg2.onnx 경로 직접 지정
#    PLAIN_ONNX            plain_vit.onnx 경로 직접 지정
#    SKIP_PLAIN=1          PlainViT 기준선 측정 스킵
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
# seg2 grid: minimum=BASELINE_BATCH_SIZE, 이후 2배 등비
BATCH_SIZES="${BATCH_SIZES:-8 16 32 64 128 256 512}"
TIMEOUT_MS="${TIMEOUT_MS:-1 2 5 10 20 50 100 200 500}"
WARMUP="${WARMUP:-50}"

ONNX_ARGS=""
[[ -n "${SEG1_ONNX:-}"  ]] && ONNX_ARGS="$ONNX_ARGS --seg1-onnx  $SEG1_ONNX"
[[ -n "${SEG2_ONNX:-}"  ]] && ONNX_ARGS="$ONNX_ARGS --seg2-onnx  $SEG2_ONNX"
[[ -n "${PLAIN_ONNX:-}" ]] && ONNX_ARGS="$ONNX_ARGS --plain-onnx $PLAIN_ONNX"

SKIP_ARGS=""
[[ "${SKIP_PLAIN:-0}" == "1" ]] && SKIP_ARGS="--skip-plain"

echo "============================================"
echo "  2-exit ONNX Real-Run Benchmark  (RTX 5090)"
echo "  DATA_ROOT           : $DATA_ROOT"
echo "  N_SAMPLES           : $N_SAMPLES  (seed=$SEED)"
echo "  BASELINE_BATCH_SIZE : $BASELINE_BATCH_SIZE  (seg1/plain static)"
echo "  BATCH_SIZES (seg2)  : $BATCH_SIZES"
echo "  TIMEOUT_MS          : $TIMEOUT_MS"
echo "  WARMUP              : $WARMUP seg1 batches"
echo "============================================"
echo ""

cd "$SRC_DIR"
python benchmark/hybrid_vit_2exit_onnx_realrun.py \
    --data-root            "$DATA_ROOT" \
    --n-samples            "$N_SAMPLES" \
    --seed                 "$SEED" \
    --baseline-batch-size  "$BASELINE_BATCH_SIZE" \
    --batch-sizes          $BATCH_SIZES \
    --timeout-ms           $TIMEOUT_MS \
    --warmup               "$WARMUP" \
    --device-label         "RTX 5090" \
    $ONNX_ARGS \
    $SKIP_ARGS \
    "$@"
