#!/usr/bin/env bash
# ============================================================
#  dynamic_benchmark.sh
#  ViT-L/16 2-exit dynamic batching goodput benchmark
#
#  timeout-based dynamic batching sweep:
#    - seg2를 dynamic batch ONNX로 실행
#    - timeout_ms sweep → 각 timeout에서 goodput & avg latency 측정
#    - SLO를 하나로 고정하여 timeout별 goodput 비교
#
#  3단계 체인:
#    Step 1. seg1.onnx + plain_vit_large.onnx export  (static_benchmark.sh와 공유 가능)
#    Step 2. seg2_dynamic.onnx export
#    Step 3. dynamic batching goodput sweep
#
#  사용법:
#    bash scripts/dynamic_benchmark.sh --threshold 0.70
#    nohup bash scripts/dynamic_benchmark.sh --threshold 0.70 \
#        > dynamic_benchmark.log 2>&1 &
#
#  환경 변수:
#    DATA_ROOT            ImageNet 루트 (기본: /home2)
#    N_SAMPLES            평가 샘플 수 (기본: 5000)
#    SEED                 랜덤 시드 (기본: 42)
#    BASELINE_BATCH_SIZE  seg1/plain 고정 batch 크기 (기본: 32)
#    MAX_BATCH_SIZES      seg2 queue 최대 batch 크기 목록 (기본: 32 64 128)
#    TIMEOUT_VALUES       sweep할 timeout 값 목록 ms (기본: 1 2 5 10 20 50 100 200 500)
#    SLO_MS               고정 SLO 임계값 ms (기본: 100)
#    WARMUP               웜업 seg1 배치 수 (기본: 50)
#    SKIP_EXPORT=1        Step1+2 export 스킵
#    SKIP_PLAIN=1         PlainViT-L 기준선 측정 스킵
#    CKPT                 체크포인트 경로 직접 지정
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

DATA_ROOT="${DATA_ROOT:-/home2}"
N_SAMPLES="${N_SAMPLES:-5000}"
SEED="${SEED:-42}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-32}"
MAX_BATCH_SIZES="${MAX_BATCH_SIZES:-1 2 4 8 16 32 64 128}"
TIMEOUT_VALUES="${TIMEOUT_VALUES:-10 20 25 50 75 100 }"
SLO_MS="${SLO_MS:-300}"
WARMUP="${WARMUP:-50}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PLAIN="${SKIP_PLAIN:-0}"

echo "============================================"
echo "  ViT-L/16 2-exit Dynamic Batching Benchmark"
echo "  DATA_ROOT           : $DATA_ROOT"
echo "  N_SAMPLES           : $N_SAMPLES  (seed=$SEED)"
echo "  BASELINE_BATCH_SIZE : $BASELINE_BATCH_SIZE  (seg1/plain)"
echo "  MAX_BATCH_SIZES     : $MAX_BATCH_SIZES"
echo "  TIMEOUT_VALUES (ms) : $TIMEOUT_VALUES"
echo "  SLO_MS (fixed)      : $SLO_MS ms"
echo "  WARMUP              : $WARMUP seg1 batches"
echo "  SKIP_EXPORT         : $SKIP_EXPORT"
echo "============================================"
echo ""

cd "$SRC_DIR"

# ── Step 1: seg1 + plain ViT-L ONNX export ────────────────────────────────────
if [[ "$SKIP_EXPORT" != "1" ]]; then
    echo ">>> [Step 1] ViT-L seg1 + plain_vit_large ONNX export ..."
    CKPT_ARG=""
    [[ -n "${CKPT:-}" ]] && CKPT_ARG="--ckpt $CKPT"
    SKIP_PLAIN_ARG=""
    [[ "$SKIP_PLAIN" == "1" ]] && SKIP_PLAIN_ARG="--skip-plain"

    python export/export_onnx_vit_large_selective.py \
        --baseline-batch-size "$BASELINE_BATCH_SIZE" \
        $CKPT_ARG \
        $SKIP_PLAIN_ARG
    echo ""

    # ── Step 2: seg2 dynamic ONNX export ──────────────────────────────────────
    echo ">>> [Step 2] ViT-L seg2 dynamic ONNX export ..."
    CKPT_ARG2=""
    [[ -n "${CKPT:-}" ]] && CKPT_ARG2="--ckpt-2exit $CKPT"

    python export/export_onnx_seg2_dynamic.py \
        --model-variant large \
        $CKPT_ARG2
    echo ""
else
    echo ">>> [Step 1+2] SKIP_EXPORT=1 — ONNX export 스킵"
    echo ""
fi

# ── Step 3: ONNX 경로 자동 탐색 ───────────────────────────────────────────────
LATEST_EXP=$(python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import paths
print(paths.EXPERIMENTS_DIR)
PYEOF
)

SEG1_ONNX="$LATEST_EXP/onnx/ee_vit_large_2exit/seg1.onnx"
SEG2_DYNAMIC_ONNX="$LATEST_EXP/onnx/ee_vit_large_2exit/seg2_dynamic.onnx"
PLAIN_ONNX="$LATEST_EXP/onnx/plain_vit_large/plain_vit_large.onnx"

echo ">>> [Step 3] Dynamic batching goodput sweep"
echo "    seg1         : $SEG1_ONNX"
echo "    seg2_dynamic : $SEG2_DYNAMIC_ONNX"
echo "    plain        : $PLAIN_ONNX"
echo ""

EXTRA_ARGS="--seg1-onnx $SEG1_ONNX --seg2-dynamic-onnx $SEG2_DYNAMIC_ONNX"
[[ "$SKIP_PLAIN" != "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --plain-onnx $PLAIN_ONNX"
[[ "$SKIP_PLAIN" == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --skip-plain"

python benchmark/dynamic_benchmark.py \
    --data-root             "$DATA_ROOT/imagenet" \
    --n-samples             "$N_SAMPLES" \
    --seed                  "$SEED" \
    --baseline-batch-size   "$BASELINE_BATCH_SIZE" \
    --max-batch-sizes       $MAX_BATCH_SIZES \
    --timeout-values        $TIMEOUT_VALUES \
    --slo-ms                "$SLO_MS" \
    --warmup                "$WARMUP" \
    --device-label          "RTX 5090 (ViT-L)" \
    $EXTRA_ARGS \
    "$@"
