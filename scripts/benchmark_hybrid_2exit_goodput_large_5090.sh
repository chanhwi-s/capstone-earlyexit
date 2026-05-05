#!/usr/bin/env bash
# ============================================================
#  benchmark_hybrid_2exit_goodput_large_5090.sh
#  ViT-L/16 2-exit goodput benchmark (학습 완료 후 실행)
#
#  4단계 체인 (자동 순서 실행):
#    Step 1. seg1.onnx + plain_vit_large.onnx export
#    Step 2. seg2_bs{N}.onnx static export (bs2별)
#    Step 3. goodput benchmark (SLO 기반)
#    Step 4. accuracy vs threshold sweep (all-exit2 기준선 대비)
#
#  사용법:
#    bash scripts/benchmark_hybrid_2exit_goodput_large_5090.sh --threshold 0.80
#    nohup bash scripts/benchmark_hybrid_2exit_goodput_large_5090.sh --threshold 0.80 \
#        > goodput_large_thr0.8.log 2>&1 &
#
#  환경 변수:
#    DATA_ROOT            ImageNet 루트 (기본: /home2)
#    N_SAMPLES            평가 샘플 수 (기본: 5000)
#    SEED                 랜덤 시드 (기본: 42)
#    BASELINE_BATCH_SIZE  seg1/plain 고정 batch 크기 (기본: 8)
#    BATCH_SIZES          seg2 static batch 목록 (기본: 1 2 4 8 16 32 64)
#    SLO_VALUES           SLO 임계값 ms (기본: 10 20 25 30 40 50 75 100)
#    WARMUP               웜업 seg1 배치 수 (기본: 50)
#    SKIP_EXPORT=1        Step1+2 export 스킵 (이미 .onnx 있을 때)
#    SKIP_PLAIN=1         PlainViT-L 기준선 측정 스킵
#    SKIP_ACC_SWEEP=1     Step4 accuracy sweep 스킵
#    ACC_N_SAMPLES        accuracy sweep 샘플 수 (기본: N_SAMPLES)
#    ACC_THRESHOLDS       accuracy sweep threshold 목록 (기본: 0.50~0.99 자동)
#    CKPT                 ee_vit_large_2exit 체크포인트 경로 직접 지정
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
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128 256}"
SLO_VALUES="${SLO_VALUES:-10 20 30 50 75 100 200 300 400 500}"
WARMUP="${WARMUP:-50}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PLAIN="${SKIP_PLAIN:-0}"
SKIP_ACC_SWEEP="${SKIP_ACC_SWEEP:-0}"
ACC_N_SAMPLES="${ACC_N_SAMPLES:-$N_SAMPLES}"
ACC_THRESHOLDS="${ACC_THRESHOLDS:-}"

echo "============================================"
echo "  ViT-L/16 2-exit Goodput Benchmark (RTX 5090)"
echo "  DATA_ROOT           : $DATA_ROOT"
echo "  N_SAMPLES           : $N_SAMPLES  (seed=$SEED)"
echo "  BASELINE_BATCH_SIZE : $BASELINE_BATCH_SIZE  (seg1/plain)"
echo "  BATCH_SIZES (seg2)  : $BATCH_SIZES"
echo "  SLO_VALUES          : $SLO_VALUES ms"
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

    # ── Step 2: seg2 static ONNX export (bs2별) ───────────────────────────────
    echo ">>> [Step 2] ViT-L seg2 static ONNX export (batch-sizes: $BATCH_SIZES) ..."
    CKPT_ARG2=""
    [[ -n "${CKPT:-}" ]] && CKPT_ARG2="--ckpt-2exit $CKPT"

    python export/export_onnx_seg2_static.py \
        --model-variant large \
        --batch-sizes $BATCH_SIZES \
        $CKPT_ARG2
    echo ""
else
    echo ">>> [Step 1+2] SKIP_EXPORT=1 — ONNX export 스킵"
    echo ""
fi

# ── Step 3: ONNX 경로 자동 탐색 ───────────────────────────────────────────────
# paths.py의 최신 exp_* 선택 로직과 동일하게 경로를 구성한다.
LATEST_EXP=$(python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import paths
print(paths.EXPERIMENTS_DIR)
PYEOF
)

SEG1_ONNX="$LATEST_EXP/onnx/ee_vit_large_2exit/seg1.onnx"
ONNX_DIR="$LATEST_EXP/onnx/ee_vit_large_2exit"
PLAIN_ONNX="$LATEST_EXP/onnx/plain_vit_large/plain_vit_large.onnx"

echo ">>> [Step 3] Goodput benchmark"
echo "    seg1  : $SEG1_ONNX"
echo "    onnx  : $ONNX_DIR"
echo "    plain : $PLAIN_ONNX"
echo ""

EXTRA_ARGS="--seg1-onnx $SEG1_ONNX --onnx-dir $ONNX_DIR"
[[ "$SKIP_PLAIN" != "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --plain-onnx $PLAIN_ONNX"
[[ "$SKIP_PLAIN" == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --skip-plain"

python benchmark/hybrid_vit_2exit_goodput.py \
    --data-root             "$DATA_ROOT/imagenet" \
    --n-samples             "$N_SAMPLES" \
    --seed                  "$SEED" \
    --baseline-batch-size   "$BASELINE_BATCH_SIZE" \
    --batch-sizes           $BATCH_SIZES \
    --slo-values            $SLO_VALUES \
    --warmup                "$WARMUP" \
    --device-label          "RTX 5090 (ViT-L)" \
    $EXTRA_ARGS \
    "$@"

# ── Step 4: Accuracy vs Threshold sweep ────────────────────────────────────────
if [[ "$SKIP_ACC_SWEEP" != "1" ]]; then
    echo ""
    echo ">>> [Step 4] Accuracy vs Threshold sweep ..."
    ACC_CKPT_ARG=""
    [[ -n "${CKPT:-}" ]] && ACC_CKPT_ARG="--ckpt $CKPT"
    ACC_THR_ARG=""
    [[ -n "$ACC_THRESHOLDS" ]] && ACC_THR_ARG="--thresholds $ACC_THRESHOLDS"

    python benchmark/sweep_accuracy_vs_threshold.py \
        --data-root     "$DATA_ROOT/imagenet" \
        --n-samples     "$ACC_N_SAMPLES" \
        --seed          "$SEED" \
        --batch-size    64 \
        --device-label  "RTX 5090 (ViT-L)" \
        $ACC_CKPT_ARG \
        $ACC_THR_ARG
    echo ""
else
    echo ">>> [Step 4] SKIP_ACC_SWEEP=1 — accuracy sweep 스킵"
fi
