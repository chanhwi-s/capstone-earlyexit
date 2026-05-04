#!/usr/bin/env bash
# ============================================================
#  benchmark_pytorch_vit_5090.sh  —  PyTorch ViT threshold sweep (RTX 5090)
#
#  TRT 불필요. 학습된 체크포인트(.pth)와 ImageNet val만으로 실행.
#  PlainViT vs 2-exit vs 3-exit vs ViT-L/16 2-exit 를 threshold sweep하여
#  accuracy / exit rate / latency(avg, p90, p95, p99) 비교.
#
#  ViT-L/16은 ee_vit_large_2exit 체크포인트가 있으면 자동으로 포함된다.
#
#  사용법:
#    bash scripts/benchmark_pytorch_vit_5090.sh
#    THRESHOLDS="0.7 0.8 0.9" bash scripts/benchmark_pytorch_vit_5090.sh
#    EXP_DIR=experiments/exp_20260414_212957 bash scripts/benchmark_pytorch_vit_5090.sh
#    SKIP_PLAIN=1 bash scripts/benchmark_pytorch_vit_5090.sh
#    SKIP_LARGE=1 bash scripts/benchmark_pytorch_vit_5090.sh   # ViT-L 제외
#    SKIP_2EXIT=1 SKIP_3EXIT=1 bash scripts/benchmark_pytorch_vit_5090.sh  # ViT-L만
#
#  환경 변수:
#    DATA_ROOT            ImageNet 루트 (기본: /home2/imagenet)
#    THRESHOLDS           sweep할 threshold 목록 (기본: 0.5 0.6 0.7 0.75 0.80 0.85 0.90)
#    EXP_DIR              실험 디렉토리 직접 지정 (기본: 최신 exp_* 자동)
#    SKIP_PLAIN=1         PlainViT-B 스킵
#    SKIP_2EXIT=1         ViT-B 2-exit 스킵
#    SKIP_3EXIT=1         ViT-B 3-exit 스킵
#    SKIP_LARGE=1         ViT-L/16 2-exit 스킵 (자동 탐지 비활성화)
#    SKIP_PLAIN_LARGE=1   PlainViT-L 기준선 스킵
#    EXIT_BLOCKS_LARGE    ViT-L exit 블록 (기본: "12 24")
#    NUM_WORKERS          DataLoader 워커 수 (기본: 8)
#    WARMUP               GPU warmup 샘플 수 (기본: 200)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

DATA_ROOT="${DATA_ROOT:-/home2/imagenet}"
THRESHOLDS="${THRESHOLDS:-0.5 0.6 0.7 0.75 0.80 0.85 0.90}"
EXIT_BLOCKS_LARGE="${EXIT_BLOCKS_LARGE:-12 24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WARMUP="${WARMUP:-200}"

# ── 실험 디렉토리 결정 ──────────────────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    [[ "$EXP_DIR" != /* ]] && EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    export EXP_DIR="$(realpath "$EXP_DIR")"
fi

# ── skip 플래그 ─────────────────────────────────────────────────────────────
SKIP_ARGS=""
[[ "${SKIP_PLAIN:-0}"       == "1" ]] && SKIP_ARGS="$SKIP_ARGS --skip-plain"
[[ "${SKIP_2EXIT:-0}"       == "1" ]] && SKIP_ARGS="$SKIP_ARGS --skip-2exit"
[[ "${SKIP_3EXIT:-0}"       == "1" ]] && SKIP_ARGS="$SKIP_ARGS --skip-3exit"
[[ "${SKIP_LARGE:-0}"       == "1" ]] && SKIP_ARGS="$SKIP_ARGS --skip-large"
[[ "${SKIP_PLAIN_LARGE:-0}" == "1" ]] && SKIP_ARGS="$SKIP_ARGS --skip-plain-large"

echo "============================================"
echo "  PyTorch ViT Threshold Sweep  (RTX 5090)"
echo "  DATA_ROOT         : $DATA_ROOT"
echo "  THRESHOLDS        : $THRESHOLDS"
echo "  EXIT_BLOCKS_LARGE : $EXIT_BLOCKS_LARGE"
echo "  NUM_WORKERS: $NUM_WORKERS  WARMUP: $WARMUP"
[[ -n "${EXP_DIR:-}" ]] && echo "  EXP_DIR    : $EXP_DIR"
echo "  (ViT-L/16은 체크포인트 있으면 자동 포함, SKIP_LARGE=1로 제외 가능)"
echo "============================================"
echo ""

cd "$SRC_DIR"
python benchmark/benchmark_pytorch_vit.py \
    --data-root          "$DATA_ROOT" \
    --thresholds         $THRESHOLDS \
    --exit-blocks-large  $EXIT_BLOCKS_LARGE \
    --num-workers        "$NUM_WORKERS" \
    --warmup             "$WARMUP" \
    --device-label       "RTX 5090" \
    $SKIP_ARGS
