#!/usr/bin/env bash
# ============================================================
#  run_vit_5090.sh  —  RTX 5090 서버 전용 EE-ViT fine-tuning 래퍼
#
#  서버 환경:
#    프로젝트 루트 : /home/cap10/capstone-earlyexit
#    ImageNet 경로 : /home2/imagenet/train, /home2/imagenet/val
#
#  사용법:
#    conda activate capstonedesign
#    bash scripts/run_vit_5090.sh            # 기본 (30 epochs, batch 128)
#    bash scripts/run_vit_5090.sh --test     # 빠른 검증 (3 epochs, batch 64)
#
#  백그라운드 실행 (권장):
#    nohup bash scripts/run_vit_5090.sh > vit_train.log 2>&1 &
#    tail -f vit_train.log
#
#  오버라이드 가능한 환경변수:
#    EPOCHS=30          학습 에포크 (기본: 30)
#    BATCH_SIZE=128     배치 크기 (기본: 128, 5090 기준)
#    LR=1e-3            learning rate (기본: 1e-3)
#    WEIGHT_DECAY=0.05  AdamW weight decay (기본: 0.05)
#    WEIGHT_MODE=equal  equal | linear (기본: equal)
#    EXP_NAME=exp_...   실험 디렉토리 이름 (기본: 자동 생성)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 서버 고정 경로 ────────────────────────────────────────────
DATA_ROOT="/home2"    # → /home2/imagenet/train, /home2/imagenet/val

# ── HuggingFace 캐시 경로 (공유 캐시 권한 오류 방지) ──────────
export HF_HOME="/home/cap10/.cache/huggingface"
mkdir -p "$HF_HOME"

# ── 5090 최적 기본값 (환경변수로 오버라이드 가능) ─────────────
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"    # 5090 24GB VRAM 기준 여유 있게
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WEIGHT_MODE="${WEIGHT_MODE:-equal}"

# ── --test 플래그: 빠른 동작 검증용 ──────────────────────────
if [[ "${1:-}" == "--test" ]]; then
    echo "[TEST MODE] epochs=3, batch=64, samples=200"
    EPOCHS=3
    BATCH_SIZE=64
    export EXP_NAME="exp_vit_test_$(date +%Y%m%d_%H%M%S)"
fi

export DATA_ROOT EPOCHS BATCH_SIZE LR WEIGHT_DECAY WEIGHT_MODE

echo "============================================"
echo "  RTX 5090 — EE-ViT Fine-tuning"
echo "  DATA_ROOT  : $DATA_ROOT"
echo "             : → $DATA_ROOT/imagenet/{train,val}"
echo "  Epochs     : $EPOCHS"
echo "  Batch size : $BATCH_SIZE"
echo "  LR (AdamW) : $LR  wd=$WEIGHT_DECAY"
echo "  Weight mode: $WEIGHT_MODE"
echo "============================================"

bash "$SCRIPT_DIR/train_vit_pipeline.sh"
