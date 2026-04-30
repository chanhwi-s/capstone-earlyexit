#!/usr/bin/env bash
# ============================================================
#  train_vit_large_5090.sh
#  ViT-L/16 2-exit exit head fine-tuning (backbone frozen)
#  exit_blocks=[12, 24], 하이퍼파라미터는 ViT-B/16과 동일
#
#  사용법:
#    bash scripts/train_vit_large_5090.sh
#    nohup bash scripts/train_vit_large_5090.sh > train_vit_large.log 2>&1 &
#
#  환경 변수:
#    DATA_ROOT     ImageNet 루트 (기본: /home2/imagenet)
#    EPOCHS        학습 에포크 수 (기본: 30)
#    BATCH_SIZE    배치 크기 (기본: 64)
#    LR            learning rate (기본: 1e-3)
#    WEIGHT_MODE   equal | linear (기본: equal)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

DATA_ROOT="${DATA_ROOT:-/home2/imagenet}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
WEIGHT_MODE="${WEIGHT_MODE:-equal}"

echo "============================================"
echo "  ViT-L/16 2-exit Fine-tuning  (RTX 5090)"
echo "  exit_blocks : [12, 24]  (backbone frozen)"
echo "  DATA_ROOT   : $DATA_ROOT"
echo "  EPOCHS      : $EPOCHS"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  LR          : $LR"
echo "  WEIGHT_MODE : $WEIGHT_MODE"
echo "============================================"
echo ""

cd "$SRC_DIR"
python train/train_vit_large_selective.py \
    --data-root   "$DATA_ROOT" \
    --epochs      "$EPOCHS" \
    --batch-size  "$BATCH_SIZE" \
    --lr          "$LR" \
    --weight-mode "$WEIGHT_MODE" \
    "$@"
