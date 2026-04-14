#!/usr/bin/env bash
# ============================================================
#  train_vit_3exit_5090.sh  —  RTX 5090 서버용 3-Exit ViT 학습 파이프라인
#
#  Exit 구성: B6 (early) + B9 (mid) + B12 (main)
#
#  실행 순서:
#    0. 타임스탬프 기반 실험 디렉토리 생성
#    1. SelectiveExitViT (exit_blocks=[6, 9, 12]) exit head fine-tuning
#       → {EXP_DIR}/train/ee_vit_3exit/
#
#  데이터셋: ImageNet (224×224, 1000 classes)
#  모델    : ViT-B/16 (pretrained backbone frozen, exit head 3개만 학습)
#  Optimizer: AdamW (lr=1e-3)
#
#  주요 환경 변수:
#    DATA_ROOT=<path>      ImageNet 루트 경로 (기본: /home2/imagenet)
#    EPOCHS=<N>            학습 에포크 수 오버라이드 (기본: 30)
#    BATCH_SIZE=<N>        배치 크기 오버라이드 (기본: 64)
#    LR=<float>            learning rate (기본: 1e-3)
#    WEIGHT_DECAY=<float>  AdamW weight decay (기본: 0.05)
#    WEIGHT_MODE=<mode>    equal | linear (기본: equal)
#    EXP_NAME=exp_...      실험 디렉토리 이름 (기본: 자동 생성)
#
#  사용법:
#    cd <project_root>
#    bash scripts/train_vit_3exit_5090.sh
#
#  백그라운드 실행 (권장):
#    nohup bash scripts/train_vit_3exit_5090.sh > vit_3exit_train.log 2>&1 &
#    tail -f vit_3exit_train.log
# ============================================================

set -euo pipefail

# ── HuggingFace / timm 캐시 경로 (서버 전용 설정) ─────────────
export HF_HOME=/home/cap10/.cache/huggingface
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── ImageNet 경로 ────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/home2}"

# ── 0. 실험 디렉토리 생성 ────────────────────────────────────
EXP_NAME="${EXP_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"
export EXP_DIR="$PROJECT_ROOT/experiments/$EXP_NAME"
export EXP_NAME

mkdir -p "$EXP_DIR"

echo "============================================"
echo "  3-Exit ViT-B/16 Fine-tuning (B6 + B9 + B12)"
echo "  Project root  : $PROJECT_ROOT"
echo "  Exp dir       : experiments/$EXP_NAME"
echo "  Exit blocks   : [6, 9, 12]"
echo "  Data root     : $DATA_ROOT"
echo "  HF_HOME       : $HF_HOME"
if [[ -n "${EPOCHS:-}"        ]]; then echo "  Epochs        : $EPOCHS (override)"; fi
if [[ -n "${BATCH_SIZE:-}"    ]]; then echo "  Batch size    : $BATCH_SIZE (override)"; fi
if [[ -n "${LR:-}"            ]]; then echo "  LR            : $LR (override)"; fi
if [[ -n "${WEIGHT_MODE:-}"   ]]; then echo "  Weight mode   : $WEIGHT_MODE (override)"; fi
echo "============================================"

cat > "$EXP_DIR/exp_info.txt" <<EOF
EXP_NAME    : $EXP_NAME
EXP_DIR     : $EXP_DIR
Model       : SelectiveExitViT-B/16 (exit_blocks=[6, 9, 12], backbone frozen)
Dataset     : ImageNet (224x224)
Start       : $(date '+%Y-%m-%d %H:%M:%S')
Host        : $(hostname)
Python      : $(python3 --version 2>&1 || echo "N/A")
EXIT_BLOCKS : 6 9 12
EPOCHS      : ${EPOCHS:-"(default 30)"}
BATCH_SIZE  : ${BATCH_SIZE:-"(default 64)"}
LR          : ${LR:-"(default 1e-3)"}
WEIGHT_DECAY: ${WEIGHT_DECAY:-"(default 0.05)"}
WEIGHT_MODE : ${WEIGHT_MODE:-"(default equal)"}
DATA_ROOT   : $DATA_ROOT
EOF
echo "  Exp info saved: $EXP_DIR/exp_info.txt"
echo ""

# ── CLI 인자 조립 ────────────────────────────────────────────
EXTRA_ARGS="--data-root $DATA_ROOT"
[[ -n "${EPOCHS:-}"       ]] && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[[ -n "${BATCH_SIZE:-}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"
[[ -n "${LR:-}"           ]] && EXTRA_ARGS="$EXTRA_ARGS --lr $LR"
[[ -n "${WEIGHT_DECAY:-}" ]] && EXTRA_ARGS="$EXTRA_ARGS --weight-decay $WEIGHT_DECAY"
[[ -n "${WEIGHT_MODE:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --weight-mode $WEIGHT_MODE"

cd "$SRC_DIR"

# ── 1. SelectiveExitViT (3-exit) 학습 ────────────────────────
echo "[1/1] 3-Exit ViT-B/16 fine-tuning (exit_blocks=[6, 9, 12]) ..."
echo "      Args: --exit-blocks 6 9 12 $EXTRA_ARGS"
python train/train_vit_selective.py --exit-blocks 6 9 12 $EXTRA_ARGS
echo "[1/1] Training complete."

echo "완료 시각: $(date '+%Y-%m-%d %H:%M:%S')" >> "$EXP_DIR/exp_info.txt"

echo ""
echo "============================================"
echo "  3-Exit ViT Training Done!"
echo "  Exp dir    : experiments/$EXP_NAME"
echo "  Checkpoints:"
echo "    experiments/$EXP_NAME/train/ee_vit_3exit/checkpoints/best.pth"
echo "    experiments/$EXP_NAME/train/ee_vit_3exit/checkpoints/final.pth"
echo "  Log:"
echo "    experiments/$EXP_NAME/train/ee_vit_3exit/train_log.csv"
echo "============================================"
