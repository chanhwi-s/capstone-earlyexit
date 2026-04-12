#!/usr/bin/env bash
# ============================================================
#  train_vit_pipeline.sh  —  RTX 5090 서버용 EE-ViT 학습 파이프라인
#
#  실행 순서:
#    0. 타임스탬프 기반 실험 디렉토리 생성
#    1. EE-ViT-B/16 exit head fine-tuning  → {EXP_DIR}/train/ee_vit/
#
#  데이터셋: ImageNet 전용 (224×224, 1000 classes)
#  모델    : ViT-B/16 (pretrained backbone frozen, exit head 12개만 학습)
#  Optimizer: AdamW (lr=1e-3)
#
#  주요 환경 변수:
#    DATA_ROOT=<path>      ImageNet 루트 경로 (기본: configs/train.yaml 값)
#    EPOCHS=<N>            학습 에포크 수 오버라이드 (기본: 30)
#    BATCH_SIZE=<N>        배치 크기 오버라이드 (기본: 64)
#    LR=<float>            learning rate 오버라이드 (기본: 1e-3)
#    WEIGHT_DECAY=<float>  AdamW weight decay (기본: 0.05)
#    WEIGHT_MODE=<mode>    equal | linear (기본: equal)
#    EXP_NAME=exp_...      실험 디렉토리 이름 직접 지정 (기본: 자동 생성)
#
#  사용법:
#    cd <project_root>
#    bash scripts/train_vit_pipeline.sh                                  # 기본
#    DATA_ROOT=/data/imagenet bash scripts/train_vit_pipeline.sh         # 경로 지정
#    EPOCHS=10 BATCH_SIZE=32 bash scripts/train_vit_pipeline.sh          # 빠른 테스트
#    WEIGHT_MODE=linear bash scripts/train_vit_pipeline.sh               # 선형 가중치
#
#  백그라운드 실행:
#    nohup bash scripts/train_vit_pipeline.sh > vit_train.log 2>&1 &
#    tail -f vit_train.log
#
#  학습 완료 후 Orin으로 전송:
#    scp -r experiments/$EXP_NAME cap6@202.30.10.85:capstone-earlyexit/experiments/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 0. 실험 디렉토리 생성 ────────────────────────────────────
EXP_NAME="${EXP_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"
export EXP_DIR="$PROJECT_ROOT/experiments/$EXP_NAME"
export EXP_NAME

mkdir -p "$EXP_DIR"

echo "============================================"
echo "  EE-ViT-B/16 Fine-tuning Pipeline"
echo "  Project root  : $PROJECT_ROOT"
echo "  실험 디렉토리 : experiments/$EXP_NAME"
echo "  모델          : ViT-B/16 (backbone frozen)"
echo "  데이터셋      : ImageNet (224×224)"
if [[ -n "${DATA_ROOT:-}"     ]]; then echo "  Data root     : $DATA_ROOT (override)"; fi
if [[ -n "${EPOCHS:-}"        ]]; then echo "  Epochs        : $EPOCHS (override)"; fi
if [[ -n "${BATCH_SIZE:-}"    ]]; then echo "  Batch size    : $BATCH_SIZE (override)"; fi
if [[ -n "${LR:-}"            ]]; then echo "  LR            : $LR (override)"; fi
if [[ -n "${WEIGHT_DECAY:-}"  ]]; then echo "  Weight decay  : $WEIGHT_DECAY (override)"; fi
if [[ -n "${WEIGHT_MODE:-}"   ]]; then echo "  Weight mode   : $WEIGHT_MODE (override)"; fi
echo "============================================"

# 실험 정보 기록
cat > "$EXP_DIR/exp_info.txt" <<EOF
EXP_NAME    : $EXP_NAME
EXP_DIR     : $EXP_DIR
Model       : EE-ViT-B/16 (exit heads x12, backbone frozen)
Dataset     : ImageNet (224x224)
시작 시각   : $(date '+%Y-%m-%d %H:%M:%S')
호스트      : $(hostname)
Python      : $(python3 --version 2>&1 || echo "N/A")
EPOCHS      : ${EPOCHS:-"(default 30)"}
BATCH_SIZE  : ${BATCH_SIZE:-"(default 64)"}
LR          : ${LR:-"(default 1e-3)"}
WEIGHT_DECAY: ${WEIGHT_DECAY:-"(default 0.05)"}
WEIGHT_MODE : ${WEIGHT_MODE:-"(default equal)"}
EOF
echo "  실험 정보 저장: $EXP_DIR/exp_info.txt"
echo ""

# ── CLI 인자 조립 ────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "${DATA_ROOT:-}"    ]] && EXTRA_ARGS="$EXTRA_ARGS --data-root $DATA_ROOT"
[[ -n "${EPOCHS:-}"       ]] && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[[ -n "${BATCH_SIZE:-}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"
[[ -n "${LR:-}"           ]] && EXTRA_ARGS="$EXTRA_ARGS --lr $LR"
[[ -n "${WEIGHT_DECAY:-}" ]] && EXTRA_ARGS="$EXTRA_ARGS --weight-decay $WEIGHT_DECAY"
[[ -n "${WEIGHT_MODE:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --weight-mode $WEIGHT_MODE"

cd "$SRC_DIR"

# ── 1. EE-ViT exit head 학습 ─────────────────────────────────
echo "[1/1] EE-ViT-B/16 exit head fine-tuning 시작..."
[[ -n "$EXTRA_ARGS" ]] && echo "      추가 인자: $EXTRA_ARGS"
python train/train_vit.py $EXTRA_ARGS
echo "[1/1] 학습 완료"

# 완료 시각 기록
echo "완료 시각   : $(date '+%Y-%m-%d %H:%M:%S')" >> "$EXP_DIR/exp_info.txt"

echo ""
echo "============================================"
echo "  EE-ViT 학습 완료!"
echo "  실험 디렉토리 : experiments/$EXP_NAME"
echo "  체크포인트    :"
echo "    experiments/$EXP_NAME/train/ee_vit/checkpoints/best.pth"
echo "    experiments/$EXP_NAME/train/ee_vit/checkpoints/final.pth"
echo "  학습 로그     :"
echo "    experiments/$EXP_NAME/train/ee_vit/train_log.csv"
echo ""
echo "  ▶ 다음 단계: Orin으로 전송 (TRT 엔진 빌드)"
echo "    scp -r experiments/$EXP_NAME \\"
echo "        cap6@202.30.10.85:capstone-earlyexit/experiments/"
echo "============================================"
