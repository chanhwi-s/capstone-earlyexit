#!/usr/bin/env bash
# ============================================================
#  train_pipeline.sh  —  RTX 5090 서버용 학습 + ONNX 변환 파이프라인
#
#  실행 순서:
#    0. 타임스탬프 기반 실험 디렉토리 생성 (experiments/exp_YYYYMMDD_HHMMSS/)
#    1. EE  ResNet-18 학습     → {EXP_DIR}/train/ee_resnet18/
#    2. Plain ResNet-18 학습   → {EXP_DIR}/train/plain_resnet18/
#    3. VEE ResNet-18 학습     → {EXP_DIR}/train/vee_resnet18/
#    4. EE  모델 ONNX 변환     → {EXP_DIR}/onnx/ee_resnet18/
#    5. Plain 모델 ONNX 변환   → {EXP_DIR}/onnx/plain_resnet18/
#    6. VEE 모델 ONNX 변환     → {EXP_DIR}/onnx/vee_resnet18/
#
#  데이터셋 선택:
#    DATASET=cifar10    (기본값) CIFAR-10  — 32×32, 10 classes, 100 epochs
#    DATASET=imagenet           ImageNet  — 224×224, 1000 classes, 90 epochs
#                               데이터 경로: {DATA_ROOT}/imagenet/train, /imagenet/val
#
#  주요 환경 변수:
#    DATASET=cifar10|imagenet   데이터셋 선택 (기본: cifar10)
#    DATA_ROOT=<path>           데이터 루트 경로 (기본: configs/train.yaml 값)
#    EPOCHS=<N>                 학습 에포크 수 오버라이드
#    BATCH_SIZE=<N>             배치 크기 오버라이드
#    LR=<float>                 초기 학습률 오버라이드
#    SKIP_EE=1                  EE 학습 스킵
#    SKIP_PLAIN=1               Plain 학습 스킵
#    SKIP_VEE=1                 VEE 학습 스킵
#    SKIP_EXPORT=1              ONNX 변환 전체 스킵
#    EXP_NAME=exp_20260401_...  실험 디렉토리 이름 직접 지정 (기본: 자동 생성)
#
#  사용법:
#    cd <project_root>
#    bash scripts/train_pipeline.sh                              # CIFAR-10 기본
#    DATASET=imagenet bash scripts/train_pipeline.sh             # ImageNet
#    DATASET=imagenet DATA_ROOT=/data bash scripts/train_pipeline.sh
#    DATASET=imagenet EPOCHS=60 BATCH_SIZE=128 bash scripts/train_pipeline.sh
#    SKIP_PLAIN=1 SKIP_VEE=1 bash scripts/train_pipeline.sh     # EE만 학습
#
#  Orin으로 전송할 때:
#    scp -r experiments/$EXP_NAME cap6@202.30.10.85:capstone-earlyexit/experiments/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 데이터셋 설정 ────────────────────────────────────────────
DATASET="${DATASET:-cifar10}"
DATASET="${DATASET,,}"   # 소문자 변환

if [[ "$DATASET" != "cifar10" && "$DATASET" != "imagenet" ]]; then
    echo "[ERROR] DATASET은 cifar10 또는 imagenet이어야 합니다. (입력값: $DATASET)"
    exit 1
fi

# ── 입력 크기 결정 (ONNX export용) ──────────────────────────
if [[ "$DATASET" == "imagenet" ]]; then
    INPUT_H=224
    INPUT_W=224
else
    INPUT_H=32
    INPUT_W=32
fi

# ── CLI 오버라이드 인자 조립 ─────────────────────────────────
# 환경변수가 설정된 항목만 인자로 추가
EXTRA_ARGS="--dataset $DATASET"
[[ -n "${DATA_ROOT:-}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --data-root $DATA_ROOT"
[[ -n "${EPOCHS:-}"      ]] && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[[ -n "${BATCH_SIZE:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"
[[ -n "${LR:-}"          ]] && EXTRA_ARGS="$EXTRA_ARGS --lr $LR"

# ── 0. 실험 디렉토리 생성 ────────────────────────────────────
EXP_NAME="${EXP_NAME:-exp_$(date +%Y%m%d_%H%M%S)}"
export EXP_DIR="$PROJECT_ROOT/experiments/$EXP_NAME"
export EXP_NAME

mkdir -p "$EXP_DIR"
echo "============================================"
echo "  EE / Plain / VEE ResNet-18 Train Pipeline"
echo "  Project root  : $PROJECT_ROOT"
echo "  실험 디렉토리 : experiments/$EXP_NAME"
echo "  Dataset       : $DATASET"
echo "  Input size    : ${INPUT_H}×${INPUT_W}"
if [[ -n "${EPOCHS:-}"     ]]; then echo "  Epochs        : $EPOCHS (override)"; fi
if [[ -n "${BATCH_SIZE:-}" ]]; then echo "  Batch size    : $BATCH_SIZE (override)"; fi
if [[ -n "${LR:-}"         ]]; then echo "  LR            : $LR (override)"; fi
echo "============================================"

# 실험 정보 기록
cat > "$EXP_DIR/exp_info.txt" <<EOF
EXP_NAME   : $EXP_NAME
EXP_DIR    : $EXP_DIR
Dataset    : $DATASET
Input size : ${INPUT_H}×${INPUT_W}
시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')
호스트     : $(hostname)
Python     : $(python3 --version 2>&1 || echo "N/A")
EOF
echo "  실험 정보 저장: $EXP_DIR/exp_info.txt"

cd "$SRC_DIR"

# ── 1. EE ResNet-18 학습 ─────────────────────────────────────
if [[ "${SKIP_EE:-0}" != "1" ]]; then
    echo ""
    echo "[1/6] EE ResNet-18 학습 시작...  ($EXTRA_ARGS)"
    python train/train.py $EXTRA_ARGS
    echo "[1/6] EE 학습 완료"
else
    echo "[1/6] EE 학습 스킵 (SKIP_EE=1)"
fi

# ── 2. Plain ResNet-18 학습 ──────────────────────────────────
if [[ "${SKIP_PLAIN:-0}" != "1" ]]; then
    echo ""
    echo "[2/6] Plain ResNet-18 학습 시작...  ($EXTRA_ARGS)"
    python train/train_plain.py $EXTRA_ARGS
    echo "[2/6] Plain 학습 완료"
else
    echo "[2/6] Plain 학습 스킵 (SKIP_PLAIN=1)"
fi

# ── 3. VEE ResNet-18 학습 ────────────────────────────────────
if [[ "${SKIP_VEE:-0}" != "1" ]]; then
    echo ""
    echo "[3/6] VEE ResNet-18 학습 시작 (exit @ layer1)...  ($EXTRA_ARGS)"
    python train/train_vee.py $EXTRA_ARGS
    echo "[3/6] VEE 학습 완료"
else
    echo "[3/6] VEE 학습 스킵 (SKIP_VEE=1)"
fi

# ── ONNX 변환 ────────────────────────────────────────────────
if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then

    # ── 4. EE 모델 ONNX 변환 ──
    echo ""
    echo "[4/6] EE 모델 ONNX 변환...  (dataset=$DATASET, input=${INPUT_H}×${INPUT_W})"
    python export/export_onnx.py \
        --mode both \
        --dataset "$DATASET" \
        --input-size "$INPUT_H" "$INPUT_W"
    echo "[4/6] EE ONNX 변환 완료"

    # ── 5. Plain 모델 ONNX 변환 ──
    echo ""
    echo "[5/6] Plain 모델 ONNX 변환..."
    python export/export_onnx_plain.py \
        --dataset "$DATASET" \
        --input-size "$INPUT_H" "$INPUT_W"
    echo "[5/6] Plain ONNX 변환 완료"

    # ── 6. VEE 모델 ONNX 변환 ──
    echo ""
    echo "[6/6] VEE 모델 ONNX 변환..."
    python export/export_onnx_vee.py \
        --mode both \
        --dataset "$DATASET" \
        --input-size "$INPUT_H" "$INPUT_W"
    echo "[6/6] VEE ONNX 변환 완료"

else
    echo "[4-6/6] ONNX 변환 전체 스킵 (SKIP_EXPORT=1)"
fi

# 완료 시각 기록
echo "완료 시각  : $(date '+%Y-%m-%d %H:%M:%S')" >> "$EXP_DIR/exp_info.txt"

echo ""
echo "============================================"
echo "  학습 + ONNX 변환 파이프라인 완료!"
echo "  실험 디렉토리 : experiments/$EXP_NAME"
echo "  Dataset       : $DATASET"
echo "  ONNX 파일 위치:"
echo "    experiments/$EXP_NAME/onnx/ee_resnet18/"
echo "    experiments/$EXP_NAME/onnx/plain_resnet18/"
echo "    experiments/$EXP_NAME/onnx/vee_resnet18/"
echo ""
echo "  ▶ 다음 단계: Orin으로 실험 디렉토리 전송"
echo "    scp -r experiments/$EXP_NAME \\"
echo "        cap6@202.30.10.85:capstone-earlyexit/experiments/"
echo ""
echo "  ▶ Orin에서 Step 1 (threshold sweep):"
echo "    bash scripts/step1_sweep.sh <N>"
echo "============================================"
