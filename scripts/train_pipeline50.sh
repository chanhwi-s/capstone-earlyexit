#!/usr/bin/env bash
# ============================================================
#  train_pipeline50.sh  —  RTX 5090 서버용 ResNet-50 학습 + ONNX 변환
#
#  실행 순서:
#    1. Plain ResNet-50 학습   → {EXP_DIR}/train/plain_resnet50/
#    2. EE    ResNet-50 학습   → {EXP_DIR}/train/ee_resnet50/
#    3. Plain ResNet-50 ONNX   → {EXP_DIR}/onnx/plain_resnet50/
#    4. EE    ResNet-50 ONNX   → {EXP_DIR}/onnx/ee_resnet50/  (4-segment)
#
#  EXP_DIR 결정 우선순위:
#    1. 환경변수 EXP_DIR 직접 지정
#    2. 없으면 experiments/ 내 가장 최신 exp_* 자동 선택
#    3. exp_* 없으면 새 디렉토리 자동 생성
#
#  주요 환경 변수:
#    DATASET=imagenet           cifar10 | imagenet (기본: imagenet)
#    DATA_ROOT=<path>           데이터 루트 경로 (기본: configs/train.yaml 값)
#    EPOCHS=<N>                 학습 에포크 수 오버라이드
#    BATCH_SIZE=<N>             배치 크기 오버라이드
#    LR=<float>                 초기 학습률 오버라이드
#    W1/W2/W3/W4=<float>        EE50 loss 가중치 오버라이드
#    SKIP_PLAIN=1               Plain 학습 스킵
#    SKIP_EE=1                  EE 학습 스킵
#    SKIP_EXPORT=1              ONNX 변환 전체 스킵
#    EXP_DIR=<path>             실험 디렉토리 직접 지정
#
#  사용법:
#    # 기본 (ImageNet, 최신 exp_* 디렉토리에 저장)
#    nohup bash scripts/train_pipeline50.sh > train50.log 2>&1 &
#    tail -f train50.log
#
#    # CIFAR-10 테스트
#    DATASET=cifar10 bash scripts/train_pipeline50.sh
#
#    # EE만 학습 스킵 없이, 특정 디렉토리에 저장
#    EXP_DIR=experiments/exp_20260401_120000 DATASET=imagenet \
#        nohup bash scripts/train_pipeline50.sh > train50.log 2>&1 &
#
#  Orin으로 전송:
#    scp -r experiments/$EXP_NAME cap6@202.30.10.85:capstone-earlyexit/experiments/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 데이터셋 설정 ────────────────────────────────────────────
DATASET="${DATASET:-imagenet}"
DATASET="${DATASET,,}"

if [[ "$DATASET" != "cifar10" && "$DATASET" != "imagenet" ]]; then
    echo "[ERROR] DATASET은 cifar10 또는 imagenet이어야 합니다."
    exit 1
fi

if [[ "$DATASET" == "imagenet" ]]; then
    INPUT_H=224; INPUT_W=224
else
    INPUT_H=32;  INPUT_W=32
fi

# ── CLI 오버라이드 인자 조립 ─────────────────────────────────
EXTRA_ARGS="--dataset $DATASET"
[[ -n "${DATA_ROOT:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --data-root $DATA_ROOT"
[[ -n "${EPOCHS:-}"     ]] && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[[ -n "${BATCH_SIZE:-}" ]] && EXTRA_ARGS="$EXTRA_ARGS --batch-size $BATCH_SIZE"
[[ -n "${LR:-}"         ]] && EXTRA_ARGS="$EXTRA_ARGS --lr $LR"

EE_EXTRA_ARGS="$EXTRA_ARGS"
[[ -n "${W1:-}" ]] && EE_EXTRA_ARGS="$EE_EXTRA_ARGS --w1 $W1"
[[ -n "${W2:-}" ]] && EE_EXTRA_ARGS="$EE_EXTRA_ARGS --w2 $W2"
[[ -n "${W3:-}" ]] && EE_EXTRA_ARGS="$EE_EXTRA_ARGS --w3 $W3"
[[ -n "${W4:-}" ]] && EE_EXTRA_ARGS="$EE_EXTRA_ARGS --w4 $W4"

# ── EXP_DIR 결정 ─────────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    if [[ "$EXP_DIR" != /* ]]; then
        EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    fi
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$LATEST_EXP" ]]; then
        export EXP_DIR="$LATEST_EXP"
    else
        export EXP_DIR="$PROJECT_ROOT/experiments/exp_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$EXP_DIR"
    fi
fi
export EXP_NAME="$(basename "$EXP_DIR")"

echo "============================================"
echo "  Plain / EE ResNet-50 Train Pipeline"
echo "  Project root  : $PROJECT_ROOT"
echo "  실험 디렉토리 : $EXP_NAME"
echo "  Dataset       : $DATASET"
echo "  Input size    : ${INPUT_H}×${INPUT_W}"
[[ -n "${EPOCHS:-}"     ]] && echo "  Epochs        : $EPOCHS (override)"
[[ -n "${BATCH_SIZE:-}" ]] && echo "  Batch size    : $BATCH_SIZE (override)"
[[ -n "${LR:-}"         ]] && echo "  LR            : $LR (override)"
echo "  시작 시각     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# 실험 정보 기록 (append 모드 — 기존 ResNet-18 내용 보존)
cat >> "$EXP_DIR/exp_info.txt" <<EOF

[ResNet-50 학습 시작]
시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')
Dataset    : $DATASET
호스트     : $(hostname)
EOF

cd "$SRC_DIR"

# ── 1. Plain ResNet-50 학습 ──────────────────────────────────
if [[ "${SKIP_PLAIN:-0}" != "1" ]]; then
    echo ""
    echo "[1/4] Plain ResNet-50 학습 시작...  ($EXTRA_ARGS)"
    python train/train_plain50.py $EXTRA_ARGS
    echo "[1/4] Plain ResNet-50 학습 완료  →  $(date '+%H:%M:%S')"
else
    echo "[1/4] Plain ResNet-50 학습 스킵 (SKIP_PLAIN=1)"
fi

# ── 2. EE ResNet-50 학습 ─────────────────────────────────────
if [[ "${SKIP_EE:-0}" != "1" ]]; then
    echo ""
    echo "[2/4] EE ResNet-50 학습 시작 (4 exits)...  ($EE_EXTRA_ARGS)"
    python train/train_ee50.py $EE_EXTRA_ARGS
    echo "[2/4] EE ResNet-50 학습 완료  →  $(date '+%H:%M:%S')"
else
    echo "[2/4] EE ResNet-50 학습 스킵 (SKIP_EE=1)"
fi

# ── ONNX 변환 ────────────────────────────────────────────────
if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then

    # ── 3. Plain ResNet-50 ONNX ──
    echo ""
    echo "[3/4] Plain ResNet-50 ONNX 변환...  (dataset=$DATASET, input=${INPUT_H}×${INPUT_W})"
    python export/export_onnx_plain50.py \
        --dataset "$DATASET" \
        --input-size "$INPUT_H" "$INPUT_W"
    echo "[3/4] Plain ResNet-50 ONNX 완료"

    # ── 4. EE ResNet-50 ONNX (4-segment) ──
    echo ""
    echo "[4/4] EE ResNet-50 ONNX 변환 (4-segment)...  (dataset=$DATASET, input=${INPUT_H}×${INPUT_W})"
    python export/export_onnx_ee50.py \
        --dataset "$DATASET" \
        --input-size "$INPUT_H" "$INPUT_W"
    echo "[4/4] EE ResNet-50 ONNX 완료"

else
    echo "[3-4/4] ONNX 변환 전체 스킵 (SKIP_EXPORT=1)"
fi

# 완료 기록
echo "완료 시각  : $(date '+%Y-%m-%d %H:%M:%S')" >> "$EXP_DIR/exp_info.txt"

echo ""
echo "============================================"
echo "  ResNet-50 학습 + ONNX 변환 완료!"
echo "  실험 디렉토리 : $EXP_NAME"
echo "  ONNX 파일 위치:"
echo "    experiments/$EXP_NAME/onnx/plain_resnet50/"
echo "    experiments/$EXP_NAME/onnx/ee_resnet50/"
echo ""
echo "  ▶ Orin으로 전송:"
echo "    scp -r experiments/$EXP_NAME \\"
echo "        cap6@202.30.10.85:capstone-earlyexit/experiments/"
echo ""
echo "  ▶ Orin에서 TRT 빌드:"
echo "    DATASET=$DATASET bash scripts/orin_pipeline.sh"
echo "============================================"
