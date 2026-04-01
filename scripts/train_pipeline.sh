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
#  환경 변수:
#    SKIP_EE=1    bash scripts/train_pipeline.sh  # EE 학습 스킵
#    SKIP_PLAIN=1 bash scripts/train_pipeline.sh  # Plain 학습 스킵
#    SKIP_VEE=1   bash scripts/train_pipeline.sh  # VEE 학습 스킵
#    EXP_NAME=exp_20260401_120000  # 디렉토리 이름 직접 지정 (기본: 자동 생성)
#
#  사용법:
#    cd <project_root>
#    bash scripts/train_pipeline.sh
#
#  Orin으로 전송할 때:
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
echo "  EE / Plain / VEE ResNet-18 Train Pipeline"
echo "  Project root : $PROJECT_ROOT"
echo "  실험 디렉토리 : experiments/$EXP_NAME"
echo "============================================"

# 실험 정보 기록 (시작 시각, 호스트 등)
cat > "$EXP_DIR/exp_info.txt" <<EOF
EXP_NAME   : $EXP_NAME
EXP_DIR    : $EXP_DIR
시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')
호스트     : $(hostname)
Python     : $(python3 --version 2>&1 || echo "N/A")
EOF
echo "  실험 정보 저장: $EXP_DIR/exp_info.txt"

cd "$SRC_DIR"

# ── 1. EE ResNet-18 학습 ─────────────────────────────────────
if [[ "${SKIP_EE:-0}" != "1" ]]; then
    echo ""
    echo "[1/6] EE ResNet-18 학습 시작..."
    python train/train.py
    echo "[1/6] EE 학습 완료"
else
    echo "[1/6] EE 학습 스킵 (SKIP_EE=1)"
fi

# ── 2. Plain ResNet-18 학습 ──────────────────────────────────
if [[ "${SKIP_PLAIN:-0}" != "1" ]]; then
    echo ""
    echo "[2/6] Plain ResNet-18 학습 시작..."
    python train/train_plain.py
    echo "[2/6] Plain 학습 완료"
else
    echo "[2/6] Plain 학습 스킵 (SKIP_PLAIN=1)"
fi

# ── 3. VEE ResNet-18 학습 ────────────────────────────────────
if [[ "${SKIP_VEE:-0}" != "1" ]]; then
    echo ""
    echo "[3/6] VEE ResNet-18 학습 시작 (exit @ layer1)..."
    python train/train_vee.py
    echo "[3/6] VEE 학습 완료"
else
    echo "[3/6] VEE 학습 스킵 (SKIP_VEE=1)"
fi

# ── 4. EE 모델 ONNX 변환 ─────────────────────────────────────
echo ""
echo "[4/6] EE 모델 ONNX 변환..."
python export/export_onnx.py --mode both
echo "[4/6] EE ONNX 변환 완료"

# ── 5. Plain 모델 ONNX 변환 ──────────────────────────────────
echo ""
echo "[5/6] Plain 모델 ONNX 변환..."
python export/export_onnx_plain.py
echo "[5/6] Plain ONNX 변환 완료"

# ── 6. VEE 모델 ONNX 변환 ────────────────────────────────────
echo ""
echo "[6/6] VEE 모델 ONNX 변환..."
python export/export_onnx_vee.py --mode both
echo "[6/6] VEE ONNX 변환 완료"

# 완료 시각 기록
echo "완료 시각  : $(date '+%Y-%m-%d %H:%M:%S')" >> "$EXP_DIR/exp_info.txt"

echo ""
echo "============================================"
echo "  학습 + ONNX 변환 파이프라인 완료!"
echo "  실험 디렉토리 : experiments/$EXP_NAME"
echo "  ONNX 파일 위치:"
echo "    experiments/$EXP_NAME/onnx/ee_resnet18/"
echo "    experiments/$EXP_NAME/onnx/plain_resnet18/"
echo "    experiments/$EXP_NAME/onnx/vee_resnet18/"
echo ""
echo "  ▶ 다음 단계: Orin으로 실험 디렉토리 전송"
echo "    scp -r experiments/$EXP_NAME \\"
echo "        cap6@202.30.10.85:capstone-earlyexit/experiments/"
echo ""
echo "  ▶ Orin에서 파이프라인 실행"
echo "    bash scripts/orin_pipeline.sh"
echo "    (최신 실험 디렉토리 자동 감지)"
echo "============================================"
