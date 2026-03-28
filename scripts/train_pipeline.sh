#!/usr/bin/env bash
# ============================================================
#  train_pipeline.sh  —  RTX 5090 서버용 학습 + ONNX 변환 파이프라인
#
#  실행 순서:
#    1. EE  ResNet-18 학습     → experiments/train/ee_resnet18/run_*/
#    2. Plain ResNet-18 학습   → experiments/train/plain_resnet18/run_*/
#    3. VEE ResNet-18 학습     → experiments/train/vee_resnet18/run_*/
#    4. EE  모델 ONNX 변환     → experiments/onnx/ee_resnet18/
#    5. Plain 모델 ONNX 변환   → experiments/onnx/plain_resnet18/
#    6. VEE 모델 ONNX 변환     → experiments/onnx/vee_resnet18/
#
#  환경 변수로 개별 단계 스킵 가능:
#    SKIP_EE=1    bash scripts/train_pipeline.sh  # EE 학습 스킵
#    SKIP_PLAIN=1 bash scripts/train_pipeline.sh  # Plain 학습 스킵
#    SKIP_VEE=1   bash scripts/train_pipeline.sh  # VEE 학습 스킵
#    SKIP_EE=1 SKIP_VEE=1 bash scripts/train_pipeline.sh  # Plain만 실행
#
#  사용법:
#    cd <project_root>
#    bash scripts/train_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "============================================"
echo "  EE / Plain / VEE ResNet-18 Train Pipeline"
echo "  Project root : $PROJECT_ROOT"
echo "============================================"

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

# ── 3. VEE ResNet-18 학습 (Very Early Exit) ──────────────────
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
echo "[4/6] EE 모델 ONNX 변환 (both: full + seg)..."
python export/export_onnx.py --mode both
echo "[4/6] EE ONNX 변환 완료"

# ── 5. Plain 모델 ONNX 변환 ──────────────────────────────────
echo ""
echo "[5/6] Plain 모델 ONNX 변환..."
python export/export_onnx_plain.py
echo "[5/6] Plain ONNX 변환 완료"

# ── 6. VEE 모델 ONNX 변환 ────────────────────────────────────
echo ""
echo "[6/6] VEE 모델 ONNX 변환 (both: full + seg)..."
python export/export_onnx_vee.py --mode both
echo "[6/6] VEE ONNX 변환 완료"

echo ""
echo "============================================"
echo "  학습 + ONNX 변환 파이프라인 완료!"
echo "  ONNX 파일 위치:"
echo "    $PROJECT_ROOT/experiments/onnx/ee_resnet18/"
echo "    $PROJECT_ROOT/experiments/onnx/plain_resnet18/"
echo "    $PROJECT_ROOT/experiments/onnx/vee_resnet18/"
echo ""
echo "  다음 단계: Orin으로 ONNX 전송 후"
echo "    bash scripts/orin_pipeline.sh"
echo "============================================"
