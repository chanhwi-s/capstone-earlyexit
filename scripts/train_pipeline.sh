#!/usr/bin/env bash
# ============================================================
#  train_pipeline.sh  —  RTX 5090 서버용 학습 + ONNX 변환 파이프라인
#
#  실행 순서:
#    1. EE ResNet-18 학습     → experiments/train/ee_resnet18/run_*/
#    2. Plain ResNet-18 학습  → experiments/train/plain_resnet18/run_*/
#    3. EE 모델 ONNX 변환     → experiments/onnx/ee_resnet18/
#    4. Plain 모델 ONNX 변환  → experiments/onnx/plain_resnet18/
#
#  사용법:
#    cd <project_root>
#    bash scripts/train_pipeline.sh
#
#    # EE 학습만 건너뛰고 Plain + ONNX만 실행하려면:
#    SKIP_EE=1 bash scripts/train_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

echo "=========================================="
echo "  Early-Exit ResNet-18  Train Pipeline"
echo "  Project root : $PROJECT_ROOT"
echo "=========================================="

cd "$SRC_DIR"

# ── 1. EE ResNet-18 학습 ──────────────────────────────────────
if [[ "${SKIP_EE:-0}" != "1" ]]; then
    echo ""
    echo "[1/4] EE ResNet-18 학습 시작..."
    python train.py
    echo "[1/4] EE 학습 완료"
else
    echo "[1/4] EE 학습 스킵 (SKIP_EE=1)"
fi

# ── 2. Plain ResNet-18 학습 ───────────────────────────────────
if [[ "${SKIP_PLAIN:-0}" != "1" ]]; then
    echo ""
    echo "[2/4] Plain ResNet-18 학습 시작..."
    python train_plain.py
    echo "[2/4] Plain 학습 완료"
else
    echo "[2/4] Plain 학습 스킵 (SKIP_PLAIN=1)"
fi

# ── 3. EE 모델 ONNX 변환 (seg 모드) ──────────────────────────
echo ""
echo "[3/4] EE 모델 ONNX 변환 (both 모드: full + seg)..."
python export_onnx.py --mode both
echo "[3/4] EE ONNX 변환 완료"

# ── 4. Plain 모델 ONNX 변환 ──────────────────────────────────
echo ""
echo "[4/4] Plain 모델 ONNX 변환..."
python export_onnx_plain.py
echo "[4/4] Plain ONNX 변환 완료"

echo ""
echo "=========================================="
echo "  파이프라인 완료!"
echo "  ONNX 파일 위치:"
echo "    $PROJECT_ROOT/experiments/onnx/ee_resnet18/"
echo "    $PROJECT_ROOT/experiments/onnx/plain_resnet18/"
echo ""
echo "  다음 단계: Orin으로 ONNX 파일 전송 후"
echo "    bash scripts/orin_pipeline.sh"
echo "=========================================="
