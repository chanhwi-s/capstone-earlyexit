#!/usr/bin/env bash
# ============================================================
#  orin_pipeline.sh  —  Jetson AGX Orin용 TRT 빌드 + 벤치마크 파이프라인
#
#  전제 조건:
#    - JetPack 6.x (TensorRT 10.x 포함)
#    - experiments/onnx/ 에 ONNX 파일이 존재 (train_pipeline.sh 이후)
#
#  실행 순서:
#    1. EE ONNX → TRT 엔진 빌드  (seg1/seg2/seg3, FP16)
#    2. Plain ONNX → TRT 엔진 빌드 (FP16)
#    3. 벤치마크 (Plain vs EE)
#    4. TRT sweep (threshold 0.50~0.95)
#    5. 엔진 레이어 fusion 분석
#
#  사용법:
#    cd <project_root>
#    bash scripts/orin_pipeline.sh
#
#    # 엔진 빌드 스킵하고 벤치마크만 다시 실행:
#    SKIP_BUILD=1 bash scripts/orin_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

EE_ONNX_DIR="$PROJECT_ROOT/experiments/onnx/ee_resnet18"
PLAIN_ONNX_DIR="$PROJECT_ROOT/experiments/onnx/plain_resnet18"
EE_ENGINE_DIR="$PROJECT_ROOT/experiments/trt_engines/ee_resnet18"
PLAIN_ENGINE_DIR="$PROJECT_ROOT/experiments/trt_engines/plain_resnet18"

echo "=========================================="
echo "  Early-Exit ResNet-18  Orin Pipeline"
echo "  Project root : $PROJECT_ROOT"
echo "=========================================="

# ── 디렉토리 확인 ─────────────────────────────────────────────
if [[ ! -d "$EE_ONNX_DIR" ]] || [[ ! -d "$PLAIN_ONNX_DIR" ]]; then
    echo "[ERROR] ONNX 파일 디렉토리 없음."
    echo "  먼저 서버에서 train_pipeline.sh 를 실행하고"
    echo "  ONNX 파일을 이 디렉토리로 복사하세요:"
    echo "    $EE_ONNX_DIR/"
    echo "    $PLAIN_ONNX_DIR/"
    exit 1
fi

mkdir -p "$EE_ENGINE_DIR" "$PLAIN_ENGINE_DIR"

# ── 1. EE TRT 엔진 빌드 ───────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo ""
    echo "[1/5] EE 세그먼트 엔진 빌드 (FP16)..."

    for seg in seg1 seg2 seg3; do
        # ONNX 파일 이름 매핑
        case $seg in
            seg1) onnx_name="seg1_stem_layer2.onnx" ;;
            seg2) onnx_name="seg2_layer3.onnx"      ;;
            seg3) onnx_name="seg3_layer4.onnx"      ;;
        esac
        onnx_path="$EE_ONNX_DIR/$onnx_name"
        engine_path="$EE_ENGINE_DIR/${seg}.engine"

        if [[ ! -f "$onnx_path" ]]; then
            echo "[SKIP] $onnx_path 없음"
            continue
        fi

        echo "  빌드: $onnx_name → ${seg}.engine"
        trtexec \
            --onnx="$onnx_path" \
            --saveEngine="$engine_path" \
            --fp16 \
            --workspace=1024 \
            --iterations=100 \
            --warmUp=500 \
            --avgRuns=100 \
            2>&1 | tail -5
    done
    echo "[1/5] EE 엔진 빌드 완료"

    # ── 2. Plain TRT 엔진 빌드 ───────────────────────────────
    echo ""
    echo "[2/5] Plain 엔진 빌드 (FP16)..."
    PLAIN_ONNX="$PLAIN_ONNX_DIR/plain_resnet18.onnx"
    PLAIN_ENGINE="$PLAIN_ENGINE_DIR/plain_resnet18.engine"

    if [[ -f "$PLAIN_ONNX" ]]; then
        trtexec \
            --onnx="$PLAIN_ONNX" \
            --saveEngine="$PLAIN_ENGINE" \
            --fp16 \
            --workspace=1024 \
            --iterations=100 \
            --warmUp=500 \
            --avgRuns=100 \
            2>&1 | tail -5
        echo "[2/5] Plain 엔진 빌드 완료"
    else
        echo "[SKIP] $PLAIN_ONNX 없음"
    fi
else
    echo "[1-2/5] TRT 빌드 스킵 (SKIP_BUILD=1)"
fi

cd "$SRC_DIR"

# ── 3. 벤치마크 (Plain vs EE) ────────────────────────────────
echo ""
echo "[3/5] 벤치마크 실행 (threshold=0.80, n=1000)..."
python benchmark_trt.py \
    --threshold 0.80 \
    --num-samples 1000
echo "[3/5] 벤치마크 완료"

# ── 4. TRT sweep ─────────────────────────────────────────────
echo ""
echo "[4/5] TRT threshold sweep (0.50~0.95)..."
python infer_trt.py \
    --seg1  "$EE_ENGINE_DIR/seg1.engine"  \
    --seg2  "$EE_ENGINE_DIR/seg2.engine"  \
    --seg3  "$EE_ENGINE_DIR/seg3.engine"  \
    --eval-cifar10 --sweep --num-samples 1000
echo "[4/5] Sweep 완료"

# ── 5. 엔진 레이어 fusion 분석 ───────────────────────────────
echo ""
echo "[5/5] 레이어 fusion 분석..."
python inspect_engines.py
echo "[5/5] 분석 완료"

echo ""
echo "=========================================="
echo "  Orin 파이프라인 완료!"
echo "  결과 위치:"
echo "    벤치마크  : $PROJECT_ROOT/experiments/eval/benchmark/"
echo "    TRT sweep : $PROJECT_ROOT/experiments/eval/trt_sweep/"
echo "    Engine 분석: $PROJECT_ROOT/experiments/eval/engine_inspect/"
echo "=========================================="
