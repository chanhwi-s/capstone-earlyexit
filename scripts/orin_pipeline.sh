#!/usr/bin/env bash
# ============================================================
#  orin_pipeline.sh  —  Jetson AGX Orin용 TRT 빌드 + 벤치마크 파이프라인
#
#  전제 조건:
#    - JetPack 6.x (TensorRT 10.x 포함)
#    - experiments/onnx/ 에 ONNX 파일 존재 (train_pipeline.sh 실행 후)
#
#  실행 순서:
#    1. EE  ONNX → TRT 엔진 빌드 (seg1/2/3, FP16)
#    2. Plain ONNX → TRT 엔진 빌드 (FP16)
#    3. VEE ONNX → TRT 엔진 빌드 (vee_seg1/vee_seg2, FP16)
#    4. 기존 벤치마크 (Plain vs EE 3-seg)
#    5. 4-Way 비교 벤치마크 (Plain / EE / VEE / Hybrid) + profiling_utils 지표
#    6. Hybrid runtime grid search (batch_size × timeout)
#    7. TRT threshold sweep
#    8. Hard sample 분석 (EE final exit → Plain 비교)
#    9. 엔진 레이어 fusion 분석
#
#  환경 변수:
#    SKIP_BUILD=1      TRT 빌드 스킵
#    SKIP_HARD=1       hard sample 분석 스킵
#    THRESHOLD=0.80    confidence threshold (기본값)
#    N_SAMPLES=1000    샘플 수
#
#  사용법:
#    cd <project_root>
#    bash scripts/orin_pipeline.sh
#    SKIP_BUILD=1 bash scripts/orin_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

EE_ONNX_DIR="$PROJECT_ROOT/experiments/onnx/ee_resnet18"
PLAIN_ONNX_DIR="$PROJECT_ROOT/experiments/onnx/plain_resnet18"
VEE_ONNX_DIR="$PROJECT_ROOT/experiments/onnx/vee_resnet18"

EE_ENGINE_DIR="$PROJECT_ROOT/experiments/trt_engines/ee_resnet18"
PLAIN_ENGINE_DIR="$PROJECT_ROOT/experiments/trt_engines/plain_resnet18"
VEE_ENGINE_DIR="$PROJECT_ROOT/experiments/trt_engines/vee_resnet18"

THRESHOLD="${THRESHOLD:-0.80}"
N_SAMPLES="${N_SAMPLES:-1000}"

echo "================================================"
echo "  EE / Plain / VEE / Hybrid  Orin Pipeline"
echo "  Project root : $PROJECT_ROOT"
echo "  Threshold    : $THRESHOLD"
echo "  Samples      : $N_SAMPLES"
echo "================================================"

mkdir -p "$EE_ENGINE_DIR" "$PLAIN_ENGINE_DIR" "$VEE_ENGINE_DIR"

# ── 1. EE TRT 엔진 빌드 ─────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo ""
    echo "[1/9] EE 세그먼트 엔진 빌드 (FP16)..."
    declare -A EE_ONNX_MAP=(
        [seg1]="seg1_stem_layer2.onnx"
        [seg2]="seg2_layer3.onnx"
        [seg3]="seg3_layer4.onnx"
    )
    for seg in seg1 seg2 seg3; do
        onnx_path="$EE_ONNX_DIR/${EE_ONNX_MAP[$seg]}"
        engine_path="$EE_ENGINE_DIR/${seg}.engine"
        if [[ ! -f "$onnx_path" ]]; then
            echo "  [SKIP] $onnx_path 없음"
            continue
        fi
        echo "  빌드: ${EE_ONNX_MAP[$seg]} → ${seg}.engine"
        trtexec --onnx="$onnx_path" --saveEngine="$engine_path" \
                --fp16 --workspace=1024 --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    done
    echo "[1/9] EE 엔진 빌드 완료"

    # ── 2. Plain TRT 엔진 빌드 ──
    echo ""
    echo "[2/9] Plain 엔진 빌드 (FP16)..."
    PLAIN_ONNX="$PLAIN_ONNX_DIR/plain_resnet18.onnx"
    if [[ -f "$PLAIN_ONNX" ]]; then
        trtexec --onnx="$PLAIN_ONNX" \
                --saveEngine="$PLAIN_ENGINE_DIR/plain_resnet18.engine" \
                --fp16 --workspace=1024 --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
        echo "[2/9] Plain 엔진 빌드 완료"
    else
        echo "[SKIP] $PLAIN_ONNX 없음"
    fi

    # ── 3. VEE TRT 엔진 빌드 ──
    echo ""
    echo "[3/9] VEE 세그먼트 엔진 빌드 (FP16)..."
    declare -A VEE_ONNX_MAP=(
        [vee_seg1]="vee_seg1_stem_layer1.onnx"
        [vee_seg2]="vee_seg2_layer2to4.onnx"
    )
    for seg in vee_seg1 vee_seg2; do
        onnx_path="$VEE_ONNX_DIR/${VEE_ONNX_MAP[$seg]}"
        engine_path="$VEE_ENGINE_DIR/${seg}.engine"
        if [[ ! -f "$onnx_path" ]]; then
            echo "  [SKIP] $onnx_path 없음"
            continue
        fi
        echo "  빌드: ${VEE_ONNX_MAP[$seg]} → ${seg}.engine"
        trtexec --onnx="$onnx_path" --saveEngine="$engine_path" \
                --fp16 --workspace=1024 --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    done
    echo "[3/9] VEE 엔진 빌드 완료"

else
    echo "[1-3/9] TRT 빌드 전체 스킵 (SKIP_BUILD=1)"
fi

cd "$SRC_DIR"

# ── 4. 기존 Plain vs EE 벤치마크 (benchmark_trt.py) ─────────
echo ""
echo "[4/9] Plain vs EE 3-Segment 벤치마크 (p50/p90/p95/p99 포함)..."
python benchmark_trt.py \
    --threshold "$THRESHOLD" \
    --num-samples "$N_SAMPLES"
echo "[4/9] 벤치마크 완료"

# ── 5. 4-Way 비교 벤치마크 (Plain / EE / VEE / Hybrid) ──────
echo ""
echo "[5/9] 4-Way 비교 벤치마크 (profiling_utils 지표 포함)..."
python benchmark_trt_hybrid.py \
    --threshold      "$THRESHOLD" \
    --num-samples    "$N_SAMPLES" \
    --hybrid-bs      8 \
    --hybrid-to-ms   10
echo "[5/9] 4-Way 벤치마크 완료"

# ── 6. Hybrid runtime grid search ────────────────────────────
echo ""
echo "[6/9] Hybrid grid search (batch_size × timeout)..."
python infer_trt_hybrid.py \
    --threshold    "$THRESHOLD" \
    --num-samples  "$N_SAMPLES" \
    --grid-search \
    --batch-sizes  1 2 4 8 16 32 \
    --timeouts     2 5 10 20 50
echo "[6/9] Grid search 완료"

# ── 7. TRT threshold sweep ────────────────────────────────────
echo ""
echo "[7/9] EE TRT threshold sweep (0.50~0.95)..."
python infer_trt.py \
    --seg1  "$EE_ENGINE_DIR/seg1.engine"  \
    --seg2  "$EE_ENGINE_DIR/seg2.engine"  \
    --seg3  "$EE_ENGINE_DIR/seg3.engine"  \
    --eval-cifar10 --sweep --num-samples "$N_SAMPLES"
echo "[7/9] Sweep 완료"

# ── 8. Hard sample 분석 ───────────────────────────────────────
if [[ "${SKIP_HARD:-0}" != "1" ]]; then
    echo ""
    echo "[8/9] Hard sample 분석 (EE exit3 → Plain 비교)..."
    python analyze_hard_samples.py --threshold "$THRESHOLD"
    echo "[8/9] Hard sample 분석 완료"
else
    echo "[8/9] Hard sample 분석 스킵 (SKIP_HARD=1)"
fi

# ── 9. 엔진 레이어 fusion 분석 ───────────────────────────────
echo ""
echo "[9/9] TRT 레이어 fusion 분석..."
python inspect_engines.py
echo "[9/9] 분석 완료"

echo ""
echo "================================================"
echo "  Orin 파이프라인 완료!"
echo "  결과 위치:"
echo "    $PROJECT_ROOT/experiments/eval/benchmark/"
echo "    $PROJECT_ROOT/experiments/eval/benchmark_comparison/"
echo "    $PROJECT_ROOT/experiments/eval/hybrid_runtime/"
echo "    $PROJECT_ROOT/experiments/eval/trt_sweep/"
echo "    $PROJECT_ROOT/experiments/eval/hard_sample_analysis/"
echo "    $PROJECT_ROOT/experiments/eval/engine_inspect/"
echo "================================================"
