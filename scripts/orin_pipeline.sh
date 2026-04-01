#!/usr/bin/env bash
# ============================================================
#  orin_pipeline.sh  —  Jetson AGX Orin용 TRT 빌드 + 벤치마크 파이프라인
#
#  전제 조건:
#    - JetPack 6.x (TensorRT 10.x 포함)
#    - 5090에서 실험 디렉토리 전송 완료:
#        scp -r experiments/exp_YYYYMMDD_HHMMSS \
#            cap6@202.30.10.85:capstone-earlyexit/experiments/
#
#  실행 순서:
#    0. 최신 exp_* 디렉토리 자동 감지 (또는 EXP_DIR 직접 지정)
#    1. EE  ONNX → TRT 엔진 빌드 (seg1/2/3, FP16)
#    2. Plain ONNX → TRT 엔진 빌드 (FP16, 동적 배치 1~32)
#    3. VEE ONNX → TRT 엔진 빌드 (vee_seg1/vee_seg2, FP16)
#    4. 기존 벤치마크 (Plain vs EE 3-seg)
#    5. 4-Way 비교 벤치마크 (Plain / EE / VEE / Hybrid) + profiling_utils 지표
#    6. Hybrid runtime grid search (batch_size × timeout)
#    7. TRT threshold sweep (EE + VEE)
#    8. 엔진 레이어 fusion 분석
#
#  NOTE: analyze_hard_samples.py 는 PyTorch 체크포인트가 필요하므로
#        5090 서버에서 별도 실행:
#          cd src && python analysis/analyze_hard_samples.py --threshold 0.80
#
#  환경 변수:
#    EXP_DIR=<path>    특정 실험 디렉토리 지정 (기본: 최신 exp_* 자동 감지)
#    SKIP_BUILD=1      TRT 빌드 스킵 (엔진이 이미 존재할 때)
#    THRESHOLD=0.80    confidence threshold (기본값)
#    N_SAMPLES=1000    샘플 수
#
#  사용법:
#    cd <project_root>
#    bash scripts/orin_pipeline.sh                          # 최신 실험 자동 감지
#    SKIP_BUILD=1 bash scripts/orin_pipeline.sh             # 빌드 스킵
#    EXP_DIR=experiments/exp_20260401_120000 bash scripts/orin_pipeline.sh  # 직접 지정
#    SKIP_BUILD=1 THRESHOLD=0.85 N_SAMPLES=2000 bash scripts/orin_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 0. 실험 디렉토리 결정 ────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    # 환경변수로 직접 지정된 경우 (절대경로 또는 project_root 상대경로 모두 허용)
    if [[ "$EXP_DIR" != /* ]]; then
        EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    fi
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    # experiments/ 내 가장 최신 exp_* 디렉토리 자동 선택
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$LATEST_EXP" ]]; then
        echo "[ERROR] experiments/ 내 exp_* 디렉토리가 없습니다."
        echo "        5090에서 먼저 전송하세요:"
        echo "        scp -r experiments/exp_YYYYMMDD_HHMMSS \\"
        echo "            cap6@202.30.10.85:capstone-earlyexit/experiments/"
        exit 1
    fi
    export EXP_DIR="$LATEST_EXP"
fi

export EXP_NAME="$(basename "$EXP_DIR")"

EE_ONNX_DIR="$EXP_DIR/onnx/ee_resnet18"
PLAIN_ONNX_DIR="$EXP_DIR/onnx/plain_resnet18"
VEE_ONNX_DIR="$EXP_DIR/onnx/vee_resnet18"

EE_ENGINE_DIR="$EXP_DIR/trt_engines/ee_resnet18"
PLAIN_ENGINE_DIR="$EXP_DIR/trt_engines/plain_resnet18"
VEE_ENGINE_DIR="$EXP_DIR/trt_engines/vee_resnet18"

THRESHOLD="${THRESHOLD:-0.80}"
N_SAMPLES="${N_SAMPLES:-1000}"

echo "================================================"
echo "  EE / Plain / VEE / Hybrid  Orin Pipeline"
echo "  Project root  : $PROJECT_ROOT"
echo "  실험 디렉토리 : $EXP_NAME"
echo "  Threshold     : $THRESHOLD"
echo "  Samples       : $N_SAMPLES"
echo "================================================"

mkdir -p "$EE_ENGINE_DIR" "$PLAIN_ENGINE_DIR" "$VEE_ENGINE_DIR"

# ── 1. EE TRT 엔진 빌드 ─────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo ""
    echo "[1/7] EE 세그먼트 엔진 빌드 (FP16)..."
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
                --fp16 --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    done
    echo "[1/7] EE 엔진 빌드 완료"

    # ── 2. Plain TRT 엔진 빌드 ──
    echo ""
    echo "[2/7] Plain 엔진 빌드 (FP16, 동적 배치 1~32)..."
    PLAIN_ONNX="$PLAIN_ONNX_DIR/plain_resnet18.onnx"
    if [[ -f "$PLAIN_ONNX" ]]; then
        trtexec --onnx="$PLAIN_ONNX" \
                --saveEngine="$PLAIN_ENGINE_DIR/plain_resnet18.engine" \
                --fp16 \
                --minShapes=input:1x3x32x32 \
                --optShapes=input:8x3x32x32 \
                --maxShapes=input:32x3x32x32 \
                --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
        echo "[2/7] Plain 엔진 빌드 완료"
    else
        echo "[SKIP] $PLAIN_ONNX 없음"
    fi

    # ── 3. VEE TRT 엔진 빌드 ──
    echo ""
    echo "[3/7] VEE 세그먼트 엔진 빌드 (FP16)..."
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
                --fp16 --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    done
    echo "[3/7] VEE 엔진 빌드 완료"

else
    echo "[1-3/7] TRT 빌드 전체 스킵 (SKIP_BUILD=1)"
fi

cd "$SRC_DIR"

# ── 4. 4-Way 비교 벤치마크 (Plain / EE / VEE / Hybrid) ──────
echo ""
echo "[4/7] 4-Way 비교 벤치마크 (latency dist + power + energy 포함)..."
python benchmark/benchmark_trt_hybrid.py \
    --threshold      "$THRESHOLD" \
    --num-samples    "$N_SAMPLES" \
    --hybrid-bs      8 \
    --hybrid-to-ms   10
echo "[5/7] 4-Way 벤치마크 완료"

# ── 5. Hybrid runtime grid search ────────────────────────────
echo ""
echo "[5/7] Hybrid grid search (batch_size × timeout_ms)..."
python benchmark/benchmark_hybrid_grid.py \
    --threshold    "$THRESHOLD" \
    --num-samples  500 \
    --batch-sizes  1 2 4 8 16 32 \
    --timeout-ms   5 10 15 20 25 30 35 40
echo "[6/8] Grid search 완료"

# ── 7. TRT threshold sweep (EE + VEE) ────────────────────────
echo ""
echo "[6/7] EE + VEE TRT threshold sweep (0.50~0.95)..."
python infer/infer_trt.py \
    --seg1     "$EE_ENGINE_DIR/seg1.engine"  \
    --seg2     "$EE_ENGINE_DIR/seg2.engine"  \
    --seg3     "$EE_ENGINE_DIR/seg3.engine"  \
    --vee-seg1 "$VEE_ENGINE_DIR/vee_seg1.engine" \
    --vee-seg2 "$VEE_ENGINE_DIR/vee_seg2.engine" \
    --eval-cifar10 --sweep --sweep-vee --num-samples "$N_SAMPLES"
echo "[7/8] Sweep 완료"

# ── 8. 엔진 레이어 fusion 분석 ───────────────────────────────
echo ""
echo "[7/7] TRT 레이어 fusion 분석 (Plain / EE / VEE)..."
python analysis/inspect_engines.py
echo "[7/7] 분석 완료"

echo ""
echo "================================================"
echo "  Orin 파이프라인 완료! (8단계)"
echo "  실험 디렉토리 : $EXP_NAME"
echo "  결과 위치:"
echo "    $EXP_DIR/eval/benchmark_comparison/"
echo "    $EXP_DIR/eval/hybrid_grid/"
echo "    $EXP_DIR/eval/trt_sweep/"
echo "    $EXP_DIR/eval/engine_inspect/"
echo ""
echo "  NOTE: Hard sample 분석은 5090에서 별도 실행"
echo "    cd src && python analysis/analyze_hard_samples.py --threshold $THRESHOLD"
echo "================================================"
