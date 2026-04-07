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
#  ★ 2-Step 워크플로우 ★
#    이 스크립트는 TRT 빌드(Step 0~3.5)만 담당합니다.
#    벤치마크는 아래 두 스크립트로 별도 실행하세요:
#
#    [Step 1] threshold 탐색 sweep (N번 반복):
#      bash scripts/step1_sweep.sh <N>
#      예) bash scripts/step1_sweep.sh 20
#
#    [Step 2] threshold 확정 후 benchmark (N번 반복):
#      bash scripts/step2_benchmark.sh <N> <THRESHOLD>
#      예) bash scripts/step2_benchmark.sh 30 0.80
#
#  실행 순서:
#    0.   최신 exp_* 디렉토리 자동 감지 (또는 EXP_DIR 직접 지정)
#    1.   EE  ONNX → TRT 엔진 빌드 (seg1/2/3, FP16)          ┐ SKIP_BUILD=1
#    2.   Plain ONNX → TRT 엔진 빌드 (FP16, 동적 배치 1~32)  │ 이면 모두
#    3.   VEE ONNX → TRT 엔진 빌드 (vee_seg1/vee_seg2, FP16) │ 스킵
#    3.5. 엔진 레이어 fusion 분석 → exp_.../engine_inspect/   ┘
#
#  NOTE: analyze_hard_samples.py 는 PyTorch 체크포인트가 필요하므로
#        5090 서버에서 별도 실행:
#          cd src && python analysis/analyze_hard_samples.py --threshold 0.80
#
#  환경 변수:
#    EXP_DIR=<path>    특정 실험 디렉토리 지정 (기본: 최신 exp_* 자동 감지)
#    SKIP_BUILD=1      TRT 빌드 스킵 (엔진이 이미 존재할 때)
#
#  사용법:
#    cd <project_root>
#    bash scripts/orin_pipeline.sh                          # 최신 실험 자동 감지
#    SKIP_BUILD=1 bash scripts/orin_pipeline.sh             # 빌드 스킵 (engine_inspect만)
#    EXP_DIR=experiments/exp_20260401_120000 bash scripts/orin_pipeline.sh  # 직접 지정
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
echo "  Orin TRT 빌드 파이프라인"
echo "  Project root  : $PROJECT_ROOT"
echo "  실험 디렉토리 : $EXP_NAME"
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

    # 데이터셋별 입력/피처 크기 결정
    # feat_layer1 shape = (B, 64, H/4, W/4)  — conv1(stride=2) + maxpool(stride=2)
    DATASET="${DATASET:-cifar10}"
    if [[ "$DATASET" == "imagenet" ]]; then
        INPUT_H=224; INPUT_W=224
    else
        INPUT_H=32;  INPUT_W=32
    fi
    FEAT_H=$(( INPUT_H / 4 ))
    FEAT_W=$(( INPUT_W / 4 ))

    # vee_seg1: batch=1 고정 (스트리밍 single-sample 추론)
    VEE_SEG1_ONNX="$VEE_ONNX_DIR/vee_seg1_stem_layer1.onnx"
    if [[ -f "$VEE_SEG1_ONNX" ]]; then
        echo "  빌드: vee_seg1_stem_layer1.onnx → vee_seg1.engine  (batch=1 고정)"
        trtexec --onnx="$VEE_SEG1_ONNX" \
                --saveEngine="$VEE_ENGINE_DIR/vee_seg1.engine" \
                --fp16 --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    else
        echo "  [SKIP] $VEE_SEG1_ONNX 없음"
    fi

    # vee_seg2: dynamic batch (Hybrid-VEE fallback 배치 처리용)
    VEE_SEG2_ONNX="$VEE_ONNX_DIR/vee_seg2_layer2to4.onnx"
    if [[ -f "$VEE_SEG2_ONNX" ]]; then
        echo "  빌드: vee_seg2_layer2to4.onnx → vee_seg2.engine  (동적 배치 1~32, feat=${FEAT_H}x${FEAT_W})"
        trtexec --onnx="$VEE_SEG2_ONNX" \
                --saveEngine="$VEE_ENGINE_DIR/vee_seg2.engine" \
                --fp16 \
                --minShapes=feat_layer1:1x64x${FEAT_H}x${FEAT_W} \
                --optShapes=feat_layer1:8x64x${FEAT_H}x${FEAT_W} \
                --maxShapes=feat_layer1:32x64x${FEAT_H}x${FEAT_W} \
                --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    else
        echo "  [SKIP] $VEE_SEG2_ONNX 없음"
    fi

    echo "[3/7] VEE 엔진 빌드 완료"

    # ── 3.5 엔진 레이어 fusion 분석 (빌드 직후, SKIP_BUILD=1 이면 스킵) ──────
    echo ""
    echo "[3.5/7] TRT 레이어 fusion 분석 (Plain / EE / VEE)..."
    cd "$SRC_DIR"
    python analysis/inspect_engines.py
    echo "[3.5/7] 분석 완료  →  $EXP_DIR/engine_inspect/"
    cd "$PROJECT_ROOT"

else
    echo "[1-3/7] TRT 빌드 + engine_inspect 전체 스킵 (SKIP_BUILD=1)"
fi

echo ""
echo "================================================"
echo "  빌드 완료!"
echo "  실험 디렉토리 : $EXP_NAME"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
echo "  엔진 분석 결과: $EXP_DIR/engine_inspect/"
fi
echo ""
echo "  다음 단계:"
echo "    [Step 1] threshold sweep (N번 반복):"
echo "      bash scripts/step1_sweep.sh <N>"
echo ""
echo "    [Step 2] threshold 확정 후 benchmark:"
echo "      bash scripts/step2_benchmark.sh <N> <THRESHOLD>"
echo ""
echo "  NOTE: Hard sample 분석은 5090에서 별도 실행"
echo "    cd src && python analysis/analyze_hard_samples.py --threshold <THRESHOLD>"
echo "================================================"
