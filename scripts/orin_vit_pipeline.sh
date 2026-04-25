#!/usr/bin/env bash
# ============================================================
#  orin_vit_pipeline.sh  —  ViT TRT 엔진 빌드 (Jetson AGX Orin)
#
#  사전 조건:
#    - 5090에서 export_vit_5090.sh 실행 후 experiments/ 전송 완료
#    - JetPack 6.x (TensorRT 10.x)
#
#  실행 순서:
#    1. PlainViT ONNX → TRT 엔진
#    2. 2-exit 세그먼트 ONNX → TRT 엔진 (seg1, seg2)
#    3. 3-exit 세그먼트 ONNX → TRT 엔진 (seg1, seg2, seg3)
#
#  벤치마크는 아래 스크립트로 별도 실행:
#    bash scripts/vit_trt_sweep.sh 8 12 10           # 2-exit sweep N=10
#    bash scripts/vit_trt_sweep.sh 6 9 12 10         # 3-exit sweep N=10
#    bash scripts/vit_trt_benchmark.sh 0.80 0.75 30  # 3-way benchmark
#
#  환경 변수:
#    EXP_DIR=<path>    실험 디렉토리 직접 지정 (기본: 최신 exp_* 자동 감지)
#    SKIP_BUILD=1      TRT 빌드 스킵
#
#  사용법:
#    bash scripts/orin_vit_pipeline.sh
#    EXP_DIR=experiments/exp_20260420_120000 bash scripts/orin_vit_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 실험 디렉토리 결정 ──────────────────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    [[ "$EXP_DIR" != /* ]] && EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$LATEST_EXP" ]]; then
        echo "[ERROR] experiments/ 내 exp_* 디렉토리가 없습니다."
        echo "        5090에서 먼저 전송하세요."
        exit 1
    fi
    export EXP_DIR="$LATEST_EXP"
fi

export EXP_NAME="$(basename "$EXP_DIR")"

PLAIN_ONNX_DIR="$EXP_DIR/onnx/plain_vit"
EE2_ONNX_DIR="$EXP_DIR/onnx/ee_vit_2exit"
EE3_ONNX_DIR="$EXP_DIR/onnx/ee_vit_3exit"

PLAIN_ENGINE_DIR="$EXP_DIR/trt_engines/plain_vit"
EE2_ENGINE_DIR="$EXP_DIR/trt_engines/ee_vit_2exit"
EE3_ENGINE_DIR="$EXP_DIR/trt_engines/ee_vit_3exit"

mkdir -p "$PLAIN_ENGINE_DIR" "$EE2_ENGINE_DIR" "$EE3_ENGINE_DIR"

echo "============================================"
echo "  ViT TRT 빌드 파이프라인  (Jetson AGX Orin)"
echo "  실험 디렉토리 : $EXP_NAME"
echo "============================================"

# ViT 입력/중간 텐서 크기 (ViT-B/16, 224×224)
#   이미지    : [1, 3, 224, 224]
#   seq feat  : [1, 197, 768]   (197 = 14×14 patches + 1 CLS)
IMG_SHAPE="1x3x224x224"
FEAT_SHAPE="1x197x768"

# ── 1. PlainViT ─────────────────────────────────────────────────────────────
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then

    echo ""
    echo "[1/3] PlainViT 엔진 빌드 (FP16) ..."
    PLAIN_ONNX="$PLAIN_ONNX_DIR/plain_vit.onnx"
    if [[ -f "$PLAIN_ONNX" ]]; then
        trtexec --onnx="$PLAIN_ONNX" \
                --saveEngine="$PLAIN_ENGINE_DIR/plain_vit.engine" \
                --fp16 \
                --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -5
        echo "[1/3] PlainViT 빌드 완료"
    else
        echo "[SKIP] $PLAIN_ONNX 없음 — export_vit_5090.sh 를 먼저 실행하세요"
    fi

    # ── 2. 2-exit 세그먼트 ──────────────────────────────────────────────────
    echo ""
    echo "[2/3] 2-exit 세그먼트 엔진 빌드 (FP16) ..."

    # seg1: image → (feat, ee_logits)  — batch=1 고정
    SEG1_ONNX="$EE2_ONNX_DIR/seg1.onnx"
    if [[ -f "$SEG1_ONNX" ]]; then
        echo "  빌드: seg1.onnx  (image → feat + ee_logits)"
        trtexec --onnx="$SEG1_ONNX" \
                --saveEngine="$EE2_ENGINE_DIR/seg1.engine" \
                --fp16 \
                --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    else
        echo "  [SKIP] $SEG1_ONNX 없음"
    fi

    # seg2: feat → ee_logits  — batch=1 고정
    SEG2_ONNX="$EE2_ONNX_DIR/seg2.onnx"
    if [[ -f "$SEG2_ONNX" ]]; then
        echo "  빌드: seg2.onnx  (feat → ee_logits)"
        trtexec --onnx="$SEG2_ONNX" \
                --saveEngine="$EE2_ENGINE_DIR/seg2.engine" \
                --fp16 \
                --iterations=100 --warmUp=500 --avgRuns=100 \
                2>&1 | tail -3
    else
        echo "  [SKIP] $SEG2_ONNX 없음"
    fi
    echo "[2/3] 2-exit 빌드 완료"

    # ── 3. 3-exit 세그먼트 ──────────────────────────────────────────────────
    echo ""
    echo "[3/3] 3-exit 세그먼트 엔진 빌드 (FP16) ..."

    for seg_num in 1 2 3; do
        SEG_ONNX="$EE3_ONNX_DIR/seg${seg_num}.onnx"
        if [[ -f "$SEG_ONNX" ]]; then
            echo "  빌드: seg${seg_num}.onnx"
            trtexec --onnx="$SEG_ONNX" \
                    --saveEngine="$EE3_ENGINE_DIR/seg${seg_num}.engine" \
                    --fp16 \
                    --iterations=100 --warmUp=500 --avgRuns=100 \
                    2>&1 | tail -3
        else
            echo "  [SKIP] $SEG_ONNX 없음"
        fi
    done
    echo "[3/3] 3-exit 빌드 완료"

else
    echo "[1-3/3] TRT 빌드 전체 스킵 (SKIP_BUILD=1)"
fi

echo ""
echo "============================================"
echo "  TRT 빌드 완료!"
echo "  실험 디렉토리 : $EXP_NAME"
echo ""
echo "  다음 단계 — Threshold Sweep:"
echo "    bash scripts/vit_trt_sweep.sh 8 12 10     # 2-exit"
echo "    bash scripts/vit_trt_sweep.sh 6 9 12 10   # 3-exit"
echo ""
echo "  다음 단계 — 3-way Benchmark:"
echo "    bash scripts/vit_trt_benchmark.sh <thr2exit> <thr3exit> <N>"
echo "    예) bash scripts/vit_trt_benchmark.sh 0.80 0.75 30"
echo "============================================"
