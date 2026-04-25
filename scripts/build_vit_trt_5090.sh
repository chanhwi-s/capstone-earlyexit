#!/usr/bin/env bash
# ============================================================
#  build_vit_trt_5090.sh  —  ViT TRT 엔진 빌드 (RTX 5090)
#
#  export_vit_5090.sh 로 ONNX export 완료 후 실행.
#  Orin 엔진과 호환되지 않음 — 반드시 같은 머신에서 벤치마크 실행.
#
#  사용법:
#    bash scripts/build_vit_trt_5090.sh
#    EXP_DIR=experiments/exp_20260420_120000 bash scripts/build_vit_trt_5090.sh
#    SKIP_PLAIN=1 bash scripts/build_vit_trt_5090.sh
#
#  환경 변수:
#    EXP_DIR=<path>   실험 디렉토리 직접 지정 (기본: 최신 exp_* 자동)
#    SKIP_PLAIN=1     PlainViT 빌드 스킵
#    SKIP_2EXIT=1     2-exit 빌드 스킵
#    SKIP_3EXIT=1     3-exit 빌드 스킵
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export HF_HOME="/home/cap10/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/home/cap10/.cache/huggingface/hub"

# ── 실험 디렉토리 결정 ──────────────────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    [[ "$EXP_DIR" != /* ]] && EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$LATEST_EXP" ]]; then
        echo "[ERROR] experiments/ 내 exp_* 디렉토리가 없습니다."
        echo "        먼저 bash scripts/export_vit_5090.sh 를 실행하세요."
        exit 1
    fi
    export EXP_DIR="$LATEST_EXP"
fi

PLAIN_ONNX_DIR="$EXP_DIR/onnx/plain_vit"
EE2_ONNX_DIR="$EXP_DIR/onnx/ee_vit_2exit"
EE3_ONNX_DIR="$EXP_DIR/onnx/ee_vit_3exit"

PLAIN_ENGINE_DIR="$EXP_DIR/trt_engines/plain_vit"
EE2_ENGINE_DIR="$EXP_DIR/trt_engines/ee_vit_2exit"
EE3_ENGINE_DIR="$EXP_DIR/trt_engines/ee_vit_3exit"

mkdir -p "$PLAIN_ENGINE_DIR" "$EE2_ENGINE_DIR" "$EE3_ENGINE_DIR"

echo "============================================"
echo "  ViT TRT Build  (RTX 5090)"
echo "  EXP : $(basename "$EXP_DIR")"
echo "============================================"

_build() {
    local onnx_path="$1"
    local engine_path="$2"
    local label="$3"

    if [[ ! -f "$onnx_path" ]]; then
        echo "  [SKIP] $label — ONNX 없음: $onnx_path"
        return
    fi
    echo "  빌드: $label"
    # trtexec는 외부 데이터(.onnx.data)가 ONNX와 같은 디렉토리에 있어야 함
    trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$engine_path" \
        --fp16 \
        --iterations=100 --warmUp=500 --avgRuns=100 \
        2>&1 | grep -E "^(&|=|\[|TRT-PERF|trtexec|Successfully)" | tail -5
    echo "  → $engine_path"
}

# ── PlainViT ────────────────────────────────────────────────────────────────
if [[ "${SKIP_PLAIN:-0}" != "1" ]]; then
    echo ""
    echo "[1/3] PlainViT 엔진 빌드 (FP16) ..."
    _build "$PLAIN_ONNX_DIR/plain_vit.onnx" \
           "$PLAIN_ENGINE_DIR/plain_vit.engine" \
           "plain_vit.onnx"
    echo "[1/3] PlainViT 완료"
fi

# ── 2-exit ──────────────────────────────────────────────────────────────────
if [[ "${SKIP_2EXIT:-0}" != "1" ]]; then
    echo ""
    echo "[2/3] 2-exit 세그먼트 엔진 빌드 (FP16) ..."
    _build "$EE2_ONNX_DIR/seg1.onnx" "$EE2_ENGINE_DIR/seg1.engine" "ee_vit_2exit/seg1"
    _build "$EE2_ONNX_DIR/seg2.onnx" "$EE2_ENGINE_DIR/seg2.engine" "ee_vit_2exit/seg2"
    echo "[2/3] 2-exit 완료"
fi

# ── 3-exit ──────────────────────────────────────────────────────────────────
if [[ "${SKIP_3EXIT:-0}" != "1" ]]; then
    echo ""
    echo "[3/3] 3-exit 세그먼트 엔진 빌드 (FP16) ..."
    for seg_num in 1 2 3; do
        _build "$EE3_ONNX_DIR/seg${seg_num}.onnx" \
               "$EE3_ENGINE_DIR/seg${seg_num}.engine" \
               "ee_vit_3exit/seg${seg_num}"
    done
    echo "[3/3] 3-exit 완료"
fi

echo ""
echo "============================================"
echo "  TRT 빌드 완료  (RTX 5090)"
echo ""
echo "  다음 단계 — Threshold Sweep:"
echo "    bash scripts/vit_trt_sweep.sh 8 12 10     # 2-exit"
echo "    bash scripts/vit_trt_sweep.sh 6 9 12 10   # 3-exit"
echo ""
echo "  다음 단계 — 3-way Benchmark:"
echo "    bash scripts/benchmark_vit_5090.sh 0.80 0.80 30"
echo "============================================"
