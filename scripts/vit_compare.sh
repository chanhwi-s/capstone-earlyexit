#!/usr/bin/env bash
# ============================================================
#  vit_compare.sh  —  PlainViT vs EE-ViT 비교 분석 (RTX 5090 서버용)
#
#  동일한 ImageNet val 샘플(기본 1000개)로
#  PlainViT-B/16(pretrained)과 EE-ViT(best.pth)를 비교.
#
#  핵심 출력:
#    vit_compare_accuracy.png  — threshold별 accuracy + plain 기준선
#    vit_compare_latency.png   — threshold별 latency + plain 기준선
#    vit_compare_tradeoff.png  — accuracy-latency tradeoff 커브
#    vit_compare_acc_heatmap.png — per-exit block accuracy heatmap
#
#  사용법:
#    bash scripts/vit_compare.sh <N>
#    N_SAMPLES=2000 bash scripts/vit_compare.sh 10
#    THRESHOLDS="0.5 0.7 0.9 0.95" bash scripts/vit_compare.sh 5
#    CHECKPOINT=/path/to/best.pth bash scripts/vit_compare.sh 5
#
#  백그라운드 실행:
#    nohup bash scripts/vit_compare.sh 5 > vit_compare.log 2>&1 &
#    tail -f vit_compare.log
#
#  환경변수:
#    N_SAMPLES=1000           샘플 수 (기본: 1000)
#    THRESHOLDS="0.3 0.5..."  threshold 목록 (기본: 0.1~0.99)
#    CHECKPOINT=<path>        EE-ViT 체크포인트 (기본: 최신 exp_*/ee_vit/best.pth)
#    DATA_ROOT=<path>         ImageNet 루트 경로
#    WARMUP=20                latency warmup 샘플 수 (기본: 20)
#    EXP_DIR=<path>           실험 디렉토리 (기본: 최신 exp_* 자동 감지)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 인자 확인 ────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "[ERROR] 반복 횟수(N)를 인자로 전달하세요."
    echo "  사용법: bash scripts/vit_compare.sh <N>"
    echo "  예시:   bash scripts/vit_compare.sh 5"
    exit 1
fi

N="$1"
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "[ERROR] N은 1 이상의 정수여야 합니다. (입력값: $N)"
    exit 1
fi

# ── 실험 디렉토리 결정 ────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    [[ "$EXP_DIR" != /* ]] && EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$LATEST_EXP" ]]; then
        echo "[ERROR] experiments/ 내 exp_* 디렉토리가 없습니다."
        echo "        먼저 train_vit_pipeline.sh 로 학습을 완료하세요."
        exit 1
    fi
    export EXP_DIR="$LATEST_EXP"
fi

export EXP_NAME="$(basename "$EXP_DIR")"

# ── 체크포인트 확인 ───────────────────────────────────────────
CKPT_DEFAULT="$EXP_DIR/train/ee_vit/checkpoints/best.pth"
CHECKPOINT="${CHECKPOINT:-$CKPT_DEFAULT}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] EE-ViT 체크포인트를 찾을 수 없습니다: $CHECKPOINT"
    echo "        먼저 train_vit_pipeline.sh 로 학습을 완료하세요."
    echo "        또는 CHECKPOINT 환경변수로 경로를 직접 지정하세요."
    exit 1
fi

# ── HuggingFace 캐시 경로 (공유 캐시 권한 오류 방지) ──────────
export HF_HOME="/home/cap10/.cache/huggingface"
mkdir -p "$HF_HOME"

# ── 파라미터 ─────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-1000}"
WARMUP="${WARMUP:-20}"
DATA_ROOT="${DATA_ROOT:-/home2}"    # → /home2/imagenet/{train,val}

echo "============================================"
echo "  PlainViT vs EE-ViT 비교 분석"
echo "  반복 횟수  : $N"
echo "  Samples    : $N_SAMPLES  (warmup: $WARMUP)"
echo "  실험 디렉  : $EXP_NAME"
echo "  EE-ViT 체크포인트: $CHECKPOINT"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  분석 내용:"
echo "    1. PlainViT-B/16 (pretrained, 수정 없음) accuracy/latency"
echo "    2. EE-ViT threshold sweep accuracy/latency"
echo "    3. threshold별 accuracy drop vs latency savings 비교"
echo "    4. per-exit block accuracy heatmap"
echo "============================================"

# ── CLI 인자 조립 ────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "${THRESHOLDS:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --thresholds $THRESHOLDS"
[[ -n "${DATA_ROOT:-}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --data-root $DATA_ROOT"

cd "$SRC_DIR"

python benchmark/run_vit_compare.py \
    --n           "$N"           \
    --num-samples "$N_SAMPLES"   \
    --checkpoint  "$CHECKPOINT"  \
    --warmup      "$WARMUP"      \
    $EXTRA_ARGS

echo ""
echo "============================================"
echo "  비교 분석 완료!  (종료: $(date '+%Y-%m-%d %H:%M:%S'))"
echo ""
echo "  ★ 핵심 그래프 확인:"
echo ""
echo "  vit_compare_accuracy.png"
echo "    → threshold별 EE-ViT accuracy + PlainViT 기준선"
echo "    → accuracy drop 1%p 이내인 threshold 후보 확인"
echo ""
echo "  vit_compare_latency.png"
echo "    → threshold별 avg/p99 latency + PlainViT 기준선"
echo "    → latency savings % 확인"
echo ""
echo "  vit_compare_tradeoff.png"
echo "    → accuracy vs latency tradeoff 커브"
echo "    → compute savings vs accuracy drop scatter"
echo "    → PlainViT 대비 이상적인 동작 지점 확인"
echo ""
echo "  vit_compare_acc_heatmap.png"
echo "    → threshold × block 별 정확도 heatmap"
echo "    → 어느 block이 high/low accuracy exit인지 확인"
echo "============================================"
