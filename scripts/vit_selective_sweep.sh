#!/usr/bin/env bash
# ============================================================
#  vit_selective_sweep.sh  —  SelectiveExitViT threshold sweep (RTX 5090 서버용)
#
#  학습 완료 후 threshold별 exit block 분포 / accuracy / latency를 분석.
#  2-exit (B8+B12) 또는 3-exit (B6+B9+B12) 모델 모두 지원.
#
#  사용법:
#    bash scripts/vit_selective_sweep.sh <EXIT_BLOCKS> <N>
#    예: bash scripts/vit_selective_sweep.sh "8 12" 10
#        bash scripts/vit_selective_sweep.sh "6 9 12" 10
#
#  환경변수:
#    N_SAMPLES=1000           샘플 수 (기본: 1000)
#    THRESHOLDS="0.3 0.5..."  threshold 목록 (기본: 0.1~0.99)
#    CHECKPOINT=<path>        체크포인트 경로
#    DATA_ROOT=<path>         ImageNet 루트 경로 (기본: /home2)
#    WARMUP=20                latency warmup 샘플 수 (기본: 20)
#    EXP_DIR=<path>           실험 디렉토리 (기본: 최신 exp_* 자동 감지)
#
#  백그라운드 실행:
#    nohup bash scripts/vit_selective_sweep.sh "8 12" 10 > vit_2exit_sweep.log 2>&1 &
#    nohup bash scripts/vit_selective_sweep.sh "6 9 12" 10 > vit_3exit_sweep.log 2>&1 &
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 인자 확인 ────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "[ERROR] exit block 목록과 반복 횟수(N)를 인자로 전달하세요."
    echo "  사용법: bash scripts/vit_selective_sweep.sh <EXIT_BLOCKS> <N>"
    echo "  예시:   bash scripts/vit_selective_sweep.sh \"8 12\" 10"
    echo "           bash scripts/vit_selective_sweep.sh \"6 9 12\" 5"
    exit 1
fi

EXIT_BLOCKS_STR="$1"
N="$2"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "[ERROR] N은 1 이상의 정수여야 합니다. (입력값: $N)"
    exit 1
fi

# exit_blocks에서 개수 계산 (모델 이름 결정용)
N_EXITS=$(echo "$EXIT_BLOCKS_STR" | wc -w | tr -d ' ')
MODEL_NAME="ee_vit_${N_EXITS}exit"

# ── 실험 디렉토리 결정 ────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    [[ "$EXP_DIR" != /* ]] && EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$LATEST_EXP" ]]; then
        echo "[ERROR] experiments/ 내 exp_* 디렉토리가 없습니다."
        exit 1
    fi
    export EXP_DIR="$LATEST_EXP"
fi

export EXP_NAME="$(basename "$EXP_DIR")"

# ── 체크포인트 확인 ───────────────────────────────────────────
CKPT_DEFAULT="$EXP_DIR/train/$MODEL_NAME/checkpoints/best.pth"
CHECKPOINT="${CHECKPOINT:-$CKPT_DEFAULT}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[ERROR] 체크포인트를 찾을 수 없습니다: $CHECKPOINT"
    echo "        먼저 학습을 완료하세요:"
    echo "          nohup bash scripts/train_vit_${N_EXITS}exit_5090.sh > train.log 2>&1 &"
    echo "        또는 CHECKPOINT 환경변수로 경로를 직접 지정하세요."
    exit 1
fi

# ── HuggingFace 캐시 경로 ─────────────────────────────────────
export HF_HOME="/home/cap10/.cache/huggingface"
export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME"

# ── 파라미터 ─────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-1000}"
WARMUP="${WARMUP:-20}"
DATA_ROOT="${DATA_ROOT:-/home2}"

echo "============================================"
echo "  SelectiveExitViT Threshold Sweep"
echo "  Exit blocks : [$EXIT_BLOCKS_STR]"
echo "  반복 횟수   : $N"
echo "  Samples     : $N_SAMPLES  (warmup: $WARMUP)"
echo "  실험 디렉   : $EXP_NAME"
echo "  체크포인트  : $CHECKPOINT"
echo "  시작 시각   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ── CLI 인자 조립 ────────────────────────────────────────────
EXTRA_ARGS="--data-root $DATA_ROOT"
[[ -n "${THRESHOLDS:-}" ]] && EXTRA_ARGS="$EXTRA_ARGS --thresholds $THRESHOLDS"

cd "$SRC_DIR"

python benchmark/run_vit_selective_sweep.py \
    --exit-blocks   $EXIT_BLOCKS_STR  \
    --n             "$N"              \
    --num-samples   "$N_SAMPLES"      \
    --checkpoint    "$CHECKPOINT"     \
    --warmup        "$WARMUP"         \
    $EXTRA_ARGS

echo ""
echo "============================================"
echo "  Sweep 완료! (종료: $(date '+%Y-%m-%d %H:%M:%S'))"
echo ""
echo "  ★ 핵심 그래프:"
echo "    sel_sweep_exit_heatmap.png  → 블록별 exit 분포"
echo "    sel_sweep_acc_heatmap.png   → 블록별 exit accuracy"
echo "    sel_sweep_latency_dist.png  → latency KDE"
echo "    sel_sweep_summary.png       → 종합 요약"
echo ""
echo "  → threshold 결정 후 compare 실행:"
echo "     nohup bash scripts/vit_selective_compare.sh \"$EXIT_BLOCKS_STR\" <N> <THRESHOLD> > cmp.log 2>&1 &"
echo "============================================"
