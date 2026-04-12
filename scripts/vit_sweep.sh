#!/usr/bin/env bash
# ============================================================
#  vit_sweep.sh  —  EE-ViT-B/16 threshold sweep (RTX 5090 서버용)
#
#  학습 완료 후 threshold별 exit block 분포 / accuracy / latency를 분석.
#  최적 threshold를 결정하기 위한 Step 1에 해당.
#
#  내부 동작:
#    - 학습된 best.pth 체크포인트 자동 탐지 (EXP_DIR 기준)
#    - N회 반복으로 latency 안정성 확보
#    - exit block 분포 heatmap이 핵심 출력물
#
#  출력 디렉토리:
#    experiments/exp_.../eval/vit_sweep_N{N}_YYYYMMDD_HHMMSS/
#      vit_sweep_raw.json
#      vit_sweep_summary.csv
#      vit_sweep_exit_heatmap.png  ← ★ 핵심: 어느 블록에서 탈출하는지
#      vit_sweep_latency_dist.png
#      vit_sweep_summary.png
#
#  사용법:
#    bash scripts/vit_sweep.sh <N>
#    N_SAMPLES=2000 bash scripts/vit_sweep.sh 10
#    THRESHOLDS="0.3 0.5 0.7 0.9" bash scripts/vit_sweep.sh 5
#    CHECKPOINT=/path/to/best.pth bash scripts/vit_sweep.sh 5
#
#  백그라운드 실행:
#    nohup bash scripts/vit_sweep.sh 10 > vit_sweep.log 2>&1 &
#    tail -f vit_sweep.log
#
#  환경변수:
#    N_SAMPLES=1000           샘플 수 (기본: 1000)
#    THRESHOLDS="0.3 0.5..."  threshold 목록 (기본: 0.1~0.99)
#    CHECKPOINT=<path>        체크포인트 경로 (기본: 최신 exp_*/ee_vit/best.pth)
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
    echo "  사용법: bash scripts/vit_sweep.sh <N>"
    echo "  예시:   bash scripts/vit_sweep.sh 10"
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
    echo "[ERROR] 체크포인트를 찾을 수 없습니다: $CHECKPOINT"
    echo "        먼저 train_vit_pipeline.sh 로 학습을 완료하세요."
    echo "        또는 CHECKPOINT 환경변수로 경로를 직접 지정하세요."
    exit 1
fi

# ── 파라미터 ─────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-1000}"
WARMUP="${WARMUP:-20}"

echo "============================================"
echo "  EE-ViT-B/16 Threshold Sweep"
echo "  반복 횟수  : $N"
echo "  Samples    : $N_SAMPLES  (warmup: $WARMUP)"
echo "  실험 디렉  : $EXP_NAME"
echo "  체크포인트 : $CHECKPOINT"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  핵심 출력  : vit_sweep_exit_heatmap.png"
echo "============================================"

# ── CLI 인자 조립 ────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "${THRESHOLDS:-}"  ]] && EXTRA_ARGS="$EXTRA_ARGS --thresholds $THRESHOLDS"
[[ -n "${DATA_ROOT:-}"   ]] && EXTRA_ARGS="$EXTRA_ARGS --data-root $DATA_ROOT"

cd "$SRC_DIR"

python benchmark/run_vit_sweep.py \
    --n           "$N"           \
    --num-samples "$N_SAMPLES"   \
    --checkpoint  "$CHECKPOINT"  \
    --warmup      "$WARMUP"      \
    $EXTRA_ARGS

echo ""
echo "============================================"
echo "  Sweep 완료!  (종료: $(date '+%Y-%m-%d %H:%M:%S'))"
echo ""
echo "  ★ 핵심 그래프 확인:"
echo "    vit_sweep_exit_heatmap.png"
echo "      → threshold별 exit block 분포"
echo "      → 어느 threshold가 early exit에 효과적인지 확인"
echo ""
echo "  → exit_heatmap 보고 적절한 threshold 선택 후"
echo "     성능 분석 스크립트 실행:"
echo "     bash scripts/vit_analysis.sh <THRESHOLD>"
echo "============================================"
