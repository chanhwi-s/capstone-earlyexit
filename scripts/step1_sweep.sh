#!/usr/bin/env bash
# ============================================================
#  step1_sweep.sh  —  EE + VEE threshold sweep N번 반복
#
#  Step 1 (이 스크립트): threshold sweep을 N번 반복하여
#                        최적 threshold 값을 탐색합니다.
#  Step 2 (step2_benchmark.sh): Step 1 결과를 보고 threshold를
#                        직접 지정하여 grid search + 4-way benchmark를 실행합니다.
#
#  내부 동작:
#    - 엔진 / 데이터를 한 번만 로드하고 N번 sweep 반복
#    - 실행마다 디렉토리 생성 없음 → 단일 디렉토리에 통합 저장
#    - 실행마다 그래프 생성 없음 → 최종 한 번만 생성
#
#  출력 디렉토리 (단일):
#    experiments/exp_.../eval/sweep_N{N}_YYYYMMDD_HHMMSS/
#      ├── ee_sweep_raw.json        ← N회 × 전 threshold 원시 latency
#      ├── ee_sweep_summary.csv     ← threshold별 통계
#      ├── vee_sweep_raw.json
#      ├── vee_sweep_summary.csv
#      ├── ee_sweep_dist.png        ← threshold별 KDE overlay (한 번만)
#      ├── vee_sweep_dist.png
#      ├── ee_sweep_summary.png     ← accuracy / exit rate / p99 요약
#      └── vee_sweep_summary.png
#
#  사용법:
#    bash scripts/step1_sweep.sh <N>
#    N_SAMPLES=500 bash scripts/step1_sweep.sh 20
#    DATASET=imagenet bash scripts/step1_sweep.sh 20
#
#  백그라운드 실행 (SSH 끊겨도 유지):
#    nohup bash scripts/step1_sweep.sh 20 > step1_sweep.log 2>&1 &
#    tail -f step1_sweep.log
#
#  환경변수:
#    N_SAMPLES=1000      sweep당 샘플 수 (기본 1000)
#    DATASET=cifar10     cifar10 | imagenet (기본 cifar10)
#    THRESHOLDS="0.70 0.75 0.80 0.85 0.90"  탐색할 threshold 목록
#    EXP_DIR=<path>      실험 디렉토리 (기본: 최신 exp_* 자동 감지)
#    NO_EE=1             EE sweep 스킵
#    NO_VEE=1            VEE sweep 스킵
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 인자 확인 ────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "[ERROR] 반복 횟수(N)를 인자로 전달하세요."
    echo "  사용법: bash scripts/step1_sweep.sh <N>"
    echo "  예시:   bash scripts/step1_sweep.sh 20"
    exit 1
fi

N="$1"
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "[ERROR] N은 1 이상의 정수여야 합니다. (입력값: $N)"
    exit 1
fi

# ── 실험 디렉토리 결정 ────────────────────────────────────────
if [[ -n "${EXP_DIR:-}" ]]; then
    if [[ "$EXP_DIR" != /* ]]; then
        EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
    fi
    export EXP_DIR="$(realpath "$EXP_DIR")"
else
    LATEST_EXP=$(ls -d "$PROJECT_ROOT/experiments"/exp_* 2>/dev/null | sort | tail -1 || true)
    if [[ -z "$LATEST_EXP" ]]; then
        echo "[ERROR] experiments/ 내 exp_* 디렉토리가 없습니다."
        echo "        EXP_DIR 환경변수로 직접 지정하세요."
        exit 1
    fi
    export EXP_DIR="$LATEST_EXP"
fi

export EXP_NAME="$(basename "$EXP_DIR")"
EE_ENGINE_DIR="$EXP_DIR/trt_engines/ee_resnet18"
VEE_ENGINE_DIR="$EXP_DIR/trt_engines/vee_resnet18"

# ── 파라미터 ─────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-1000}"
DATASET="${DATASET:-cifar10}"

# ── 엔진 파일 존재 확인 ──────────────────────────────────────
for engine_file in \
    "$EE_ENGINE_DIR/seg1.engine" \
    "$EE_ENGINE_DIR/seg2.engine" \
    "$EE_ENGINE_DIR/seg3.engine" \
    "$VEE_ENGINE_DIR/vee_seg1.engine" \
    "$VEE_ENGINE_DIR/vee_seg2.engine"
do
    if [[ ! -f "$engine_file" ]]; then
        echo "[ERROR] 엔진 파일 없음: $engine_file"
        echo "        먼저 TRT 빌드를 실행하세요: bash scripts/orin_pipeline.sh"
        exit 1
    fi
done

echo "================================================"
echo "  Step 1: EE + VEE Threshold Sweep"
echo "  반복 횟수  : $N"
echo "  Samples    : $N_SAMPLES"
echo "  Dataset    : $DATASET"
echo "  실험 디렉  : $EXP_NAME"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  출력 위치  : eval/sweep_N${N}_YYYYMMDD_HHMMSS/ (단일 디렉토리)"
echo "================================================"

# ── 추가 인자 조립 ────────────────────────────────────────────
EXTRA_ARGS=""
[[ -n "${THRESHOLDS:-}" ]] && EXTRA_ARGS="$EXTRA_ARGS --thresholds $THRESHOLDS"
[[ "${NO_EE:-0}"  == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --no-ee"
[[ "${NO_VEE:-0}" == "1" ]] && EXTRA_ARGS="$EXTRA_ARGS --no-vee"

cd "$SRC_DIR"

# ── run_sweep_n.py 호출 (단일 호출, 내부에서 N번 반복) ────────
python benchmark/run_sweep_n.py \
    --n            "$N" \
    --dataset      "$DATASET" \
    --num-samples  "$N_SAMPLES" \
    --seg1         "$EE_ENGINE_DIR/seg1.engine" \
    --seg2         "$EE_ENGINE_DIR/seg2.engine" \
    --seg3         "$EE_ENGINE_DIR/seg3.engine" \
    --vee-seg1     "$VEE_ENGINE_DIR/vee_seg1.engine" \
    --vee-seg2     "$VEE_ENGINE_DIR/vee_seg2.engine" \
    $EXTRA_ARGS

echo ""
echo "================================================"
echo "  Step 1 완료!"
echo "  결과를 확인하여 threshold를 결정한 뒤 Step 2를 실행하세요:"
echo "    bash scripts/step2_benchmark.sh <N> <THRESHOLD>"
echo "    예) bash scripts/step2_benchmark.sh 30 0.80"
echo "  종료 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
