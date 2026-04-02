#!/usr/bin/env bash
# ============================================================
#  step1_sweep.sh  —  EE + VEE threshold sweep N번 반복
#
#  Step 1 (이 스크립트): threshold sweep을 N번 반복하여
#                        최적 threshold 값을 탐색합니다.
#  Step 2 (step2_benchmark.sh): Step 1 결과를 보고 threshold를
#                        직접 지정하여 grid search + 4-way benchmark를 실행합니다.
#
#  실행 순서:
#    1.  EE (3-segment) threshold sweep (0.50~0.95)  ┐ N번
#    2.  VEE (2-segment) threshold sweep (0.50~0.95) ┘ 반복
#    3.  N번 실행 결과를 CSV로 자동 집계
#
#  사용법:
#    bash scripts/step1_sweep.sh <N>
#    N_SAMPLES=500 bash scripts/step1_sweep.sh 20
#
#  백그라운드 실행 (SSH 끊겨도 유지):
#    nohup bash scripts/step1_sweep.sh 20 > step1_sweep.log 2>&1 &
#    tail -f step1_sweep.log   # 실시간 로그 확인
#
#  환경변수:
#    N_SAMPLES=1000    sweep당 샘플 수 (기본 1000)
#    EXP_DIR=<path>    실험 디렉토리 (기본: 최신 exp_* 자동 감지)
#
#  결과 위치:
#    experiments/exp_.../eval/run_YYYYMMDD_HHMMSS/trt_sweep/
#      ├── ee_sweep_results.json
#      ├── ee_sweep_results.png
#      ├── vee_sweep_results.json
#      └── vee_sweep_results.png
#    experiments/exp_.../eval/
#      ├── aggregate_sweep_ee_YYYYMMDD_HHMMSS.csv   ← EE 집계
#      └── aggregate_sweep_vee_YYYYMMDD_HHMMSS.csv  ← VEE 집계
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

N_SAMPLES="${N_SAMPLES:-1000}"
AGGREGATE_TS="$(date +%Y%m%d_%H%M%S)"

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
echo "  실험 디렉  : $EXP_NAME"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

PASS=0
FAIL=0

cd "$SRC_DIR"

for i in $(seq 1 "$N"); do
    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Sweep Run $i / $N   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────"

    export RUN_TIMESTAMP="run_$(date +%Y%m%d_%H%M%S)"

    if python infer/infer_trt.py \
        --seg1     "$EE_ENGINE_DIR/seg1.engine" \
        --seg2     "$EE_ENGINE_DIR/seg2.engine" \
        --seg3     "$EE_ENGINE_DIR/seg3.engine" \
        --vee-seg1 "$VEE_ENGINE_DIR/vee_seg1.engine" \
        --vee-seg2 "$VEE_ENGINE_DIR/vee_seg2.engine" \
        --eval-cifar10 --sweep --sweep-vee \
        --num-samples "$N_SAMPLES"; then
        PASS=$((PASS + 1))
        echo "  [OK] Sweep Run $i 완료  →  eval/$RUN_TIMESTAMP/trt_sweep/"
    else
        FAIL=$((FAIL + 1))
        echo "  [FAIL] Sweep Run $i 실패 (계속 진행)"
    fi

    # 동일 초 타임스탬프 충돌 방지
    sleep 1
done

echo ""
echo "================================================"
echo "  Sweep 반복 완료: 성공 $PASS / $N   실패 $FAIL / $N"
echo "  집계 중..."
echo "================================================"

# ── 결과 집계 ────────────────────────────────────────────────
OUT_EE="$EXP_DIR/eval/aggregate_sweep_ee_${AGGREGATE_TS}.csv"
OUT_VEE="$EXP_DIR/eval/aggregate_sweep_vee_${AGGREGATE_TS}.csv"

python analysis/aggregate_results.py \
    --mode     sweep \
    --exp-dir  "$EXP_DIR" \
    --out-ee   "$OUT_EE" \
    --out-vee  "$OUT_VEE"

echo ""
echo "================================================"
echo "  Step 1 완료!"
echo "  집계 결과:"
echo "    EE  → $OUT_EE"
echo "    VEE → $OUT_VEE"
echo ""
echo "  결과를 확인하여 threshold를 결정한 뒤 Step 2를 실행하세요:"
echo "    bash scripts/step2_benchmark.sh <N> <THRESHOLD>"
echo "    예) bash scripts/step2_benchmark.sh 30 0.80"
echo "================================================"
