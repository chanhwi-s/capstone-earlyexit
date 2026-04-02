#!/usr/bin/env bash
# ============================================================
#  step2_benchmark.sh  —  Hybrid Grid Search + 4-Way Benchmark N번 반복
#
#  Step 1 (step1_sweep.sh) 결과를 보고 threshold를 결정한 뒤,
#  이 스크립트로 Grid Search + 4-Way Benchmark를 N번 반복합니다.
#
#  내부 동작:
#    - 엔진 / 데이터를 한 번만 로드하고 N번 반복
#    - 각 실행: Grid Search → 최적 (bs, timeout) → 4-Way Benchmark
#    - 실행마다 디렉토리 생성 없음 → 단일 디렉토리에 통합 저장
#    - 실행마다 그래프 생성 없음 → 최종 한 번만 생성
#
#  출력 디렉토리 (단일):
#    experiments/exp_.../eval/benchmark_N{N}_thr{thr}_YYYYMMDD_HHMMSS/
#      ├── grid_raw.json            ← N회 × grid 탐색 결과
#      ├── grid_summary.csv         ← (bs, to)별 통계
#      ├── benchmark_raw.json       ← N회 × 4-way 비교 결과
#      ├── benchmark_summary.csv    ← 모델별 통계 (mean/std)
#      └── benchmark_comparison.png ← 모델별 latency distribution (한 번만)
#
#  사용법:
#    bash scripts/step2_benchmark.sh <N> <THRESHOLD>
#    bash scripts/step2_benchmark.sh 30 0.80
#    N_SAMPLES=1000 bash scripts/step2_benchmark.sh 30 0.80
#    DATASET=imagenet bash scripts/step2_benchmark.sh 30 0.80
#
#  백그라운드 실행 (SSH 끊겨도 유지):
#    nohup bash scripts/step2_benchmark.sh 30 0.80 > step2_benchmark.log 2>&1 &
#    tail -f step2_benchmark.log
#
#  환경변수:
#    N_SAMPLES=1000        benchmark당 샘플 수 (기본 1000)
#    GRID_SAMPLES=500      grid search당 샘플 수 (기본 500)
#    DATASET=cifar10       cifar10 | imagenet (기본 cifar10)
#    BATCH_SIZES="2 4 8 16"         grid 탐색 batch_size 후보
#    TIMEOUT_MS="5 10 15 20 25 30 35 40"   grid 탐색 timeout 후보
#    EXP_DIR=<path>        실험 디렉토리 (기본: 최신 exp_* 자동 감지)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"

# ── 인자 확인 ────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "[ERROR] 인자가 부족합니다."
    echo "  사용법: bash scripts/step2_benchmark.sh <N> <THRESHOLD>"
    echo "  예시:   bash scripts/step2_benchmark.sh 30 0.80"
    exit 1
fi

N="$1"
THRESHOLD="$2"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
    echo "[ERROR] N은 1 이상의 정수여야 합니다. (입력값: $N)"
    exit 1
fi
if ! [[ "$THRESHOLD" =~ ^0?\.[0-9]+$ ]] && ! [[ "$THRESHOLD" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "[ERROR] THRESHOLD는 소수점 숫자여야 합니다. (입력값: $THRESHOLD)"
    echo "  예) 0.80  0.85  0.90"
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

# ── 파라미터 기본값 ──────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-1000}"
GRID_SAMPLES="${GRID_SAMPLES:-500}"
DATASET="${DATASET:-cifar10}"
BATCH_SIZES="${BATCH_SIZES:-2 4 8 16}"
TIMEOUT_MS="${TIMEOUT_MS:-1 1.25 1.5 1.75 2 2.25 2.5 2.75 3}"

echo "================================================"
echo "  Step 2: Grid Search + 4-Way Benchmark"
echo "  반복 횟수  : $N"
echo "  Threshold  : $THRESHOLD"
echo "  Samples    : $N_SAMPLES  (grid: $GRID_SAMPLES)"
echo "  Dataset    : $DATASET"
echo "  Batch sizes: $BATCH_SIZES"
echo "  Timeout ms : $TIMEOUT_MS"
echo "  실험 디렉  : $EXP_NAME"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  출력 위치  : eval/benchmark_N${N}_thr${THRESHOLD}_YYYYMMDD_HHMMSS/ (단일 디렉토리)"
echo "================================================"

cd "$SRC_DIR"

# ── run_benchmark_n.py 호출 (단일 호출, 내부에서 N번 반복) ───
python benchmark/run_benchmark_n.py \
    --n            "$N" \
    --threshold    "$THRESHOLD" \
    --dataset      "$DATASET" \
    --num-samples  "$N_SAMPLES" \
    --grid-samples "$GRID_SAMPLES" \
    --batch-sizes  $BATCH_SIZES \
    --timeout-ms   $TIMEOUT_MS

echo ""
echo "================================================"
echo "  Step 2 완료!"
echo "  종료 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
