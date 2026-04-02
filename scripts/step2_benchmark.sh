#!/usr/bin/env bash
# ============================================================
#  step2_benchmark.sh  —  Hybrid Grid Search + 4-Way Benchmark N번 반복
#
#  Step 1 (step1_sweep.sh) 결과를 보고 threshold를 결정한 뒤,
#  이 스크립트로 Grid Search + 4-Way Benchmark를 N번 반복합니다.
#
#  실행 순서 (1회):
#    1.  Hybrid Grid Search (batch_size × timeout_ms)
#        → P99 latency 기준으로 최적 (bs, timeout) 자동 선택
#    2.  4-Way Benchmark: Plain / EE / VEE / Hybrid (최적 파라미터 사용)
#  위 과정을 N번 반복 후 전체 결과를 CSV로 자동 집계
#
#  사용법:
#    bash scripts/step2_benchmark.sh <N> <THRESHOLD>
#    bash scripts/step2_benchmark.sh 30 0.80
#    N_SAMPLES=1000 bash scripts/step2_benchmark.sh 30 0.80
#
#  백그라운드 실행 (SSH 끊겨도 유지):
#    nohup bash scripts/step2_benchmark.sh 30 0.80 > step2_benchmark.log 2>&1 &
#    tail -f step2_benchmark.log
#
#  환경변수:
#    N_SAMPLES=1000        benchmark당 샘플 수 (기본 1000)
#    GRID_SAMPLES=500      grid search당 샘플 수 (기본 500)
#    BATCH_SIZES="2 4 8 16"        grid 탐색 batch_size 후보
#    TIMEOUT_MS="5 10 15 20 25 30 35 40"   grid 탐색 timeout 후보
#    EXP_DIR=<path>        실험 디렉토리 (기본: 최신 exp_* 자동 감지)
#
#  결과 위치:
#    experiments/exp_.../eval/run_YYYYMMDD_HHMMSS/
#      ├── hybrid_grid/
#      │   ├── grid_search_thr{thr}.json
#      │   └── grid_search_thr{thr}.png
#      └── benchmark_comparison/
#          ├── compare_thr{thr}.json
#          └── compare_thr{thr}.png
#    experiments/exp_.../eval/
#      ├── aggregate_grid_thr{thr}_YYYYMMDD.csv       ← Grid 집계
#      └── aggregate_benchmark_thr{thr}_YYYYMMDD.csv  ← Benchmark 집계
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
BATCH_SIZES="${BATCH_SIZES:-2 4 8 16}"
TIMEOUT_MS="${TIMEOUT_MS:-5 10 15 20 25 30 35 40}"
AGGREGATE_TS="$(date +%Y%m%d_%H%M%S)"

echo "================================================"
echo "  Step 2: Grid Search + 4-Way Benchmark"
echo "  반복 횟수  : $N"
echo "  Threshold  : $THRESHOLD"
echo "  Samples    : $N_SAMPLES  (grid: $GRID_SAMPLES)"
echo "  Batch sizes: $BATCH_SIZES"
echo "  Timeout ms : $TIMEOUT_MS"
echo "  실험 디렉  : $EXP_NAME"
echo "  시작 시각  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

PASS=0
FAIL=0

cd "$SRC_DIR"

for i in $(seq 1 "$N"); do
    echo ""
    echo "────────────────────────────────────────────────"
    echo "  Benchmark Run $i / $N   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────"

    export RUN_TIMESTAMP="run_$(date +%Y%m%d_%H%M%S)"
    echo "  📁 결과 디렉: eval/$RUN_TIMESTAMP/"

    RUN_FAILED=false

    # ── [1/2] Hybrid Grid Search ──────────────────────────────
    echo ""
    echo "  [1/2] Hybrid Grid Search  (bs: $BATCH_SIZES | to: $TIMEOUT_MS)"

    # tee로 실시간 출력 유지하면서 BEST_BS= 라인만 추출
    GRID_LOG="$(mktemp)"
    if python benchmark/benchmark_hybrid_grid.py \
        --threshold         "$THRESHOLD" \
        --num-samples       "$GRID_SAMPLES" \
        --batch-sizes       $BATCH_SIZES \
        --timeout-ms        $TIMEOUT_MS \
        --print-best-params 2>&1 | tee "$GRID_LOG"; then

        BEST_LINE=$(grep '^BEST_BS=' "$GRID_LOG" | tail -1 || true)
        BEST_BS=$(echo "$BEST_LINE" | grep -oP 'BEST_BS=\K[0-9]+' || true)
        BEST_TO=$(echo "$BEST_LINE" | grep -oP 'BEST_TO=\K[0-9.]+' || true)
        rm -f "$GRID_LOG"

        BEST_BS="${BEST_BS:-8}"
        BEST_TO="${BEST_TO:-10}"
        echo "  → Grid Search 완료: 최적 hybrid-bs=${BEST_BS}, hybrid-to-ms=${BEST_TO}"
    else
        rm -f "$GRID_LOG"
        BEST_BS=8
        BEST_TO=10
        echo "  [WARN] Grid Search 실패, fallback: bs=${BEST_BS}, to=${BEST_TO}"
        RUN_FAILED=true
    fi

    # ── [2/2] 4-Way Benchmark ────────────────────────────────
    echo ""
    echo "  [2/2] 4-Way Benchmark  (hybrid-bs=${BEST_BS}, to=${BEST_TO}ms)"

    if python benchmark/benchmark_trt_hybrid.py \
        --threshold    "$THRESHOLD" \
        --num-samples  "$N_SAMPLES" \
        --hybrid-bs    "$BEST_BS" \
        --hybrid-to-ms "$BEST_TO"; then
        echo "  [OK] Benchmark Run $i 완료  →  eval/$RUN_TIMESTAMP/"
    else
        echo "  [FAIL] 4-Way Benchmark 실패"
        RUN_FAILED=true
    fi

    if $RUN_FAILED; then
        FAIL=$((FAIL + 1))
    else
        PASS=$((PASS + 1))
    fi

    # 동일 초 타임스탬프 충돌 방지
    sleep 1
done

echo ""
echo "================================================"
echo "  Benchmark 반복 완료: 성공 $PASS / $N   실패 $FAIL / $N"
echo "  집계 중..."
echo "================================================"

# ── 결과 집계 ────────────────────────────────────────────────
OUT_GRID="$EXP_DIR/eval/aggregate_grid_thr${THRESHOLD}_${AGGREGATE_TS}.csv"
OUT_BENCH="$EXP_DIR/eval/aggregate_benchmark_thr${THRESHOLD}_${AGGREGATE_TS}.csv"

python analysis/aggregate_results.py \
    --mode       benchmark \
    --exp-dir    "$EXP_DIR" \
    --threshold  "$THRESHOLD" \
    --out-grid   "$OUT_GRID" \
    --out-bench  "$OUT_BENCH"

echo ""
echo "================================================"
echo "  Step 2 완료!"
echo "  집계 결과:"
echo "    Grid Search → $OUT_GRID"
echo "    Benchmark   → $OUT_BENCH"
echo "  종료 시각: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
