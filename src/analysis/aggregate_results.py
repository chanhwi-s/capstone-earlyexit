"""
aggregate_results.py  —  N번 반복 실험 결과 JSON → CSV 취합

실험 디렉토리 내 모든 run_* 폴더를 스캔하여
sweep / grid / benchmark 결과를 각각 CSV로 취합합니다.

지원 모드:
  --mode sweep     : trt_sweep/ 내 ee_sweep_results.json, vee_sweep_results.json 취합
  --mode benchmark : hybrid_grid/, benchmark_comparison/ 취합
  --mode all       : sweep + benchmark 모두 취합

출력 CSV 예시:
  sweep 모드:
    aggregate_sweep_ee.csv   (run_id, threshold, accuracy, ee1_rate, ee2_rate, main_rate, avg_ms, p50_ms, p99_ms)
    aggregate_sweep_vee.csv  (run_id, threshold, accuracy, exit1_rate, main_rate, avg_ms, p50_ms, p99_ms)

  benchmark 모드:
    aggregate_grid.csv       (run_id, threshold, batch_size, timeout_ms, accuracy, exit1_rate, fallback_rate, avg_ms, p99_ms, avg_throughput, p99_goodput, tail_ratio)
    aggregate_benchmark.csv  (run_id, threshold, model, accuracy, exit_info, power_mw, energy_mj, eff_inf_per_j, gpu_util_pct, avg_ms, p50_ms, p90_ms, p95_ms, p99_ms, avg_throughput, p99_goodput, tail_ratio)

사용법:
  # step1_sweep.sh / step2_benchmark.sh 에서 자동 호출됨
  # 단독 실행도 가능:

  cd src
  python analysis/aggregate_results.py --mode sweep --exp-dir ../experiments/exp_YYYYMMDD
  python analysis/aggregate_results.py --mode benchmark --threshold 0.80
  python analysis/aggregate_results.py --mode all
"""

import os
import sys
import json
import csv
import glob
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths


# ── 경로 유틸 ────────────────────────────────────────────────────────────────

def find_run_dirs(exp_dir: str) -> list[str]:
    """eval/ 내 run_* 디렉토리를 시간순 정렬로 반환."""
    eval_base = os.path.join(exp_dir, "eval")
    if not os.path.isdir(eval_base):
        return []
    dirs = sorted(
        d for d in os.listdir(eval_base)
        if d.startswith("run_") and
        os.path.isdir(os.path.join(eval_base, d))
    )
    return [os.path.join(eval_base, d) for d in dirs]


def run_id(run_dir: str) -> str:
    """run_YYYYMMDD_HHMMSS 형태에서 ID 추출."""
    return os.path.basename(run_dir)


# ── Sweep 집계 ───────────────────────────────────────────────────────────────

def aggregate_ee_sweep(run_dirs: list[str]) -> list[dict]:
    """모든 run에서 ee_sweep_results.json 수집."""
    rows = []
    for rd in run_dirs:
        json_path = os.path.join(rd, "trt_sweep", "ee_sweep_results.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] EE sweep JSON 읽기 실패: {json_path} — {e}")
            continue

        rid = run_id(rd)
        for thr_str, r in sorted(data.items(), key=lambda x: float(x[0])):
            exit_rate = r.get("exit_rate", [None, None, None])
            rows.append({
                "run_id":       rid,
                "threshold":    r.get("threshold", float(thr_str)),
                "accuracy":     r.get("accuracy"),
                "ee1_rate_pct": exit_rate[0] if len(exit_rate) > 0 else None,
                "ee2_rate_pct": exit_rate[1] if len(exit_rate) > 1 else None,
                "main_rate_pct":exit_rate[2] if len(exit_rate) > 2 else None,
                "avg_ms":       r.get("avg_ms"),
                "p50_ms":       r.get("p50_ms"),
                "p99_ms":       r.get("p99_ms"),
            })
    return rows


def aggregate_vee_sweep(run_dirs: list[str]) -> list[dict]:
    """모든 run에서 vee_sweep_results.json 수집."""
    rows = []
    for rd in run_dirs:
        json_path = os.path.join(rd, "trt_sweep", "vee_sweep_results.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] VEE sweep JSON 읽기 실패: {json_path} — {e}")
            continue

        rid = run_id(rd)
        for thr_str, r in sorted(data.items(), key=lambda x: float(x[0])):
            exit_rate = r.get("exit_rate", [None, None])
            rows.append({
                "run_id":        rid,
                "threshold":     r.get("threshold", float(thr_str)),
                "accuracy":      r.get("accuracy"),
                "exit1_rate_pct":exit_rate[0] if len(exit_rate) > 0 else None,
                "main_rate_pct": exit_rate[1] if len(exit_rate) > 1 else None,
                "avg_ms":        r.get("avg_ms"),
                "p50_ms":        r.get("p50_ms"),
                "p99_ms":        r.get("p99_ms"),
            })
    return rows


# ── Grid Search 집계 ─────────────────────────────────────────────────────────

def aggregate_grid(run_dirs: list[str], threshold: float = None) -> list[dict]:
    """모든 run에서 grid_search_thr*.json 수집."""
    rows = []
    for rd in run_dirs:
        grid_dir = os.path.join(rd, "hybrid_grid")
        if not os.path.isdir(grid_dir):
            continue

        # threshold 지정 시 해당 파일만, 아니면 전부
        if threshold is not None:
            pattern = os.path.join(grid_dir, f"grid_search_thr{threshold:.2f}.json")
            json_files = glob.glob(pattern)
        else:
            json_files = sorted(glob.glob(os.path.join(grid_dir, "grid_search_thr*.json")))

        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARN] Grid JSON 읽기 실패: {json_path} — {e}")
                continue

            # threshold 값 파일명에서 추출 (예: grid_search_thr0.80.json → 0.80)
            fname = os.path.basename(json_path)
            try:
                thr_val = float(fname.replace("grid_search_thr", "").replace(".json", ""))
            except ValueError:
                thr_val = None

            rid = run_id(rd)
            for key, r in data.items():
                if r is None or "error" in r:
                    continue
                rows.append({
                    "run_id":         rid,
                    "threshold":      thr_val,
                    "batch_size":     r.get("batch_size"),
                    "timeout_ms":     r.get("timeout_ms"),
                    "accuracy":       r.get("accuracy"),
                    "exit1_rate_pct": r.get("exit1_rate"),
                    "fallback_rate_pct": r.get("fallback_rate"),
                    "avg_ms":         r.get("avg_ms"),
                    "p50_ms":         r.get("p50_ms"),
                    "p90_ms":         r.get("p90_ms"),
                    "p95_ms":         r.get("p95_ms"),
                    "p99_ms":         r.get("p99_ms"),
                    "avg_throughput": r.get("avg_throughput"),
                    "p99_goodput":    r.get("p99_goodput"),
                    "tail_ratio_p99_p50": r.get("tail_ratio_p99_p50"),
                })
    return rows


# ── Benchmark Comparison 집계 ────────────────────────────────────────────────

def aggregate_benchmark(run_dirs: list[str], threshold: float = None) -> list[dict]:
    """모든 run에서 compare_thr*.json 수집."""
    rows = []
    for rd in run_dirs:
        bench_dir = os.path.join(rd, "benchmark_comparison")
        if not os.path.isdir(bench_dir):
            continue

        if threshold is not None:
            pattern = os.path.join(bench_dir, f"compare_thr{threshold:.2f}.json")
            json_files = glob.glob(pattern)
        else:
            json_files = sorted(glob.glob(os.path.join(bench_dir, "compare_thr*.json")))

        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [WARN] Benchmark JSON 읽기 실패: {json_path} — {e}")
                continue

            fname = os.path.basename(json_path)
            try:
                thr_val = float(fname.replace("compare_thr", "").replace(".json", ""))
            except ValueError:
                thr_val = None

            rid = run_id(rd)
            for model_label, r in data.items():
                rows.append({
                    "run_id":        rid,
                    "threshold":     thr_val,
                    "model":         model_label,
                    "accuracy":      r.get("accuracy"),
                    "exit_info":     r.get("exit_info"),
                    "power_mw":      r.get("power_mw"),
                    "energy_mj":     r.get("energy_mj"),
                    "eff_inf_per_j": r.get("eff_inf_per_j"),
                    "gpu_util_pct":  r.get("gpu_util_pct"),
                    "avg_ms":        r.get("avg_ms"),
                    "p50_ms":        r.get("p50_ms"),
                    "p90_ms":        r.get("p90_ms"),
                    "p95_ms":        r.get("p95_ms"),
                    "p99_ms":        r.get("p99_ms"),
                    "avg_throughput":r.get("avg_throughput"),
                    "p90_goodput":   r.get("p90_goodput"),
                    "p95_goodput":   r.get("p95_goodput"),
                    "p99_goodput":   r.get("p99_goodput"),
                    "tail_ratio_p99_p50": r.get("tail_ratio_p99_p50"),
                })
    return rows


# ── CSV 저장 ─────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_path: str):
    if not rows:
        print(f"  [SKIP] 집계 데이터 없음: {out_path}")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  저장 완료 ({len(rows)}행): {out_path}")


# ── 요약 통계 출력 ───────────────────────────────────────────────────────────

def print_sweep_summary(ee_rows: list[dict], vee_rows: list[dict]):
    """threshold별 평균 accuracy / p99 latency 출력."""
    def summarize(rows, label):
        if not rows:
            return
        from collections import defaultdict
        by_thr = defaultdict(list)
        for r in rows:
            by_thr[r["threshold"]].append(r)
        print(f"\n  [{label} — threshold별 평균 (run {len(set(r['run_id'] for r in rows))}회)]")
        print(f"  {'thr':>6}  {'acc_mean':>10}  {'acc_std':>9}  {'p99_mean_ms':>12}  {'p99_std_ms':>11}  {'n_runs':>6}")
        print(f"  {'-'*65}")
        import statistics
        for thr in sorted(by_thr.keys()):
            grp = by_thr[thr]
            accs = [r["accuracy"] for r in grp if r["accuracy"] is not None]
            p99s = [r["p99_ms"]   for r in grp if r["p99_ms"]   is not None]
            acc_mean = statistics.mean(accs) if accs else float("nan")
            acc_std  = statistics.stdev(accs) if len(accs) > 1 else 0.0
            p99_mean = statistics.mean(p99s) if p99s else float("nan")
            p99_std  = statistics.stdev(p99s) if len(p99s) > 1 else 0.0
            print(f"  {thr:>6.2f}  {acc_mean:>10.4f}  {acc_std:>9.4f}  "
                  f"{p99_mean:>12.2f}  {p99_std:>11.2f}  {len(grp):>6}")

    summarize(ee_rows,  "EE  (3-seg)")
    summarize(vee_rows, "VEE (2-seg)")


def print_benchmark_summary(bench_rows: list[dict]):
    """모델별 평균 accuracy / p99 latency 출력."""
    if not bench_rows:
        return
    from collections import defaultdict
    import statistics
    by_model = defaultdict(list)
    for r in bench_rows:
        by_model[r["model"]].append(r)
    n_runs = len(set(r["run_id"] for r in bench_rows))
    print(f"\n  [Benchmark — 모델별 평균 (run {n_runs}회)]")
    print(f"  {'model':>12}  {'acc_mean':>10}  {'p99_mean_ms':>12}  {'p99_std_ms':>11}  {'throughput':>10}")
    print(f"  {'-'*65}")
    for model in sorted(by_model.keys()):
        grp = by_model[model]
        accs = [r["accuracy"]      for r in grp if r["accuracy"]      is not None]
        p99s = [r["p99_ms"]        for r in grp if r["p99_ms"]        is not None]
        tps  = [r["avg_throughput"] for r in grp if r["avg_throughput"] is not None]
        print(f"  {model:>12}  "
              f"{(statistics.mean(accs) if accs else float('nan')):>10.4f}  "
              f"{(statistics.mean(p99s) if p99s else float('nan')):>12.2f}  "
              f"{(statistics.stdev(p99s) if len(p99s) > 1 else 0.0):>11.2f}  "
              f"{(statistics.mean(tps) if tps else float('nan')):>10.1f}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N번 반복 실험 결과 JSON → CSV 취합")
    parser.add_argument("--mode",      choices=["sweep", "benchmark", "all"],
                        default="all",
                        help="집계 모드: sweep / benchmark / all (기본: all)")
    parser.add_argument("--exp-dir",   type=str, default=None,
                        help="실험 디렉토리 경로 (기본: 최신 exp_* 자동 감지)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="benchmark 모드: 특정 threshold만 집계 (기본: 전체)")
    # sweep 출력 경로
    parser.add_argument("--out-ee",    type=str, default=None,
                        help="EE sweep CSV 출력 경로")
    parser.add_argument("--out-vee",   type=str, default=None,
                        help="VEE sweep CSV 출력 경로")
    # benchmark 출력 경로
    parser.add_argument("--out-grid",  type=str, default=None,
                        help="Grid search CSV 출력 경로")
    parser.add_argument("--out-bench", type=str, default=None,
                        help="Benchmark comparison CSV 출력 경로")
    args = parser.parse_args()

    # 실험 디렉토리 결정
    if args.exp_dir:
        exp_dir = os.path.realpath(args.exp_dir)
    else:
        exp_dir = paths.EXPERIMENTS_DIR
    exp_name = os.path.basename(exp_dir)

    print(f"\n{'='*60}")
    print(f"  aggregate_results.py")
    print(f"  실험 디렉: {exp_name}")
    print(f"  모드:      {args.mode}")
    print(f"{'='*60}")

    # run 디렉토리 수집
    run_dirs = find_run_dirs(exp_dir)
    print(f"\n  발견된 run_* 디렉토리: {len(run_dirs)}개")
    if not run_dirs:
        print("  [ERROR] eval/ 내 run_* 디렉토리가 없습니다. 먼저 실험을 실행하세요.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(exp_dir, "eval")

    # ── Sweep 집계 ────────────────────────────────────────────
    if args.mode in ("sweep", "all"):
        print("\n  [EE Sweep 집계 중...]")
        ee_rows = aggregate_ee_sweep(run_dirs)
        print(f"  → {len(ee_rows)}개 row 수집")

        print("\n  [VEE Sweep 집계 중...]")
        vee_rows = aggregate_vee_sweep(run_dirs)
        print(f"  → {len(vee_rows)}개 row 수집")

        out_ee  = args.out_ee  or os.path.join(eval_dir, f"aggregate_sweep_ee_{ts}.csv")
        out_vee = args.out_vee or os.path.join(eval_dir, f"aggregate_sweep_vee_{ts}.csv")
        save_csv(ee_rows,  out_ee)
        save_csv(vee_rows, out_vee)

        print_sweep_summary(ee_rows, vee_rows)

    # ── Benchmark 집계 ────────────────────────────────────────
    if args.mode in ("benchmark", "all"):
        thr_label = f"thr{args.threshold:.2f}_" if args.threshold else ""

        print("\n  [Grid Search 집계 중...]")
        grid_rows = aggregate_grid(run_dirs, threshold=args.threshold)
        print(f"  → {len(grid_rows)}개 row 수집")

        print("\n  [Benchmark Comparison 집계 중...]")
        bench_rows = aggregate_benchmark(run_dirs, threshold=args.threshold)
        print(f"  → {len(bench_rows)}개 row 수집")

        out_grid  = args.out_grid  or os.path.join(eval_dir, f"aggregate_grid_{thr_label}{ts}.csv")
        out_bench = args.out_bench or os.path.join(eval_dir, f"aggregate_benchmark_{thr_label}{ts}.csv")
        save_csv(grid_rows,  out_grid)
        save_csv(bench_rows, out_bench)

        print_benchmark_summary(bench_rows)

    print(f"\n{'='*60}")
    print(f"  집계 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
