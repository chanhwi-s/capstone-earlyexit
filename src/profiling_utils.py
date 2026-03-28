"""
프로파일링 유틸리티 — 딥러닝 논문/엣지 추론 표준 지표

지표 목록:
  ── Latency Percentiles ──
  P50, P90, P95, P99, P999
  Mean, Std, Min, Max

  ── Throughput ──
  Avg Throughput        : 1000 / avg_latency_ms  (inferences/sec)
  P90 Goodput           : 1000 / P90_latency_ms  (inferences/sec)
  P95 Goodput           : 1000 / P95_latency_ms  (inferences/sec)
  P99 Goodput           : 1000 / P99_latency_ms  (inferences/sec)

  ── Goodput 개념 ──
  "Goodput" (effective throughput, SLO-constrained throughput):
  worst-case 보장 처리량. P90 goodput이란 "90%의 추론이 해당 시간 이내에
  완료되는 조건에서 달성 가능한 최대 처리량"을 의미한다.
  엣지 디바이스에서는 실시간성 보장이 중요하므로 avg throughput보다
  P90/P95 goodput이 실질적 성능 지표로 사용됨.

  ── Tail Latency Ratio ──
  P99/P50: tail latency가 median 대비 얼마나 나쁜지 (1에 가까울수록 안정적)
  P99/avg: tail latency가 평균 대비 얼마나 나쁜지

사용법:
  from profiling_utils import compute_latency_stats, print_latency_report, format_stats_row
"""

import numpy as np


def compute_latency_stats(latencies_ms: list | np.ndarray) -> dict:
    """레이턴시 배열로부터 전체 프로파일링 지표 계산.

    Args:
        latencies_ms: 밀리초 단위 레이턴시 리스트/배열

    Returns:
        dict: 모든 프로파일링 지표
    """
    lat = np.asarray(latencies_ms, dtype=np.float64)
    n   = len(lat)
    if n == 0:
        return {}

    avg = float(np.mean(lat))
    std = float(np.std(lat))

    p50  = float(np.percentile(lat, 50))
    p90  = float(np.percentile(lat, 90))
    p95  = float(np.percentile(lat, 95))
    p99  = float(np.percentile(lat, 99))
    p999 = float(np.percentile(lat, 99.9))

    return {
        # ── Count ──
        "n":             n,

        # ── Latency (ms) ──
        "avg_ms":        avg,
        "std_ms":        std,
        "min_ms":        float(np.min(lat)),
        "max_ms":        float(np.max(lat)),
        "p50_ms":        p50,
        "p90_ms":        p90,
        "p95_ms":        p95,
        "p99_ms":        p99,
        "p999_ms":       p999,

        # ── Throughput (inferences/sec) ──
        "avg_throughput":  1000.0 / avg  if avg  > 0 else 0,
        "p90_goodput":     1000.0 / p90  if p90  > 0 else 0,
        "p95_goodput":     1000.0 / p95  if p95  > 0 else 0,
        "p99_goodput":     1000.0 / p99  if p99  > 0 else 0,

        # ── Tail Latency Ratio ──
        "tail_ratio_p99_p50":  p99 / p50 if p50 > 0 else float("inf"),
        "tail_ratio_p99_avg":  p99 / avg if avg > 0 else float("inf"),
        "tail_ratio_p95_p50":  p95 / p50 if p50 > 0 else float("inf"),

        # ── Jitter (variance indicator) ──
        "iqr_ms":        float(np.percentile(lat, 75) - np.percentile(lat, 25)),
    }


def print_latency_report(stats: dict, label: str = ""):
    """프로파일링 결과를 가독성 좋게 출력."""
    if not stats:
        print(f"[{label}] 데이터 없음")
        return

    title = f"  {label}  " if label else "  Latency Report  "
    print(f"\n{'='*60}")
    print(f"{title:=^60}")
    print(f"{'='*60}")

    print(f"  Samples        : {stats['n']}")
    print()

    print("  ── Latency (ms) ──")
    print(f"  Mean ± Std     : {stats['avg_ms']:.3f} ± {stats['std_ms']:.3f}")
    print(f"  Min / Max      : {stats['min_ms']:.3f} / {stats['max_ms']:.3f}")
    print(f"  P50 (median)   : {stats['p50_ms']:.3f}")
    print(f"  P90            : {stats['p90_ms']:.3f}")
    print(f"  P95            : {stats['p95_ms']:.3f}")
    print(f"  P99            : {stats['p99_ms']:.3f}")
    print(f"  P99.9          : {stats['p999_ms']:.3f}")
    print(f"  IQR            : {stats['iqr_ms']:.3f}")
    print()

    print("  ── Throughput (inf/sec) ──")
    print(f"  Avg Throughput  : {stats['avg_throughput']:.1f}")
    print(f"  P90 Goodput     : {stats['p90_goodput']:.1f}")
    print(f"  P95 Goodput     : {stats['p95_goodput']:.1f}")
    print(f"  P99 Goodput     : {stats['p99_goodput']:.1f}")
    print()

    print("  ── Tail Ratio ──")
    print(f"  P99/P50        : {stats['tail_ratio_p99_p50']:.2f}x")
    print(f"  P99/Avg        : {stats['tail_ratio_p99_avg']:.2f}x")
    print(f"  P95/P50        : {stats['tail_ratio_p95_p50']:.2f}x")
    print(f"{'='*60}\n")


def format_stats_row(stats: dict, label: str = "") -> str:
    """한 줄 요약 포맷 (테이블 행으로 사용)."""
    if not stats:
        return f"{label:<20} (no data)"
    return (
        f"{label:<20} "
        f"avg={stats['avg_ms']:7.3f}ms  "
        f"p50={stats['p50_ms']:7.3f}ms  "
        f"p90={stats['p90_ms']:7.3f}ms  "
        f"p95={stats['p95_ms']:7.3f}ms  "
        f"p99={stats['p99_ms']:7.3f}ms  "
        f"goodput(p90)={stats['p90_goodput']:7.1f}inf/s  "
        f"tail={stats['tail_ratio_p99_p50']:.2f}x"
    )


def merge_stats_to_dict(stats: dict, prefix: str = "") -> dict:
    """stats dict의 키에 prefix를 붙여서 반환 (CSV/JSON 저장용)."""
    if not stats:
        return {}
    return {f"{prefix}{k}": v for k, v in stats.items()}
