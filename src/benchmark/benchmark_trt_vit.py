"""
benchmark_trt_vit.py  —  PlainViT vs 2-exit vs 3-exit TRT 3-way 비교 (Orin)

고정된 threshold에서 3개 모델을 N회 반복 추론하여 latency/accuracy/exit-rate 비교.
sweep 결과에서 최적 threshold 결정 후 실행하는 최종 벤치마크.

생성 파일 ({EXP_DIR}/eval/vit_trt_benchmark_N{N}_YYYYMMDD/):
  trt_benchmark_raw.json
  trt_benchmark_summary.csv
  trt_benchmark_latency.png          ← avg/p50/p99 bar chart
  trt_benchmark_tradeoff.png         ← accuracy vs avg latency scatter
  trt_benchmark_exit_rate.png        ← exit block 분포 bar chart

사용법 (Orin):
  cd src
  python benchmark/benchmark_trt_vit.py --thr-2exit 0.80 --thr-3exit 0.75 --n 30
  python benchmark/benchmark_trt_vit.py --thr-2exit 0.80 --n 30 --latency-only

인자:
  --thr-2exit      2-exit 모델 threshold (기본: 0.80)
  --thr-3exit      3-exit 모델 threshold (기본: 0.80)
  --n              반복 횟수 (기본: 30)
  --num-samples    샘플 수 (기본: 1000)
  --exit-blocks-2  2-exit 블록 번호 (기본: 8 12)
  --exit-blocks-3  3-exit 블록 번호 (기본: 6 9 12)
  --skip-2exit     2-exit 모델 제외
  --skip-3exit     3-exit 모델 제외
  --data-root      ImageNet val 루트
  --latency-only   accuracy 생략, latency만 측정
  --warmup         warmup 샘플 수 (기본: 20)
  --out-dir        결과 저장 디렉토리
"""

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from infer.infer_trt_vit_selective import (
    SelectiveViTTRT, PlainViTTRT,
    load_selective_vit_trt, load_plain_vit_trt,
)
from benchmark.run_vit_trt_sweep import load_imagenet_samples, make_dummy_samples


# ── 단일 모델 N회 벤치마크 ───────────────────────────────────────────────────

def bench_selective(engine: SelectiveViTTRT, images, labels,
                    threshold: float, N: int,
                    warmup: int = 20, latency_only: bool = False):
    exit_blocks = engine.exit_blocks
    n_exits     = engine.n_segs

    all_latencies   = []
    all_exit_counts = [0] * n_exits
    all_correct     = 0
    n_total         = 0

    for run_idx in range(N):
        for i, (img, lbl) in enumerate(zip(images, labels)):
            img_cuda = img.cuda()
            logits, exit_block, lat_ms = engine.infer(img_cuda, threshold=threshold)

            if i >= warmup:
                all_latencies.append(lat_ms)
                head_idx = exit_blocks.index(exit_block)
                all_exit_counts[head_idx] += 1
                n_total += 1

                if not latency_only:
                    pred = logits.argmax(dim=1).item()
                    if pred == lbl:
                        all_correct += 1

    lat = np.array(all_latencies) if all_latencies else np.array([0.0])
    return {
        'model':           f"EE-ViT-{n_exits}exit",
        'threshold':        threshold,
        'accuracy':        (all_correct / n_total) if (n_total > 0 and not latency_only) else None,
        'exit_blocks':      exit_blocks,
        'exit_counts':      all_exit_counts,
        'exit_rate':       [c / n_total * 100 for c in all_exit_counts] if n_total > 0 else [0.0]*n_exits,
        'avg_ms':          float(np.mean(lat)),
        'p50_ms':          float(np.percentile(lat, 50)),
        'p99_ms':          float(np.percentile(lat, 99)),
        'std_ms':          float(np.std(lat)),
        'avg_exit_block':  sum(exit_blocks[i] * all_exit_counts[i] for i in range(n_exits)) / n_total if n_total > 0 else exit_blocks[-1],
        'n_runs':           N,
        'n_samples':        len(images),
        'all_latencies_ms': all_latencies,
    }


def bench_plain(engine: PlainViTTRT, images, labels,
                N: int, warmup: int = 20, latency_only: bool = False):
    all_latencies = []
    all_correct   = 0
    n_total       = 0

    for run_idx in range(N):
        for i, (img, lbl) in enumerate(zip(images, labels)):
            img_cuda = img.cuda()
            logits, lat_ms = engine.infer(img_cuda)

            if i >= warmup:
                all_latencies.append(lat_ms)
                n_total += 1
                if not latency_only:
                    pred = logits.argmax(dim=1).item()
                    if pred == lbl:
                        all_correct += 1

    lat = np.array(all_latencies) if all_latencies else np.array([0.0])
    return {
        'model':           'PlainViT',
        'threshold':        None,
        'accuracy':        (all_correct / n_total) if (n_total > 0 and not latency_only) else None,
        'exit_blocks':      [12],
        'exit_counts':      [n_total],
        'exit_rate':       [100.0],
        'avg_ms':          float(np.mean(lat)),
        'p50_ms':          float(np.percentile(lat, 50)),
        'p99_ms':          float(np.percentile(lat, 99)),
        'std_ms':          float(np.std(lat)),
        'avg_exit_block':  12.0,
        'n_runs':           N,
        'n_samples':        len(images),
        'all_latencies_ms': all_latencies,
    }


# ── 결과 저장 ─────────────────────────────────────────────────────────────────

def save_results_json(results: dict, out_path: str):
    payload = {k: {kk: vv for kk, vv in v.items() if kk != 'all_latencies_ms'}
               for k, v in results.items()}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  raw JSON saved: {out_path}")


def save_results_csv(results: dict, out_path: str):
    fields = ['model', 'threshold', 'accuracy', 'avg_ms', 'p50_ms',
              'p99_ms', 'std_ms', 'avg_exit_block', 'n_runs', 'n_samples']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results.values():
            row = {k: round(r[k], 4) if isinstance(r[k], float) else r[k]
                   for k in fields if k in r}
            if row.get('accuracy') is not None:
                row['accuracy'] = round(row['accuracy'] * 100, 3)
            writer.writerow(row)
    print(f"  summary CSV saved: {out_path}")


# ── 플롯 ─────────────────────────────────────────────────────────────────────

def plot_latency(results: dict, N: int, out_path: str):
    models  = list(results.keys())
    colors  = ['steelblue', 'darkorange', 'seagreen']
    metrics = ['avg_ms', 'p50_ms', 'p99_ms']
    labels  = ['avg', 'p50', 'p99']

    x     = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    for mi, (metric, label) in enumerate(zip(metrics, labels)):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(x + (mi - 1) * width, vals, width,
                      label=label, color=colors[mi], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'TRT Latency Comparison  (N={N} runs, Jetson AGX Orin)', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  latency bar chart saved: {out_path}")


def plot_tradeoff(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'PlainViT': 'steelblue', 'EE-ViT-2exit': 'darkorange', 'EE-ViT-3exit': 'seagreen'}
    markers = {'PlainViT': 'D', 'EE-ViT-2exit': 'o', 'EE-ViT-3exit': 's'}

    for key, r in results.items():
        if r['accuracy'] is None:
            continue
        color  = colors.get(r['model'], 'gray')
        marker = markers.get(r['model'], 'o')
        label  = r['model']
        if r['threshold'] is not None:
            label += f" (thr={r['threshold']:.2f})"
        ax.scatter(r['avg_ms'], r['accuracy'] * 100,
                   s=100, color=color, marker=marker, label=label, zorder=5)

    ax.set_xlabel('Avg Latency (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Latency Tradeoff  (Jetson AGX Orin TRT)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  tradeoff plot saved: {out_path}")


def plot_exit_rate(results: dict, out_path: str):
    ee_results = {k: r for k, r in results.items() if r['model'] != 'PlainViT'}
    if not ee_results:
        return

    fig, axes = plt.subplots(1, len(ee_results), figsize=(6 * len(ee_results), 5))
    if len(ee_results) == 1:
        axes = [axes]

    for ax, (key, r) in zip(axes, ee_results.items()):
        blocks = [f'B{b}' for b in r['exit_blocks']]
        rates  = r['exit_rate']
        colors = ['royalblue', 'darkorange', 'seagreen'][:len(blocks)]
        bars   = ax.bar(blocks, rates, color=colors, alpha=0.85, edgecolor='white')
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Exit Rate (%)')
        ax.set_title(f"{r['model']}  (thr={r['threshold']:.2f})", fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Exit Block Distribution  (Jetson AGX Orin TRT)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  exit rate chart saved: {out_path}")


def print_comparison_table(results: dict):
    print(f"\n{'='*80}")
    print(f"  TRT Benchmark — Jetson AGX Orin")
    print(f"{'='*80}")
    header = f"  {'Model':<22} {'Thr':>6} {'Acc':>8} {'avg_ms':>8} {'p50_ms':>8} {'p99_ms':>8} {'AvgBlock':>10}"
    print(header)
    print(f"  {'-'*74}")
    for r in results.values():
        acc_str  = f"{r['accuracy']*100:>7.2f}%" if r['accuracy'] is not None else f"{'N/A':>8}"
        thr_str  = f"{r['threshold']:.2f}" if r['threshold'] is not None else "  -  "
        avg_blk  = r.get('avg_exit_block', 12)
        print(f"  {r['model']:<22} {thr_str:>6}  {acc_str}  "
              f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
              f"{avg_blk:>9.2f}")
    print(f"{'='*80}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PlainViT vs SelectiveExitViT TRT 3-way Benchmark (Orin)'
    )
    parser.add_argument('--thr-2exit',     type=float, default=0.80)
    parser.add_argument('--thr-3exit',     type=float, default=0.80)
    parser.add_argument('--n',             type=int,   default=30)
    parser.add_argument('--num-samples',   type=int,   default=1000)
    parser.add_argument('--exit-blocks-2', type=int,   nargs='+', default=[8, 12])
    parser.add_argument('--exit-blocks-3', type=int,   nargs='+', default=[6, 9, 12])
    parser.add_argument('--skip-2exit',    action='store_true')
    parser.add_argument('--skip-3exit',    action='store_true')
    parser.add_argument('--data-root',     type=str,   default=None)
    parser.add_argument('--latency-only',  action='store_true')
    parser.add_argument('--num-workers',   type=int,   default=2)
    parser.add_argument('--warmup',        type=int,   default=20)
    parser.add_argument('--out-dir',       type=str,   default=None)
    args = parser.parse_args()

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval',
        f'vit_trt_benchmark_N{args.n}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"N={args.n}  samples={args.num_samples}  warmup={args.warmup}")
    print(f"Output: {out_dir}\n")

    # ── 데이터 로드 ──
    latency_only = args.latency_only
    if not latency_only and args.data_root is None:
        from utils import load_config
        cfg = load_config('configs/train.yaml')
        dr = cfg.get('vit', {}).get('data_root',
             cfg.get('imagenet', {}).get('data_root', None))
        if dr is None or not os.path.isdir(str(dr)):
            print("[WARN] data_root 없음 → latency-only 모드")
            latency_only = True
        else:
            args.data_root = dr

    if latency_only:
        images, labels = make_dummy_samples(args.num_samples)
    else:
        images, labels = load_imagenet_samples(
            args.num_samples, args.data_root, args.num_workers)

    results = {}

    # ── PlainViT ──
    print("Loading PlainViT TRT engine ...")
    plain_engine = load_plain_vit_trt()
    print(f"  Benchmarking PlainViT (N={args.n}) ...")
    r = bench_plain(plain_engine, images, labels,
                    args.n, args.warmup, latency_only)
    results['plain'] = r
    print(f"  → avg={r['avg_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms"
          + (f"  acc={r['accuracy']*100:.2f}%" if r['accuracy'] else ""))

    # ── 2-exit ──
    if not args.skip_2exit:
        print(f"\nLoading 2-exit TRT engines (B{args.exit_blocks_2[0]}+B{args.exit_blocks_2[-1]}) ...")
        ee2_engine = load_selective_vit_trt(args.exit_blocks_2)
        print(f"  Benchmarking 2-exit (thr={args.thr_2exit}, N={args.n}) ...")
        r = bench_selective(ee2_engine, images, labels,
                            args.thr_2exit, args.n, args.warmup, latency_only)
        results['2exit'] = r
        print(f"  → avg={r['avg_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms"
              + (f"  acc={r['accuracy']*100:.2f}%" if r['accuracy'] else "")
              + f"  exit_rate={[f'{x:.0f}%' for x in r['exit_rate']]}")

    # ── 3-exit ──
    if not args.skip_3exit:
        print(f"\nLoading 3-exit TRT engines ({'B'+'+B'.join(str(b) for b in args.exit_blocks_3)}) ...")
        ee3_engine = load_selective_vit_trt(args.exit_blocks_3)
        print(f"  Benchmarking 3-exit (thr={args.thr_3exit}, N={args.n}) ...")
        r = bench_selective(ee3_engine, images, labels,
                            args.thr_3exit, args.n, args.warmup, latency_only)
        results['3exit'] = r
        print(f"  → avg={r['avg_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms"
              + (f"  acc={r['accuracy']*100:.2f}%" if r['accuracy'] else "")
              + f"  exit_rate={[f'{x:.0f}%' for x in r['exit_rate']]}")

    # ── 저장 ──
    print("\nSaving results ...")
    save_results_json(results, os.path.join(out_dir, 'trt_benchmark_raw.json'))
    save_results_csv(results, os.path.join(out_dir, 'trt_benchmark_summary.csv'))

    print("Generating plots ...")
    plot_latency(results, args.n, os.path.join(out_dir, 'trt_benchmark_latency.png'))
    if not latency_only:
        plot_tradeoff(results, os.path.join(out_dir, 'trt_benchmark_tradeoff.png'))
    plot_exit_rate(results, os.path.join(out_dir, 'trt_benchmark_exit_rate.png'))

    print_comparison_table(results)
    print(f"Done! Results at:\n  {out_dir}")


if __name__ == '__main__':
    main()
