"""
run_vit_compare.py  —  PlainViT vs EE-ViT 비교 분석 (PyTorch 기반)

학습된 EE-ViT 체크포인트와 timm pretrained PlainViT-B/16을 동일한
ImageNet val 샘플로 비교하여 threshold별 accuracy/latency 트레이드오프를 분석.

비교 목적:
  - PlainViT-B/16 (pretrained, 수정 없음) 의 accuracy/latency 를 baseline으로 삼아
    어느 threshold에서 EE-ViT 가 거의 같은 accuracy를 유지하면서 latency를 줄이는지 확인.

생성 파일 ({EXP_DIR}/eval/vit_compare_N{N}_YYYYMMDD_HHMMSS/):
  vit_compare_raw.json           ← 원시 결과 (Plain + EE-ViT)
  vit_compare_summary.csv        ← threshold별 비교 통계
  vit_compare_accuracy.png       ← accuracy vs threshold (plain 기준선 포함)
  vit_compare_latency.png        ← latency vs threshold (plain 기준선 포함)
  vit_compare_tradeoff.png       ← accuracy-latency tradeoff 커브
  vit_compare_exit_heatmap.png   ← EE-ViT exit block 분포 heatmap
  vit_compare_acc_heatmap.png    ← EE-ViT per-exit accuracy heatmap

사용법:
  cd src
  python benchmark/run_vit_compare.py --n 5
  python benchmark/run_vit_compare.py --n 3 --num-samples 1000
  python benchmark/run_vit_compare.py --n 5 --thresholds 0.5 0.7 0.9 0.95
  python benchmark/run_vit_compare.py --n 5 --checkpoint /path/to/best.pth

인자:
  --n              반복 횟수 (기본: 5)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     EE-ViT 탐색 threshold 목록 (기본: 0.1~0.99)
  --checkpoint     EE-ViT 체크포인트 경로 (기본: 최신 exp_*/ee_vit best.pth)
  --data-root      ImageNet 루트 경로
  --num-workers    DataLoader num_workers (기본: 4)
  --warmup         latency 측정 제외 warmup 샘플 수 (기본: 20)
  --out-dir        결과 저장 디렉토리 (기본: {EXP_DIR}/eval/vit_compare_N{N}_...)
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
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.plain_vit import PlainViT, build_model as build_plain
from models.ee_vit import EEViT, build_model as build_ee
from datasets.dataloader import get_dataloader
from utils import load_config


# ── 기본 threshold 목록 ───────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data(num_samples: int, data_root: str = None, num_workers: int = 4):
    """ImageNet val에서 num_samples개 로드 (batch_size=1, 샘플별 latency 측정용)."""
    cfg = load_config('configs/train.yaml')
    if data_root is None:
        data_root = cfg.get('vit', {}).get('data_root',
                    cfg.get('imagenet', {}).get('data_root',
                    cfg['dataset']['data_root']))

    _, test_loader, _ = get_dataloader(
        dataset='imagenet',
        batch_size=1,
        data_root=data_root,
        num_workers=num_workers,
        seed=cfg['train']['seed'],
    )

    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images.append(img)
        labels.append(lbl[0].item())

    print(f"  데이터 로드 완료: {len(images)}개 샘플 (ImageNet val)")
    return images, labels


# ── PlainViT 단일 실행 ────────────────────────────────────────────────────────

def run_plain_once(model: PlainViT, images, labels,
                   device: torch.device, warmup: int = 20):
    """
    PlainViT로 전체 샘플 추론 1회.
    Returns: accuracy, avg_ms, p50_ms, p99_ms, latencies_ms
    """
    latencies = []
    correct   = 0

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_dev = img.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(img_dev)           # [1, num_classes]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

        if logits.argmax(dim=1).item() == lbl:
            correct += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])
    return {
        'accuracy': correct / n,
        'avg_ms':   float(np.mean(lat)),
        'p50_ms':   float(np.percentile(lat, 50)),
        'p99_ms':   float(np.percentile(lat, 99)),
        'latencies_ms': latencies,
    }


def run_plain_n(model: PlainViT, images, labels,
                N: int, device: torch.device, warmup: int = 20):
    """PlainViT N회 반복. accuracy(1회)와 latency(N회 집계)."""
    runs     = []
    accuracy = None

    print(f"\n  [PlainViT]  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}")
    print(f"  {'-'*46}")

    for run_idx in range(N):
        r = run_plain_once(model, images, labels, device, warmup)
        if accuracy is None:
            accuracy = r['accuracy']
        runs.append({
            'run_idx':      run_idx,
            'avg_ms':       r['avg_ms'],
            'p50_ms':       r['p50_ms'],
            'p99_ms':       r['p99_ms'],
            'latencies_ms': r['latencies_ms'],
        })
        print(f"  [PlainViT]  run {run_idx+1:>3}/{N}  "
              f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}")

    p99s = [r['p99_ms'] for r in runs]
    avgs = [r['avg_ms'] for r in runs]
    p50s = [r['p50_ms'] for r in runs]
    return {
        'accuracy': accuracy,
        'runs': runs,
        'summary': {
            'p99_mean': float(np.mean(p99s)),
            'p99_std':  float(np.std(p99s)),
            'avg_mean': float(np.mean(avgs)),
            'avg_std':  float(np.std(avgs)),
            'p50_mean': float(np.mean(p50s)),
            'p50_std':  float(np.std(p50s)),
        }
    }


# ── EE-ViT 단일 threshold 1회 실행 ───────────────────────────────────────────

def run_ee_once(model: EEViT, images, labels, threshold: float,
                device: torch.device, warmup: int = 20):
    """
    threshold 고정 후 EE-ViT로 전체 샘플 추론 1회.
    Returns: accuracy, exit_counts, acc_per_exit, avg_exit_block, latencies, ...
    """
    n_exits          = model.NUM_BLOCKS
    exit_counts      = [0] * n_exits
    correct_per_exit = [0] * n_exits
    latencies        = []
    correct          = 0

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_dev = img.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits, exit_idx = model(img_dev, threshold=threshold)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

        block_i = exit_idx - 1
        exit_counts[block_i] += 1
        if logits.argmax(dim=1).item() == lbl:
            correct += 1
            correct_per_exit[block_i] += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])
    acc_per_exit = [
        correct_per_exit[i] / exit_counts[i] if exit_counts[i] > 0 else None
        for i in range(n_exits)
    ]
    return {
        'accuracy':       correct / n,
        'exit_counts':    exit_counts,
        'exit_rate':      [c / n * 100 for c in exit_counts],
        'acc_per_exit':   acc_per_exit,
        'avg_exit_block': sum((i + 1) * exit_counts[i] for i in range(n_exits)) / n,
        'avg_ms':  float(np.mean(lat)),
        'p50_ms':  float(np.percentile(lat, 50)),
        'p99_ms':  float(np.percentile(lat, 99)),
        'latencies_ms': latencies,
    }


def run_ee_n_sweeps(model: EEViT, images, labels,
                    thresholds: list, N: int,
                    device: torch.device, warmup: int = 20):
    """thresholds × N회 EE-ViT sweep."""
    results = {str(round(t, 2)): {
        'threshold':      round(t, 2),
        'accuracy':       None,
        'exit_rate':      None,
        'acc_per_exit':   None,
        'avg_exit_block': None,
        'runs': [],
    } for t in thresholds}

    total = N * len(thresholds)
    done  = 0

    print(f"\n  [EE-ViT]  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  "
          f"{'p50_ms':>8}  {'p99_ms':>8}  {'avg_exit':>9}  {'acc':>7}")
    print(f"  {'-'*70}")

    for run_idx in range(N):
        for thr in thresholds:
            key = str(round(thr, 2))
            r   = run_ee_once(model, images, labels, thr, device, warmup)

            if results[key]['accuracy'] is None:
                results[key]['accuracy']       = r['accuracy']
                results[key]['exit_rate']      = r['exit_rate']
                results[key]['acc_per_exit']   = r['acc_per_exit']
                results[key]['avg_exit_block'] = r['avg_exit_block']

            results[key]['runs'].append({
                'run_idx':      run_idx,
                'avg_ms':       r['avg_ms'],
                'p50_ms':       r['p50_ms'],
                'p99_ms':       r['p99_ms'],
                'latencies_ms': r['latencies_ms'],
            })

            done += 1
            print(f"  [EE-ViT]  {thr:.2f}  run {run_idx+1:>3}/{N}  "
                  f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
                  f"{r['avg_exit_block']:>9.2f}  {r['accuracy']*100:>6.2f}%"
                  f"  [{done}/{total}]")

    for key in results:
        runs  = results[key]['runs']
        p99s  = [r['p99_ms'] for r in runs]
        avgs  = [r['avg_ms'] for r in runs]
        p50s  = [r['p50_ms'] for r in runs]
        results[key]['summary'] = {
            'p99_mean': float(np.mean(p99s)),
            'p99_std':  float(np.std(p99s)),
            'avg_mean': float(np.mean(avgs)),
            'avg_std':  float(np.std(avgs)),
            'p50_mean': float(np.mean(p50s)),
            'p50_std':  float(np.std(p50s)),
        }

    return results


# ── 저장: raw JSON ────────────────────────────────────────────────────────────

def save_raw_json(plain_result: dict, ee_results: dict,
                  n_samples: int, device_str: str,
                  thresholds: list, N: int, out_path: str):
    payload = {
        'meta': {
            'n_runs':     N,
            'n_samples':  n_samples,
            'thresholds': thresholds,
            'device':     device_str,
            'timestamp':  datetime.now().isoformat(),
        },
        'plain_vit': plain_result,
        'ee_vit':    ee_results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  raw JSON 저장: {out_path}")


# ── 저장: summary CSV ─────────────────────────────────────────────────────────

def save_summary_csv(plain_result: dict, ee_results: dict,
                     n_exits: int, out_path: str):
    """
    model, threshold, accuracy, acc_drop_vs_plain,
    avg_ms, p99_mean, p99_std, avg_exit_block, compute_savings_pct,
    exit_rate_b1..b12, acc_b1..b12
    """
    er_fields  = [f'exit_rate_b{i+1}' for i in range(n_exits)]
    acc_fields = [f'acc_b{i+1}'        for i in range(n_exits)]
    fieldnames = (
        ['model', 'threshold', 'accuracy', 'acc_drop_vs_plain',
         'avg_ms', 'avg_std', 'p50_mean', 'p99_mean', 'p99_std',
         'avg_exit_block', 'compute_savings_pct']
        + er_fields + acc_fields
    )

    rows = []
    plain_acc = plain_result['accuracy']
    ps        = plain_result['summary']

    # PlainViT row (threshold = 1.0 / full network)
    plain_row = {
        'model':               'plain_vit',
        'threshold':           '-',
        'accuracy':            round(plain_acc, 6),
        'acc_drop_vs_plain':   0.0,
        'avg_ms':              round(ps['avg_mean'], 4),
        'avg_std':             round(ps['avg_std'],  4),
        'p50_mean':            round(ps['p50_mean'], 4),
        'p99_mean':            round(ps['p99_mean'], 4),
        'p99_std':             round(ps['p99_std'],  4),
        'avg_exit_block':      n_exits,
        'compute_savings_pct': 0.0,
    }
    for i in range(n_exits):
        plain_row[f'exit_rate_b{i+1}'] = 100.0 if i == n_exits - 1 else 0.0
        plain_row[f'acc_b{i+1}']       = ''
    rows.append(plain_row)

    # EE-ViT rows
    for key, data in sorted(ee_results.items(), key=lambda x: float(x[0])):
        s       = data['summary']
        er      = data['exit_rate']    or [0.0]  * n_exits
        ape     = data['acc_per_exit'] or [None] * n_exits
        avg_blk = data['avg_exit_block'] or 0.0
        savings = (1 - avg_blk / n_exits) * 100
        acc     = data['accuracy'] or 0.0
        row = {
            'model':               'ee_vit',
            'threshold':           data['threshold'],
            'accuracy':            round(acc, 6),
            'acc_drop_vs_plain':   round((plain_acc - acc) * 100, 3),  # % point
            'avg_ms':              round(s['avg_mean'], 4),
            'avg_std':             round(s['avg_std'],  4),
            'p50_mean':            round(s['p50_mean'], 4),
            'p99_mean':            round(s['p99_mean'], 4),
            'p99_std':             round(s['p99_std'],  4),
            'avg_exit_block':      round(avg_blk, 4),
            'compute_savings_pct': round(savings, 2),
        }
        for i, rate in enumerate(er):
            row[f'exit_rate_b{i+1}'] = round(rate, 2)
        for i, a in enumerate(ape):
            row[f'acc_b{i+1}'] = round(a * 100, 2) if a is not None else ''
        rows.append(row)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  summary CSV 저장: {out_path}")


# ── Plot 1: Accuracy 비교 ─────────────────────────────────────────────────────

def plot_accuracy(plain_result: dict, ee_results: dict, save_path: str):
    """EE-ViT accuracy vs threshold + PlainViT 기준선."""
    thresholds = sorted(ee_results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]
    ee_accs    = [(ee_results[t]['accuracy'] or 0) * 100 for t in thresholds]
    plain_acc  = plain_result['accuracy'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PlainViT vs EE-ViT  —  Accuracy Comparison', fontsize=13)

    # ── subplot 1: accuracy vs threshold ──────────────────────────────────
    ax = axes[0]
    ax.plot(thr_vals, ee_accs, 'o-', color='steelblue', linewidth=2,
            markersize=6, label='EE-ViT (threshold sweep)')
    ax.axhline(plain_acc, color='tomato', linestyle='--', linewidth=2,
               label=f'PlainViT baseline ({plain_acc:.2f}%)')
    ax.fill_between(thr_vals, ee_accs, plain_acc,
                    where=[a < plain_acc for a in ee_accs],
                    alpha=0.12, color='tomato', label='Accuracy drop')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy vs Threshold')
    ax.set_ylim(max(0, min(ee_accs) - 5), min(100, plain_acc + 5))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── subplot 2: accuracy drop vs threshold ─────────────────────────────
    ax = axes[1]
    drops = [plain_acc - a for a in ee_accs]
    colors = ['tomato' if d > 1.0 else 'steelblue' for d in drops]
    bars = ax.bar(thr_vals, drops, width=0.04, color=colors, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5,
               label='±1% threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy Drop vs PlainViT (% point)')
    ax.set_title('Accuracy Drop  (blue = within 1%p, red = over 1%p)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    # 막대 위 수치
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{drop:.1f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  accuracy 비교 저장: {save_path}")


# ── Plot 2: Latency 비교 ─────────────────────────────────────────────────────

def plot_latency(plain_result: dict, ee_results: dict, N: int, save_path: str):
    """EE-ViT latency vs threshold + PlainViT 기준선."""
    thresholds = sorted(ee_results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]
    ee_avgs    = [ee_results[t]['summary']['avg_mean'] for t in thresholds]
    ee_p99s    = [ee_results[t]['summary']['p99_mean'] for t in thresholds]
    ee_p99stds = [ee_results[t]['summary']['p99_std']  for t in thresholds]
    plain_avg  = plain_result['summary']['avg_mean']
    plain_p99  = plain_result['summary']['p99_mean']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'PlainViT vs EE-ViT  —  Latency Comparison  (N={N}회)', fontsize=13)

    # ── subplot 1: avg latency ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(thr_vals, ee_avgs, 'o-', color='steelblue', linewidth=2,
            markersize=6, label='EE-ViT avg latency')
    ax.axhline(plain_avg, color='tomato', linestyle='--', linewidth=2,
               label=f'PlainViT avg ({plain_avg:.2f} ms)')
    savings_avg = [(plain_avg - e) / plain_avg * 100 for e in ee_avgs]
    ax2 = ax.twinx()
    ax2.plot(thr_vals, savings_avg, 's--', color='seagreen', linewidth=1.5,
             markersize=4, alpha=0.7, label='Latency savings (%)')
    ax2.set_ylabel('Latency Savings (%)', color='seagreen')
    ax2.tick_params(axis='y', labelcolor='seagreen')
    ax2.axhline(0, color='seagreen', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Avg Latency (ms)', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_title('Avg Latency vs Threshold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(alpha=0.3)

    # ── subplot 2: p99 latency ────────────────────────────────────────────
    ax = axes[1]
    ax.errorbar(thr_vals, ee_p99s, yerr=ee_p99stds,
                fmt='o-', color='steelblue', linewidth=2, markersize=6,
                capsize=5, capthick=1.5, label='EE-ViT P99 (mean ± std)')
    ax.fill_between(thr_vals,
                    [m - s for m, s in zip(ee_p99s, ee_p99stds)],
                    [m + s for m, s in zip(ee_p99s, ee_p99stds)],
                    alpha=0.12, color='steelblue')
    ax.axhline(plain_p99, color='tomato', linestyle='--', linewidth=2,
               label=f'PlainViT P99 ({plain_p99:.2f} ms)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency vs Threshold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  latency 비교 저장: {save_path}")


# ── Plot 3: Accuracy-Latency Tradeoff ────────────────────────────────────────

def plot_tradeoff(plain_result: dict, ee_results: dict, save_path: str):
    """
    x-axis: avg latency (ms)  y-axis: accuracy (%)
    각 threshold가 하나의 점. PlainViT는 별도 마커.
    이상적인 동작 영역(PlainViT 정확도에 가까우면서 latency가 낮은 곳)을 시각화.
    """
    thresholds = sorted(ee_results.keys(), key=float)
    ee_lats  = [ee_results[t]['summary']['avg_mean'] for t in thresholds]
    ee_accs  = [(ee_results[t]['accuracy'] or 0) * 100 for t in thresholds]
    thr_vals = [float(t) for t in thresholds]
    plain_lat = plain_result['summary']['avg_mean']
    plain_acc = plain_result['accuracy'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PlainViT vs EE-ViT  —  Accuracy-Latency Tradeoff', fontsize=13)

    # ── subplot 1: scatter + threshold 라벨 ──────────────────────────────
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    sc = ax.scatter(ee_lats, ee_accs, c=thr_vals, cmap='viridis',
                    s=80, zorder=3, label='EE-ViT thresholds')
    ax.plot(ee_lats, ee_accs, '-', color='steelblue', linewidth=1.2,
            alpha=0.5, zorder=2)
    ax.scatter([plain_lat], [plain_acc], marker='*', s=250, color='tomato',
               zorder=4, label=f'PlainViT ({plain_acc:.1f}%, {plain_lat:.1f}ms)')

    # threshold 라벨 (홀수 인덱스만)
    for i, (lat, acc, thr) in enumerate(zip(ee_lats, ee_accs, thr_vals)):
        if i % 2 == 0:
            ax.annotate(f'{thr:.2f}', (lat, acc),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)

    # PlainViT 기준선 (수평)
    ax.axhline(plain_acc, color='tomato', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(plain_lat, color='tomato', linestyle=':', linewidth=1, alpha=0.5)

    fig.colorbar(sc, ax=ax, label='Threshold')
    ax.set_xlabel('Avg Latency (ms)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy vs Latency\n(← 왼쪽 위가 이상적)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── subplot 2: compute savings vs accuracy drop ───────────────────────
    ax = axes[1]
    n_exits  = 12
    savings  = [(1 - ee_results[t]['avg_exit_block'] / n_exits) * 100
                if ee_results[t]['avg_exit_block'] else 0
                for t in thresholds]
    acc_drop = [plain_acc - acc for acc in ee_accs]

    sc2 = ax.scatter(savings, acc_drop, c=thr_vals, cmap='viridis',
                     s=80, zorder=3)
    ax.plot(savings, acc_drop, '-', color='steelblue', linewidth=1.2, alpha=0.5)
    for i, (sv, dr, thr) in enumerate(zip(savings, acc_drop, thr_vals)):
        if i % 2 == 0:
            ax.annotate(f'{thr:.2f}', (sv, dr),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5,
               label='1%p drop limit')
    ax.axvline(0, color='black', linewidth=0.8)

    fig.colorbar(sc2, ax=ax, label='Threshold')
    ax.set_xlabel('Compute Savings (%, based on avg exit block)')
    ax.set_ylabel('Accuracy Drop vs PlainViT (%p)')
    ax.set_title('Compute Savings vs Accuracy Drop\n(→ 오른쪽 아래가 이상적)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  tradeoff 커브 저장: {save_path}")


# ── Plot 4: Exit Block 분포 Heatmap (EE-ViT) ─────────────────────────────────

def plot_exit_heatmap(ee_results: dict, n_exits: int, save_path: str):
    thresholds = sorted(ee_results.keys(), key=float)
    matrix = np.array([
        ee_results[k]['exit_rate'] or [0.0] * n_exits
        for k in thresholds
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels([f'B{i+1}' for i in range(n_exits)], fontsize=8)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('EE-ViT  —  Exit Rate (%) per Block by Threshold')
    fig.colorbar(im, ax=ax, label='Exit Rate (%)')

    for row_i, thr in enumerate(thresholds):
        for col_j in range(n_exits):
            val = matrix[row_i, col_j]
            if val >= 5.0:
                ax.text(col_j, row_i, f'{val:.0f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if val < 60 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  exit heatmap 저장: {save_path}")


# ── Plot 5: Per-Exit Accuracy Heatmap (EE-ViT) ───────────────────────────────

def plot_acc_heatmap(ee_results: dict, n_exits: int, save_path: str):
    thresholds  = sorted(ee_results.keys(), key=float)
    acc_matrix  = np.full((len(thresholds), n_exits), np.nan)
    rate_matrix = np.zeros_like(acc_matrix)

    for row_i, key in enumerate(thresholds):
        ape = ee_results[key]['acc_per_exit'] or [None] * n_exits
        er  = ee_results[key]['exit_rate']    or [0.0]  * n_exits
        for col_j in range(n_exits):
            rate_matrix[row_i, col_j] = er[col_j]
            if ape[col_j] is not None:
                acc_matrix[row_i, col_j] = ape[col_j] * 100

    masked = np.ma.array(acc_matrix, mask=np.isnan(acc_matrix))
    cmap   = plt.cm.RdYlGn
    cmap.set_bad(color='#cccccc')

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels([f'B{i+1}' for i in range(n_exits)], fontsize=8)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('EE-ViT  —  Accuracy (%) of Samples Exiting at Each Block\n'
                 '(grey = no samples exited here)')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')

    for row_i in range(len(thresholds)):
        for col_j in range(n_exits):
            if rate_matrix[row_i, col_j] >= 1.0 and not np.isnan(acc_matrix[row_i, col_j]):
                val = acc_matrix[row_i, col_j]
                ax.text(col_j, row_i, f'{val:.0f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if 20 < val < 80 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  per-exit accuracy heatmap 저장: {save_path}")


# ── 콘솔 결과 출력 ────────────────────────────────────────────────────────────

def print_comparison_table(plain_result: dict, ee_results: dict, n_exits: int):
    thresholds = sorted(ee_results.keys(), key=float)
    plain_acc  = plain_result['accuracy'] * 100
    plain_avg  = plain_result['summary']['avg_mean']
    plain_p99  = plain_result['summary']['p99_mean']

    print(f"\n{'='*95}")
    print(f"{'PlainViT vs EE-ViT Comparison Summary':^95}")
    print(f"{'='*95}")
    print(f"{'Model/Thr':>12} {'Accuracy':>10} {'AccDrop':>9} {'AvgExit':>8} "
          f"{'Savings':>8} {'Avg(ms)':>9} {'P99(ms)':>9} {'P99±std':>8}")
    print(f"  {'-'*85}")

    # PlainViT row
    print(f"  {'PlainViT':>10}  "
          f"{plain_acc:>8.2f}%  "
          f"{'—':>7}  "
          f"{n_exits:>7}  "
          f"{'0.0%':>7}  "
          f"{plain_avg:>9.2f}  "
          f"{plain_p99:>9.2f}  "
          f"{'±'+str(round(plain_result['summary']['p99_std'],2)):>8}")

    # EE-ViT rows
    for key in thresholds:
        d       = ee_results[key]
        s       = d['summary']
        acc     = (d['accuracy'] or 0) * 100
        drop    = plain_acc - acc
        avg_blk = d['avg_exit_block'] or 0
        savings = (1 - avg_blk / n_exits) * 100
        marker  = ' ←' if abs(drop) <= 1.0 else ''
        print(f"  EE thr={d['threshold']:.2f}  "
              f"{acc:>8.2f}%  "
              f"{drop:>+7.2f}%p  "
              f"{avg_blk:>7.2f}  "
              f"{savings:>6.1f}%  "
              f"{s['avg_mean']:>9.2f}  "
              f"{s['p99_mean']:>9.2f}  "
              f"±{s['p99_std']:>5.2f}"
              f"{marker}")

    print(f"{'='*95}")
    print(f"  ← : PlainViT 대비 accuracy drop 1%p 이내 (threshold 후보)\n")

    # per-exit accuracy 표
    print(f"  EE-ViT Per-Exit Block Accuracy (%) — 각 block 탈출 샘플의 정확도")
    print(f"  (- = 해당 threshold에서 해당 block으로 탈출한 샘플 없음)")
    header = f"{'thr':>6}" + "".join(f"  B{i+1:>2}" for i in range(n_exits))
    print(f"  {'-'*len(header)}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for key in thresholds:
        d   = ee_results[key]
        ape = d['acc_per_exit'] or [None] * n_exits
        row = f"{d['threshold']:>6.2f}"
        for acc in ape:
            row += f"  {acc*100:>4.0f}" if acc is not None else f"    -"
        print(f"  {row}")
    print(f"  {'-'*len(header)}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PlainViT vs EE-ViT 비교 분석 (ImageNet 1000 샘플)')
    parser.add_argument('--n',           type=int,   default=5,
                        help='반복 횟수 (기본: 5)')
    parser.add_argument('--num-samples', type=int,   default=1000,
                        help='샘플 수 (기본: 1000)')
    parser.add_argument('--thresholds',  type=float, nargs='+',
                        default=DEFAULT_THRESHOLDS,
                        help='EE-ViT threshold 목록')
    parser.add_argument('--checkpoint',  type=str,   default=None,
                        help='EE-ViT 체크포인트 경로')
    parser.add_argument('--data-root',   type=str,   default=None,
                        help='ImageNet 루트 경로')
    parser.add_argument('--num-workers', type=int,   default=4)
    parser.add_argument('--warmup',      type=int,   default=20,
                        help='latency 측정 제외 warmup 샘플 수 (기본: 20)')
    parser.add_argument('--out-dir',     type=str,   default=None,
                        help='결과 저장 디렉토리')
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── EE-ViT 체크포인트 ────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or paths.latest_checkpoint('ee_vit', 'best.pth')
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] EE-ViT 체크포인트를 찾을 수 없습니다.")
        print(f"        --checkpoint 인자로 직접 지정하거나,")
        print(f"        먼저 train_vit_pipeline.sh 로 학습을 완료하세요.")
        sys.exit(1)
    print(f"EE-ViT 체크포인트: {ckpt_path}")

    # ── 출력 디렉토리 ────────────────────────────────────────────────────────
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'vit_compare_N{args.n}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"출력 디렉토리: {out_dir}")

    thresholds = sorted(set(args.thresholds))

    print(f"\n설정:")
    print(f"  N={args.n}회  samples={args.num_samples}  warmup={args.warmup}")
    print(f"  thresholds={thresholds}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n데이터 로드 중...")
    images, labels = load_data(args.num_samples, args.data_root, args.num_workers)

    num_classes = 1000

    # ── PlainViT 로드 및 실행 ────────────────────────────────────────────────
    print("\nPlainViT-B/16 로드 중 (pretrained)...")
    plain_model = build_plain(num_classes=num_classes).to(device)
    plain_model.eval()
    print("  PlainViT 로드 완료")

    print(f"\nPlainViT 추론 시작 ({args.n}회 × {len(images)}샘플)...")
    plain_result = run_plain_n(plain_model, images, labels,
                               args.n, device, args.warmup)
    print(f"\n  PlainViT  accuracy: {plain_result['accuracy']*100:.2f}%  "
          f"avg: {plain_result['summary']['avg_mean']:.2f}ms  "
          f"p99: {plain_result['summary']['p99_mean']:.2f}ms")

    # 메모리 절약: PlainViT 해제
    del plain_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── EE-ViT 로드 및 sweep ─────────────────────────────────────────────────
    print("\nEE-ViT-B/16 로드 중...")
    ee_model = build_ee(num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    ee_model.load_state_dict(state)
    ee_model.eval()
    print(f"  EE-ViT 로드 완료  (exit heads: {ee_model.NUM_BLOCKS}개)")

    print(f"\nEE-ViT Sweep 시작 ({args.n}회 × {len(thresholds)} thresholds × {len(images)}샘플)...")
    ee_results = run_ee_n_sweeps(ee_model, images, labels, thresholds,
                                 args.n, device, args.warmup)

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print("\n결과 저장 중...")
    save_raw_json(plain_result, ee_results, len(images), str(device),
                  thresholds, args.n,
                  os.path.join(out_dir, 'vit_compare_raw.json'))
    save_summary_csv(plain_result, ee_results, ee_model.NUM_BLOCKS,
                     os.path.join(out_dir, 'vit_compare_summary.csv'))

    print("그래프 생성 중...")
    plot_accuracy(plain_result, ee_results,
                  os.path.join(out_dir, 'vit_compare_accuracy.png'))
    plot_latency(plain_result, ee_results, args.n,
                 os.path.join(out_dir, 'vit_compare_latency.png'))
    plot_tradeoff(plain_result, ee_results,
                  os.path.join(out_dir, 'vit_compare_tradeoff.png'))
    plot_exit_heatmap(ee_results, ee_model.NUM_BLOCKS,
                      os.path.join(out_dir, 'vit_compare_exit_heatmap.png'))
    plot_acc_heatmap(ee_results, ee_model.NUM_BLOCKS,
                     os.path.join(out_dir, 'vit_compare_acc_heatmap.png'))

    print_comparison_table(plain_result, ee_results, ee_model.NUM_BLOCKS)

    print(f"완료! 결과 위치:\n  {out_dir}")
    print(f"\n  핵심 그래프:")
    print(f"    vit_compare_accuracy.png  — accuracy vs threshold + plain 기준선")
    print(f"    vit_compare_latency.png   — latency vs threshold + plain 기준선")
    print(f"    vit_compare_tradeoff.png  — accuracy-latency tradeoff 커브")
    print(f"    vit_compare_acc_heatmap.png — per-exit block accuracy heatmap")


if __name__ == '__main__':
    main()
