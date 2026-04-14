"""
run_vit_selective_compare.py  —  PlainViT vs SelectiveExitViT 비교 분석

학습된 SelectiveExitViT 체크포인트와 timm pretrained PlainViT-B/16을 동일한
ImageNet val 샘플로 비교하여 threshold별 accuracy/latency 트레이드오프를 분석.

2-exit (B8+B12) 또는 3-exit (B6+B9+B12) 모두 지원.

생성 파일 ({EXP_DIR}/eval/vit_sel{N}exit_compare_N{N}_YYYYMMDD/):
  sel_compare_raw.json             ← 원시 결과 (Plain + SelectiveExitViT)
  sel_compare_summary.csv          ← threshold별 비교 통계
  sel_compare_accuracy.png         ← accuracy vs threshold (plain 기준선 포함)
  sel_compare_latency.png          ← latency vs threshold (plain 기준선 포함)
  sel_compare_tradeoff.png         ← accuracy-latency tradeoff 커브
  sel_compare_exit_heatmap.png     ← exit block 분포 heatmap
  sel_compare_acc_heatmap.png      ← per-exit accuracy heatmap

사용법:
  cd src
  python benchmark/run_vit_selective_compare.py --exit-blocks 8 12 --n 5
  python benchmark/run_vit_selective_compare.py --exit-blocks 6 9 12 --n 5
  python benchmark/run_vit_selective_compare.py --exit-blocks 8 12 --n 3 --num-samples 1000
  python benchmark/run_vit_selective_compare.py --exit-blocks 8 12 --checkpoint /path/to/best.pth

인자:
  --exit-blocks    모델의 exit 블록 번호 목록 (필수, 예: 8 12 또는 6 9 12)
  --n              반복 횟수 (기본: 5)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     EE 탐색 threshold 목록 (기본: 0.1~0.99)
  --checkpoint     SelectiveExitViT 체크포인트 경로
  --data-root      ImageNet 루트 경로
  --num-workers    DataLoader num_workers (기본: 4)
  --warmup         latency warmup 샘플 수 (기본: 20)
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
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.plain_vit import PlainViT, build_model as build_plain
from models.ee_vit_selective import SelectiveExitViT, build_model as build_selective
from datasets.dataloader import get_dataloader
from utils import load_config


DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data(num_samples: int, data_root: str = None, num_workers: int = 4):
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

    print(f"  Data loaded: {len(images)} samples (ImageNet val)")
    return images, labels


# ── PlainViT 실행 ─────────────────────────────────────────────────────────────

def run_plain_once(model: PlainViT, images, labels,
                   device: torch.device, warmup: int = 20):
    latencies = []
    correct   = 0

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_dev = img.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = model(img_dev)

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
        'accuracy':     correct / n,
        'avg_ms':       float(np.mean(lat)),
        'p50_ms':       float(np.percentile(lat, 50)),
        'p99_ms':       float(np.percentile(lat, 99)),
        'latencies_ms': latencies,
    }


def run_plain_n(model: PlainViT, images, labels,
                N: int, device: torch.device, warmup: int = 20):
    runs     = []
    accuracy = None

    print(f"\n  [PlainViT]  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}")
    print(f"  {'-'*50}")

    for run_idx in range(N):
        r = run_plain_once(model, images, labels, device, warmup)
        if accuracy is None:
            accuracy = r['accuracy']
        runs.append({'run_idx': run_idx, 'avg_ms': r['avg_ms'],
                     'p50_ms': r['p50_ms'], 'p99_ms': r['p99_ms'],
                     'latencies_ms': r['latencies_ms']})
        print(f"  [PlainViT]  run {run_idx+1:>3}/{N}  "
              f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}")

    p99s = [r['p99_ms'] for r in runs]
    avgs = [r['avg_ms'] for r in runs]
    p50s = [r['p50_ms'] for r in runs]
    return {
        'accuracy': accuracy,
        'runs': runs,
        'summary': {
            'p99_mean': float(np.mean(p99s)), 'p99_std': float(np.std(p99s)),
            'avg_mean': float(np.mean(avgs)), 'avg_std': float(np.std(avgs)),
            'p50_mean': float(np.mean(p50s)), 'p50_std': float(np.std(p50s)),
        }
    }


# ── SelectiveExitViT 실행 ─────────────────────────────────────────────────────

def run_sel_once(model: SelectiveExitViT, images, labels, threshold: float,
                 device: torch.device, warmup: int = 20):
    n_exits          = model.NUM_BLOCKS
    exit_blocks      = model.exit_blocks
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
            logits, exit_block = model(img_dev, threshold=threshold)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

        head_idx = exit_blocks.index(exit_block)
        exit_counts[head_idx] += 1
        if logits.argmax(dim=1).item() == lbl:
            correct += 1
            correct_per_exit[head_idx] += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])
    acc_per_exit = [
        correct_per_exit[i] / exit_counts[i] if exit_counts[i] > 0 else None
        for i in range(n_exits)
    ]
    avg_exit_block = sum(exit_blocks[i] * exit_counts[i] for i in range(n_exits)) / n

    return {
        'accuracy':       correct / n,
        'exit_counts':    exit_counts,
        'exit_rate':      [c / n * 100 for c in exit_counts],
        'acc_per_exit':   acc_per_exit,
        'avg_exit_block': avg_exit_block,
        'avg_ms':         float(np.mean(lat)),
        'p50_ms':         float(np.percentile(lat, 50)),
        'p99_ms':         float(np.percentile(lat, 99)),
        'latencies_ms':   latencies,
    }


def run_sel_n_sweeps(model: SelectiveExitViT, images, labels,
                     thresholds: list, N: int,
                     device: torch.device, warmup: int = 20):
    results = {str(round(t, 2)): {
        'threshold':      round(t, 2),
        'accuracy':       None,
        'exit_rate':      None,
        'acc_per_exit':   None,
        'avg_exit_block': None,
        'runs': [],
    } for t in thresholds}

    total      = N * len(thresholds)
    done       = 0
    labels_str = model.exit_block_labels

    print(f"\n  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}  {'acc':>7}")
    print(f"  {'-'*60}")

    for run_idx in range(N):
        for thr in thresholds:
            key = str(round(thr, 2))
            r   = run_sel_once(model, images, labels, thr, device, warmup)

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
            exit_str = "  ".join(
                f"{labels_str[i]}={r['exit_rate'][i]:.0f}%"
                for i in range(model.NUM_BLOCKS)
            )
            print(f"  {thr:.2f}  run {run_idx+1:>3}/{N}  "
                  f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
                  f"{r['accuracy']*100:>6.2f}%  [{done}/{total}]  {exit_str}")

    for key in results:
        runs  = results[key]['runs']
        p99s  = [r['p99_ms'] for r in runs]
        avgs  = [r['avg_ms'] for r in runs]
        p50s  = [r['p50_ms'] for r in runs]
        results[key]['summary'] = {
            'p99_mean': float(np.mean(p99s)), 'p99_std': float(np.std(p99s)),
            'avg_mean': float(np.mean(avgs)), 'avg_std': float(np.std(avgs)),
            'p50_mean': float(np.mean(p50s)), 'p50_std': float(np.std(p50s)),
        }

    return results


# ── 저장: raw JSON ────────────────────────────────────────────────────────────

def save_raw_json(plain_result: dict, sel_results: dict,
                  exit_blocks: list, N: int, n_samples: int,
                  device_str: str, thresholds: list, out_path: str):
    payload = {
        'meta': {
            'n_runs':      N,
            'n_samples':   n_samples,
            'device':      device_str,
            'exit_blocks': exit_blocks,
            'thresholds':  thresholds,
            'timestamp':   datetime.now().isoformat(),
        },
        'plain_vit': plain_result,
        'sel_ee_vit': sel_results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  raw JSON saved: {out_path}")


# ── 저장: summary CSV ─────────────────────────────────────────────────────────

def save_summary_csv(plain_result: dict, sel_results: dict,
                     exit_blocks: list, out_path: str):
    labels      = [f'B{b}' for b in exit_blocks]
    er_fields   = [f'exit_rate_{l.lower()}' for l in labels]
    acc_fields  = [f'acc_{l.lower()}' for l in labels]

    fieldnames = (
        ['model', 'threshold',
         'accuracy', 'acc_drop_vs_plain',
         'avg_exit_block', 'compute_savings_pct',
         'p99_mean', 'p99_std', 'avg_mean', 'avg_std',
         'latency_savings_pct']
        + er_fields + acc_fields
    )

    plain_acc     = plain_result['accuracy']
    plain_p99     = plain_result['summary']['p99_mean']
    plain_avg     = plain_result['summary']['avg_mean']
    rows          = []

    # Plain row
    rows.append({
        'model': 'PlainViT', 'threshold': '-',
        'accuracy': round(plain_acc, 6),
        'acc_drop_vs_plain': 0.0,
        'avg_exit_block': 12, 'compute_savings_pct': 0.0,
        'p99_mean': round(plain_p99, 4), 'p99_std': round(plain_result['summary']['p99_std'], 4),
        'avg_mean': round(plain_avg, 4), 'avg_std': round(plain_result['summary']['avg_std'], 4),
        'latency_savings_pct': 0.0,
        **{f: '' for f in er_fields},
        **{f: '' for f in acc_fields},
    })

    # SelectiveExit rows
    for key, data in sorted(sel_results.items(), key=lambda x: float(x[0])):
        s       = data['summary']
        er      = data['exit_rate']    or [0.0]  * len(exit_blocks)
        ape     = data['acc_per_exit'] or [None] * len(exit_blocks)
        avg_blk = data['avg_exit_block'] or exit_blocks[-1]
        acc     = data['accuracy'] or 0.0
        lat_sav = (1 - s['avg_mean'] / plain_avg) * 100 if plain_avg > 0 else 0.0

        row = {
            'model':                f'SelectiveExit({"+".join(labels)})',
            'threshold':            data['threshold'],
            'accuracy':             round(acc, 6),
            'acc_drop_vs_plain':    round((plain_acc - acc) * 100, 4),
            'avg_exit_block':       round(avg_blk, 4),
            'compute_savings_pct':  round((1 - avg_blk / 12) * 100, 2),
            'p99_mean':             round(s['p99_mean'], 4),
            'p99_std':              round(s['p99_std'],  4),
            'avg_mean':             round(s['avg_mean'], 4),
            'avg_std':              round(s['avg_std'],  4),
            'latency_savings_pct':  round(lat_sav, 2),
        }
        for i, lbl in enumerate(labels):
            row[f'exit_rate_{lbl.lower()}'] = round(er[i], 2)
        for i, lbl in enumerate(labels):
            a = ape[i]
            row[f'acc_{lbl.lower()}'] = round(a * 100, 2) if a is not None else ''
        rows.append(row)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  summary CSV saved: {out_path}")


# ── Plot 1: Accuracy 비교 ─────────────────────────────────────────────────────

def plot_accuracy(plain_result: dict, sel_results: dict,
                  exit_blocks: list, model_tag: str, save_path: str):
    labels     = [f'B{b}' for b in exit_blocks]
    plain_acc  = plain_result['accuracy'] * 100
    thresholds = sorted(sel_results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]
    sel_accs   = [sel_results[t]['accuracy'] * 100 for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'PlainViT vs {model_tag}  —  Accuracy Comparison', fontsize=13)

    ax = axes[0]
    ax.axhline(plain_acc, color='gray', linestyle='--', linewidth=2,
               label=f'PlainViT ({plain_acc:.2f}%)')
    ax.plot(thr_vals, sel_accs, 'o-', color='steelblue', linewidth=2,
            markersize=6, label=f'SelectiveExit ({"+".join(labels)})')
    ax.fill_between(thr_vals, plain_acc - 1, plain_acc + 0.5,
                    alpha=0.1, color='gray', label='±1%p band')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy vs Threshold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    drops = [plain_acc - a for a in sel_accs]
    colors = ['steelblue' if d <= 1.0 else 'tomato' for d in drops]
    ax.bar(range(len(thresholds)), drops, color=colors, alpha=0.8)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='1%p threshold')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{float(t):.2f}' for t in thresholds], rotation=45, fontsize=8)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy Drop vs PlainViT (%p)')
    ax.set_title('Accuracy Drop by Threshold\n(blue = within 1%p, red = over)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  accuracy plot saved: {save_path}")


# ── Plot 2: Latency 비교 ──────────────────────────────────────────────────────

def plot_latency(plain_result: dict, sel_results: dict,
                 model_tag: str, save_path: str):
    plain_p99  = plain_result['summary']['p99_mean']
    plain_avg  = plain_result['summary']['avg_mean']
    thresholds = sorted(sel_results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]

    sel_avg  = [sel_results[t]['summary']['avg_mean'] for t in thresholds]
    sel_p99  = [sel_results[t]['summary']['p99_mean'] for t in thresholds]
    sel_p99s = [sel_results[t]['summary']['p99_std']  for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'PlainViT vs {model_tag}  —  Latency Comparison', fontsize=13)

    ax = axes[0]
    ax.axhline(plain_avg, color='gray', linestyle='--', linewidth=2,
               label=f'PlainViT avg ({plain_avg:.1f} ms)')
    ax.axhline(plain_p99, color='lightcoral', linestyle=':', linewidth=2,
               label=f'PlainViT P99 ({plain_p99:.1f} ms)')
    ax.plot(thr_vals, sel_avg, 'o-', color='steelblue', linewidth=2,
            markersize=6, label='SelectiveExit avg')
    ax.plot(thr_vals, sel_p99, 's--', color='tomato', linewidth=2,
            markersize=5, label='SelectiveExit P99')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Avg & P99 Latency vs Threshold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    avg_savings = [(1 - a / plain_avg) * 100 if plain_avg > 0 else 0 for a in sel_avg]
    p99_savings = [(1 - p / plain_p99) * 100 if plain_p99 > 0 else 0 for p in sel_p99]
    ax.plot(thr_vals, avg_savings, 'o-', color='steelblue', linewidth=2,
            markersize=6, label='Avg latency savings')
    ax.plot(thr_vals, p99_savings, 's--', color='tomato', linewidth=2,
            markersize=5, label='P99 latency savings')
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Latency Savings (%)')
    ax.set_title('Latency Savings vs PlainViT (%)\n(positive = faster than PlainViT)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  latency plot saved: {save_path}")


# ── Plot 3: Accuracy-Latency Tradeoff ────────────────────────────────────────

def plot_tradeoff(plain_result: dict, sel_results: dict,
                  exit_blocks: list, model_tag: str, save_path: str):
    labels     = [f'B{b}' for b in exit_blocks]
    plain_acc  = plain_result['accuracy'] * 100
    plain_avg  = plain_result['summary']['avg_mean']
    thresholds = sorted(sel_results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]

    sel_accs    = [sel_results[t]['accuracy'] * 100 for t in thresholds]
    sel_avg_ms  = [sel_results[t]['summary']['avg_mean'] for t in thresholds]
    avg_blks    = [sel_results[t]['avg_exit_block'] or exit_blocks[-1] for t in thresholds]
    savings     = [(1 - b / 12) * 100 for b in avg_blks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'PlainViT vs {model_tag}  —  Accuracy-Latency Tradeoff', fontsize=13)

    # ── scatter: latency vs accuracy ──────────────────────────────────────────
    ax = axes[0]
    cmap   = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
    sc     = ax.scatter(sel_avg_ms, sel_accs, c=thr_vals, cmap='viridis',
                        s=80, zorder=5)
    ax.scatter([plain_avg], [plain_acc], c='red', marker='*', s=200,
               zorder=6, label='PlainViT', edgecolors='black')
    ax.plot(sel_avg_ms, sel_accs, '-', color='gray', alpha=0.4, linewidth=1)

    # annotate threshold values
    for i, (x, y, t) in enumerate(zip(sel_avg_ms, sel_accs, thr_vals)):
        ax.annotate(f'{t:.2f}', (x, y), xytext=(3, 4), textcoords='offset points',
                    fontsize=7, alpha=0.8)

    fig.colorbar(sc, ax=ax, label='Threshold')
    ax.set_xlabel('Avg Latency (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Latency\n(upper-left is ideal: high acc, low latency)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── scatter: compute savings vs accuracy drop ──────────────────────────────
    ax = axes[1]
    drops = [plain_acc - a for a in sel_accs]
    sc    = ax.scatter(savings, drops, c=thr_vals, cmap='viridis', s=80, zorder=5)
    ax.plot(savings, drops, '-', color='gray', alpha=0.4, linewidth=1)
    ax.axhline(1.0, color='tomato', linestyle='--', linewidth=1.5,
               label='1%p acc drop limit')

    for i, (x, y, t) in enumerate(zip(savings, drops, thr_vals)):
        ax.annotate(f'{t:.2f}', (x, y), xytext=(3, 4), textcoords='offset points',
                    fontsize=7, alpha=0.8)

    fig.colorbar(sc, ax=ax, label='Threshold')
    ax.set_xlabel('Compute Savings (%) vs 12-block full inference')
    ax.set_ylabel('Accuracy Drop vs PlainViT (%p)')
    ax.set_title('Compute Savings vs Accuracy Drop\n(lower-right is ideal: high savings, low drop)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  tradeoff plot saved: {save_path}")


# ── Plot 4: Exit Heatmap ──────────────────────────────────────────────────────

def plot_exit_heatmap(sel_results: dict, exit_blocks: list,
                      model_tag: str, save_path: str):
    labels     = [f'B{b}' for b in exit_blocks]
    n_exits    = len(exit_blocks)
    thresholds = sorted(sel_results.keys(), key=float)
    matrix     = []
    for key in thresholds:
        er = sel_results[key]['exit_rate'] or [0.0] * n_exits
        matrix.append(er)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'{model_tag}  —  Exit Block Distribution', fontsize=13)

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=9)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('Exit Rate (%) per Block')
    fig.colorbar(im, ax=ax, label='Exit Rate (%)')

    for row_i in range(len(thresholds)):
        for col_j in range(n_exits):
            val = matrix[row_i, col_j]
            if val >= 5.0:
                ax.text(col_j, row_i, f'{val:.0f}',
                        ha='center', va='center', fontsize=10,
                        color='black' if val < 60 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  exit heatmap saved: {save_path}")


# ── Plot 5: Per-Exit Accuracy Heatmap ────────────────────────────────────────

def plot_acc_heatmap(sel_results: dict, exit_blocks: list,
                     model_tag: str, save_path: str):
    labels     = [f'B{b}' for b in exit_blocks]
    n_exits    = len(exit_blocks)
    thresholds = sorted(sel_results.keys(), key=float)

    acc_matrix  = np.full((len(thresholds), n_exits), np.nan)
    rate_matrix = np.zeros((len(thresholds), n_exits))

    for row_i, key in enumerate(thresholds):
        ape = sel_results[key]['acc_per_exit'] or [None] * n_exits
        er  = sel_results[key]['exit_rate']    or [0.0]  * n_exits
        for col_j in range(n_exits):
            rate_matrix[row_i, col_j] = er[col_j]
            if ape[col_j] is not None:
                acc_matrix[row_i, col_j] = ape[col_j] * 100

    masked = np.ma.array(acc_matrix, mask=np.isnan(acc_matrix))
    cmap   = plt.cm.RdYlGn
    cmap.set_bad(color='#cccccc')

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'{model_tag}  —  Per-Exit Block Accuracy', fontsize=13)

    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=9)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('Accuracy (%) per Exit Block\n(grey = no samples exited here)')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')

    for row_i in range(len(thresholds)):
        for col_j in range(n_exits):
            if rate_matrix[row_i, col_j] >= 1.0 and not np.isnan(acc_matrix[row_i, col_j]):
                val = acc_matrix[row_i, col_j]
                ax.text(col_j, row_i, f'{val:.1f}',
                        ha='center', va='center', fontsize=10,
                        color='black' if 20 < val < 80 else 'white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  per-exit accuracy heatmap saved: {save_path}")


# ── 콘솔 결과 출력 ────────────────────────────────────────────────────────────

def print_result_table(plain_result: dict, sel_results: dict,
                       exit_blocks: list):
    labels     = [f'B{b}' for b in exit_blocks]
    plain_acc  = plain_result['accuracy'] * 100
    plain_avg  = plain_result['summary']['avg_mean']
    plain_p99  = plain_result['summary']['p99_mean']
    thresholds = sorted(sel_results.keys(), key=float)

    print(f"\n{'='*90}")
    print(f"  PlainViT vs SelectiveExitViT ({'+'.join(labels)})  —  Comparison")
    print(f"{'='*90}")
    print(f"  {'Model/Thr':>14} {'Acc':>8} {'Drop':>7} {'AvgBlk':>8} "
          f"{'Savings':>8} {'Avg(ms)':>9} {'P99(ms)':>9}")
    print(f"  {'-'*75}")
    print(f"  {'PlainViT':>14} {plain_acc:>7.2f}%  {'---':>6}  "
          f"{'12.0':>7}  {'0.0%':>7}  {plain_avg:>9.2f}  {plain_p99:>9.2f}")
    print(f"  {'-'*75}")

    for key in thresholds:
        d       = sel_results[key]
        s       = d['summary']
        avg_blk = d['avg_exit_block'] or exit_blocks[-1]
        acc     = (d['accuracy'] or 0) * 100
        drop    = plain_acc - acc
        savings = (1 - avg_blk / 12) * 100
        marker  = " <--" if drop <= 1.0 else ""
        print(f"  {'thr='+str(d['threshold']):>14} {acc:>7.2f}%  "
              f"{drop:>+6.2f}%p  {avg_blk:>7.2f}  {savings:>6.1f}%  "
              f"{s['avg_mean']:>9.2f}  {s['p99_mean']:>9.2f}{marker}")

    print(f"{'='*90}")
    print(f"  '<--' marks thresholds with accuracy drop <= 1%p vs PlainViT\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PlainViT vs SelectiveExitViT comparison'
    )
    parser.add_argument('--exit-blocks',  type=int,   nargs='+', required=True,
                        help='Exit block positions (1-indexed, must end with 12). '
                             'Example: --exit-blocks 8 12  or  --exit-blocks 6 9 12')
    parser.add_argument('--n',            type=int,   default=5)
    parser.add_argument('--num-samples',  type=int,   default=1000)
    parser.add_argument('--thresholds',   type=float, nargs='+',
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument('--checkpoint',   type=str,   default=None)
    parser.add_argument('--data-root',    type=str,   default=None)
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--warmup',       type=int,   default=20)
    parser.add_argument('--out-dir',      type=str,   default=None)
    args = parser.parse_args()

    exit_blocks = args.exit_blocks
    if exit_blocks[-1] != 12:
        parser.error("Last exit block must be 12.")
    if exit_blocks != sorted(exit_blocks):
        parser.error("--exit-blocks must be ascending.")

    n_exits    = len(exit_blocks)
    model_name = f"ee_vit_{n_exits}exit"
    model_tag  = f"SelectiveExitViT ({'+'.join(f'B{b}' for b in exit_blocks)})"
    labels_str = [f'B{b}' for b in exit_blocks]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or paths.latest_checkpoint(model_name, 'best.pth')
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found for '{model_name}'.")
        print(f"        Use --checkpoint or run training first.")
        sys.exit(1)
    print(f"Checkpoint: {ckpt_path}")

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval',
        f'vit_sel{n_exits}exit_compare_N{args.n}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    thresholds = sorted(set(args.thresholds))

    print(f"\nConfig:")
    print(f"  Model       : {model_tag}")
    print(f"  N={args.n} runs  samples={args.num_samples}  warmup={args.warmup}")
    print(f"  thresholds  : {thresholds}")

    print("\nLoading data...")
    images, labels = load_data(args.num_samples, args.data_root, args.num_workers)

    # ── PlainViT ──────────────────────────────────────────────────────────────
    print("\n[1/2] Running PlainViT-B/16 ...")
    plain_model = build_plain(num_classes=1000).to(device)
    plain_model.eval()
    plain_result = run_plain_n(plain_model, images, labels, args.n, device, args.warmup)
    plain_acc_pct = plain_result['accuracy'] * 100
    print(f"  PlainViT acc={plain_acc_pct:.2f}%  "
          f"avg={plain_result['summary']['avg_mean']:.2f}ms  "
          f"P99={plain_result['summary']['p99_mean']:.2f}ms")

    # Free PlainViT VRAM before loading SelectiveExitViT
    del plain_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── SelectiveExitViT ──────────────────────────────────────────────────────
    print(f"\n[2/2] Running {model_tag} ...")
    sel_model = build_selective(exit_blocks=exit_blocks, num_classes=1000)
    state     = torch.load(ckpt_path, map_location=device, weights_only=True)
    sel_model.load_state_dict(state)
    sel_model.eval()
    sel_model  = sel_model.to(device)

    sel_results = run_sel_n_sweeps(
        sel_model, images, labels, thresholds, args.n, device, args.warmup
    )

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print("\nSaving results...")
    save_raw_json(plain_result, sel_results, exit_blocks, args.n, len(images),
                  str(device), thresholds,
                  os.path.join(out_dir, 'sel_compare_raw.json'))
    save_summary_csv(plain_result, sel_results, exit_blocks,
                     os.path.join(out_dir, 'sel_compare_summary.csv'))

    print("Generating plots...")
    plot_accuracy(plain_result, sel_results, exit_blocks, model_tag,
                  os.path.join(out_dir, 'sel_compare_accuracy.png'))
    plot_latency(plain_result, sel_results, model_tag,
                 os.path.join(out_dir, 'sel_compare_latency.png'))
    plot_tradeoff(plain_result, sel_results, exit_blocks, model_tag,
                  os.path.join(out_dir, 'sel_compare_tradeoff.png'))
    plot_exit_heatmap(sel_results, exit_blocks, model_tag,
                      os.path.join(out_dir, 'sel_compare_exit_heatmap.png'))
    plot_acc_heatmap(sel_results, exit_blocks, model_tag,
                     os.path.join(out_dir, 'sel_compare_acc_heatmap.png'))

    print_result_table(plain_result, sel_results, exit_blocks)

    print(f"Done! Results at:\n  {out_dir}")


if __name__ == '__main__':
    main()
