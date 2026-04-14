"""
run_vit_selective_sweep.py  —  SelectiveExitViT threshold sweep (PyTorch)

학습된 SelectiveExitViT 체크포인트를 로드하여 confidence threshold를 바꿔가며
각 exit 블록의 exit 분포, accuracy, latency를 분석한다.

2-exit (B8+B12) 또는 3-exit (B6+B9+B12) 모델 모두 지원.
x축 레이블에 실제 블록 번호(B8, B12 등)를 표시.

생성 파일 ({EXP_DIR}/eval/vit_sel{N}exit_sweep_N{N}_YYYYMMDD/):
  sel_sweep_raw.json             ← N회 × threshold별 원시 결과
  sel_sweep_summary.csv          ← threshold별 통계 + per-exit accuracy
  sel_sweep_exit_heatmap.png     ← exit block 분포 heatmap (threshold × block)
  sel_sweep_acc_heatmap.png      ← per-exit block accuracy heatmap
  sel_sweep_latency_dist.png     ← threshold별 KDE latency overlay
  sel_sweep_summary.png          ← accuracy / avg_exit_block / p99 / compute_savings

사용법:
  cd src
  python benchmark/run_vit_selective_sweep.py --exit-blocks 8 12 --n 5
  python benchmark/run_vit_selective_sweep.py --exit-blocks 6 9 12 --n 5
  python benchmark/run_vit_selective_sweep.py --exit-blocks 8 12 --n 10 --num-samples 2000
  python benchmark/run_vit_selective_sweep.py --exit-blocks 8 12 --checkpoint /path/to/best.pth

인자:
  --exit-blocks    모델의 exit 블록 번호 목록 (필수, 1-indexed, 예: 8 12 또는 6 9 12)
  --n              반복 횟수 (기본: 5)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     탐색 threshold 목록 (기본: 0.1~0.99)
  --checkpoint     체크포인트 경로 (기본: 최신 exp_*/ee_vit_{N}exit best.pth)
  --data-root      ImageNet 루트 경로
  --num-workers    DataLoader num_workers (기본: 4)
  --out-dir        결과 저장 디렉토리
  --warmup         latency 측정 제외 warmup 샘플 수 (기본: 20)
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
from models.ee_vit_selective import SelectiveExitViT, build_model
from datasets.dataloader import get_dataloader
from utils import load_config


DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, exit_blocks: list,
               num_classes: int, device: torch.device) -> SelectiveExitViT:
    model = build_model(exit_blocks=exit_blocks, num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


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


# ── 단일 threshold 1회 실행 ───────────────────────────────────────────────────

def run_sweep_once(model: SelectiveExitViT, images, labels, threshold: float,
                   device: torch.device, warmup: int = 20):
    """
    threshold 고정 후 전체 샘플에 대해 추론 1회 실행.

    Returns dict with:
        accuracy       : overall top-1 accuracy
        exit_counts    : list[int] len=n_exits, samples exiting at each head
        exit_rate      : list[float] % per exit
        acc_per_exit   : list[float|None] accuracy of samples exiting at each head
        avg_exit_block : weighted average of actual block positions (1~12 scale)
        latencies_ms   : list[float] (warmup excluded)
        avg_ms, p50_ms, p99_ms
    """
    n_exits          = model.NUM_BLOCKS
    exit_blocks      = model.exit_blocks   # e.g. [8, 12]
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

        # exit_block is 1-indexed actual block number (e.g. 8 or 12)
        head_idx = exit_blocks.index(exit_block)
        exit_counts[head_idx] += 1
        pred = logits.argmax(dim=1).item()
        if pred == lbl:
            correct += 1
            correct_per_exit[head_idx] += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])

    acc_per_exit = [
        correct_per_exit[i] / exit_counts[i] if exit_counts[i] > 0 else None
        for i in range(n_exits)
    ]

    # avg_exit_block: weighted avg of actual block positions (on 1~12 scale)
    avg_exit_block = sum(exit_blocks[i] * exit_counts[i] for i in range(n_exits)) / n

    return {
        'accuracy':       correct / n,
        'exit_counts':    exit_counts,
        'exit_rate':      [c / n * 100 for c in exit_counts],
        'acc_per_exit':   acc_per_exit,
        'avg_exit_block': avg_exit_block,
        'latencies_ms':   latencies,
        'avg_ms':  float(np.mean(lat)),
        'p50_ms':  float(np.percentile(lat, 50)),
        'p99_ms':  float(np.percentile(lat, 99)),
    }


# ── N회 반복 sweep ────────────────────────────────────────────────────────────

def run_n_sweeps(model: SelectiveExitViT, images, labels,
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

    total = N * len(thresholds)
    done  = 0
    labels_str = model.exit_block_labels   # ['B8','B12'] or ['B6','B9','B12']

    print(f"\n  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}  {'avg_block':>10}")
    print(f"  {'-'*60}")

    for run_idx in range(N):
        for thr in thresholds:
            key = str(round(thr, 2))
            r   = run_sweep_once(model, images, labels, thr, device, warmup)

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
                  f"avg={r['avg_exit_block']:>5.1f}  [{done}/{total}]  {exit_str}")

    # N회 집계
    for key in results:
        runs  = results[key]['runs']
        p99s  = [r['p99_ms'] for r in runs]
        avgs  = [r['avg_ms'] for r in runs]
        p50s  = [r['p50_ms'] for r in runs]
        results[key]['summary'] = {
            'p99_mean': float(np.mean(p99s)),
            'p99_std':  float(np.std(p99s)),
            'p99_min':  float(np.min(p99s)),
            'p99_max':  float(np.max(p99s)),
            'avg_mean': float(np.mean(avgs)),
            'avg_std':  float(np.std(avgs)),
            'p50_mean': float(np.mean(p50s)),
            'p50_std':  float(np.std(p50s)),
        }

    return results


# ── 저장: raw JSON ────────────────────────────────────────────────────────────

def save_raw_json(results: dict, n_samples: int, device_str: str,
                  exit_blocks: list, thresholds: list, N: int, out_path: str):
    payload = {
        'meta': {
            'model':       f'ee_vit_{len(exit_blocks)}exit',
            'exit_blocks': exit_blocks,
            'n_runs':      N,
            'n_samples':   n_samples,
            'thresholds':  thresholds,
            'device':      device_str,
            'timestamp':   datetime.now().isoformat(),
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  raw JSON saved: {out_path}")


# ── 저장: summary CSV ─────────────────────────────────────────────────────────

def save_summary_csv(results: dict, exit_blocks: list, out_path: str):
    labels = [f'B{b}' for b in exit_blocks]
    exit_rate_fields  = [f'exit_rate_{l.lower()}' for l in labels]
    acc_per_exit_fields = [f'acc_{l.lower()}' for l in labels]

    fieldnames = (
        ['threshold', 'accuracy', 'avg_exit_block', 'compute_savings_pct',
         'n_runs', 'p99_mean', 'p99_std', 'avg_mean', 'avg_std', 'p50_mean']
        + exit_rate_fields + acc_per_exit_fields
    )
    rows = []
    for key, data in sorted(results.items(), key=lambda x: float(x[0])):
        s   = data['summary']
        er  = data['exit_rate']    or [0.0] * len(exit_blocks)
        ape = data['acc_per_exit'] or [None] * len(exit_blocks)
        # compute savings relative to full 12-block inference
        avg_blk = data['avg_exit_block'] or exit_blocks[-1]
        compute_savings = (1 - avg_blk / 12) * 100

        row = {
            'threshold':           data['threshold'],
            'accuracy':            round(data['accuracy'], 6) if data['accuracy'] is not None else None,
            'avg_exit_block':      round(avg_blk, 4),
            'compute_savings_pct': round(compute_savings, 2),
            'n_runs':              len(data['runs']),
            'p99_mean':            round(s['p99_mean'], 4),
            'p99_std':             round(s['p99_std'],  4),
            'avg_mean':            round(s['avg_mean'], 4),
            'avg_std':             round(s['avg_std'],  4),
            'p50_mean':            round(s['p50_mean'], 4),
        }
        for i, lbl in enumerate(labels):
            row[f'exit_rate_{lbl.lower()}'] = round(er[i], 2)
        for i, lbl in enumerate(labels):
            acc = ape[i]
            row[f'acc_{lbl.lower()}'] = round(acc * 100, 2) if acc is not None else ''
        rows.append(row)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  summary CSV saved: {out_path}")


# ── Plot 1: Exit Block 분포 Heatmap ──────────────────────────────────────────

def plot_exit_heatmap(results: dict, exit_blocks: list, model_tag: str,
                      save_path: str):
    labels     = [f'B{b}' for b in exit_blocks]
    n_exits    = len(exit_blocks)
    thresholds = sorted(results.keys(), key=float)
    matrix     = []
    for key in thresholds:
        er = results[key]['exit_rate'] or [0.0] * n_exits
        matrix.append(er)
    matrix = np.array(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_tag}  —  Exit Block Distribution by Threshold', fontsize=13)

    ax = axes[0]
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('Exit Rate (%) per Block')
    fig.colorbar(im, ax=ax, label='Exit Rate (%)')

    for row_i in range(len(thresholds)):
        for col_j in range(n_exits):
            val = matrix[row_i, col_j]
            if val >= 5.0:
                ax.text(col_j, row_i, f'{val:.0f}',
                        ha='center', va='center', fontsize=9,
                        color='black' if val < 60 else 'white')

    ax = axes[1]
    thr_vals    = [float(t) for t in thresholds]
    avg_exits   = [results[t]['avg_exit_block'] or exit_blocks[-1] for t in thresholds]
    savings_pct = [(1 - a / 12) * 100 for a in avg_exits]

    color1, color2 = 'steelblue', 'darkorange'
    ax.plot(thr_vals, avg_exits, 'o-', color=color1, linewidth=2, markersize=5,
            label='Avg exit block')
    # Mark actual exit block positions
    for b in exit_blocks:
        ax.axhline(b, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(thr_vals[0], b + 0.1, f'B{b}', fontsize=8, color='gray')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Avg Exit Block (block #)', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim(0, 13)
    ax.set_yticks(exit_blocks)
    ax.set_yticklabels(labels)

    ax2 = ax.twinx()
    ax2.plot(thr_vals, savings_pct, 's--', color=color2, linewidth=2, markersize=5,
             label='Compute savings (vs 12-block)')
    ax2.set_ylabel('Compute Savings (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)

    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=9)
    ax.set_title('Avg Exit Block & Compute Savings vs Threshold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  exit heatmap saved: {save_path}")


# ── Plot 1b: Per-Exit Accuracy Heatmap ───────────────────────────────────────

def plot_acc_heatmap(results: dict, exit_blocks: list, model_tag: str,
                     save_path: str):
    labels     = [f'B{b}' for b in exit_blocks]
    n_exits    = len(exit_blocks)
    thresholds = sorted(results.keys(), key=float)

    acc_matrix  = np.full((len(thresholds), n_exits), np.nan)
    rate_matrix = np.zeros((len(thresholds), n_exits))

    for row_i, key in enumerate(thresholds):
        ape = results[key]['acc_per_exit'] or [None] * n_exits
        er  = results[key]['exit_rate']    or [0.0]  * n_exits
        for col_j in range(n_exits):
            rate_matrix[row_i, col_j] = er[col_j]
            if ape[col_j] is not None:
                acc_matrix[row_i, col_j] = ape[col_j] * 100

    masked = np.ma.array(acc_matrix, mask=np.isnan(acc_matrix))
    cmap   = plt.cm.RdYlGn
    cmap.set_bad(color='#cccccc')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_tag}  —  Per-Exit Block Accuracy by Threshold', fontsize=13)

    ax = axes[0]
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('Accuracy (%) of Samples Exiting at Each Block\n(grey = no samples exited here)')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')

    for row_i in range(len(thresholds)):
        for col_j in range(n_exits):
            if rate_matrix[row_i, col_j] >= 1.0 and not np.isnan(acc_matrix[row_i, col_j]):
                val = acc_matrix[row_i, col_j]
                ax.text(col_j, row_i, f'{val:.1f}',
                        ha='center', va='center', fontsize=9,
                        color='black' if 20 < val < 80 else 'white')

    ax = axes[1]
    thr_vals = [float(t) for t in thresholds]
    overall  = [results[t]['accuracy'] or 0 for t in thresholds]
    ax.plot(thr_vals, [a * 100 for a in overall], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Overall accuracy', zorder=5)

    colors_exit = ['darkorange', 'seagreen', 'tomato']
    for col_j, (lbl, color) in enumerate(zip(labels, colors_exit)):
        valid_thr = [thr_vals[i] for i in range(len(thresholds))
                     if not np.isnan(acc_matrix[i, col_j])]
        valid_acc = [acc_matrix[i, col_j] for i in range(len(thresholds))
                     if not np.isnan(acc_matrix[i, col_j])]
        if valid_thr:
            ax.plot(valid_thr, valid_acc, 's--', color=color, linewidth=1.5,
                    markersize=5, alpha=0.8, label=f'{lbl} exit acc')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall vs Per-Exit Block Accuracy')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  per-exit accuracy heatmap saved: {save_path}")


# ── Plot 2: Latency Distribution KDE ─────────────────────────────────────────

def plot_latency_dist(results: dict, N: int, model_tag: str, save_path: str):
    thresholds = sorted(results.keys(), key=float)
    cmap       = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{model_tag}  —  Latency Distribution by Threshold  (N={N} runs)',
                 fontsize=13)

    ax = axes[0]
    for key, color in zip(thresholds, cmap):
        runs = results[key]['runs']
        lats = []
        for r in runs:
            lats.extend(r['latencies_ms'])
        arr = np.array(lats)
        if len(arr) < 2:
            continue
        try:
            kde = gaussian_kde(arr, bw_method='scott')
            x_r = np.linspace(arr.min(), np.percentile(arr, 99.5), 300)
            ax.plot(x_r, kde(x_r), color=color, linewidth=1.8,
                    label=f'thr={float(key):.2f}')
            ax.axvline(np.median(arr), color=color, linestyle='--',
                       linewidth=0.8, alpha=0.6)
        except Exception:
            pass

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('KDE per Threshold  (dashed = median)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    thr_vals  = [float(k) for k in thresholds]
    p99_means = [results[k]['summary']['p99_mean'] for k in thresholds]
    p99_stds  = [results[k]['summary']['p99_std']  for k in thresholds]
    ax.errorbar(thr_vals, p99_means, yerr=p99_stds,
                fmt='o-', color='tomato', linewidth=2, markersize=6,
                capsize=5, capthick=1.5, label='P99 mean +/- std')
    ax.fill_between(thr_vals,
                    [m - s for m, s in zip(p99_means, p99_stds)],
                    [m + s for m, s in zip(p99_means, p99_stds)],
                    alpha=0.15, color='tomato')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency (mean +/- std) vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  latency KDE saved: {save_path}")


# ── Plot 3: 종합 요약 ─────────────────────────────────────────────────────────

def plot_summary(results: dict, exit_blocks: list, model_tag: str, save_path: str):
    thresholds = sorted(results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]

    accs      = [results[t]['accuracy'] or 0 for t in thresholds]
    avg_exits = [results[t]['avg_exit_block'] or exit_blocks[-1] for t in thresholds]
    savings   = [(1 - a / 12) * 100 for a in avg_exits]
    p99s      = [results[t]['summary']['p99_mean'] for t in thresholds]
    p99_stds  = [results[t]['summary']['p99_std']  for t in thresholds]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_tag}  —  Threshold Sweep Summary', fontsize=13)

    ax = axes[0, 0]
    ax.plot(thr_vals, [a * 100 for a in accs], 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-1 Accuracy vs Threshold')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(thr_vals, avg_exits, 's-', color='darkorange', linewidth=2)
    labels = [f'B{b}' for b in exit_blocks]
    for b, lbl in zip(exit_blocks, labels):
        ax.axhline(b, color='gray', linestyle='--', linewidth=1, alpha=0.6, label=lbl)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Avg Exit Block (block #)')
    ax.set_title('Avg Exit Block vs Threshold')
    ax.set_ylim(0, 13)
    ax.set_yticks(exit_blocks)
    ax.set_yticklabels(labels)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(thr_vals, p99s, yerr=p99_stds,
                fmt='o-', color='tomato', linewidth=2, markersize=6,
                capsize=4, capthick=1.5)
    ax.set_xlabel('Threshold'); ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency vs Threshold')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(thr_vals, savings, '^-', color='seagreen', linewidth=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Compute Savings (%)')
    ax.set_title(f'Compute Savings vs Threshold\n(1 - avg_exit_block / 12)')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  summary saved: {save_path}")


# ── 콘솔 결과 출력 ────────────────────────────────────────────────────────────

def print_result_table(results: dict, exit_blocks: list):
    labels     = [f'B{b}' for b in exit_blocks]
    n_exits    = len(exit_blocks)
    thresholds = sorted(results.keys(), key=float)

    print(f"\n{'='*80}")
    print(f"  Selective EE-ViT-B/16 ({'+'.join(labels)})  Threshold Sweep Summary")
    print(f"{'='*80}")
    print(f"  {'Threshold':>10} {'Accuracy':>10} {'AvgBlock':>9} {'Savings':>8} "
          f"{'P99(ms)':>9} {'±std':>6}")
    print(f"  {'-'*65}")
    for key in thresholds:
        d = results[key]
        s = d['summary']
        avg_blk = d['avg_exit_block'] or exit_blocks[-1]
        savings = (1 - avg_blk / 12) * 100
        print(f"  {d['threshold']:>10.2f}  "
              f"{(d['accuracy'] or 0)*100:>8.2f}%  "
              f"{avg_blk:>8.2f}  "
              f"{savings:>6.1f}%  "
              f"{s['p99_mean']:>9.2f}  "
              f"±{s['p99_std']:>5.2f}")
    print(f"{'='*80}")

    # per-exit accuracy
    header = f"  {'thr':>6}" + "".join(f"  {lbl:>6}" for lbl in labels)
    print(f"\n  Per-Exit Block Accuracy (%) — accuracy of samples exiting at each block")
    print(f"  (no samples = '-')")
    print(f"  {'-'*(len(header)-2)}")
    print(header)
    print(f"  {'-'*(len(header)-2)}")
    for key in thresholds:
        d   = results[key]
        ape = d['acc_per_exit'] or [None] * n_exits
        row = f"  {d['threshold']:>6.2f}"
        for acc in ape:
            if acc is None:
                row += f"       -"
            else:
                row += f"  {acc*100:>5.1f}%"
        print(row)
    print(f"  {'-'*(len(header)-2)}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT-B/16 Threshold Sweep'
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
    parser.add_argument('--out-dir',      type=str,   default=None)
    parser.add_argument('--warmup',       type=int,   default=20)
    args = parser.parse_args()

    exit_blocks = args.exit_blocks
    if exit_blocks[-1] != 12:
        parser.error("Last exit block must be 12.")
    if exit_blocks != sorted(exit_blocks):
        parser.error("--exit-blocks must be ascending.")

    n_exits    = len(exit_blocks)
    model_name = f"ee_vit_{n_exits}exit"
    model_tag  = f"SelectiveExitViT ({'+'.join(f'B{b}' for b in exit_blocks)})"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or paths.latest_checkpoint(model_name, 'best.pth')
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found for model '{model_name}'.")
        print(f"        Use --checkpoint to specify path, or train first.")
        sys.exit(1)
    print(f"Checkpoint: {ckpt_path}")

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval',
        f'vit_sel{n_exits}exit_sweep_N{args.n}_{ts}'
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

    print("Loading model...")
    model = load_model(ckpt_path, exit_blocks, num_classes=1000, device=device)
    print(f"  {model_tag} loaded  (exit heads: {model.NUM_BLOCKS})")

    print(f"\nSweep start ({args.n} runs x {len(thresholds)} thresholds x {len(images)} samples)...")
    results = run_n_sweeps(model, images, labels, thresholds,
                           args.n, device, args.warmup)

    print("\nSaving results...")
    save_raw_json(results, len(images), str(device), exit_blocks, thresholds, args.n,
                  os.path.join(out_dir, 'sel_sweep_raw.json'))
    save_summary_csv(results, exit_blocks,
                     os.path.join(out_dir, 'sel_sweep_summary.csv'))

    print("Generating plots...")
    plot_exit_heatmap(results, exit_blocks, model_tag,
                      os.path.join(out_dir, 'sel_sweep_exit_heatmap.png'))
    plot_acc_heatmap(results, exit_blocks, model_tag,
                     os.path.join(out_dir, 'sel_sweep_acc_heatmap.png'))
    plot_latency_dist(results, args.n, model_tag,
                      os.path.join(out_dir, 'sel_sweep_latency_dist.png'))
    plot_summary(results, exit_blocks, model_tag,
                 os.path.join(out_dir, 'sel_sweep_summary.png'))

    print_result_table(results, exit_blocks)

    print(f"Done! Results at:\n  {out_dir}")


if __name__ == '__main__':
    main()
