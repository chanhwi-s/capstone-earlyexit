"""
run_vit_sweep.py  —  EE-ViT-B/16 threshold sweep (PyTorch 기반)

학습된 EE-ViT 체크포인트를 로드하여 confidence threshold를 바꿔가며
각 블록의 exit 분포, accuracy, latency를 분석한다.

ResNet sweep과의 차이:
  - TRT 엔진 없이 PyTorch 직접 추론 (ViT TRT export 전 분석용)
  - 12개 exit point → exit block 분포 heatmap이 핵심 플롯
  - latency 측정 시 torch.cuda.synchronize() + warmup 처리

생성 파일 ({EXP_DIR}/eval/vit_sweep_N{N}_YYYYMMDD_HHMMSS/):
  vit_sweep_raw.json         ← N회 × threshold별 원시 결과
  vit_sweep_summary.csv      ← threshold별 통계 + per-exit block accuracy
  vit_sweep_exit_heatmap.png ← ★ exit block 분포 heatmap (threshold × block)
  vit_sweep_acc_heatmap.png  ← ★ per-exit block accuracy heatmap (threshold × block)
  vit_sweep_latency_dist.png ← threshold별 KDE latency overlay
  vit_sweep_summary.png      ← accuracy / avg_exit_block / p99 / compute_savings

사용법:
  cd src
  python benchmark/run_vit_sweep.py --n 5
  python benchmark/run_vit_sweep.py --n 10 --num-samples 2000
  python benchmark/run_vit_sweep.py --n 5 --thresholds 0.3 0.5 0.7 0.9
  python benchmark/run_vit_sweep.py --n 5 --checkpoint /path/to/best.pth

인자:
  --n              반복 횟수 (기본: 5)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     탐색 threshold 목록 (기본: 0.1~0.99)
  --checkpoint     체크포인트 경로 (기본: 최신 exp_*/train/ee_vit/checkpoints/best.pth)
  --data-root      ImageNet 루트 경로 (기본: configs/train.yaml 값)
  --num-workers    DataLoader num_workers (기본: 4)
  --out-dir        결과 저장 디렉토리 (기본: {EXP_DIR}/eval/vit_sweep_N{N}_...)
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
from models.ee_vit import EEViT, build_model
from datasets.dataloader import get_dataloader
from utils import load_config


# ── 기본 threshold 목록 ───────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> EEViT:
    model = build_model(num_classes=num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data(num_samples: int, data_root: str = None, num_workers: int = 4):
    """
    ImageNet val 셋에서 num_samples개 로드.
    batch_size=1 (샘플별 latency 측정을 위해).
    Returns: images (list of Tensor [1,3,224,224]), labels (list of int)
    """
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


# ── 단일 threshold 1회 실행 ───────────────────────────────────────────────────

def run_sweep_once(model: EEViT, images, labels, threshold: float,
                   device: torch.device, warmup: int = 20):
    """
    threshold 고정 후 전체 샘플에 대해 추론 1회 실행.

    Returns:
        accuracy      : float
        exit_counts   : list[int] 길이 12, 각 블록에서 탈출한 샘플 수
        acc_per_exit  : list[float] 길이 12, 각 블록에서 탈출한 샘플들의 accuracy
        latencies_ms  : list[float] 길이 (num_samples - warmup), warmup 제외
        avg_ms, p50_ms, p99_ms : float
    """
    n_exits         = model.NUM_BLOCKS
    exit_counts     = [0] * n_exits
    correct_per_exit = [0] * n_exits   # 각 블록에서 탈출한 샘플 중 정답 수
    latencies       = []
    correct         = 0

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_dev = img.to(device)

        # ── latency 측정 (warmup 이후) ────────────────────────────────────
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

        block_i = exit_idx - 1          # exit_idx는 1-indexed → 0-indexed
        exit_counts[block_i] += 1
        pred = logits.argmax(dim=1).item()
        if pred == lbl:
            correct += 1
            correct_per_exit[block_i] += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])
    # 해당 블록을 통과한 샘플이 없으면 None (0으로 나누기 방지)
    acc_per_exit = [
        correct_per_exit[i] / exit_counts[i] if exit_counts[i] > 0 else None
        for i in range(n_exits)
    ]
    return {
        'accuracy':       correct / n,
        'exit_counts':    exit_counts,
        'exit_rate':      [c / n * 100 for c in exit_counts],  # % per block
        'acc_per_exit':   acc_per_exit,
        'avg_exit_block': sum((i + 1) * exit_counts[i] for i in range(n_exits)) / n,
        'latencies_ms':   latencies,
        'avg_ms':  float(np.mean(lat)),
        'p50_ms':  float(np.percentile(lat, 50)),
        'p99_ms':  float(np.percentile(lat, 99)),
    }


# ── N회 반복 sweep ────────────────────────────────────────────────────────────

def run_n_sweeps(model: EEViT, images, labels,
                 thresholds: list, N: int,
                 device: torch.device, warmup: int = 20):
    """
    thresholds × N회 전체 sweep 실행.
    accuracy/exit_rate는 첫 번째 run에서만 저장 (이후 동일).
    latency는 N회 누적.

    Returns:
        results: {threshold_str: {threshold, accuracy, exit_rate,
                                  avg_exit_block, runs:[...], summary:{...}}}
    """
    results = {str(round(t, 2)): {
        'threshold':      round(t, 2),
        'accuracy':       None,
        'exit_rate':      None,
        'acc_per_exit':   None,   # 각 exit block에서 탈출한 샘플들의 accuracy
        'avg_exit_block': None,
        'runs': [],
    } for t in thresholds}

    total = N * len(thresholds)
    done  = 0

    print(f"\n  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}  {'avg_exit':>9}")
    print(f"  {'-'*55}")

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
            print(f"  {thr:.2f}  run {run_idx+1:>3}/{N}  "
                  f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
                  f"{r['avg_exit_block']:>9.2f}  [{done}/{total}]")

    # N회 집계 통계
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
                  thresholds: list, N: int, out_path: str):
    payload = {
        'meta': {
            'model':      'ee_vit_b16',
            'n_runs':     N,
            'n_samples':  n_samples,
            'thresholds': thresholds,
            'device':     device_str,
            'timestamp':  datetime.now().isoformat(),
        },
        'results': results,
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  raw JSON 저장: {out_path}")


# ── 저장: summary CSV ─────────────────────────────────────────────────────────

def save_summary_csv(results: dict, n_exits: int, out_path: str):
    exit_rate_fields = [f'exit_rate_b{i+1}' for i in range(n_exits)]
    acc_per_exit_fields = [f'acc_b{i+1}' for i in range(n_exits)]
    fieldnames = (
        ['threshold', 'accuracy', 'avg_exit_block', 'compute_savings_pct',
         'n_runs', 'p99_mean', 'p99_std', 'avg_mean', 'avg_std', 'p50_mean']
        + exit_rate_fields + acc_per_exit_fields
    )
    rows = []
    for key, data in sorted(results.items(), key=lambda x: float(x[0])):
        s   = data['summary']
        er  = data['exit_rate']    or [0.0]  * n_exits
        ape = data['acc_per_exit'] or [None] * n_exits
        compute_savings = (1 - data['avg_exit_block'] / n_exits) * 100 if data['avg_exit_block'] else 0.0
        row = {
            'threshold':            data['threshold'],
            'accuracy':             round(data['accuracy'], 6) if data['accuracy'] is not None else None,
            'avg_exit_block':       round(data['avg_exit_block'], 4) if data['avg_exit_block'] else None,
            'compute_savings_pct':  round(compute_savings, 2),
            'n_runs':               len(data['runs']),
            'p99_mean':             round(s['p99_mean'], 4),
            'p99_std':              round(s['p99_std'],  4),
            'avg_mean':             round(s['avg_mean'], 4),
            'avg_std':              round(s['avg_std'],  4),
            'p50_mean':             round(s['p50_mean'], 4),
        }
        for i, rate in enumerate(er):
            row[f'exit_rate_b{i+1}'] = round(rate, 2)
        for i, acc in enumerate(ape):
            row[f'acc_b{i+1}'] = round(acc * 100, 2) if acc is not None else ''
        rows.append(row)

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  summary CSV 저장: {out_path}")


# ── Plot 1: Exit Block 분포 Heatmap (★ 핵심 플롯) ────────────────────────────

def plot_exit_heatmap(results: dict, n_exits: int, save_path: str):
    """
    x-axis: exit block (1~12)
    y-axis: threshold
    color : exit rate (%) — 어느 block에서 얼마나 많이 탈출하는지
    """
    thresholds = sorted(results.keys(), key=float)
    matrix = []   # shape: [n_thresholds, n_exits]
    for key in thresholds:
        er = results[key]['exit_rate'] or [0.0] * n_exits
        matrix.append(er)
    matrix = np.array(matrix)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('EE-ViT-B/16  —  Exit Block Distribution by Threshold', fontsize=13)

    # ── subplot 1: heatmap ──────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels([f'B{i+1}' for i in range(n_exits)], fontsize=8)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('Exit Rate (%) per Block')
    fig.colorbar(im, ax=ax, label='Exit Rate (%)')

    # 셀 값 표시 (5% 이상인 경우만)
    for row_i, thr in enumerate(thresholds):
        for col_j in range(n_exits):
            val = matrix[row_i, col_j]
            if val >= 5.0:
                ax.text(col_j, row_i, f'{val:.0f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if val < 60 else 'white')

    # ── subplot 2: avg exit block + compute savings vs threshold ────────────
    ax = axes[1]
    thr_vals    = [float(t) for t in thresholds]
    avg_exits   = [results[t]['avg_exit_block'] or 0 for t in thresholds]
    savings_pct = [(1 - a / n_exits) * 100 for a in avg_exits]

    color1, color2 = 'steelblue', 'darkorange'
    ax.plot(thr_vals, avg_exits, 'o-', color=color1, linewidth=2,
            markersize=5, label='Avg exit block')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Avg Exit Block', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim(0, n_exits + 0.5)
    ax.set_yticks(range(0, n_exits + 1, 2))

    ax2 = ax.twinx()
    ax2.plot(thr_vals, savings_pct, 's--', color=color2, linewidth=2,
             markersize=5, label='Compute savings')
    ax2.set_ylabel('Compute Savings (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.set_title('Avg Exit Block & Compute Savings vs Threshold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  exit heatmap 저장: {save_path}")


# ── Plot 1b: Per-Exit Accuracy Heatmap ───────────────────────────────────────

def plot_acc_heatmap(results: dict, n_exits: int, save_path: str):
    """
    x-axis: exit block (1~12)
    y-axis: threshold
    color : accuracy (%) of samples that exited at each block
            — 샘플 수 0인 셀은 회색으로 표시
    """
    thresholds = sorted(results.keys(), key=float)
    acc_matrix  = np.full((len(thresholds), n_exits), np.nan)
    rate_matrix = np.zeros((len(thresholds), n_exits))

    for row_i, key in enumerate(thresholds):
        ape = results[key]['acc_per_exit'] or [None] * n_exits
        er  = results[key]['exit_rate']    or [0.0]  * n_exits
        for col_j in range(n_exits):
            rate_matrix[row_i, col_j] = er[col_j]
            if ape[col_j] is not None:
                acc_matrix[row_i, col_j] = ape[col_j] * 100  # → %

    # NaN(샘플 없음) 처리용 masked array
    masked = np.ma.array(acc_matrix, mask=np.isnan(acc_matrix))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='#cccccc')  # 샘플 없는 셀 = 회색

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('EE-ViT-B/16  —  Per-Exit Block Accuracy by Threshold', fontsize=13)

    # ── subplot 1: accuracy heatmap ─────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(n_exits))
    ax.set_xticklabels([f'B{i+1}' for i in range(n_exits)], fontsize=8)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block')
    ax.set_ylabel('Threshold')
    ax.set_title('Accuracy (%) of Samples Exiting at Each Block\n(grey = no samples exited here)')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')

    # 셀 값 표시 (exit rate >= 1% 이상 셀만)
    for row_i in range(len(thresholds)):
        for col_j in range(n_exits):
            if rate_matrix[row_i, col_j] >= 1.0 and not np.isnan(acc_matrix[row_i, col_j]):
                val = acc_matrix[row_i, col_j]
                ax.text(col_j, row_i, f'{val:.0f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if 20 < val < 80 else 'white')

    # ── subplot 2: overall accuracy vs threshold (참고용) ─────────────────
    ax = axes[1]
    thr_vals = [float(t) for t in thresholds]
    overall  = [results[t]['accuracy'] or 0 for t in thresholds]
    ax.plot(thr_vals, [a * 100 for a in overall], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Overall accuracy')

    # 가장 많이 사용되는 exit block의 accuracy를 overlay
    for col_j in range(n_exits):
        col_accs = [acc_matrix[row_i, col_j] for row_i in range(len(thresholds))]
        col_rates = [rate_matrix[row_i, col_j] for row_i in range(len(thresholds))]
        # 어떤 threshold에서든 exit rate >= 5% 인 block만 표시
        if max(col_rates) >= 5.0:
            # NaN 제거 후 plot
            valid_thr = [thr_vals[i] for i in range(len(thresholds))
                         if not np.isnan(col_accs[i])]
            valid_acc = [col_accs[i] for i in range(len(thresholds))
                         if not np.isnan(col_accs[i])]
            if valid_thr:
                ax.plot(valid_thr, valid_acc, '--', linewidth=1, alpha=0.6,
                        label=f'B{col_j+1} exit acc')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall vs Per-Exit Block Accuracy\n(dashed = acc of samples exiting at that block)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  per-exit accuracy heatmap 저장: {save_path}")


# ── Plot 2: Latency Distribution KDE overlay ─────────────────────────────────

def plot_latency_dist(results: dict, N: int, save_path: str):
    """threshold별 latency KDE overlay + p99 error bar."""
    thresholds = sorted(results.keys(), key=float)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'EE-ViT-B/16  —  Latency Distribution by Threshold  (N={N}회)', fontsize=13)

    # ── KDE overlay ──────────────────────────────────────────────────────────
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

    # ── p99 error bar ─────────────────────────────────────────────────────────
    ax = axes[1]
    thr_vals  = [float(k) for k in thresholds]
    p99_means = [results[k]['summary']['p99_mean'] for k in thresholds]
    p99_stds  = [results[k]['summary']['p99_std']  for k in thresholds]
    ax.errorbar(thr_vals, p99_means, yerr=p99_stds,
                fmt='o-', color='tomato', linewidth=2, markersize=6,
                capsize=5, capthick=1.5, label='P99 mean ± std')
    ax.fill_between(thr_vals,
                    [m - s for m, s in zip(p99_means, p99_stds)],
                    [m + s for m, s in zip(p99_means, p99_stds)],
                    alpha=0.15, color='tomato')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency (mean ± std) vs Threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  latency KDE 저장: {save_path}")


# ── Plot 3: 종합 요약 ─────────────────────────────────────────────────────────

def plot_summary(results: dict, n_exits: int, save_path: str):
    """accuracy / avg_exit_block / p99 latency / compute_savings vs threshold."""
    thresholds = sorted(results.keys(), key=float)
    thr_vals = [float(t) for t in thresholds]

    accs       = [results[t]['accuracy'] or 0 for t in thresholds]
    avg_exits  = [results[t]['avg_exit_block'] or 0 for t in thresholds]
    savings    = [(1 - a / n_exits) * 100 for a in avg_exits]
    p99s       = [results[t]['summary']['p99_mean'] for t in thresholds]
    p99_stds   = [results[t]['summary']['p99_std']  for t in thresholds]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EE-ViT-B/16  —  Threshold Sweep Summary', fontsize=13)

    # ── accuracy ──────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(thr_vals, [a * 100 for a in accs], 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top-1 Accuracy vs Threshold')
    ax.grid(alpha=0.3)

    # ── avg exit block ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(thr_vals, avg_exits, 's-', color='darkorange', linewidth=2)
    ax.axhline(n_exits, color='gray', linestyle='--', linewidth=1, label=f'Full ({n_exits})')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Avg Exit Block')
    ax.set_title('Avg Exit Block vs Threshold')
    ax.set_ylim(0, n_exits + 1)
    ax.legend(); ax.grid(alpha=0.3)

    # ── p99 latency ───────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.errorbar(thr_vals, p99s, yerr=p99_stds,
                fmt='o-', color='tomato', linewidth=2, markersize=6,
                capsize=4, capthick=1.5)
    ax.set_xlabel('Threshold'); ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency vs Threshold')
    ax.grid(alpha=0.3)

    # ── compute savings ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(thr_vals, savings, '^-', color='seagreen', linewidth=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Compute Savings (%)')
    ax.set_title('Compute Savings vs Threshold\n(1 - avg_exit_block / 12)')
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  summary 저장: {save_path}")


# ── 콘솔 결과 출력 ────────────────────────────────────────────────────────────

def print_result_table(results: dict, n_exits: int):
    thresholds = sorted(results.keys(), key=float)

    # ── 전체 요약 테이블 ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'EE-ViT-B/16 Threshold Sweep Summary':^80}")
    print(f"{'='*80}")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'AvgExit':>8} {'Savings':>8} "
          f"{'P99(ms)':>9} {'±std':>6}")
    print(f"  {'-'*70}")
    for key in thresholds:
        d = results[key]
        s = d['summary']
        savings = (1 - (d['avg_exit_block'] or 0) / n_exits) * 100
        print(f"  {d['threshold']:>8.2f}  "
              f"{(d['accuracy'] or 0)*100:>8.2f}%  "
              f"{(d['avg_exit_block'] or 0):>7.2f}  "
              f"{savings:>6.1f}%  "
              f"{s['p99_mean']:>9.2f}  "
              f"±{s['p99_std']:>5.2f}")
    print(f"{'='*80}")

    # ── per-exit accuracy 테이블 ──────────────────────────────────────────────
    # 헤더: block 번호
    header = f"{'thr':>6}" + "".join(f"  B{i+1:>2}" for i in range(n_exits))
    print(f"\n  Per-Exit Block Accuracy (%)  — 각 block에서 탈출한 샘플의 정확도")
    print(f"  (샘플 없는 block = '  -')")
    print(f"  {'-'*len(header)}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for key in thresholds:
        d   = results[key]
        ape = d['acc_per_exit'] or [None] * n_exits
        row = f"{d['threshold']:>6.2f}"
        for acc in ape:
            if acc is None:
                row += f"    -"
            else:
                row += f"  {acc*100:>4.0f}"
        print(f"  {row}")
    print(f"  {'-'*len(header)}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='EE-ViT-B/16 Threshold Sweep')
    parser.add_argument('--n',            type=int,   default=5,
                        help='반복 횟수 (기본: 5)')
    parser.add_argument('--num-samples',  type=int,   default=1000,
                        help='샘플 수 (기본: 1000)')
    parser.add_argument('--thresholds',   type=float, nargs='+',
                        default=DEFAULT_THRESHOLDS,
                        help='탐색할 threshold 목록')
    parser.add_argument('--checkpoint',   type=str,   default=None,
                        help='체크포인트 경로 (기본: 최신 exp_*/ee_vit best.pth)')
    parser.add_argument('--data-root',    type=str,   default=None,
                        help='ImageNet 루트 경로')
    parser.add_argument('--num-workers',  type=int,   default=4)
    parser.add_argument('--out-dir',      type=str,   default=None,
                        help='결과 저장 디렉토리')
    parser.add_argument('--warmup',       type=int,   default=20,
                        help='latency 측정 제외 warmup 샘플 수 (기본: 20)')
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── checkpoint 결정 ──────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or paths.latest_checkpoint('ee_vit', 'best.pth')
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[ERROR] EE-ViT 체크포인트를 찾을 수 없습니다.")
        print(f"        --checkpoint 인자로 직접 지정하거나,")
        print(f"        먼저 train_vit_pipeline.sh 로 학습을 완료하세요.")
        sys.exit(1)
    print(f"체크포인트: {ckpt_path}")

    # ── 출력 디렉토리 ────────────────────────────────────────────────────────
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'vit_sweep_N{args.n}_{ts}'
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

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print("모델 로드 중...")
    num_classes = 1000
    model = load_model(ckpt_path, num_classes, device)
    print(f"  EE-ViT-B/16 로드 완료  (exit heads: {model.NUM_BLOCKS}개)")

    # ── sweep 실행 ────────────────────────────────────────────────────────────
    print(f"\nSweep 시작 ({args.n}회 × {len(thresholds)} thresholds × {len(images)}샘플)...")
    results = run_n_sweeps(model, images, labels, thresholds,
                           args.n, device, args.warmup)

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print("\n결과 저장 중...")
    save_raw_json(results, len(images), str(device), thresholds, args.n,
                  os.path.join(out_dir, 'vit_sweep_raw.json'))
    save_summary_csv(results, model.NUM_BLOCKS,
                     os.path.join(out_dir, 'vit_sweep_summary.csv'))

    print("그래프 생성 중...")
    plot_exit_heatmap(results, model.NUM_BLOCKS,
                      os.path.join(out_dir, 'vit_sweep_exit_heatmap.png'))
    plot_acc_heatmap(results, model.NUM_BLOCKS,
                     os.path.join(out_dir, 'vit_sweep_acc_heatmap.png'))
    plot_latency_dist(results, args.n,
                      os.path.join(out_dir, 'vit_sweep_latency_dist.png'))
    plot_summary(results, model.NUM_BLOCKS,
                 os.path.join(out_dir, 'vit_sweep_summary.png'))

    print_result_table(results, model.NUM_BLOCKS)

    print(f"완료! 결과 위치:\n  {out_dir}")


if __name__ == '__main__':
    main()
