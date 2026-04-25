"""
run_vit_trt_sweep.py  —  SelectiveExitViT TRT threshold sweep  (Jetson AGX Orin)

run_vit_selective_sweep.py 의 TRT 버전.
PyTorch 대신 TRT 세그먼트 엔진으로 추론하며, 동일한 분석·플롯을 생성한다.

생성 파일 ({EXP_DIR}/eval/vit_trt_sel{N}exit_sweep_N{N}_YYYYMMDD/):
  trt_sel_sweep_raw.json
  trt_sel_sweep_summary.csv
  trt_sel_sweep_exit_heatmap.png
  trt_sel_sweep_acc_heatmap.png
  trt_sel_sweep_latency_dist.png
  trt_sel_sweep_summary.png

사용법 (Orin):
  cd src
  python benchmark/run_vit_trt_sweep.py --exit-blocks 8 12 --n 10
  python benchmark/run_vit_trt_sweep.py --exit-blocks 6 9 12 --n 10
  python benchmark/run_vit_trt_sweep.py --exit-blocks 8 12 --n 5 --latency-only

인자:
  --exit-blocks    exit 블록 번호 (필수, 예: 8 12 또는 6 9 12)
  --n              반복 횟수 (기본: 10)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     threshold 목록 (기본: 0.1~0.99)
  --data-root      ImageNet val 루트 (없으면 --latency-only 자동 전환)
  --latency-only   랜덤 노이즈로 latency만 측정 (accuracy 생략)
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
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from infer.infer_trt_vit_selective import SelectiveViTTRT, load_selective_vit_trt
# timm 의존성 없이 독립적으로 동작 (Orin에 timm 불필요)

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ── 저장: raw JSON ────────────────────────────────────────────────────────────

def save_raw_json(results, n_samples, device_str, exit_blocks, thresholds, N, out_path):
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

def save_summary_csv(results, exit_blocks, out_path):
    labels = [f'B{b}' for b in exit_blocks]
    fieldnames = (
        ['threshold', 'accuracy', 'avg_exit_block', 'compute_savings_pct',
         'n_runs', 'p99_mean', 'p99_std', 'avg_mean', 'avg_std', 'p50_mean']
        + [f'exit_rate_{l.lower()}' for l in labels]
        + [f'acc_{l.lower()}' for l in labels]
    )
    rows = []
    for key, data in sorted(results.items(), key=lambda x: float(x[0])):
        s   = data['summary']
        er  = data['exit_rate']    or [0.0] * len(exit_blocks)
        ape = data['acc_per_exit'] or [None] * len(exit_blocks)
        avg_blk = data['avg_exit_block'] or exit_blocks[-1]
        row = {
            'threshold':           data['threshold'],
            'accuracy':            round(data['accuracy'], 6) if data['accuracy'] is not None else None,
            'avg_exit_block':      round(avg_blk, 4),
            'compute_savings_pct': round((1 - avg_blk / 12) * 100, 2),
            'n_runs':              len(data['runs']),
            'p99_mean':            round(s['p99_mean'], 4),
            'p99_std':             round(s['p99_std'],  4),
            'avg_mean':            round(s['avg_mean'], 4),
            'avg_std':             round(s['avg_std'],  4),
            'p50_mean':            round(s['p50_mean'], 4),
        }
        for i, lbl in enumerate(labels):
            row[f'exit_rate_{lbl.lower()}'] = round(er[i], 2)
            acc = ape[i]
            row[f'acc_{lbl.lower()}'] = round(acc * 100, 2) if acc is not None else ''
        rows.append(row)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  summary CSV saved: {out_path}")


# ── Plot: Exit Block 분포 Heatmap ─────────────────────────────────────────────

def plot_exit_heatmap(results, exit_blocks, model_tag, save_path):
    labels     = [f'B{b}' for b in exit_blocks]
    thresholds = sorted(results.keys(), key=float)
    matrix     = np.array([results[k]['exit_rate'] or [0.0]*len(exit_blocks) for k in thresholds])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_tag}  —  Exit Block Distribution by Threshold', fontsize=13)

    ax = axes[0]
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    ax.set_xticks(range(len(exit_blocks))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block'); ax.set_ylabel('Threshold')
    ax.set_title('Exit Rate (%) per Block')
    fig.colorbar(im, ax=ax, label='Exit Rate (%)')
    for ri in range(len(thresholds)):
        for ci in range(len(exit_blocks)):
            if matrix[ri, ci] >= 5.0:
                ax.text(ci, ri, f'{matrix[ri,ci]:.0f}', ha='center', va='center',
                        fontsize=9, color='black' if matrix[ri,ci] < 60 else 'white')

    ax = axes[1]
    thr_vals  = [float(t) for t in thresholds]
    avg_exits = [results[t]['avg_exit_block'] or exit_blocks[-1] for t in thresholds]
    savings   = [(1 - a / 12) * 100 for a in avg_exits]
    c1, c2    = 'steelblue', 'darkorange'
    ax.plot(thr_vals, avg_exits, 'o-', color=c1, linewidth=2, markersize=5, label='Avg exit block')
    for b in exit_blocks:
        ax.axhline(b, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(thr_vals[0], b + 0.1, f'B{b}', fontsize=8, color='gray')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Avg Exit Block', color=c1)
    ax.tick_params(axis='y', labelcolor=c1); ax.set_ylim(0, 13)
    ax2 = ax.twinx()
    ax2.plot(thr_vals, savings, 's--', color=c2, linewidth=2, markersize=5, label='Compute savings')
    ax2.set_ylabel('Compute Savings (%)', color=c2)
    ax2.tick_params(axis='y', labelcolor=c2); ax2.set_ylim(0, 100)
    l1, lb1 = ax.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=9); ax.set_title('Avg Exit Block & Compute Savings'); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  exit heatmap saved: {save_path}")


# ── Plot: Per-Exit Accuracy Heatmap ──────────────────────────────────────────

def plot_acc_heatmap(results, exit_blocks, model_tag, save_path):
    labels     = [f'B{b}' for b in exit_blocks]
    thresholds = sorted(results.keys(), key=float)
    n          = len(exit_blocks)
    acc_m  = np.full((len(thresholds), n), np.nan)
    rate_m = np.zeros((len(thresholds), n))
    for ri, key in enumerate(thresholds):
        ape = results[key]['acc_per_exit'] or [None]*n
        er  = results[key]['exit_rate']    or [0.0]*n
        for ci in range(n):
            rate_m[ri, ci] = er[ci]
            if ape[ci] is not None:
                acc_m[ri, ci] = ape[ci] * 100

    masked = np.ma.array(acc_m, mask=np.isnan(acc_m))
    cmap   = plt.cm.RdYlGn; cmap.set_bad(color='#cccccc')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{model_tag}  —  Per-Exit Block Accuracy by Threshold', fontsize=13)
    ax = axes[0]
    im = ax.imshow(masked, aspect='auto', cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f'{float(t):.2f}' for t in thresholds], fontsize=8)
    ax.set_xlabel('Exit Block'); ax.set_ylabel('Threshold')
    ax.set_title('Accuracy (%) per Block\n(grey = no samples)')
    fig.colorbar(im, ax=ax, label='Accuracy (%)')
    for ri in range(len(thresholds)):
        for ci in range(n):
            if rate_m[ri,ci] >= 1.0 and not np.isnan(acc_m[ri,ci]):
                v = acc_m[ri,ci]
                ax.text(ci, ri, f'{v:.1f}', ha='center', va='center', fontsize=9,
                        color='black' if 20 < v < 80 else 'white')
    ax = axes[1]
    thr_vals = [float(t) for t in thresholds]
    overall  = [results[t]['accuracy'] or 0 for t in thresholds]
    ax.plot(thr_vals, [a*100 for a in overall], 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Overall accuracy', zorder=5)
    for ci, (lbl, color) in enumerate(zip(labels, ['darkorange','seagreen','tomato'])):
        vt = [thr_vals[ri] for ri in range(len(thresholds)) if not np.isnan(acc_m[ri,ci])]
        va = [acc_m[ri,ci] for ri in range(len(thresholds)) if not np.isnan(acc_m[ri,ci])]
        if vt: ax.plot(vt, va, 's--', color=color, linewidth=1.5, markersize=5, alpha=0.8, label=f'{lbl} exit acc')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall vs Per-Exit Accuracy'); ax.set_ylim(0, 105)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  per-exit accuracy heatmap saved: {save_path}")


# ── Plot: Latency Distribution KDE ────────────────────────────────────────────

def plot_latency_dist(results, N, model_tag, save_path):
    thresholds = sorted(results.keys(), key=float)
    cmap       = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))
    fig, axes  = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{model_tag}  —  Latency Distribution  (N={N})', fontsize=13)
    ax = axes[0]
    for key, color in zip(thresholds, cmap):
        lats = []
        for r in results[key]['runs']: lats.extend(r['latencies_ms'])
        arr = np.array(lats)
        if len(arr) < 2: continue
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(arr, bw_method='scott')
            xr  = np.linspace(arr.min(), np.percentile(arr, 99.5), 300)
            ax.plot(xr, kde(xr), color=color, linewidth=1.8, label=f'thr={float(key):.2f}')
            ax.axvline(np.median(arr), color=color, linestyle='--', linewidth=0.8, alpha=0.6)
        except Exception: pass
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Density')
    ax.set_title('KDE per Threshold'); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax = axes[1]
    thr_vals  = [float(k) for k in thresholds]
    p99_means = [results[k]['summary']['p99_mean'] for k in thresholds]
    p99_stds  = [results[k]['summary']['p99_std']  for k in thresholds]
    ax.errorbar(thr_vals, p99_means, yerr=p99_stds, fmt='o-', color='tomato',
                linewidth=2, markersize=6, capsize=5, capthick=1.5, label='P99 mean±std')
    ax.fill_between(thr_vals, [m-s for m,s in zip(p99_means,p99_stds)],
                    [m+s for m,s in zip(p99_means,p99_stds)], alpha=0.15, color='tomato')
    ax.set_xlabel('Threshold'); ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency vs Threshold'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  latency KDE saved: {save_path}")


# ── Plot: 종합 요약 ───────────────────────────────────────────────────────────

def plot_summary(results, exit_blocks, model_tag, save_path):
    thresholds = sorted(results.keys(), key=float)
    thr_vals   = [float(t) for t in thresholds]
    accs       = [results[t]['accuracy'] or 0 for t in thresholds]
    avg_exits  = [results[t]['avg_exit_block'] or exit_blocks[-1] for t in thresholds]
    savings    = [(1 - a / 12) * 100 for a in avg_exits]
    p99s       = [results[t]['summary']['p99_mean'] for t in thresholds]
    p99_stds   = [results[t]['summary']['p99_std']  for t in thresholds]
    labels     = [f'B{b}' for b in exit_blocks]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_tag}  —  Threshold Sweep Summary', fontsize=13)

    ax = axes[0,0]
    ax.plot(thr_vals, [a*100 for a in accs], 'o-', color='steelblue', linewidth=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy (%)'); ax.set_title('Top-1 Accuracy'); ax.grid(alpha=0.3)

    ax = axes[0,1]
    ax.plot(thr_vals, avg_exits, 's-', color='darkorange', linewidth=2)
    for b, lbl in zip(exit_blocks, labels):
        ax.axhline(b, color='gray', linestyle='--', linewidth=1, alpha=0.6, label=lbl)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Avg Exit Block'); ax.set_title('Avg Exit Block')
    ax.set_ylim(0, 13); ax.set_yticks(exit_blocks); ax.set_yticklabels(labels)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1,0]
    ax.errorbar(thr_vals, p99s, yerr=p99_stds, fmt='o-', color='tomato',
                linewidth=2, markersize=6, capsize=4, capthick=1.5)
    ax.set_xlabel('Threshold'); ax.set_ylabel('P99 Latency (ms)'); ax.set_title('P99 Latency'); ax.grid(alpha=0.3)

    ax = axes[1,1]
    ax.plot(thr_vals, savings, '^-', color='seagreen', linewidth=2)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Compute Savings (%)')
    ax.set_title('Compute Savings (1 - avg_block/12)'); ax.set_ylim(0, 100); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  summary saved: {save_path}")


# ── 콘솔 결과 출력 ────────────────────────────────────────────────────────────

def print_result_table(results, exit_blocks):
    labels     = [f'B{b}' for b in exit_blocks]
    thresholds = sorted(results.keys(), key=float)
    print(f"\n{'='*80}")
    print(f"  Selective EE-ViT-B/16 TRT  ({'+'.join(labels)})  Threshold Sweep")
    print(f"{'='*80}")
    print(f"  {'Threshold':>10} {'Accuracy':>10} {'AvgBlock':>9} {'Savings':>8} {'P99(ms)':>9} {'±std':>6}")
    print(f"  {'-'*65}")
    for key in thresholds:
        d = results[key]; s = d['summary']
        avg_blk = d['avg_exit_block'] or exit_blocks[-1]
        print(f"  {d['threshold']:>10.2f}  {(d['accuracy'] or 0)*100:>8.2f}%  "
              f"{avg_blk:>8.2f}  {(1-avg_blk/12)*100:>6.1f}%  "
              f"{s['p99_mean']:>9.2f}  ±{s['p99_std']:>5.2f}")
    print(f"{'='*80}")


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_imagenet_samples(num_samples: int, data_root: str, num_workers: int = 2):
    from datasets.dataloader import get_dataloader
    from utils import load_config

    cfg = load_config('configs/train.yaml')
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

    print(f"  ImageNet 데이터 로드: {len(images)} 샘플")
    return images, labels


def make_dummy_samples(num_samples: int):
    images = [torch.randn(1, 3, 224, 224) for _ in range(num_samples)]
    labels = [0] * num_samples   # 더미 레이블 (accuracy 무의미)
    print(f"  랜덤 더미 데이터: {num_samples} 샘플  (latency-only 모드)")
    return images, labels


# ── 단일 threshold 1회 실행 (TRT) ────────────────────────────────────────────

def run_sweep_once_trt(engine: SelectiveViTTRT,
                       images, labels,
                       threshold: float,
                       warmup: int = 20,
                       latency_only: bool = False):
    exit_blocks = engine.exit_blocks
    n_exits     = engine.n_segs
    exit_counts      = [0] * n_exits
    correct_per_exit = [0] * n_exits
    latencies        = []
    correct          = 0

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_cuda = img.cuda()

        logits, exit_block, lat_ms = engine.infer(img_cuda, threshold=threshold)

        if i >= warmup:
            latencies.append(lat_ms)

        head_idx = exit_blocks.index(exit_block)
        exit_counts[head_idx] += 1

        if not latency_only:
            pred = logits.argmax(dim=1).item()
            if pred == lbl:
                correct += 1
                correct_per_exit[head_idx] += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])

    accuracy = (correct / n) if not latency_only else None
    acc_per_exit = [
        (correct_per_exit[i] / exit_counts[i]) if (exit_counts[i] > 0 and not latency_only) else None
        for i in range(n_exits)
    ]
    avg_exit_block = sum(exit_blocks[i] * exit_counts[i] for i in range(n_exits)) / n

    return {
        'accuracy':       accuracy,
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

def run_n_sweeps_trt(engine: SelectiveViTTRT,
                     images, labels,
                     thresholds: list, N: int,
                     warmup: int = 20,
                     latency_only: bool = False):
    exit_blocks  = engine.exit_blocks
    labels_str   = [f'B{b}' for b in exit_blocks]

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

    print(f"\n  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}  {'avg_block':>10}")
    print(f"  {'-'*60}")

    for run_idx in range(N):
        for thr in thresholds:
            key = str(round(thr, 2))
            r   = run_sweep_once_trt(engine, images, labels, thr,
                                     warmup=warmup, latency_only=latency_only)

            if results[key]['accuracy'] is None and not latency_only:
                results[key]['accuracy']       = r['accuracy']
                results[key]['exit_rate']      = r['exit_rate']
                results[key]['acc_per_exit']   = r['acc_per_exit']
                results[key]['avg_exit_block'] = r['avg_exit_block']

            if results[key]['exit_rate'] is None:
                results[key]['exit_rate']      = r['exit_rate']
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
                for i in range(engine.n_segs)
            )
            print(f"  {thr:.2f}  run {run_idx+1:>3}/{N}  "
                  f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
                  f"avg={r['avg_exit_block']:>5.1f}  [{done}/{total}]  {exit_str}")

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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT TRT Threshold Sweep (Orin)'
    )
    parser.add_argument('--exit-blocks',  type=int, nargs='+', required=True)
    parser.add_argument('--n',            type=int, default=10)
    parser.add_argument('--num-samples',  type=int, default=1000)
    parser.add_argument('--thresholds',   type=float, nargs='+',
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument('--data-root',    type=str,   default=None)
    parser.add_argument('--latency-only', action='store_true',
                        help='랜덤 노이즈로 latency만 측정 (ImageNet 없을 때)')
    parser.add_argument('--num-workers',  type=int,   default=2)
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
    model_tag  = f"SelectiveExitViT-TRT ({'+'.join(f'B{b}' for b in exit_blocks)})"

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval',
        f'vit_trt_sel{n_exits}exit_sweep_N{args.n}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model   : {model_tag}")
    print(f"N       : {args.n} runs  samples={args.num_samples}  warmup={args.warmup}")
    print(f"Output  : {out_dir}")

    # ── 데이터 로드 ──
    latency_only = args.latency_only
    if not latency_only and args.data_root is None:
        from utils import load_config
        cfg = load_config('configs/train.yaml')
        dr = cfg.get('vit', {}).get('data_root',
             cfg.get('imagenet', {}).get('data_root', None))
        if dr is None or not os.path.isdir(dr):
            print("[WARN] data_root 없음 → latency-only 모드로 전환")
            latency_only = True
        else:
            args.data_root = dr

    if latency_only:
        images, labels = make_dummy_samples(args.num_samples)
    else:
        images, labels = load_imagenet_samples(
            args.num_samples, args.data_root, args.num_workers)

    # ── TRT 엔진 로드 ──
    print("\nLoading TRT engines ...")
    engine = load_selective_vit_trt(exit_blocks)

    # ── Sweep ──
    thresholds = sorted(set(args.thresholds))
    print(f"\nSweep start ({args.n} runs × {len(thresholds)} thresholds × {len(images)} samples) ...")
    results = run_n_sweeps_trt(engine, images, labels, thresholds,
                                args.n, warmup=args.warmup,
                                latency_only=latency_only)

    # ── 저장 ──
    print("\nSaving results ...")
    save_raw_json(results, len(images), "orin_trt", exit_blocks, thresholds, args.n,
                  os.path.join(out_dir, 'trt_sel_sweep_raw.json'))
    save_summary_csv(results, exit_blocks,
                     os.path.join(out_dir, 'trt_sel_sweep_summary.csv'))

    print("Generating plots ...")
    plot_exit_heatmap(results, exit_blocks, model_tag,
                      os.path.join(out_dir, 'trt_sel_sweep_exit_heatmap.png'))
    if not latency_only:
        plot_acc_heatmap(results, exit_blocks, model_tag,
                         os.path.join(out_dir, 'trt_sel_sweep_acc_heatmap.png'))
    plot_latency_dist(results, args.n, model_tag,
                      os.path.join(out_dir, 'trt_sel_sweep_latency_dist.png'))
    plot_summary(results, exit_blocks, model_tag,
                 os.path.join(out_dir, 'trt_sel_sweep_summary.png'))

    if not latency_only:
        print_result_table(results, exit_blocks)

    print(f"\nDone! Results at:\n  {out_dir}")


if __name__ == '__main__':
    main()
