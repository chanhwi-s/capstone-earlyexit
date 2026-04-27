"""
benchmark_pytorch_vit.py — PlainViT vs SelectiveExitViT PyTorch threshold sweep

TRT 없이 PyTorch 체크포인트만으로 실행하는 벤치마크.
ImageNet val 전체(50k)를 1회 프로파일링 후 threshold별 통계를 후처리(post-hoc)로 계산.

핵심 아이디어:
  EE 모델을 한 번 full-forward(모든 블록 통과)로 프로파일링하여
  exit point별 confidence / prediction / 누적 latency를 수집.
  이후 각 threshold에 대해 재실행 없이 통계 계산 → N_threshold 배 효율적.

출력 ({EXP_DIR}/eval/pytorch_sweep_YYYYMMDD_HHMMSS/):
  pytorch_sweep_summary.csv       — model × threshold별 핵심 수치
  pytorch_sweep_raw.json          — 모델별 전체 집계 결과
  pytorch_sweep_accuracy.png      — accuracy vs threshold
  pytorch_sweep_latency.png       — avg/p90/p95/p99 latency vs threshold
  pytorch_sweep_exit_rate.png     — exit block 분포 vs threshold (stacked bar)
  pytorch_sweep_tradeoff.png      — accuracy vs avg latency scatter

사용법:
  cd src
  python benchmark/benchmark_pytorch_vit.py
  python benchmark/benchmark_pytorch_vit.py --thresholds 0.5 0.6 0.7 0.8 0.9
  python benchmark/benchmark_pytorch_vit.py --skip-plain --out-dir /tmp/results

인자:
  --data-root      ImageNet 루트 디렉토리 (기본: /home2/imagenet)
  --thresholds     sweep할 threshold 목록 (기본: 0.5 0.6 0.7 0.75 0.80 0.85 0.90)
  --exit-blocks-2  2-exit 블록 번호 (기본: 8 12)
  --exit-blocks-3  3-exit 블록 번호 (기본: 6 9 12)
  --skip-plain     PlainViT 제외
  --skip-2exit     2-exit 제외
  --skip-3exit     3-exit 제외
  --warmup         GPU warmup 샘플 수 (기본: 200)
  --num-workers    DataLoader 워커 수 (기본: 8)
  --out-dir        결과 저장 디렉토리 (기본: auto)
  --device-label   플롯/테이블 제목 디바이스 이름 (기본: RTX 5090)
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.plain_vit import build_model as build_plain
from models.ee_vit_selective import build_model as build_selective


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Data ──────────────────────────────────────────────────────────────────────

def build_val_loader(data_root: str, num_workers: int = 8) -> DataLoader:
    val_dir = os.path.join(data_root, 'val')
    if not os.path.isdir(val_dir):
        val_dir = data_root  # fallback: data_root 자체가 val 디렉토리인 경우
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    print(f"  ImageNet val: {len(dataset):,} samples  ({val_dir})")
    return loader


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(model, ckpt_path: str):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in state:
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    return model


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_plain(model, loader: DataLoader, device, warmup: int = 200):
    """PlainViT: per-sample latency + accuracy (warmup 이후 집계)."""
    model.eval()
    latencies, correct = [], []

    with torch.no_grad():
        for i, (x, lbl) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            logits = model(x)
            e.record()
            torch.cuda.synchronize()

            if i >= warmup:
                latencies.append(s.elapsed_time(e))
                correct.append(int(logits.argmax(1).item() == lbl.item()))

            n_measured = i - warmup + 1
            if i >= warmup and n_measured % 5000 == 0:
                print(f"    {n_measured:>6} / {len(loader) - warmup}")

    return latencies, correct


def profile_selective(model, loader: DataLoader, device, warmup: int = 200):
    """
    SelectiveExitViT 1회 full-forward 프로파일링.

    모든 블록을 통과시키면서 각 exit point의 confidence / prediction /
    누적 latency(ms)를 기록.  latency 는 embedding 시작 → 해당 exit head
    완료 시점까지의 시간이므로, 실제 추론에서 해당 exit에서 종료했을 때의
    레이턴시와 정확히 일치한다 (중간 exit head도 실제 추론과 동일하게 실행됨).

    Returns:
        confs_all:    [[float]]  shape [N, n_exits]  — 각 exit의 max softmax
        preds_all:    [[int]]    shape [N, n_exits]  — 각 exit의 argmax
        cum_lats_all: [[float]]  shape [N, n_exits]  — 누적 latency (ms)
        labels_all:   [int]      shape [N]
    """
    model.eval()
    n_exits = model.NUM_BLOCKS

    confs_all, preds_all, cum_lats_all, labels_all = [], [], [], []

    with torch.no_grad():
        for i, (x, lbl) in enumerate(loader):
            x = x.to(device, non_blocking=True)

            start_ev = torch.cuda.Event(enable_timing=True)
            exit_evs = [torch.cuda.Event(enable_timing=True) for _ in range(n_exits)]

            start_ev.record()
            x_feat = model._embed(x)

            ei = 0
            sample_confs, sample_preds = [], []

            for bi, block in enumerate(model.blocks):
                x_feat = block(x_feat)
                if bi in model._exit_set:
                    logits = model.exit_heads[ei](x_feat)
                    exit_evs[ei].record()
                    sample_confs.append(F.softmax(logits, dim=1).max(1).values.item())
                    sample_preds.append(logits.argmax(1).item())
                    ei += 1

            torch.cuda.synchronize()
            cum_lats = [start_ev.elapsed_time(ev) for ev in exit_evs]

            if i >= warmup:
                confs_all.append(sample_confs)
                preds_all.append(sample_preds)
                cum_lats_all.append(cum_lats)
                labels_all.append(lbl.item())

            n_measured = i - warmup + 1
            if i >= warmup and n_measured % 5000 == 0:
                print(f"    {n_measured:>6} / {len(loader) - warmup}")

    return confs_all, preds_all, cum_lats_all, labels_all


# ── Statistics ────────────────────────────────────────────────────────────────

def sweep_stats(confs_all, preds_all, cum_lats_all, labels_all,
                exit_blocks: list, threshold: float) -> dict:
    """Post-hoc threshold 적용 → 한 threshold의 통계 계산."""
    n_exits = len(exit_blocks)
    n = len(labels_all)

    latencies    = []
    exit_counts  = [0] * n_exits
    exit_correct = [0] * n_exits
    n_correct    = 0

    for i in range(n):
        chosen = n_exits - 1
        for j in range(n_exits):
            if confs_all[i][j] >= threshold:
                chosen = j
                break
        latencies.append(cum_lats_all[i][chosen])
        exit_counts[chosen] += 1
        if preds_all[i][chosen] == labels_all[i]:
            n_correct += 1
            exit_correct[chosen] += 1

    lat = np.array(latencies)
    early_exit_pct = (n - exit_counts[-1]) / n * 100
    exit_accuracy = [
        exit_correct[j] / exit_counts[j] if exit_counts[j] > 0 else float('nan')
        for j in range(n_exits)
    ]
    return {
        'threshold':      threshold,
        'accuracy':       n_correct / n,
        'exit_rate':      [c / n * 100 for c in exit_counts],
        'exit_counts':    exit_counts,
        'exit_accuracy':  exit_accuracy,
        'exit_blocks':    exit_blocks,
        'early_exit_pct': early_exit_pct,
        'n_samples':      n,
        'avg_ms': float(np.mean(lat)),
        'p50_ms': float(np.percentile(lat, 50)),
        'p90_ms': float(np.percentile(lat, 90)),
        'p95_ms': float(np.percentile(lat, 95)),
        'p99_ms': float(np.percentile(lat, 99)),
        'std_ms': float(np.std(lat)),
    }


def plain_stats(latencies: list, correct: list) -> dict:
    lat = np.array(latencies)
    n   = len(latencies)
    acc = sum(correct) / n
    return {
        'threshold':      None,
        'accuracy':       acc,
        'exit_rate':      [100.0],
        'exit_counts':    [n],
        'exit_accuracy':  [acc],
        'exit_blocks':    [12],
        'early_exit_pct': 0.0,
        'n_samples':      n,
        'avg_ms': float(np.mean(lat)),
        'p50_ms': float(np.percentile(lat, 50)),
        'p90_ms': float(np.percentile(lat, 90)),
        'p95_ms': float(np.percentile(lat, 95)),
        'p99_ms': float(np.percentile(lat, 99)),
        'std_ms': float(np.std(lat)),
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def save_json(data: dict, path: str):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  JSON: {path}")


def save_csv(rows: list, path: str):
    fields = ['model', 'threshold', 'accuracy_pct', 'avg_ms', 'p50_ms',
              'p90_ms', 'p95_ms', 'p99_ms', 'std_ms', 'early_exit_pct', 'n_samples']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV:  {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

_EE_COLORS = {'EE-ViT-2exit': 'darkorange', 'EE-ViT-3exit': 'seagreen'}


def plot_accuracy(ee_sweeps: dict, plain_acc: float, out_path: str, device_label: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rows in ee_sweeps.items():
        thrs = [r['threshold'] for r in rows]
        accs = [r['accuracy'] * 100 for r in rows]
        ax.plot(thrs, accs, 'o-', label=name, color=_EE_COLORS.get(name, 'gray'))
    ax.axhline(plain_acc * 100, ls='--', color='steelblue',
               label=f'PlainViT ({plain_acc*100:.2f}%)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(f'Accuracy vs Threshold  ({device_label}, PyTorch, ImageNet val)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  accuracy plot:   {out_path}")


def plot_accuracy_heatmap(ee_sweeps: dict, out_path: str, device_label: str):
    """exit block × threshold → 해당 block에서 탈출한 sample들의 accuracy heatmap."""
    n_models = len(ee_sweeps)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(max(8, len(next(iter(ee_sweeps.values()))) + 2) * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, rows) in zip(axes, ee_sweeps.items()):
        thrs   = [str(r['threshold']) for r in rows]
        blocks = [f'B{b}' for b in rows[0]['exit_blocks']]
        # data shape: [n_exits, n_thresholds]
        data = np.array([[r['exit_accuracy'][j] * 100
                          if not np.isnan(r['exit_accuracy'][j]) else np.nan
                          for r in rows]
                         for j in range(len(blocks))])

        valid = data[~np.isnan(data)]
        vmin  = float(np.floor(valid.min() - 1)) if len(valid) else 70.0
        vmax  = float(np.ceil(valid.max()  + 1)) if len(valid) else 90.0

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Accuracy (%)')
        ax.set_xticks(range(len(thrs)))
        ax.set_xticklabels(thrs, rotation=45)
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels(blocks)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Exit Block')
        ax.set_title(f'{name}')
        for i in range(len(blocks)):
            for j in range(len(thrs)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1f}%',
                            ha='center', va='center', fontsize=8)
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center',
                            fontsize=7, color='gray')

    fig.suptitle(f'Per-Exit Accuracy Heatmap  ({device_label})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  accuracy heatmap: {out_path}")


def plot_latency_split(ee_sweeps: dict, plain_st: dict, out_dir: str, device_label: str):
    """2-exit / 3-exit 각각 PlainViT와 1:1 비교하는 별도 latency plot."""
    metrics = [
        ('avg_ms', '-',  1.0, 'avg'),
        ('p90_ms', '--', 0.85, 'p90'),
        ('p95_ms', '-.', 0.85, 'p95'),
        ('p99_ms', ':',  0.85, 'p99'),
    ]
    plain_color = 'steelblue'

    for name, rows in ee_sweeps.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        thrs  = [r['threshold'] for r in rows]
        color = _EE_COLORS.get(name, 'gray')

        # PlainViT: 수평선 (threshold 없음)
        if plain_st:
            for metric, ls, alpha, tag in metrics:
                ax.axhline(plain_st[metric], ls=ls, color=plain_color,
                           alpha=alpha, label=f'PlainViT ({tag})')

        # EE 모델: threshold별 변화 선
        for metric, ls, alpha, tag in metrics:
            vals = [r[metric] for r in rows]
            ax.plot(thrs, vals, ls=ls, color=color,
                    alpha=alpha, label=f'{name} ({tag})')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Latency vs Threshold — PlainViT vs {name}  ({device_label})')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        suffix   = '2exit' if '2exit' in name else '3exit'
        out_path = os.path.join(out_dir, f'pytorch_sweep_latency_{suffix}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  latency plot ({suffix}): {out_path}")


def plot_exit_rate_heatmap(ee_sweeps: dict, out_path: str, device_label: str):
    """exit block × threshold → exit rate (%) heatmap."""
    n_models = len(ee_sweeps)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(max(8, len(next(iter(ee_sweeps.values()))) + 2) * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, rows) in zip(axes, ee_sweeps.items()):
        thrs   = [str(r['threshold']) for r in rows]
        blocks = [f'B{b}' for b in rows[0]['exit_blocks']]
        # data shape: [n_exits, n_thresholds]
        data = np.array([[r['exit_rate'][j] for r in rows]
                         for j in range(len(blocks))])
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label='Exit Rate (%)')
        ax.set_xticks(range(len(thrs)))
        ax.set_xticklabels(thrs, rotation=45)
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels(blocks)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Exit Block')
        ax.set_title(f'{name}')
        for i in range(len(blocks)):
            for j in range(len(thrs)):
                ax.text(j, i, f'{data[i, j]:.1f}%',
                        ha='center', va='center', fontsize=8,
                        color='black' if data[i, j] < 70 else 'white')

    fig.suptitle(f'Exit Rate Heatmap  ({device_label})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  exit rate heatmap: {out_path}")


def plot_tradeoff(ee_sweeps: dict, plain_st: dict, out_path: str, device_label: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    if plain_st:
        ax.scatter(plain_st['avg_ms'], plain_st['accuracy'] * 100,
                   s=150, color='steelblue', marker='D', label='PlainViT', zorder=5)
    for name, rows in ee_sweeps.items():
        color = _EE_COLORS.get(name, 'gray')
        xs = [r['avg_ms'] for r in rows]
        ys = [r['accuracy'] * 100 for r in rows]
        ax.scatter(xs, ys, s=80, color=color, label=name, zorder=4)
        for r, x, y in zip(rows, xs, ys):
            ax.annotate(f"{r['threshold']:.2f}", (x, y),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax.set_xlabel('Avg Latency (ms)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(f'Accuracy vs Latency Tradeoff  ({device_label}, PyTorch)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  tradeoff plot:   {out_path}")


# ── Console Table ─────────────────────────────────────────────────────────────

def print_table(all_rows: list, device_label: str):
    print(f"\n{'='*100}")
    print(f"  PyTorch Benchmark — {device_label}  (ImageNet val, 50k samples)")
    print(f"{'='*100}")
    hdr = (f"  {'Model':<20} {'Thr':>6}  {'Acc':>8}  "
           f"{'avg':>7}  {'p90':>7}  {'p95':>7}  {'p99':>7}  {'EarlyExit%':>11}")
    print(hdr)
    print(f"  {'-'*96}")
    for r in all_rows:
        thr_s = f"{r['threshold']:.2f}" if r['threshold'] is not None else "  -  "
        ee_s  = f"{r.get('early_exit_pct', 0.0):>10.1f}%"
        print(f"  {r['model']:<20} {thr_s:>6}  {r['accuracy_pct']:>7.2f}%  "
              f"{r['avg_ms']:>7.2f}  {r['p90_ms']:>7.2f}  "
              f"{r['p95_ms']:>7.2f}  {r['p99_ms']:>7.2f}  {ee_s}")
    print(f"{'='*100}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PlainViT vs SelectiveExitViT PyTorch threshold sweep (ImageNet val)'
    )
    parser.add_argument('--data-root',     type=str,   default='/home2/imagenet')
    parser.add_argument('--thresholds',    type=float, nargs='+',
                        default=[0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument('--exit-blocks-2', type=int,   nargs='+', default=[8, 12])
    parser.add_argument('--exit-blocks-3', type=int,   nargs='+', default=[6, 9, 12])
    parser.add_argument('--skip-plain',    action='store_true')
    parser.add_argument('--skip-2exit',    action='store_true')
    parser.add_argument('--skip-3exit',    action='store_true')
    parser.add_argument('--warmup',        type=int,   default=200)
    parser.add_argument('--num-workers',   type=int,   default=8)
    parser.add_argument('--out-dir',       type=str,   default=None)
    parser.add_argument('--device-label',  type=str,   default='RTX 5090')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts     = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'pytorch_sweep_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device     : {device}  ({args.device_label})")
    print(f"Thresholds : {args.thresholds}")
    print(f"Output     : {out_dir}\n")

    loader   = build_val_loader(args.data_root, args.num_workers)
    all_rows = []
    all_json = {}
    ee_sweeps = {}
    plain_st  = None

    # ── PlainViT ──────────────────────────────────────────────────────────────
    if not args.skip_plain:
        print("[PlainViT] loading timm pretrained ViT-B/16 ...")
        model = build_plain().to(device)
        model.eval()
        print(f"[PlainViT] profiling {len(loader):,} samples (warmup={args.warmup}) ...")
        lats, correct = profile_plain(model, loader, device, args.warmup)
        del model
        torch.cuda.empty_cache()

        plain_st = plain_stats(lats, correct)
        row = {'model': 'PlainViT', **plain_st,
               'accuracy_pct': plain_st['accuracy'] * 100}
        all_rows.append(row)
        all_json['plain'] = plain_st
        print(f"  → acc={plain_st['accuracy']*100:.2f}%  "
              f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms\n")

    # ── 2-exit ────────────────────────────────────────────────────────────────
    if not args.skip_2exit:
        ckpt = paths.latest_checkpoint('ee_vit_2exit')
        if ckpt is None:
            print("[WARN] ee_vit_2exit checkpoint 없음 → skip")
        else:
            print(f"[2-exit] checkpoint: {ckpt}")
            model = build_selective(args.exit_blocks_2).to(device)
            load_checkpoint(model, ckpt)
            model.eval()
            print(f"[2-exit] profiling {len(loader):,} samples ...")
            confs, preds, lats, labels = profile_selective(
                model, loader, device, args.warmup)
            del model
            torch.cuda.empty_cache()

            sweep_rows = []
            for thr in args.thresholds:
                st  = sweep_stats(confs, preds, lats, labels, args.exit_blocks_2, thr)
                row = {'model': 'EE-ViT-2exit', **st,
                       'accuracy_pct': st['accuracy'] * 100}
                all_rows.append(row)
                sweep_rows.append(st)
                rate_s = '  '.join(f'{x:.1f}%' for x in st['exit_rate'])
                print(f"  thr={thr:.2f}  acc={st['accuracy']*100:.2f}%  "
                      f"avg={st['avg_ms']:.2f}ms  exit=[{rate_s}]")
            ee_sweeps['EE-ViT-2exit'] = sweep_rows
            all_json['2exit'] = sweep_rows
            print()

    # ── 3-exit ────────────────────────────────────────────────────────────────
    if not args.skip_3exit:
        ckpt = paths.latest_checkpoint('ee_vit_3exit')
        if ckpt is None:
            print("[WARN] ee_vit_3exit checkpoint 없음 → skip")
        else:
            print(f"[3-exit] checkpoint: {ckpt}")
            model = build_selective(args.exit_blocks_3).to(device)
            load_checkpoint(model, ckpt)
            model.eval()
            print(f"[3-exit] profiling {len(loader):,} samples ...")
            confs, preds, lats, labels = profile_selective(
                model, loader, device, args.warmup)
            del model
            torch.cuda.empty_cache()

            sweep_rows = []
            for thr in args.thresholds:
                st  = sweep_stats(confs, preds, lats, labels, args.exit_blocks_3, thr)
                row = {'model': 'EE-ViT-3exit', **st,
                       'accuracy_pct': st['accuracy'] * 100}
                all_rows.append(row)
                sweep_rows.append(st)
                rate_s = '  '.join(f'{x:.1f}%' for x in st['exit_rate'])
                print(f"  thr={thr:.2f}  acc={st['accuracy']*100:.2f}%  "
                      f"avg={st['avg_ms']:.2f}ms  exit=[{rate_s}]")
            ee_sweeps['EE-ViT-3exit'] = sweep_rows
            all_json['3exit'] = sweep_rows
            print()

    # ── Save & Plot ───────────────────────────────────────────────────────────
    print("Saving ...")
    save_json(all_json, os.path.join(out_dir, 'pytorch_sweep_raw.json'))
    save_csv(all_rows,  os.path.join(out_dir, 'pytorch_sweep_summary.csv'))

    dl = args.device_label
    if ee_sweeps:
        plain_acc = plain_st['accuracy'] if plain_st else 0.0
        plot_accuracy(ee_sweeps, plain_acc,
                      os.path.join(out_dir, 'pytorch_sweep_accuracy.png'), dl)
        plot_accuracy_heatmap(ee_sweeps,
                              os.path.join(out_dir, 'pytorch_sweep_accuracy_heatmap.png'), dl)
        plot_latency_split(ee_sweeps, plain_st, out_dir, dl)
        plot_exit_rate_heatmap(ee_sweeps,
                               os.path.join(out_dir, 'pytorch_sweep_exit_rate_heatmap.png'), dl)
        plot_tradeoff(ee_sweeps, plain_st,
                      os.path.join(out_dir, 'pytorch_sweep_tradeoff.png'), dl)

    print_table(all_rows, dl)
    print(f"Done! → {out_dir}")


if __name__ == '__main__':
    main()
