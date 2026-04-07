"""
run_benchmark_n.py  —  Hybrid Grid Search + 4-Way Benchmark N회 반복 + 단일 파일 취합

step2_benchmark.sh에서 사용. run_sweep_n.py와 같은 철학:
  ✓ 엔진 / 데이터를 한 번만 로드
  ✓ N회 실행 결과를 하나의 디렉토리에 단일 JSON + CSV로 취합
  ✓ 실행마다 그래프 생성하지 않음 → 취합 후 한 번만 그래프 생성
  ✓ cifar10 / imagenet 모두 지원

각 실행(run):
  1. Hybrid Grid Search (batch_size × timeout_ms) → 최적 (bs, to) 탐색
  2. 4-Way Benchmark (Plain / EE-3Seg / VEE-2Seg / Hybrid) with 최적 (bs, to)

생성되는 출력:
  {EXP_DIR}/eval/benchmark_N{N}_thr{thr}_YYYYMMDD_HHMMSS/
    grid_raw.json           ← N회 × grid 탐색 결과
    grid_summary.csv        ← (bs, to)별 통계
    benchmark_raw.json      ← N회 × 4-way 비교 결과
    benchmark_summary.csv   ← 모델별 통계 (mean/std)
    benchmark_comparison.png ← 모델별 latency error bar + distribution

사용법:
  cd src
  python benchmark/run_benchmark_n.py --n 30 --threshold 0.80
  python benchmark/run_benchmark_n.py --n 30 --threshold 0.80 --dataset imagenet

인자:
  --n              반복 횟수 (기본: 10)
  --threshold      confidence threshold (필수)
  --dataset        cifar10 | imagenet (기본: cifar10)
  --data-root      데이터 루트 경로
  --num-samples    benchmark 샘플 수 (기본: 1000)
  --grid-samples   grid search 샘플 수 (기본: 500)
  --batch-sizes    grid 탐색 batch_size 후보 (기본: 2 4 8 16)
  --timeout-ms     grid 탐색 timeout 후보 (기본: 5 10 15 20 25 30 35 40)
"""

import os
import sys
import json
import csv
import argparse
import time
import threading
import subprocess
import shutil
import re
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from profiling_utils import compute_latency_stats


# ── TRT Engine ────────────────────────────────────────────────────────────────

class TRTEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names  = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = {self.input_names[0]: inputs}
        input_tensors = {}
        for name, tensor in inputs.items():
            t = tensor.contiguous().cuda().float()
            self.context.set_input_shape(name, list(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors[name] = t
        output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.zeros(shape, dtype=torch.float32, device='cuda')
            self.context.set_tensor_address(name, t.data_ptr())
            output_tensors[name] = t
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.synchronize()
        return {name: t.cpu() for name, t in output_tensors.items()}


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_test_data(dataset: str, num_samples: int, data_root=None):
    from datasets.dataloader import get_dataloader
    from utils import load_config
    cfg = load_config('configs/train.yaml')
    if data_root is None:
        if dataset == 'imagenet' and 'imagenet' in cfg:
            data_root = cfg['imagenet']['data_root']
        else:
            data_root = cfg['dataset']['data_root']
    _, test_loader, _ = get_dataloader(
        dataset=dataset, batch_size=1, data_root=data_root,
        num_workers=0, seed=cfg['train']['seed'],
    )
    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images.append(img)
        labels.append(lbl[0].item())
    return images, labels


# ── 벤치마크 함수들 ───────────────────────────────────────────────────────────

def bench_plain(engine, images, labels):
    correct, latencies = 0, []
    for img, lbl in zip(images, labels):
        t0  = time.perf_counter()
        out = engine.infer(img)
        latencies.append((time.perf_counter() - t0) * 1000)
        if list(out.values())[0].argmax(dim=1).item() == lbl:
            correct += 1
    return correct / len(labels), latencies


def bench_ee(seg1, seg2, seg3, images, labels, threshold):
    correct, latencies, exit_counts = 0, [], [0, 0, 0]
    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()
        out1       = seg1.infer(img)
        ee1_logits = out1.get('ee1_logits', list(out1.values())[-1])
        feat       = out1.get('feat_layer2', list(out1.values())[0])
        conf = F.softmax(ee1_logits, dim=1).max().item()
        if conf >= threshold:
            pred = ee1_logits.argmax(dim=1).item(); exit_counts[0] += 1
        else:
            out2       = seg2.infer(feat)
            ee2_logits = out2.get('ee2_logits', list(out2.values())[-1])
            feat2      = out2.get('feat_layer3', list(out2.values())[0])
            conf = F.softmax(ee2_logits, dim=1).max().item()
            if conf >= threshold:
                pred = ee2_logits.argmax(dim=1).item(); exit_counts[1] += 1
            else:
                out3 = seg3.infer(feat2)
                pred = list(out3.values())[0].argmax(dim=1).item(); exit_counts[2] += 1
        latencies.append((time.perf_counter() - t0) * 1000)
        if pred == lbl: correct += 1
    n = len(labels)
    return correct / n, latencies, [c / n * 100 for c in exit_counts]


def bench_vee(seg1, seg2, images, labels, threshold):
    correct, latencies, exit_counts = 0, [], [0, 0]
    for img, lbl in zip(images, labels):
        t0  = time.perf_counter()
        out1       = seg1.infer(img)
        ee1_logits = out1.get('ee1_logits', list(out1.values())[-1])
        feat       = out1.get('feat_layer1', list(out1.values())[0])
        conf = F.softmax(ee1_logits, dim=1).max().item()
        if conf >= threshold:
            pred = ee1_logits.argmax(dim=1).item(); exit_counts[0] += 1
        else:
            out2 = seg2.infer(feat)
            pred = list(out2.values())[0].argmax(dim=1).item(); exit_counts[1] += 1
        latencies.append((time.perf_counter() - t0) * 1000)
        if pred == lbl: correct += 1
    n = len(labels)
    return correct / n, latencies, [c / n * 100 for c in exit_counts]


def bench_hybrid(vee_seg1, plain_engine, images, labels,
                 threshold, batch_size, timeout_ms):
    from infer.infer_trt_hybrid import HybridOrchestrator
    orch = HybridOrchestrator(vee_seg1, plain_engine,
                              batch_size=batch_size, timeout_ms=timeout_ms)
    run  = orch.run_stream(images, labels, threshold)
    n    = len(labels)
    correct = sum(
        1 for i in range(n)
        if run['results'][i] is not None and run['results'][i]['pred'] == labels[i]
    )
    exits = [run['exit1_count'] / n * 100, run['fallback_count'] / n * 100]
    return correct / n, run['latencies_ms'], exits


def bench_hybrid_vee(vee_seg1, vee_seg2, images, labels,
                     threshold, batch_size, timeout_ms):
    from infer.infer_trt_hybrid import HybridVEEOrchestrator
    orch = HybridVEEOrchestrator(vee_seg1, vee_seg2,
                                 batch_size=batch_size, timeout_ms=timeout_ms)
    run  = orch.run_stream(images, labels, threshold)
    n    = len(labels)
    correct = sum(
        1 for i in range(n)
        if run['results'][i] is not None and run['results'][i]['pred'] == labels[i]
    )
    exits = [run['exit1_count'] / n * 100, run['fallback_count'] / n * 100]
    return correct / n, run['latencies_ms'], exits


# ── Grid Search (단일 실행) ───────────────────────────────────────────────────

def run_grid_once(vee_seg1, plain_engine, images_grid, labels_grid,
                  threshold, batch_sizes, timeout_ms_list):
    """
    Warm-up: 첫 번째 bs × 전체 timeout 한 사이클
    Grid: 전체 (bs × to) 탐색
    Returns: {(bs, to): result_dict}, best_bs, best_to
    """
    from benchmark.benchmark_hybrid_grid import bench_hybrid_once

    # warm-up
    for to_ms in timeout_ms_list:
        try:
            bench_hybrid_once(vee_seg1, plain_engine, images_grid, labels_grid,
                              threshold, batch_sizes[0], to_ms)
        except Exception:
            pass

    grid = {}
    for bs in batch_sizes:
        for to_ms in timeout_ms_list:
            try:
                r = bench_hybrid_once(vee_seg1, plain_engine, images_grid, labels_grid,
                                      threshold, bs, to_ms)
                grid[(bs, to_ms)] = r
            except Exception as e:
                grid[(bs, to_ms)] = None

    # 최적: p99 latency 최소
    valid = {k: v for k, v in grid.items() if v is not None}
    if valid:
        best_key = min(valid, key=lambda k: valid[k].get('p99_ms', float('inf')))
        best_bs, best_to = best_key
    else:
        best_bs, best_to = batch_sizes[0], timeout_ms_list[0]

    return grid, best_bs, best_to


def run_grid_once_vee(vee_seg1, vee_seg2, images_grid, labels_grid,
                      threshold, batch_sizes, timeout_ms_list):
    """
    Hybrid-VEE 전용 grid search.
    run_grid_once과 동일 구조, bench_hybrid_vee_once 사용.
    Returns: {(bs, to): result_dict}, best_bs, best_to
    """
    from benchmark.benchmark_hybrid_grid import bench_hybrid_vee_once

    # warm-up
    for to_ms in timeout_ms_list:
        try:
            bench_hybrid_vee_once(vee_seg1, vee_seg2, images_grid, labels_grid,
                                  threshold, batch_sizes[0], to_ms)
        except Exception:
            pass

    grid = {}
    for bs in batch_sizes:
        for to_ms in timeout_ms_list:
            try:
                r = bench_hybrid_vee_once(vee_seg1, vee_seg2, images_grid, labels_grid,
                                          threshold, bs, to_ms)
                grid[(bs, to_ms)] = r
            except Exception:
                grid[(bs, to_ms)] = None

    valid = {k: v for k, v in grid.items() if v is not None}
    if valid:
        best_key = min(valid, key=lambda k: valid[k].get('p99_ms', float('inf')))
        best_bs, best_to = best_key
    else:
        best_bs, best_to = batch_sizes[0], timeout_ms_list[0]

    return grid, best_bs, best_to


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_benchmark_csv(all_runs: list, out_path: str):
    """모델별 × run별 통계 CSV."""
    if not all_runs:
        return
    models = list(all_runs[0].keys())
    rows = []
    for run_idx, run_data in enumerate(all_runs):
        for model, data in run_data.items():
            stats = compute_latency_stats(data['latencies_ms'])
            rows.append({
                'run_idx':    run_idx,
                'model':      model,
                'accuracy':   round(data['accuracy'], 6),
                'exit_info':  str(data.get('exit_info', 'N/A')),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in stats.items()},
            })
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Benchmark CSV 저장: {out_path}')


def save_grid_csv(all_grid_runs: list, out_path: str):
    """grid search 전체 결과 CSV."""
    rows = []
    for run_idx, (grid, best_bs, best_to) in enumerate(all_grid_runs):
        for (bs, to_ms), r in grid.items():
            if r is None:
                continue
            rows.append({
                'run_idx':    run_idx,
                'batch_size': bs,
                'timeout_ms': to_ms,
                'is_best':    (bs == best_bs and to_ms == best_to),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in r.items()},
            })
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Grid CSV 저장: {out_path}')


# ── Grid Best 통계 ────────────────────────────────────────────────────────────

def save_grid_best_csv(all_grid_runs: list, out_path: str):
    """
    N회 grid search에서 best로 선택된 (bs, to) 조합의 빈도 CSV.
    columns: batch_size, timeout_ms, count, pct
    """
    from collections import Counter
    counter = Counter(
        (best_bs, best_to)
        for _, best_bs, best_to in all_grid_runs
    )
    n_total = len(all_grid_runs)
    rows = [
        {
            'batch_size':  bs,
            'timeout_ms':  to,
            'count':       cnt,
            'pct':         round(cnt / n_total * 100, 2),
        }
        for (bs, to), cnt in sorted(counter.items(), key=lambda x: -x[1])
    ]
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['batch_size', 'timeout_ms', 'count', 'pct'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Grid Best CSV 저장: {out_path}')


def print_grid_best_summary(all_grid_runs: list):
    """터미널에 best (bs, to) 빈도 테이블 출력."""
    from collections import Counter
    counter = Counter(
        (best_bs, best_to)
        for _, best_bs, best_to in all_grid_runs
    )
    n_total = len(all_grid_runs)
    print(f'\n  {"─" * 44}')
    print(f'  Grid Best 조합 빈도  (총 {n_total}회)')
    print(f'  {"─" * 44}')
    print(f'  {"bs":>5}  {"timeout(ms)":>12}  {"count":>7}  {"pct":>8}')
    print(f'  {"─" * 44}')
    for (bs, to), cnt in sorted(counter.items(), key=lambda x: -x[1]):
        bar = '█' * int(cnt / n_total * 20)
        print(f'  {bs:>5}  {to:>12}  {cnt:>7}  {cnt/n_total*100:>7.1f}%  {bar}')
    print(f'  {"─" * 44}')

    # 최빈 조합 강조
    if counter:
        (top_bs, top_to), top_cnt = counter.most_common(1)[0]
        print(f'  ★ 최빈 조합: bs={top_bs}, timeout={top_to}ms  '
              f'({top_cnt}/{n_total}회, {top_cnt/n_total*100:.1f}%)')


def plot_grid_best_heatmap(all_grid_runs: list, save_path: str):
    """
    N회 grid search에서 best 선택 빈도를 히트맵으로 시각화.
    X축: timeout_ms, Y축: batch_size
    """
    from collections import Counter
    import matplotlib.ticker as mticker

    counter = Counter(
        (best_bs, best_to)
        for _, best_bs, best_to in all_grid_runs
    )
    if not counter:
        return

    all_bs  = sorted(set(bs for bs, _ in counter))
    all_to  = sorted(set(to for _, to in counter))
    n_total = len(all_grid_runs)

    # 히트맵 행렬 (행=bs, 열=to)
    matrix = np.zeros((len(all_bs), len(all_to)))
    for (bs, to), cnt in counter.items():
        r = all_bs.index(bs)
        c = all_to.index(to)
        matrix[r, c] = cnt / n_total * 100  # %로 표시

    fig, ax = plt.subplots(figsize=(max(6, len(all_to) * 0.9), max(4, len(all_bs) * 0.8)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(all_to)))
    ax.set_yticks(range(len(all_bs)))
    ax.set_xticklabels([str(t) for t in all_to], rotation=45, ha='right')
    ax.set_yticklabels([str(b) for b in all_bs])
    ax.set_xlabel('Timeout (ms)')
    ax.set_ylabel('Batch Size')
    ax.set_title(f'Grid Best Selection Frequency (%)  —  N={n_total}회')

    # 셀마다 수치 표시
    for r in range(len(all_bs)):
        for c in range(len(all_to)):
            val = matrix[r, c]
            if val > 0:
                ax.text(c, r, f'{val:.0f}%',
                        ha='center', va='center',
                        fontsize=9, color='black' if val < 60 else 'white',
                        fontweight='bold')

    plt.colorbar(im, ax=ax, label='선택 빈도 (%)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grid Best 히트맵 저장: {save_path}')


# ── Plot ──────────────────────────────────────────────────────────────────────

def _collect_model_data(all_runs: list) -> dict:
    """모델별 latency/accuracy 집계."""
    model_data = defaultdict(lambda: {
        'latencies': [], 'accuracy': [],
        'avgs': [], 'p90s': [], 'p95s': [], 'p99s': [],
    })
    for run in all_runs:
        for model, data in run.items():
            lats = np.array(data['latencies_ms'])
            model_data[model]['latencies'].extend(data['latencies_ms'])
            model_data[model]['accuracy'].append(data['accuracy'])
            model_data[model]['avgs'].append(float(np.mean(lats)))
            model_data[model]['p90s'].append(float(np.percentile(lats, 90)))
            model_data[model]['p95s'].append(float(np.percentile(lats, 95)))
            model_data[model]['p99s'].append(float(np.percentile(lats, 99)))
    return model_data


def plot_benchmark_results(all_runs: list, threshold: float, save_path: str):
    """
    모델별 latency error bar + 개별 KDE distribution.
    Row 1: Avg latency bar | Accuracy | p90/p95/p99 grouped bar
    Row 2: 4모델 개별 KDE (각 모델 1칸씩)
    """
    if not all_runs:
        return

    models = list(all_runs[0].keys())
    colors = ['steelblue', 'tomato', 'orange', 'mediumpurple', 'seagreen', 'crimson']
    n_runs = len(all_runs)
    model_data = _collect_model_data(all_runs)

    n_models = len(models)
    fig, axes = plt.subplots(2, max(3, n_models), figsize=(5 * max(3, n_models), 10))
    fig.suptitle(f'4-Way Benchmark  (threshold={threshold}, N={n_runs}회)', fontsize=13)

    x = np.arange(n_models)
    col_colors = [colors[i % 4] for i in range(n_models)]

    # ── Row 1-1: Avg latency error bar ──────────────────────────────────────
    ax = axes[0][0]
    avgs  = [np.mean(model_data[m]['avgs']) for m in models]
    a_std = [np.std(model_data[m]['avgs'])  for m in models]
    ax.bar(x, avgs, 0.5, color=col_colors, alpha=0.8, yerr=a_std, capsize=5)
    for xi, (v, s) in enumerate(zip(avgs, a_std)):
        ax.text(xi, v + s + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Latency (ms)'); ax.set_title('Avg Latency  mean ± std')
    ax.grid(alpha=0.3, axis='y')

    # ── Row 1-2: Accuracy ───────────────────────────────────────────────────
    ax = axes[0][1]
    acc_means = [np.mean(model_data[m]['accuracy']) for m in models]
    ax.bar(x, [a * 100 for a in acc_means], 0.5, color=col_colors, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Accuracy (%)'); ax.set_title('Accuracy')
    ax.set_ylim([max(0, min(a * 100 for a in acc_means) - 5), 100])
    ax.grid(alpha=0.3, axis='y')

    # ── Row 1-3: p90/p95/p99 grouped bar ────────────────────────────────────
    ax = axes[0][2]
    pct_keys = [('p90s', 'P90', '//'), ('p95s', 'P95', ''), ('p99s', 'P99', 'xx')]
    width = 0.22
    offsets = [-width, 0, width]
    for (key, label, hatch), off in zip(pct_keys, offsets):
        vals = [np.mean(model_data[m][key]) for m in models]
        stds = [np.std(model_data[m][key])  for m in models]
        bars = ax.bar(x + off, vals, width, label=label,
                      color=col_colors, alpha=0.75, yerr=stds, capsize=3, hatch=hatch)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Latency (ms)'); ax.set_title('P90 / P95 / P99  mean ± std')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # 4번째 이상 칸은 빈 칸 처리
    for col_idx in range(3, max(3, n_models)):
        axes[0][col_idx].axis('off')

    # ── Row 2: 모델별 개별 KDE ──────────────────────────────────────────────
    for col_idx, (model, color) in enumerate(zip(models, colors)):
        ax = axes[1][col_idx]
        lats = np.array(model_data[model]['latencies'])
        clip = np.percentile(lats, 99.5)
        lats_c = lats[lats <= clip]
        try:
            kde = gaussian_kde(lats_c, bw_method='scott')
            xr  = np.linspace(lats_c.min(), clip, 300)
            ax.fill_between(xr, kde(xr), alpha=0.35, color=color)
            ax.plot(xr, kde(xr), color=color, linewidth=2)
            for pct, ls, lbl in [
                (50,  '--', f'Median={np.median(lats):.1f}ms'),
                (90,  ':',  f'P90={np.percentile(lats, 90):.1f}ms'),
                (95,  '-.',  f'P95={np.percentile(lats, 95):.1f}ms'),
                (99,  (0,(3,1)), f'P99={np.percentile(lats, 99):.1f}ms'),
            ]:
                ax.axvline(np.percentile(lats, pct), color='black' if pct == 50 else 'red',
                           linestyle=ls, linewidth=1, alpha=0.8, label=lbl)
        except Exception:
            ax.hist(lats_c, bins=40, color=color, alpha=0.6, density=True)
        ax.set_title(f'{model}')
        ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Density')
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    # 남는 칸 숨기기
    for col_idx in range(n_models, max(3, n_models)):
        axes[1][col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Benchmark plot 저장: {save_path}')


def plot_kde_overlay_large(all_runs: list, threshold: float, save_path: str):
    """
    4모델 KDE overlay — 단독 대형 그래프.
    benchmark_kde_overlay.png 로 별도 저장.
    """
    if not all_runs:
        return

    models = list(all_runs[0].keys())
    colors = ['steelblue', 'tomato', 'orange', 'mediumpurple', 'seagreen', 'crimson']
    n_runs = len(all_runs)
    model_data = _collect_model_data(all_runs)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f'Latency Distribution — All Models  (threshold={threshold}, N={n_runs}회)',
        fontsize=14
    )

    global_max_clip = 0.0
    for model in models:
        lats = np.array(model_data[model]['latencies'])
        global_max_clip = max(global_max_clip, float(np.percentile(lats, 99.5)))

    for model, color in zip(models, colors):
        lats = np.array(model_data[model]['latencies'])
        lats_c = lats[lats <= global_max_clip]
        avg  = np.mean(model_data[model]['avgs'])
        p90  = np.mean(model_data[model]['p90s'])
        p95  = np.mean(model_data[model]['p95s'])
        p99  = np.mean(model_data[model]['p99s'])
        try:
            kde    = gaussian_kde(lats_c, bw_method='scott')
            x_range = np.linspace(0, global_max_clip, 500)
            label  = (f'{model}  '
                      f'avg={avg:.1f}  P90={p90:.1f}  P95={p95:.1f}  P99={p99:.1f} ms')
            ax.fill_between(x_range, kde(x_range), alpha=0.2, color=color)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2.5, label=label)
        except Exception:
            pass

    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  KDE overlay 저장: {save_path}')


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Grid Search + 4-Way Benchmark N회 반복')
    parser.add_argument('--n',           type=int,   default=10,
                        help='반복 횟수 (기본: 10)')
    parser.add_argument('--threshold',   type=float, required=True,
                        help='confidence threshold (필수)')
    parser.add_argument('--dataset',     type=str,   default='cifar10',
                        choices=['cifar10', 'imagenet'])
    parser.add_argument('--data-root',   type=str,   default=None)
    parser.add_argument('--num-samples', type=int,   default=1000)
    parser.add_argument('--grid-samples',type=int,   default=500)
    parser.add_argument('--batch-sizes', type=int,   nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--timeout-ms',  type=float, nargs='+',
                        default=[5, 10, 15, 20, 25, 30, 35, 40])
    # 엔진 경로
    parser.add_argument('--plain',    type=str, default=None)
    parser.add_argument('--seg1',     type=str, default=None)
    parser.add_argument('--seg2',     type=str, default=None)
    parser.add_argument('--seg3',     type=str, default=None)
    parser.add_argument('--vee-seg1', type=str, default=None)
    parser.add_argument('--vee-seg2', type=str, default=None)
    parser.add_argument('--out-dir',  type=str, default=None)
    args = parser.parse_args()

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    thr_str = f'{args.threshold:.2f}'.replace('.', '')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval',
        f'benchmark_N{args.n}_thr{args.threshold:.2f}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    # 엔진 경로 자동 선택
    plain    = args.plain    or paths.engine_path('plain_resnet18', 'plain_resnet18.engine')
    seg1     = args.seg1     or paths.engine_path('ee_resnet18',    'seg1.engine')
    seg2     = args.seg2     or paths.engine_path('ee_resnet18',    'seg2.engine')
    seg3     = args.seg3     or paths.engine_path('ee_resnet18',    'seg3.engine')
    vee_seg1 = args.vee_seg1 or paths.engine_path('vee_resnet18',   'vee_seg1.engine')
    vee_seg2 = args.vee_seg2 or paths.engine_path('vee_resnet18',   'vee_seg2.engine')

    print('=' * 60)
    print(f'  6-Way Benchmark  ×  {args.n}회')
    print(f'  Threshold   : {args.threshold}')
    print(f'  Dataset     : {args.dataset}')
    print(f'  Samples     : {args.num_samples}  (grid: {args.grid_samples})')
    print(f'  출력 디렉토리: {out_dir}')
    print('=' * 60)

    # ── 엔진 로드 ─────────────────────────────────────────────────────────────
    print('\n=== TRT 엔진 로드 ===')
    engines = {}
    for name, path in [('plain', plain), ('seg1', seg1), ('seg2', seg2),
                       ('seg3', seg3), ('vee_seg1', vee_seg1), ('vee_seg2', vee_seg2)]:
        if os.path.exists(path):
            engines[name] = TRTEngine(path)
            print(f'  [OK] {name}: {os.path.basename(path)}')
        else:
            engines[name] = None
            print(f'  [SKIP] {name}: {path} 없음')

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    print(f'\n데이터 로드 중 (benchmark: {args.num_samples}, grid: {args.grid_samples})...')
    images_bench, labels_bench = load_test_data(args.dataset, args.num_samples, args.data_root)
    images_grid,  labels_grid  = load_test_data(args.dataset, args.grid_samples, args.data_root)
    print(f'  로드 완료: benchmark={len(images_bench)}, grid={len(images_grid)}\n')

    # ── N회 반복 ─────────────────────────────────────────────────────────────
    all_benchmark_runs    = []   # list of {model: {accuracy, latencies_ms, exit_info}}
    all_grid_plain_runs   = []   # list of (grid_dict, best_bs, best_to) — Hybrid-Plain
    all_grid_vee_runs     = []   # list of (grid_dict, best_bs, best_to) — Hybrid-VEE

    for run_idx in range(args.n):
        print(f'\n{"─"*50}')
        print(f'  Run {run_idx + 1} / {args.n}')
        print(f'{"─"*50}')

        run_results = {}

        # 1-a) Grid Search — Hybrid-Plain
        if engines.get('vee_seg1') and engines.get('plain'):
            print('  [Grid Search — Hybrid-Plain]')
            grid_p, best_bs_plain, best_to_plain = run_grid_once(
                engines['vee_seg1'], engines['plain'],
                images_grid, labels_grid,
                args.threshold, args.batch_sizes, args.timeout_ms,
            )
            all_grid_plain_runs.append((grid_p, best_bs_plain, best_to_plain))
            print(f'  → Plain 최적: bs={best_bs_plain}, to={best_to_plain}ms')
        else:
            best_bs_plain = args.batch_sizes[0]
            best_to_plain = args.timeout_ms[0]
            all_grid_plain_runs.append(({}, best_bs_plain, best_to_plain))
            print(f'  [Grid Skip — Plain] fallback: bs={best_bs_plain}, to={best_to_plain}ms')

        # 1-b) Grid Search — Hybrid-VEE
        if all(engines.get(k) for k in ['vee_seg1', 'vee_seg2']):
            print('  [Grid Search — Hybrid-VEE]')
            grid_v, best_bs_vee, best_to_vee = run_grid_once_vee(
                engines['vee_seg1'], engines['vee_seg2'],
                images_grid, labels_grid,
                args.threshold, args.batch_sizes, args.timeout_ms,
            )
            all_grid_vee_runs.append((grid_v, best_bs_vee, best_to_vee))
            print(f'  → VEE  최적: bs={best_bs_vee}, to={best_to_vee}ms')
        else:
            best_bs_vee = args.batch_sizes[0]
            best_to_vee = args.timeout_ms[0]
            all_grid_vee_runs.append(({}, best_bs_vee, best_to_vee))
            print(f'  [Grid Skip — VEE] fallback: bs={best_bs_vee}, to={best_to_vee}ms')

        # 2) Plain
        if engines.get('plain'):
            acc, lats = bench_plain(engines['plain'], images_bench, labels_bench)
            run_results['Plain'] = {'accuracy': acc, 'latencies_ms': lats, 'exit_info': 'N/A'}

        # 3) EE
        if all(engines.get(k) for k in ['seg1', 'seg2', 'seg3']):
            acc, lats, exits = bench_ee(
                engines['seg1'], engines['seg2'], engines['seg3'],
                images_bench, labels_bench, args.threshold,
            )
            run_results['EE-3Seg'] = {
                'accuracy': acc, 'latencies_ms': lats,
                'exit_info': f'EE1={exits[0]:.1f}% EE2={exits[1]:.1f}% Main={exits[2]:.1f}%',
            }

        # 4) VEE
        if all(engines.get(k) for k in ['vee_seg1', 'vee_seg2']):
            acc, lats, exits = bench_vee(
                engines['vee_seg1'], engines['vee_seg2'],
                images_bench, labels_bench, args.threshold,
            )
            run_results['VEE-2Seg'] = {
                'accuracy': acc, 'latencies_ms': lats,
                'exit_info': f'Exit1={exits[0]:.1f}% Main={exits[1]:.1f}%',
            }

        # 5) Hybrid-Plain
        if engines.get('vee_seg1') and engines.get('plain'):
            acc, lats, exits = bench_hybrid(
                engines['vee_seg1'], engines['plain'],
                images_bench, labels_bench, args.threshold,
                batch_size=best_bs_plain, timeout_ms=best_to_plain,
            )
            run_results['Hybrid-Plain'] = {
                'accuracy': acc, 'latencies_ms': lats,
                'exit_info': f'Exit1={exits[0]:.1f}% Fallback={exits[1]:.1f}% bs={best_bs_plain} to={best_to_plain}ms',
                'hybrid_bs': best_bs_plain, 'hybrid_to_ms': best_to_plain,
            }

        # 6) Hybrid-VEE
        if all(engines.get(k) for k in ['vee_seg1', 'vee_seg2']):
            acc, lats, exits = bench_hybrid_vee(
                engines['vee_seg1'], engines['vee_seg2'],
                images_bench, labels_bench, args.threshold,
                batch_size=best_bs_vee, timeout_ms=best_to_vee,
            )
            run_results['Hybrid-VEE'] = {
                'accuracy': acc, 'latencies_ms': lats,
                'exit_info': f'Exit1={exits[0]:.1f}% Fallback={exits[1]:.1f}% bs={best_bs_vee} to={best_to_vee}ms',
                'hybrid_bs': best_bs_vee, 'hybrid_to_ms': best_to_vee,
            }

        # 요약 출력
        for model, data in run_results.items():
            lats_arr = np.array(data['latencies_ms'])
            print(f'    {model:12s}  acc={data["accuracy"]:.4f}  '
                  f'avg={np.mean(lats_arr):.2f}  '
                  f'p90={np.percentile(lats_arr, 90):.2f}  '
                  f'p95={np.percentile(lats_arr, 95):.2f}  '
                  f'p99={np.percentile(lats_arr, 99):.2f} ms  '
                  f'{data["exit_info"]}')

        all_benchmark_runs.append(run_results)

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    print(f'\n결과 저장 중...')

    metadata = {
        'n': args.n, 'threshold': args.threshold,
        'dataset': args.dataset, 'num_samples': args.num_samples,
        'timestamp': ts,
    }

    def _serialize_grid(grid_runs):
        out = []
        for run_idx, (grid, best_bs, best_to) in enumerate(grid_runs):
            entry = {'run_idx': run_idx, 'best_bs': best_bs, 'best_to_ms': best_to, 'grid': {}}
            for (bs, to), r in grid.items():
                entry['grid'][f'bs={bs}_to={to}'] = (
                    {k: v for k, v in r.items() if k != 'latencies_ms'} if r else None
                )
            out.append(entry)
        return out

    # Grid JSON — Hybrid-Plain
    with open(os.path.join(out_dir, 'grid_plain_raw.json'), 'w') as f:
        json.dump({'metadata': metadata, 'runs': _serialize_grid(all_grid_plain_runs)}, f, indent=2)
    print(f'  Grid Plain JSON 저장: {out_dir}/grid_plain_raw.json')

    # Grid JSON — Hybrid-VEE
    with open(os.path.join(out_dir, 'grid_vee_raw.json'), 'w') as f:
        json.dump({'metadata': metadata, 'runs': _serialize_grid(all_grid_vee_runs)}, f, indent=2)
    print(f'  Grid VEE   JSON 저장: {out_dir}/grid_vee_raw.json')

    # Benchmark JSON (latencies_ms 포함)
    bench_json = []
    for run_idx, run_data in enumerate(all_benchmark_runs):
        entry = {'run_idx': run_idx, 'models': {}}
        for model, data in run_data.items():
            entry['models'][model] = {k: v for k, v in data.items()}
        bench_json.append(entry)
    with open(os.path.join(out_dir, 'benchmark_raw.json'), 'w') as f:
        json.dump({'metadata': metadata, 'runs': bench_json}, f, indent=2)
    print(f'  Benchmark JSON 저장: {out_dir}/benchmark_raw.json')

    # CSV — Grid (분리)
    save_grid_csv(all_grid_plain_runs, os.path.join(out_dir, 'grid_plain_summary.csv'))
    save_grid_csv(all_grid_vee_runs,   os.path.join(out_dir, 'grid_vee_summary.csv'))
    save_grid_best_csv(all_grid_plain_runs, os.path.join(out_dir, 'grid_plain_best_stats.csv'))
    save_grid_best_csv(all_grid_vee_runs,   os.path.join(out_dir, 'grid_vee_best_stats.csv'))

    # CSV — Benchmark
    save_benchmark_csv(all_benchmark_runs, os.path.join(out_dir, 'benchmark_summary.csv'))

    # Plot — Grid heatmap (분리)
    plot_grid_best_heatmap(
        all_grid_plain_runs,
        os.path.join(out_dir, 'grid_plain_best_heatmap.png'),
    )
    plot_grid_best_heatmap(
        all_grid_vee_runs,
        os.path.join(out_dir, 'grid_vee_best_heatmap.png'),
    )

    # Plot — Benchmark
    plot_benchmark_results(
        all_benchmark_runs, args.threshold,
        os.path.join(out_dir, 'benchmark_comparison.png'),
    )
    plot_kde_overlay_large(
        all_benchmark_runs, args.threshold,
        os.path.join(out_dir, 'benchmark_kde_overlay.png'),
    )

    # 요약 통계 출력
    print(f'\n{"=" * 60}')
    print(f'  {args.n}회 실행 모델별 P99 요약  (threshold={args.threshold})')
    print(f'  {"model":14s}  {"p99_mean":>10}  {"p99_std":>10}  {"acc_mean":>10}')
    print(f'  {"-" * 52}')
    if all_benchmark_runs:
        models = list(all_benchmark_runs[0].keys())
        for model in models:
            p99s = [np.percentile(run[model]['latencies_ms'], 99)
                    for run in all_benchmark_runs if model in run]
            accs = [run[model]['accuracy']
                    for run in all_benchmark_runs if model in run]
            print(f'  {model:14s}  '
                  f'{np.mean(p99s):>10.2f}ms  '
                  f'{np.std(p99s):>10.2f}ms  '
                  f'{np.mean(accs):>10.4f}')

    # Grid best 조합 빈도 터미널 출력
    if all_grid_plain_runs:
        print('\n  [Hybrid-Plain Grid Best]')
        print_grid_best_summary(all_grid_plain_runs)
    if all_grid_vee_runs:
        print('\n  [Hybrid-VEE Grid Best]')
        print_grid_best_summary(all_grid_vee_runs)

    print(f'\n  결과 저장: {out_dir}/')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
