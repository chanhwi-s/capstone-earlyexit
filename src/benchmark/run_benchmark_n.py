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


# ── Tegrastats 전력/GPU 모니터 ────────────────────────────────────────────────

class TegrastatsMonitor:
    """
    tegrastats를 백그라운드로 실행하며 전력/GPU 이용률을 수집.

    tegrastats 출력 예 (JetPack 6.x):
      ... GR3D_FREQ 99% ... VDD_IN 14098mW VDD_CPU_GPU_CV 4943mW VDD_SOC 1893mW
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc    = None
        self._thread  = None
        self._samples: list[dict] = []
        self._running = False

    def start(self):
        self._samples.clear()
        self._running = True
        try:
            self._proc = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        except FileNotFoundError:
            # tegrastats 없는 환경(개발 PC)에서는 조용히 비활성화
            self._running = False

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=2)
            self._proc = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _reader(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            s = self._parse(line)
            if s:
                self._samples.append(s)

    @staticmethod
    def _parse(line: str) -> dict | None:
        """tegrastats 한 줄을 파싱하여 전력/GPU 이용률 dict 반환."""
        d = {}
        # GR3D_FREQ (GPU 이용률)
        m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if m:
            d['gpu_util_pct'] = int(m.group(1))
        # VDD_IN (전체 보드 전력)
        m = re.search(r'VDD_IN\s+(\d+)mW', line)
        if m:
            d['vdd_in_mw'] = int(m.group(1))
        # VDD_CPU_GPU_CV
        m = re.search(r'VDD_CPU_GPU_CV\s+(\d+)mW', line)
        if m:
            d['vdd_cpu_gpu_mw'] = int(m.group(1))
        # VDD_SOC
        m = re.search(r'VDD_SOC\s+(\d+)mW', line)
        if m:
            d['vdd_soc_mw'] = int(m.group(1))
        return d if d else None

    def get_stats(self, total_wall_ms: float, n_samples: int) -> dict:
        """수집된 샘플로부터 평균 전력, energy/inference, GPU utilization 계산."""
        if not self._samples:
            return {
                'power_available':      False,
                'avg_vdd_in_mw':        None,
                'avg_gpu_util_pct':     None,
                'energy_per_inf_mj':    None,
            }
        vdd_in  = [s['vdd_in_mw']    for s in self._samples if 'vdd_in_mw'    in s]
        gpu_util = [s['gpu_util_pct'] for s in self._samples if 'gpu_util_pct' in s]
        avg_vdd_in  = float(np.mean(vdd_in))   if vdd_in   else None
        avg_gpu_util = float(np.mean(gpu_util)) if gpu_util else None
        # energy/inference (mJ) = 평균전력(mW) × 총시간(ms) / 1000 / N
        energy_per_inf_mj = (avg_vdd_in * total_wall_ms / 1000 / n_samples
                             if avg_vdd_in and total_wall_ms > 0 and n_samples > 0
                             else None)
        return {
            'power_available':      True,
            'avg_vdd_in_mw':        round(avg_vdd_in,    2) if avg_vdd_in    is not None else None,
            'avg_gpu_util_pct':     round(avg_gpu_util,  2) if avg_gpu_util  is not None else None,
            'energy_per_inf_mj':    round(energy_per_inf_mj, 4) if energy_per_inf_mj is not None else None,
        }


# ── 벤치마크 함수들 ───────────────────────────────────────────────────────────

def bench_plain(engine, images, labels, monitor: TegrastatsMonitor = None):
    correct, latencies = 0, []
    if monitor: monitor.start()
    t_wall = time.perf_counter()
    for img, lbl in zip(images, labels):
        t0  = time.perf_counter()
        out = engine.infer(img)
        latencies.append((time.perf_counter() - t0) * 1000)
        if list(out.values())[0].argmax(dim=1).item() == lbl:
            correct += 1
    total_wall_ms = (time.perf_counter() - t_wall) * 1000
    if monitor: monitor.stop()
    throughput_fps = len(labels) / (total_wall_ms / 1000)
    power_stats = monitor.get_stats(total_wall_ms, len(labels)) if monitor else {}
    return correct / len(labels), latencies, throughput_fps, power_stats


def bench_ee(seg1, seg2, seg3, images, labels, threshold, monitor: TegrastatsMonitor = None):
    correct, latencies, exit_counts = 0, [], [0, 0, 0]
    if monitor: monitor.start()
    t_wall = time.perf_counter()
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
    total_wall_ms = (time.perf_counter() - t_wall) * 1000
    if monitor: monitor.stop()
    throughput_fps = n / (total_wall_ms / 1000)
    power_stats = monitor.get_stats(total_wall_ms, n) if monitor else {}
    return correct / n, latencies, [c / n * 100 for c in exit_counts], throughput_fps, power_stats


def bench_vee(seg1, seg2, images, labels, threshold, monitor: TegrastatsMonitor = None):
    correct, latencies, exit_counts = 0, [], [0, 0]
    if monitor: monitor.start()
    t_wall = time.perf_counter()
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
    total_wall_ms = (time.perf_counter() - t_wall) * 1000
    if monitor: monitor.stop()
    throughput_fps = n / (total_wall_ms / 1000)
    power_stats = monitor.get_stats(total_wall_ms, n) if monitor else {}
    return correct / n, latencies, [c / n * 100 for c in exit_counts], throughput_fps, power_stats


def bench_hybrid(vee_seg1, plain_engine, images, labels,
                 threshold, batch_size, timeout_ms, monitor: TegrastatsMonitor = None):
    from infer.infer_trt_hybrid import HybridOrchestrator
    orch = HybridOrchestrator(vee_seg1, plain_engine,
                              batch_size=batch_size, timeout_ms=timeout_ms)
    if monitor: monitor.start()
    run  = orch.run_stream(images, labels, threshold)
    if monitor: monitor.stop()
    n    = len(labels)
    correct = sum(
        1 for i in range(n)
        if run['results'][i] is not None and run['results'][i]['pred'] == labels[i]
    )
    exits = [run['exit1_count'] / n * 100, run['fallback_count'] / n * 100]
    # total_wall_ms = 전체 스트림의 실제 경과 시간 → 올바른 throughput 계산 기준
    throughput_fps = n / (run['total_wall_ms'] / 1000)
    power_stats = monitor.get_stats(run['total_wall_ms'], n) if monitor else {}
    return correct / n, run['latencies_ms'], exits, throughput_fps, power_stats


def bench_hybrid_vee(vee_seg1, vee_seg2, images, labels,
                     threshold, batch_size, timeout_ms, monitor: TegrastatsMonitor = None):
    from infer.infer_trt_hybrid import HybridVEEOrchestrator
    orch = HybridVEEOrchestrator(vee_seg1, vee_seg2,
                                 batch_size=batch_size, timeout_ms=timeout_ms)
    if monitor: monitor.start()
    run  = orch.run_stream(images, labels, threshold)
    if monitor: monitor.stop()
    n    = len(labels)
    correct = sum(
        1 for i in range(n)
        if run['results'][i] is not None and run['results'][i]['pred'] == labels[i]
    )
    exits = [run['exit1_count'] / n * 100, run['fallback_count'] / n * 100]
    throughput_fps = n / (run['total_wall_ms'] / 1000)
    power_stats = monitor.get_stats(run['total_wall_ms'], n) if monitor else {}
    return correct / n, run['latencies_ms'], exits, throughput_fps, power_stats


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

    # 최적: p95 latency 최소
    valid = {k: v for k, v in grid.items() if v is not None}
    if valid:
        best_key = min(valid, key=lambda k: valid[k].get('p95_ms', float('inf')))
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
        best_key = min(valid, key=lambda k: valid[k].get('p95_ms', float('inf')))
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
            pwr = data.get('power', {})
            rows.append({
                'run_idx':           run_idx,
                'model':             model,
                'accuracy':          round(data['accuracy'], 6),
                'throughput_fps':    round(data.get('throughput_fps', 0.0), 2),
                'avg_vdd_in_mw':     pwr.get('avg_vdd_in_mw'),
                'avg_gpu_util_pct':  pwr.get('avg_gpu_util_pct'),
                'energy_per_inf_mj': pwr.get('energy_per_inf_mj'),
                'exit_info':         str(data.get('exit_info', 'N/A')),
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
    ax.set_title(f'Grid Best Selection Frequency (%)  —  N={n_total} runs')

    # 셀마다 수치 표시
    for r in range(len(all_bs)):
        for c in range(len(all_to)):
            val = matrix[r, c]
            if val > 0:
                ax.text(c, r, f'{val:.0f}%',
                        ha='center', va='center',
                        fontsize=9, color='black' if val < 60 else 'white',
                        fontweight='bold')

    plt.colorbar(im, ax=ax, label='Selection Freq. (%)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Grid Best 히트맵 저장: {save_path}')


# ── Plot ──────────────────────────────────────────────────────────────────────

def _collect_model_data(all_runs: list) -> dict:
    """모델별 latency/accuracy/throughput 집계."""
    model_data = defaultdict(lambda: {
        'latencies': [], 'accuracy': [], 'throughputs': [],
        'avgs': [], 'p90s': [], 'p95s': [], 'p99s': [],
        'vdd_in': [], 'gpu_util': [], 'energy_per_inf': [],
    })
    for run in all_runs:
        for model, data in run.items():
            lats = np.array(data['latencies_ms'])
            model_data[model]['latencies'].extend(data['latencies_ms'])
            model_data[model]['accuracy'].append(data['accuracy'])
            model_data[model]['throughputs'].append(data.get('throughput_fps', 0.0))
            model_data[model]['avgs'].append(float(np.mean(lats)))
            model_data[model]['p90s'].append(float(np.percentile(lats, 90)))
            model_data[model]['p95s'].append(float(np.percentile(lats, 95)))
            model_data[model]['p99s'].append(float(np.percentile(lats, 99)))
            pwr = data.get('power', {})
            if pwr.get('avg_vdd_in_mw')    is not None: model_data[model]['vdd_in'].append(pwr['avg_vdd_in_mw'])
            if pwr.get('avg_gpu_util_pct') is not None: model_data[model]['gpu_util'].append(pwr['avg_gpu_util_pct'])
            if pwr.get('energy_per_inf_mj') is not None: model_data[model]['energy_per_inf'].append(pwr['energy_per_inf_mj'])
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
    n_cols = max(4, n_models)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    fig.suptitle(f'6-Way Benchmark  (threshold={threshold}, N={n_runs} runs)', fontsize=13)

    x = np.arange(n_models)
    col_colors = [colors[i % len(colors)] for i in range(n_models)]

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
        ax.bar(x + off, vals, width, label=label,
               color=col_colors, alpha=0.75, yerr=stds, capsize=3, hatch=hatch)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Latency (ms)'); ax.set_title('P90 / P95 / P99  mean ± std')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # ── Row 1-4: Throughput (fps) ────────────────────────────────────────────
    ax = axes[0][3]
    thr_means = [np.mean(model_data[m]['throughputs']) for m in models]
    thr_stds  = [np.std(model_data[m]['throughputs'])  for m in models]
    ax.bar(x, thr_means, 0.5, color=col_colors, alpha=0.8, yerr=thr_stds, capsize=5)
    for xi, (v, s) in enumerate(zip(thr_means, thr_stds)):
        ax.text(xi, v + s + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('Throughput (fps)'); ax.set_title('Throughput  mean ± std\n(N / wall-clock time)')
    ax.grid(alpha=0.3, axis='y')

    # 5번째 이상 칸 처리
    for col_idx in range(4, n_cols):
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
        f'Latency Distribution — All Models  (threshold={threshold}, N={n_runs} runs)',
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
        thr  = np.mean(model_data[model]['throughputs'])
        try:
            kde    = gaussian_kde(lats_c, bw_method='scott')
            x_range = np.linspace(0, global_max_clip, 500)
            label  = (f'{model}  '
                      f'avg={avg:.1f}  P90={p90:.1f}  P95={p95:.1f}  P99={p99:.1f} ms  '
                      f'thr={thr:.1f} fps')
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

        mon = TegrastatsMonitor(interval_ms=100)

        # 2) Plain
        if engines.get('plain'):
            acc, lats, thr, pwr = bench_plain(engines['plain'], images_bench, labels_bench, mon)
            run_results['Plain'] = {'accuracy': acc, 'latencies_ms': lats,
                                    'throughput_fps': thr, 'power': pwr, 'exit_info': 'N/A'}

        # 3) EE
        if all(engines.get(k) for k in ['seg1', 'seg2', 'seg3']):
            acc, lats, exits, thr, pwr = bench_ee(
                engines['seg1'], engines['seg2'], engines['seg3'],
                images_bench, labels_bench, args.threshold, mon,
            )
            run_results['EE-3Seg'] = {
                'accuracy': acc, 'latencies_ms': lats, 'throughput_fps': thr, 'power': pwr,
                'exit_info': f'EE1={exits[0]:.1f}% EE2={exits[1]:.1f}% Main={exits[2]:.1f}%',
            }

        # 4) VEE
        if all(engines.get(k) for k in ['vee_seg1', 'vee_seg2']):
            acc, lats, exits, thr, pwr = bench_vee(
                engines['vee_seg1'], engines['vee_seg2'],
                images_bench, labels_bench, args.threshold, mon,
            )
            run_results['VEE-2Seg'] = {
                'accuracy': acc, 'latencies_ms': lats, 'throughput_fps': thr, 'power': pwr,
                'exit_info': f'Exit1={exits[0]:.1f}% Main={exits[1]:.1f}%',
            }

        # 5) Hybrid-Plain
        if engines.get('vee_seg1') and engines.get('plain'):
            acc, lats, exits, thr, pwr = bench_hybrid(
                engines['vee_seg1'], engines['plain'],
                images_bench, labels_bench, args.threshold,
                batch_size=best_bs_plain, timeout_ms=best_to_plain, monitor=mon,
            )
            run_results['Hybrid-Plain'] = {
                'accuracy': acc, 'latencies_ms': lats, 'throughput_fps': thr, 'power': pwr,
                'exit_info': f'Exit1={exits[0]:.1f}% Fallback={exits[1]:.1f}% bs={best_bs_plain} to={best_to_plain}ms',
                'hybrid_bs': best_bs_plain, 'hybrid_to_ms': best_to_plain,
            }

        # 6) Hybrid-VEE
        if all(engines.get(k) for k in ['vee_seg1', 'vee_seg2']):
            acc, lats, exits, thr, pwr = bench_hybrid_vee(
                engines['vee_seg1'], engines['vee_seg2'],
                images_bench, labels_bench, args.threshold,
                batch_size=best_bs_vee, timeout_ms=best_to_vee, monitor=mon,
            )
            run_results['Hybrid-VEE'] = {
                'accuracy': acc, 'latencies_ms': lats, 'throughput_fps': thr, 'power': pwr,
                'exit_info': f'Exit1={exits[0]:.1f}% Fallback={exits[1]:.1f}% bs={best_bs_vee} to={best_to_vee}ms',
                'hybrid_bs': best_bs_vee, 'hybrid_to_ms': best_to_vee,
            }

        # 요약 출력
        for model, data in run_results.items():
            lats_arr = np.array(data['latencies_ms'])
            pwr = data.get('power', {})
            if (pwr.get('power_available')
                    and pwr.get('avg_vdd_in_mw') is not None
                    and pwr.get('avg_gpu_util_pct') is not None
                    and pwr.get('energy_per_inf_mj') is not None):
                pwr_str = (f'  pwr={pwr["avg_vdd_in_mw"]:.0f}mW  '
                           f'gpu={pwr["avg_gpu_util_pct"]:.0f}%  '
                           f'e/inf={pwr["energy_per_inf_mj"]:.2f}mJ')
            else:
                pwr_str = ''
            print(f'    {model:12s}  acc={data["accuracy"]:.4f}  '
                  f'avg={np.mean(lats_arr):.2f}  '
                  f'p90={np.percentile(lats_arr, 90):.2f}  '
                  f'p95={np.percentile(lats_arr, 95):.2f}  '
                  f'p99={np.percentile(lats_arr, 99):.2f} ms  '
                  f'thr={data["throughput_fps"]:.1f} fps'
                  f'{pwr_str}  '
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
    print(f'\n{"=" * 80}')
    print(f'  {args.n}회 실행 모델별 요약  (threshold={args.threshold})')
    print(f'  {"model":14s}  {"avg_ms":>8}  {"p90_ms":>8}  {"p95_ms":>8}  {"p99_ms":>8}  {"thr_fps":>9}  {"acc":>7}')
    print(f'  {"-" * 72}')
    if all_benchmark_runs:
        models = list(all_benchmark_runs[0].keys())
        model_summary = {}
        for model in models:
            lats_all = [lat for run in all_benchmark_runs if model in run
                        for lat in run[model]['latencies_ms']]
            thrs = [run[model]['throughput_fps']
                    for run in all_benchmark_runs if model in run]
            accs = [run[model]['accuracy']
                    for run in all_benchmark_runs if model in run]
            model_summary[model] = {
                'avg': np.mean(lats_all), 'p90': np.percentile(lats_all, 90),
                'p95': np.percentile(lats_all, 95), 'p99': np.percentile(lats_all, 99),
                'thr': np.mean(thrs), 'acc': np.mean(accs),
            }
            s = model_summary[model]
            print(f'  {model:14s}  '
                  f'{s["avg"]:>8.2f}  {s["p90"]:>8.2f}  '
                  f'{s["p95"]:>8.2f}  {s["p99"]:>8.2f}  '
                  f'{s["thr"]:>9.1f}  {s["acc"]:>7.4f}')

        # plain 대비 speedup 출력
        if 'Plain' in model_summary:
            print(f'\n  Plain 대비 speedup (배율 — 클수록 좋음):')
            print(f'  {"model":14s}  {"avg_x":>7}  {"p90_x":>7}  {"p95_x":>7}  {"p99_x":>7}  {"thr_x":>7}')
            print(f'  {"-" * 56}')
            pb = model_summary['Plain']
            for model in models:
                if model == 'Plain':
                    continue
                s = model_summary[model]
                print(f'  {model:14s}  '
                      f'{pb["avg"]/s["avg"]:>7.2f}x  '
                      f'{pb["p90"]/s["p90"]:>7.2f}x  '
                      f'{pb["p95"]/s["p95"]:>7.2f}x  '
                      f'{pb["p99"]/s["p99"]:>7.2f}x  '
                      f'{s["thr"]/pb["thr"]:>7.2f}x')

        # 전력 요약 출력
        power_rows = [
            (model, run[model].get('power', {}))
            for run in all_benchmark_runs[:1]   # 첫 run에서 availability 확인
            for model in run
        ]
        if any(p.get('power_available') for _, p in power_rows):
            print(f'\n  전력 / Energy 요약  (N회 평균):')
            print(f'  {"model":14s}  {"vdd_in_mw":>10}  {"gpu_util%":>10}  {"mJ/inf":>9}')
            print(f'  {"-" * 52}')
            for model in models:
                pwr_list = [run[model].get('power', {})
                            for run in all_benchmark_runs if model in run]
                vdd_vals = [p['avg_vdd_in_mw']    for p in pwr_list if p.get('avg_vdd_in_mw')    is not None]
                gpu_vals = [p['avg_gpu_util_pct']  for p in pwr_list if p.get('avg_gpu_util_pct') is not None]
                eng_vals = [p['energy_per_inf_mj'] for p in pwr_list if p.get('energy_per_inf_mj') is not None]
                vdd_str = f'{np.mean(vdd_vals):>10.0f}' if vdd_vals else f'{"N/A":>10}'
                gpu_str = f'{np.mean(gpu_vals):>10.1f}' if gpu_vals else f'{"N/A":>10}'
                eng_str = f'{np.mean(eng_vals):>9.3f}'  if eng_vals else f'{"N/A":>9}'
                print(f'  {model:14s}  {vdd_str}  {gpu_str}  {eng_str}')

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
