"""
Hybrid Runtime Benchmark — 4가지 런타임 방식 비교

비교 대상:
  1. Plain ResNet-18 (single TRT engine)
  2. EE  ResNet-18   (3-segment orchestration, 기존)
  3. VEE ResNet-18   (2-segment orchestration)
  4. Hybrid          (VEE seg1 + batched plain fallback)

측정 지표 (profiling_utils.py 활용):
  - P50, P90, P95, P99, P99.9 Latency
  - Avg/P90/P95/P99 Goodput (inf/sec)
  - Tail Latency Ratio (P99/P50)
  - Accuracy, Exit Rate

사용법:
  python benchmark_trt_hybrid.py \
      --plain    ../experiments/trt_engines/plain_resnet18/plain_resnet18.engine \
      --seg1     ../experiments/trt_engines/ee_resnet18/seg1.engine \
      --seg2     ../experiments/trt_engines/ee_resnet18/seg2.engine \
      --seg3     ../experiments/trt_engines/ee_resnet18/seg3.engine \
      --vee-seg1 ../experiments/trt_engines/vee_resnet18/vee_seg1.engine \
      --vee-seg2 ../experiments/trt_engines/vee_resnet18/vee_seg2.engine \
      --threshold 0.80 --num-samples 1000

      # 엔진 경로 미지정 시 paths.py에서 자동 선택
      python benchmark_trt_hybrid.py --threshold 0.80
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from profiling_utils import (
    compute_latency_stats, print_latency_report, format_stats_row,
)


# ── TRT Engine ───────────────────────────────────────────────────────────────

class TRTEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
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


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_test_data(num_samples):
    from datasets.dataloader import get_dataloader
    from utils import load_config
    cfg = load_config('configs/train.yaml')
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'], batch_size=1,
        data_root=cfg['dataset']['data_root'], num_workers=0,
        seed=cfg['train']['seed'],
    )
    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images.append(img)
        labels.append(lbl[0].item())
    return images, labels


# ── 1. Plain ResNet-18 벤치마크 ──────────────────────────────────────────────

def bench_plain(engine, images, labels):
    correct = 0
    latencies = []
    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()
        out = engine.infer(img)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        logits = list(out.values())[0]
        if logits.argmax(dim=1).item() == lbl:
            correct += 1
    return correct / len(labels), latencies


# ── 2. EE 3-segment 벤치마크 ─────────────────────────────────────────────────

def bench_ee_3seg(seg1, seg2, seg3, images, labels, threshold):
    correct = 0
    latencies = []
    exit_counts = [0, 0, 0]

    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()

        out1 = seg1.infer(img)
        ee1_logits = out1.get('ee1_logits', list(out1.values())[-1])
        feat = out1.get('feat_layer2', list(out1.values())[0])

        conf1 = F.softmax(ee1_logits, dim=1).max().item()
        if conf1 >= threshold:
            pred = ee1_logits.argmax(dim=1).item()
            exit_counts[0] += 1
        else:
            out2 = seg2.infer(feat)
            ee2_logits = out2.get('ee2_logits', list(out2.values())[-1])
            feat2 = out2.get('feat_layer3', list(out2.values())[0])

            conf2 = F.softmax(ee2_logits, dim=1).max().item()
            if conf2 >= threshold:
                pred = ee2_logits.argmax(dim=1).item()
                exit_counts[1] += 1
            else:
                out3 = seg3.infer(feat2)
                main_logits = list(out3.values())[0]
                pred = main_logits.argmax(dim=1).item()
                exit_counts[2] += 1

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if pred == lbl:
            correct += 1

    n = len(labels)
    return correct / n, latencies, [c / n * 100 for c in exit_counts]


# ── 3. VEE 2-segment 벤치마크 ────────────────────────────────────────────────

def bench_vee_2seg(seg1, seg2, images, labels, threshold):
    correct = 0
    latencies = []
    exit_counts = [0, 0]  # exit1, main

    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()

        out1 = seg1.infer(img)
        ee1_logits = out1.get('ee1_logits', list(out1.values())[-1])
        feat = out1.get('feat_layer1', list(out1.values())[0])

        conf = F.softmax(ee1_logits, dim=1).max().item()
        if conf >= threshold:
            pred = ee1_logits.argmax(dim=1).item()
            exit_counts[0] += 1
        else:
            out2 = seg2.infer(feat)
            main_logits = list(out2.values())[0]
            pred = main_logits.argmax(dim=1).item()
            exit_counts[1] += 1

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if pred == lbl:
            correct += 1

    n = len(labels)
    return correct / n, latencies, [c / n * 100 for c in exit_counts]


# ── 4. Hybrid 벤치마크 ───────────────────────────────────────────────────────

def bench_hybrid(vee_seg1, plain_engine, images, labels, threshold,
                 batch_size=8, timeout_ms=10.0):
    """Hybrid: VEE seg1 exit → batched plain fallback."""
    from infer.infer_trt_hybrid import HybridOrchestrator, eval_cifar10_hybrid

    orch = HybridOrchestrator(vee_seg1, plain_engine,
                              batch_size=batch_size, timeout_ms=timeout_ms)
    run = orch.run_stream(images, labels, threshold)

    correct = sum(
        1 for i in range(len(labels))
        if run['results'][i] is not None and
        run['results'][i]['pred'] == labels[i]
    )
    n = len(labels)
    exit1_rate = run['exit1_count'] / n * 100
    fb_rate    = run['fallback_count'] / n * 100

    return correct / n, run['latencies_ms'], [exit1_rate, fb_rate]


# ── 비교 출력 & 그래프 ───────────────────────────────────────────────────────

def print_comparison(all_stats, threshold, n):
    print(f"\n{'='*110}")
    print(f"  4-Way Runtime Comparison  |  threshold={threshold}  n={n}")
    print(f"{'='*110}")
    for label, data in all_stats.items():
        print(format_stats_row(data['stats'], label))
        print(f"{'':20} acc={data['accuracy']:.4f}  exits={data.get('exit_info', 'N/A')}")
    print(f"{'='*110}")


def plot_comparison(all_stats, threshold, save_path):
    labels = list(all_stats.keys())
    n = len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'4-Way Runtime Comparison  (threshold={threshold})', fontsize=13)

    # 1) Latency bar chart (avg, p50, p90, p95, p99)
    ax = axes[0]
    metrics = ['avg_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms']
    x = np.arange(len(metrics))
    width = 0.18
    colors = ['steelblue', 'tomato', 'orange', 'mediumpurple']
    for i, label in enumerate(labels):
        vals = [all_stats[label]['stats'].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i % 4], alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([m.replace('_ms', '') for m in metrics])
    ax.set_ylabel('ms')
    ax.set_title('Latency')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # 2) Goodput bar chart
    ax = axes[1]
    gp_metrics = ['avg_throughput', 'p90_goodput', 'p95_goodput', 'p99_goodput']
    x2 = np.arange(len(gp_metrics))
    for i, label in enumerate(labels):
        vals = [all_stats[label]['stats'].get(m, 0) for m in gp_metrics]
        ax.bar(x2 + i * width, vals, width, label=label, color=colors[i % 4], alpha=0.8)
    ax.set_xticks(x2 + width * (n - 1) / 2)
    ax.set_xticklabels(['Avg TP', 'P90 GP', 'P95 GP', 'P99 GP'])
    ax.set_ylabel('inf/sec')
    ax.set_title('Throughput / Goodput')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    # 3) Accuracy + Tail Ratio
    ax = axes[2]
    acc_vals = [all_stats[l]['accuracy'] * 100 for l in labels]
    tail_vals = [all_stats[l]['stats'].get('tail_ratio_p99_p50', 0) for l in labels]
    x3 = np.arange(len(labels))
    ax.bar(x3 - 0.2, acc_vals, 0.35, label='Accuracy (%)', color='seagreen', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x3 + 0.2, tail_vals, 0.35, label='P99/P50 Tail Ratio', color='coral', alpha=0.8)
    ax.set_xticks(x3)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel('Accuracy (%)')
    ax2.set_ylabel('Tail Ratio (x)')
    ax.set_title('Accuracy & Tail Stability')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"그래프 저장: {save_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plain',     type=str, default=None)
    parser.add_argument('--seg1',      type=str, default=None, help='EE seg1')
    parser.add_argument('--seg2',      type=str, default=None, help='EE seg2')
    parser.add_argument('--seg3',      type=str, default=None, help='EE seg3')
    parser.add_argument('--vee-seg1',  type=str, default=None, help='VEE seg1')
    parser.add_argument('--vee-seg2',  type=str, default=None, help='VEE seg2')
    parser.add_argument('--threshold',    type=float, default=0.80)
    parser.add_argument('--num-samples',  type=int,   default=1000)
    parser.add_argument('--hybrid-bs',    type=int,   default=8)
    parser.add_argument('--hybrid-to-ms', type=float, default=10.0)
    args = parser.parse_args()

    # ── 경로 자동 선택 ──
    if args.plain    is None: args.plain    = paths.engine_path("plain_resnet18", "plain_resnet18.engine")
    if args.seg1     is None: args.seg1     = paths.engine_path("ee_resnet18", "seg1.engine")
    if args.seg2     is None: args.seg2     = paths.engine_path("ee_resnet18", "seg2.engine")
    if args.seg3     is None: args.seg3     = paths.engine_path("ee_resnet18", "seg3.engine")
    if args.vee_seg1 is None: args.vee_seg1 = paths.engine_path("vee_resnet18", "vee_seg1.engine")
    if args.vee_seg2 is None: args.vee_seg2 = paths.engine_path("vee_resnet18", "vee_seg2.engine")

    # ── 엔진 로드 ──
    print("\n=== TRT 엔진 로드 ===")
    engines = {}
    for name, path in [
        ('plain',    args.plain),
        ('ee_seg1',  args.seg1),
        ('ee_seg2',  args.seg2),
        ('ee_seg3',  args.seg3),
        ('vee_seg1', args.vee_seg1),
        ('vee_seg2', args.vee_seg2),
    ]:
        if os.path.exists(path):
            engines[name] = TRTEngine(path)
        else:
            print(f"[SKIP] {name}: {path} 없음")
            engines[name] = None

    # ── 테스트 데이터 로드 ──
    print(f"\n테스트 데이터 로드 (n={args.num_samples})...")
    images, labels = load_test_data(args.num_samples)
    n = len(images)
    print(f"  로드 완료: {n}개\n")

    all_stats = {}

    # ── 1) Plain ──
    if engines['plain']:
        print("▶ [1/4] Plain ResNet-18 벤치마크...")
        acc, lats = bench_plain(engines['plain'], images, labels)
        stats = compute_latency_stats(lats)
        all_stats['Plain'] = {'accuracy': acc, 'stats': stats, 'exit_info': 'N/A'}
        print_latency_report(stats, "Plain ResNet-18")

    # ── 2) EE 3-segment ──
    if all(engines.get(k) for k in ['ee_seg1', 'ee_seg2', 'ee_seg3']):
        print("▶ [2/4] EE 3-Segment 벤치마크...")
        acc, lats, exits = bench_ee_3seg(
            engines['ee_seg1'], engines['ee_seg2'], engines['ee_seg3'],
            images, labels, args.threshold,
        )
        stats = compute_latency_stats(lats)
        exit_str = f"EE1={exits[0]:.1f}% EE2={exits[1]:.1f}% Main={exits[2]:.1f}%"
        all_stats['EE-3Seg'] = {'accuracy': acc, 'stats': stats, 'exit_info': exit_str}
        print_latency_report(stats, "EE 3-Segment")

    # ── 3) VEE 2-segment ──
    if all(engines.get(k) for k in ['vee_seg1', 'vee_seg2']):
        print("▶ [3/4] VEE 2-Segment 벤치마크...")
        acc, lats, exits = bench_vee_2seg(
            engines['vee_seg1'], engines['vee_seg2'],
            images, labels, args.threshold,
        )
        stats = compute_latency_stats(lats)
        exit_str = f"Exit1={exits[0]:.1f}% Main={exits[1]:.1f}%"
        all_stats['VEE-2Seg'] = {'accuracy': acc, 'stats': stats, 'exit_info': exit_str}
        print_latency_report(stats, "VEE 2-Segment")

    # ── 4) Hybrid ──
    if engines.get('vee_seg1') and engines.get('plain'):
        print("▶ [4/4] Hybrid (VEE + batched plain) 벤치마크...")
        acc, lats, exits = bench_hybrid(
            engines['vee_seg1'], engines['plain'],
            images, labels, args.threshold,
            batch_size=args.hybrid_bs, timeout_ms=args.hybrid_to_ms,
        )
        stats = compute_latency_stats(lats)
        exit_str = f"Exit1={exits[0]:.1f}% Fallback={exits[1]:.1f}%"
        all_stats['Hybrid'] = {'accuracy': acc, 'stats': stats, 'exit_info': exit_str}
        print_latency_report(stats, "Hybrid Runtime")

    # ── 비교 출력 ──
    if all_stats:
        print_comparison(all_stats, args.threshold, n)

    # ── 저장 ──
    out_dir = paths.eval_dir("benchmark_comparison")

    # JSON
    json_data = {}
    for label, data in all_stats.items():
        json_data[label] = {
            'accuracy': data['accuracy'],
            'exit_info': data['exit_info'],
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in data['stats'].items()},
        }
    json_path = os.path.join(out_dir, f"compare_thr{args.threshold:.2f}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON 저장: {json_path}")

    # 그래프
    png_path = os.path.join(out_dir, f"compare_thr{args.threshold:.2f}.png")
    plot_comparison(all_stats, args.threshold, png_path)


if __name__ == '__main__':
    main()
