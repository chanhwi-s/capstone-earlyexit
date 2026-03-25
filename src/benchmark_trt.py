"""
TRT Latency Benchmark: Plain ResNet-18 vs Early Exit ResNet-18

Plain과 EE 모델의 레이턴시를 직접 비교.
결과를 테이블 + 그래프로 저장.

사용법 (Orin에서):
  python benchmark_trt.py \
    --plain  ../onnx_plain/plain_resnet18.engine \
    --seg1   ../onnx/seg1.engine \
    --seg2   ../onnx/seg2.engine \
    --seg3   ../onnx/seg3.engine \
    --threshold 0.80 \
    --num-samples 1000
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))


# ── TRT Engine (torch 기반, infer_trt.py와 동일) ─────────────────────────────

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


# ── Plain 모델 추론 ───────────────────────────────────────────────────────────

def infer_plain(engine: TRTEngine, image: torch.Tensor):
    t0  = time.perf_counter()
    out = engine.infer(image)
    lat = (time.perf_counter() - t0) * 1000
    logits = out['logits']
    return logits.argmax(dim=1).item(), lat


# ── EE 모델 추론 ──────────────────────────────────────────────────────────────

def infer_ee(seg1, seg2, seg3, image: torch.Tensor, threshold: float):
    t0 = time.perf_counter()

    out1        = seg1.infer(image)
    feat_layer2 = out1['feat_layer2']
    ee1_logits  = out1['ee1_logits']
    conf1 = F.softmax(ee1_logits, dim=1).max(dim=1).values.item()
    if conf1 >= threshold:
        return ee1_logits.argmax(dim=1).item(), 1, (time.perf_counter() - t0) * 1000

    out2        = seg2.infer(feat_layer2)
    feat_layer3 = out2['feat_layer3']
    ee2_logits  = out2['ee2_logits']
    conf2 = F.softmax(ee2_logits, dim=1).max(dim=1).values.item()
    if conf2 >= threshold:
        return ee2_logits.argmax(dim=1).item(), 2, (time.perf_counter() - t0) * 1000

    out3        = seg3.infer(feat_layer3)
    main_logits = out3['main_logits']
    return main_logits.argmax(dim=1).item(), 3, (time.perf_counter() - t0) * 1000


# ── 벤치마크 실행 ─────────────────────────────────────────────────────────────

def run_benchmark(plain_engine, seg1, seg2, seg3, threshold, num_samples):
    from datasets.dataloader import get_dataloader
    from utils import load_config

    cfg = load_config('configs/train.yaml')
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'],
        batch_size=1,
        data_root=cfg['dataset']['data_root'],
        num_workers=0,
        seed=cfg['train']['seed']
    )

    plain_lats  = []
    ee_lats     = []
    exit_counts = [0, 0, 0]

    plain_correct = 0
    ee_correct    = 0

    print(f"벤치마크 실행 중... (n={num_samples}, threshold={threshold})")

    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        label = labels[0].item()

        # Plain 추론
        plain_pred, plain_lat = infer_plain(plain_engine, images)
        plain_lats.append(plain_lat)
        if plain_pred == label:
            plain_correct += 1

        # EE 추론
        ee_pred, exit_idx, ee_lat = infer_ee(seg1, seg2, seg3, images, threshold)
        ee_lats.append(ee_lat)
        exit_counts[exit_idx - 1] += 1
        if ee_pred == label:
            ee_correct += 1

    n = min(num_samples, i + 1)

    plain_stats = {
        'accuracy': plain_correct / n,
        'avg_ms':   float(np.mean(plain_lats)),
        'p50_ms':   float(np.percentile(plain_lats, 50)),
        'p99_ms':   float(np.percentile(plain_lats, 99)),
        'latencies': plain_lats,
    }
    ee_stats = {
        'accuracy':   ee_correct / n,
        'avg_ms':     float(np.mean(ee_lats)),
        'p50_ms':     float(np.percentile(ee_lats, 50)),
        'p99_ms':     float(np.percentile(ee_lats, 99)),
        'exit_rate':  [c / n * 100 for c in exit_counts],
        'latencies':  ee_lats,
    }

    return plain_stats, ee_stats, n


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_comparison(plain, ee, threshold, n):
    speedup_avg = plain['avg_ms'] / ee['avg_ms']
    speedup_p50 = plain['p50_ms'] / ee['p50_ms']

    print(f"\n{'='*65}")
    print(f"  Benchmark Results  (n={n}, threshold={threshold})")
    print(f"{'='*65}")
    print(f"{'':20s} {'Plain ResNet-18':>20s} {'EE ResNet-18':>20s}")
    print(f"{'-'*65}")
    print(f"{'Accuracy':20s} {plain['accuracy']:>20.4f} {ee['accuracy']:>20.4f}")
    print(f"{'Avg Latency (ms)':20s} {plain['avg_ms']:>20.3f} {ee['avg_ms']:>20.3f}")
    print(f"{'P50 Latency (ms)':20s} {plain['p50_ms']:>20.3f} {ee['p50_ms']:>20.3f}")
    print(f"{'P99 Latency (ms)':20s} {plain['p99_ms']:>20.3f} {ee['p99_ms']:>20.3f}")
    print(f"{'-'*65}")
    print(f"{'Speedup (avg)':20s} {'—':>20s} {speedup_avg:>19.2f}x")
    print(f"{'Speedup (p50)':20s} {'—':>20s} {speedup_p50:>19.2f}x")
    print(f"{'-'*65}")
    print(f"{'Exit1 rate':20s} {'—':>20s} {ee['exit_rate'][0]:>19.1f}%")
    print(f"{'Exit2 rate':20s} {'—':>20s} {ee['exit_rate'][1]:>19.1f}%")
    print(f"{'Main  rate':20s} {'—':>20s} {ee['exit_rate'][2]:>19.1f}%")
    print(f"{'='*65}\n")


# ── 그래프 저장 ───────────────────────────────────────────────────────────────

def plot_comparison(plain, ee, threshold, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Plain ResNet-18 vs Early Exit ResNet-18  (threshold={threshold})\nJetson AGX Orin · TRT FP16',
        fontsize=12
    )

    # 1) Latency 분포 (히스토그램)
    ax = axes[0]
    bins = np.linspace(0, max(np.percentile(plain['latencies'], 99),
                               np.percentile(ee['latencies'], 99)) * 1.1, 40)
    ax.hist(plain['latencies'], bins=bins, alpha=0.6, label='Plain', color='steelblue')
    ax.hist(ee['latencies'],    bins=bins, alpha=0.6, label='EE',    color='tomato')
    ax.axvline(plain['avg_ms'], color='steelblue', linestyle='--', linewidth=1.5)
    ax.axvline(ee['avg_ms'],    color='tomato',    linestyle='--', linewidth=1.5)
    ax.set_title('Latency Distribution')
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count')
    ax.legend(); ax.grid(alpha=0.3)

    # 2) 지표 막대 비교 (avg, p50, p99)
    ax = axes[1]
    metrics = ['avg', 'p50', 'p99']
    plain_vals = [plain['avg_ms'], plain['p50_ms'], plain['p99_ms']]
    ee_vals    = [ee['avg_ms'],    ee['p50_ms'],    ee['p99_ms']]
    x = np.arange(len(metrics))
    w = 0.35
    bars1 = ax.bar(x - w/2, plain_vals, w, label='Plain', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + w/2, ee_vals,    w, label='EE',    color='tomato',    alpha=0.8)
    ax.set_title('Latency Comparison')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('ms'); ax.legend(); ax.grid(alpha=0.3, axis='y')
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    # 3) EE exit rate 파이차트
    ax = axes[2]
    labels  = [f"Exit1\n(layer2)\n{ee['exit_rate'][0]:.1f}%",
               f"Exit2\n(layer3)\n{ee['exit_rate'][1]:.1f}%",
               f"Main\n(layer4)\n{ee['exit_rate'][2]:.1f}%"]
    sizes   = ee['exit_rate']
    colors  = ['#4C8EFF', '#FF8C42', '#4CAF50']
    wedges, texts = ax.pie(sizes, labels=labels, colors=colors,
                           startangle=90, wedgeprops=dict(width=0.5))
    ax.set_title(f'EE Exit Distribution\n(threshold={threshold})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"그래프 저장: {save_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plain',       type=str, required=True,
                        help='plain_resnet18.engine 경로')
    parser.add_argument('--seg1',        type=str, required=True)
    parser.add_argument('--seg2',        type=str, required=True)
    parser.add_argument('--seg3',        type=str, required=True)
    parser.add_argument('--threshold',   type=float, default=0.80)
    parser.add_argument('--num-samples', type=int,   default=1000)
    parser.add_argument('--out-dir',     type=str,   default='../benchmark_results',
                        help='결과 저장 디렉토리')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 엔진 로드 ──
    print("\n=== TRT 엔진 로드 ===")
    plain_engine = TRTEngine(args.plain)
    seg1         = TRTEngine(args.seg1)
    seg2         = TRTEngine(args.seg2)
    seg3         = TRTEngine(args.seg3)
    print()

    # ── 벤치마크 ──
    plain_stats, ee_stats, n = run_benchmark(
        plain_engine, seg1, seg2, seg3,
        threshold=args.threshold,
        num_samples=args.num_samples
    )

    # ── 결과 출력 ──
    print_comparison(plain_stats, ee_stats, args.threshold, n)

    # ── 그래프 저장 ──
    save_path = os.path.join(
        args.out_dir,
        f'benchmark_thr{args.threshold:.2f}.png'
    )
    plot_comparison(plain_stats, ee_stats, args.threshold, save_path)


if __name__ == '__main__':
    main()
