"""
TRT Latency Benchmark: Plain ResNet-18 vs Early Exit ResNet-18

Plain과 EE 모델의 레이턴시 / 전력 / 에너지를 직접 비교.
결과를 테이블 + 그래프로 저장.

사용법 (Orin에서):
  python benchmark_trt.py \
    --plain  ../onnx_plain/plain_resnet18.engine \
    --seg1   ../onnx/seg1.engine \
    --seg2   ../onnx/seg2.engine \
    --seg3   ../onnx/seg3.engine \
    --threshold 0.80 \
    --num-samples 1000

측정 지표:
  - Accuracy, Avg/P50/P99 Latency, FPS
  - Avg Power (mW), Energy per inference (mJ)   ← tegrastats
  - GPU Utilization (%)                          ← tegrastats
"""

import os
import re
import sys
import csv
import json
import shutil
import argparse
import threading
import subprocess
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


# ── tegrastats 모니터 ─────────────────────────────────────────────────────────

def _parse_tegrastats_line(line: str) -> dict:
    """tegrastats 한 줄을 파싱해 전력/GPU 사용률 반환.

    Jetson AGX Orin 실제 포맷 (JetPack 6.x):
      VDD_GPU_SOC 1600mW/1600mW  VDD_CPU_CV 0mW/0mW  VIN_SYS_5V0 2616mW/2616mW
      형식: FIELD_NAME 현재값mW/평균값mW  → 현재값(첫 번째) 사용

    이전 포맷 호환 (JetPack 5.x):
      VDD_GPU_SOC 10340mW  TOT_PWR 16549mW  (슬래시 없음)
    """
    result = {}

    # 전력 파싱 헬퍼: "FIELD XmW/YmW" 또는 "FIELD XmW" 모두 처리 → 현재값 반환
    def _pw(field):
        m = re.search(rf'{field}\s+(\d+)\s*mW', line)
        return int(m.group(1)) if m else None

    # GPU utilization: GR3D_FREQ 92%@1300  또는  GR3D_FREQ 92%
    m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
    if m:
        result['gpu_util'] = int(m.group(1))

    # GPU+SOC 전력
    v = _pw('VDD_GPU_SOC')
    if v is not None:
        result['gpu_soc_power_mw'] = v

    # CPU+CV 전력
    v = _pw('VDD_CPU_CV')
    if v is not None:
        result['cpu_cv_power_mw'] = v

    # 총 전력: 우선순위 VIN_SYS_5V0 > TOT_PWR > VDD_IN > rail 합산
    for field in ('VIN_SYS_5V0', 'TOT_PWR', 'VDD_IN'):
        v = _pw(field)
        if v is not None:
            result['tot_power_mw'] = v
            break
    if 'tot_power_mw' not in result:
        total = sum(result.get(k, 0) for k in ('gpu_soc_power_mw', 'cpu_cv_power_mw'))
        if total > 0:
            result['tot_power_mw'] = total

    return result if result else None


class TegraStatsMonitor:
    """백그라운드 스레드에서 tegrastats를 실행해 전력/GPU 사용률 수집."""

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc   = None
        self._thread = None
        self._running = False
        self.samples: list[dict] = []
        self.available = shutil.which('tegrastats') is not None
        if not self.available:
            print("[WARNING] tegrastats를 찾을 수 없음. 전력/GPU 지표는 수집되지 않습니다.")

    def start(self):
        if not self.available:
            return
        self.samples = []
        self._running = True
        self._proc = subprocess.Popen(
            ['tegrastats', '--interval', str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self):
        if not self.available:
            return
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _reader(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            parsed = _parse_tegrastats_line(line)
            if parsed:
                self.samples.append(parsed)

    def get_stats(self) -> dict:
        """수집된 샘플로부터 평균 통계를 반환."""
        if not self.samples:
            return {'avg_power_mw': None, 'avg_gpu_util': None,
                    'avg_gpu_soc_mw': None}
        tot_powers   = [s['tot_power_mw']    for s in self.samples if 'tot_power_mw'    in s]
        gpu_utils    = [s['gpu_util']         for s in self.samples if 'gpu_util'         in s]
        gpu_soc      = [s['gpu_soc_power_mw'] for s in self.samples if 'gpu_soc_power_mw' in s]
        return {
            'avg_power_mw':  float(np.mean(tot_powers)) if tot_powers else None,
            'avg_gpu_util':  float(np.mean(gpu_utils))  if gpu_utils  else None,
            'avg_gpu_soc_mw':float(np.mean(gpu_soc))    if gpu_soc    else None,
        }


# ── TRT Engine ────────────────────────────────────────────────────────────────

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


# ── 추론 함수 ─────────────────────────────────────────────────────────────────

def infer_plain(engine: TRTEngine, image: torch.Tensor):
    t0  = time.perf_counter()
    out = engine.infer(image)
    lat = (time.perf_counter() - t0) * 1000
    return out['logits'].argmax(dim=1).item(), lat


def infer_ee(seg1, seg2, seg3, image: torch.Tensor, threshold: float):
    t0 = time.perf_counter()

    out1        = seg1.infer(image)
    feat_layer2 = out1['feat_layer2']
    ee1_logits  = out1['ee1_logits']
    if F.softmax(ee1_logits, dim=1).max(dim=1).values.item() >= threshold:
        return ee1_logits.argmax(dim=1).item(), 1, (time.perf_counter() - t0) * 1000

    out2        = seg2.infer(feat_layer2)
    feat_layer3 = out2['feat_layer3']
    ee2_logits  = out2['ee2_logits']
    if F.softmax(ee2_logits, dim=1).max(dim=1).values.item() >= threshold:
        return ee2_logits.argmax(dim=1).item(), 2, (time.perf_counter() - t0) * 1000

    out3        = seg3.infer(feat_layer3)
    return out3['main_logits'].argmax(dim=1).item(), 3, (time.perf_counter() - t0) * 1000


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

    # 데이터 미리 로드 (tegrastats 측정 구간에서 IO 제외)
    print("데이터 로드 중...")
    samples = []
    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break
        samples.append((images, labels[0].item()))
    n = len(samples)
    print(f"로드 완료: {n}개 샘플\n")

    # ── Phase 1: Plain 벤치마크 ──────────────────────────────────
    print("=== Plain ResNet-18 벤치마크 ===")
    plain_monitor = TegraStatsMonitor()
    plain_monitor.start()

    plain_lats, plain_correct = [], 0
    for images, label in samples:
        pred, lat = infer_plain(plain_engine, images)
        plain_lats.append(lat)
        if pred == label:
            plain_correct += 1

    plain_monitor.stop()
    plain_hw = plain_monitor.get_stats()
    print(f"완료 (avg {np.mean(plain_lats):.3f}ms)\n")

    # ── Phase 2: EE 벤치마크 ────────────────────────────────────
    print("=== EE ResNet-18 벤치마크 ===")
    ee_monitor = TegraStatsMonitor()
    ee_monitor.start()

    ee_lats, ee_correct = [], 0
    exit_counts = [0, 0, 0]
    for images, label in samples:
        pred, exit_idx, lat = infer_ee(seg1, seg2, seg3, images, threshold)
        ee_lats.append(lat)
        exit_counts[exit_idx - 1] += 1
        if pred == label:
            ee_correct += 1

    ee_monitor.stop()
    ee_hw = ee_monitor.get_stats()
    print(f"완료 (avg {np.mean(ee_lats):.3f}ms)\n")

    # ── 통계 계산 ────────────────────────────────────────────────
    def make_stats(lats, correct, hw, exit_rate=None):
        avg_ms = float(np.mean(lats))
        stats = {
            'accuracy':     correct / n,
            'avg_ms':       avg_ms,
            'p50_ms':       float(np.percentile(lats, 50)),
            'p99_ms':       float(np.percentile(lats, 99)),
            'fps':          1000.0 / avg_ms,
            'latencies':    lats,
            # 전력 (mW)
            'avg_power_mw': hw['avg_power_mw'],
            'avg_gpu_util': hw['avg_gpu_util'],
            # 에너지/추론 (mJ) = power(mW) × latency(ms) / 1000
            'energy_mj':    (hw['avg_power_mw'] * avg_ms / 1000.0)
                            if hw['avg_power_mw'] is not None else None,
        }
        if exit_rate is not None:
            stats['exit_rate'] = exit_rate
        return stats

    plain_stats = make_stats(plain_lats, plain_correct, plain_hw)
    ee_stats    = make_stats(ee_lats,    ee_correct,    ee_hw,
                             exit_rate=[c / n * 100 for c in exit_counts])

    return plain_stats, ee_stats, n


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_comparison(plain, ee, threshold, n):
    W = 65
    sp_avg = plain['avg_ms'] / ee['avg_ms']
    sp_p50 = plain['p50_ms'] / ee['p50_ms']

    def _pw(v): return f"{v:.1f} mW" if v is not None else "N/A"
    def _mj(v): return f"{v:.4f} mJ" if v is not None else "N/A"
    def _gu(v): return f"{v:.1f}%"   if v is not None else "N/A"

    print(f"\n{'='*W}")
    print(f"  Benchmark Results  (n={n}, threshold={threshold})")
    print(f"{'='*W}")
    print(f"{'':22s} {'Plain ResNet-18':>20s} {'EE ResNet-18':>20s}")
    print(f"{'-'*W}")
    print(f"{'Accuracy':22s} {plain['accuracy']:>20.4f} {ee['accuracy']:>20.4f}")
    print(f"{'Avg Latency (ms)':22s} {plain['avg_ms']:>20.3f} {ee['avg_ms']:>20.3f}")
    print(f"{'P50 Latency (ms)':22s} {plain['p50_ms']:>20.3f} {ee['p50_ms']:>20.3f}")
    print(f"{'P99 Latency (ms)':22s} {plain['p99_ms']:>20.3f} {ee['p99_ms']:>20.3f}")
    print(f"{'FPS':22s} {plain['fps']:>20.1f} {ee['fps']:>20.1f}")
    print(f"{'-'*W}")
    print(f"{'Speedup (avg)':22s} {'—':>20s} {sp_avg:>19.2f}x")
    print(f"{'Speedup (p50)':22s} {'—':>20s} {sp_p50:>19.2f}x")
    print(f"{'-'*W}")
    print(f"{'Avg Power':22s} {_pw(plain['avg_power_mw']):>20s} {_pw(ee['avg_power_mw']):>20s}")
    print(f"{'Energy/Inference':22s} {_mj(plain['energy_mj']):>20s} {_mj(ee['energy_mj']):>20s}")
    print(f"{'GPU Utilization':22s} {_gu(plain['avg_gpu_util']):>20s} {_gu(ee['avg_gpu_util']):>20s}")
    print(f"{'-'*W}")
    print(f"{'Exit1 rate':22s} {'—':>20s} {ee['exit_rate'][0]:>19.1f}%")
    print(f"{'Exit2 rate':22s} {'—':>20s} {ee['exit_rate'][1]:>19.1f}%")
    print(f"{'Main  rate':22s} {'—':>20s} {ee['exit_rate'][2]:>19.1f}%")
    print(f"{'='*W}\n")


# ── JSON 저장 ─────────────────────────────────────────────────────────────────

def save_json(plain, ee, threshold, n, save_path):
    def _clean(d):
        out = {}
        for k, v in d.items():
            if k == 'latencies':
                continue
            out[k] = round(v, 6) if isinstance(v, float) else v
        return out
    data = {
        'threshold': threshold, 'n': n,
        'plain': _clean(plain), 'ee': _clean(ee),
        'speedup_avg': round(plain['avg_ms'] / ee['avg_ms'], 4),
        'speedup_p50': round(plain['p50_ms'] / ee['p50_ms'], 4),
    }
    if plain['energy_mj'] is not None and ee['energy_mj'] is not None:
        data['energy_saving_pct'] = round(
            (plain['energy_mj'] - ee['energy_mj']) / plain['energy_mj'] * 100, 2)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"JSON 저장: {save_path}")


# ── 그래프 저장 ───────────────────────────────────────────────────────────────

def plot_comparison(plain, ee, threshold, save_path):
    has_power = plain['avg_power_mw'] is not None

    ncols = 4 if has_power else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(
        f'Plain ResNet-18 vs Early Exit ResNet-18  (threshold={threshold})\n'
        f'Jetson AGX Orin · TRT FP16  (n=1000)',
        fontsize=12
    )

    # 1) Latency 분포 히스토그램
    ax = axes[0]
    p99_max = max(np.percentile(plain['latencies'], 99),
                  np.percentile(ee['latencies'],    99))
    bins = np.linspace(0, p99_max * 1.1, 40)
    ax.hist(plain['latencies'], bins=bins, alpha=0.6, label='Plain', color='steelblue')
    ax.hist(ee['latencies'],    bins=bins, alpha=0.6, label='EE',    color='tomato')
    ax.axvline(plain['avg_ms'], color='steelblue', linestyle='--', linewidth=1.5,
               label=f"Plain avg {plain['avg_ms']:.2f}ms")
    ax.axvline(ee['avg_ms'],    color='tomato',    linestyle='--', linewidth=1.5,
               label=f"EE avg {ee['avg_ms']:.2f}ms")
    ax.set_title('Latency Distribution')
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('Count')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 2) Latency 비교 막대 (avg / p50 / p99)
    ax = axes[1]
    metrics    = ['avg', 'p50', 'p99']
    plain_vals = [plain['avg_ms'], plain['p50_ms'], plain['p99_ms']]
    ee_vals    = [ee['avg_ms'],    ee['p50_ms'],    ee['p99_ms']]
    x = np.arange(len(metrics)); w = 0.35
    bars1 = ax.bar(x - w/2, plain_vals, w, label='Plain', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + w/2, ee_vals,    w, label='EE',    color='tomato',    alpha=0.8)
    ax.set_title('Latency Comparison')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel('ms'); ax.legend(); ax.grid(alpha=0.3, axis='y')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    # 3) EE Exit rate 파이차트
    ax = axes[2]
    labels = [f"Exit1\n(layer2)\n{ee['exit_rate'][0]:.1f}%",
              f"Exit2\n(layer3)\n{ee['exit_rate'][1]:.1f}%",
              f"Main\n(layer4)\n{ee['exit_rate'][2]:.1f}%"]
    ax.pie(ee['exit_rate'], labels=labels,
           colors=['#4C8EFF', '#FF8C42', '#4CAF50'],
           startangle=90, wedgeprops=dict(width=0.5))
    ax.set_title(f'EE Exit Distribution\n(threshold={threshold})')

    # 4) 전력 / 에너지 비교 (tegrastats 있을 때만)
    if has_power:
        ax = axes[3]
        cats   = ['Avg Power\n(mW)', 'Energy/Inf\n(mJ × 100)']
        p_vals = [plain['avg_power_mw'], plain['energy_mj'] * 100]
        e_vals = [ee['avg_power_mw'],    ee['energy_mj']    * 100]
        x2 = np.arange(len(cats)); w2 = 0.35
        b1 = ax.bar(x2 - w2/2, p_vals, w2, label='Plain', color='steelblue', alpha=0.8)
        b2 = ax.bar(x2 + w2/2, e_vals, w2, label='EE',    color='tomato',    alpha=0.8)
        ax.set_title('Power & Energy')
        ax.set_xticks(x2); ax.set_xticklabels(cats)
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        # GPU util 텍스트
        ax.text(0.5, 0.02,
                f"GPU Util — Plain: {plain['avg_gpu_util']:.1f}%  EE: {ee['avg_gpu_util']:.1f}%",
                transform=ax.transAxes, ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"그래프 저장: {save_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plain',       type=str, default=None,
                        help='plain_resnet18.engine 경로 (미지정 시 paths.py 자동 선택)')
    parser.add_argument('--seg1',        type=str, default=None)
    parser.add_argument('--seg2',        type=str, default=None)
    parser.add_argument('--seg3',        type=str, default=None)
    parser.add_argument('--threshold',   type=float, default=0.80)
    parser.add_argument('--num-samples', type=int,   default=1000)
    parser.add_argument('--out-dir',     type=str,   default=None,
                        help='결과 저장 디렉토리 (미지정 시 experiments/eval/benchmark/)')
    args = parser.parse_args()

    # ── 엔진 경로 자동 선택 ──
    if args.plain is None:
        args.plain = paths.engine_path("plain_resnet18", "plain_resnet18.engine")
    if args.seg1 is None:
        args.seg1  = paths.engine_path("ee_resnet18", "seg1.engine")
    if args.seg2 is None:
        args.seg2  = paths.engine_path("ee_resnet18", "seg2.engine")
    if args.seg3 is None:
        args.seg3  = paths.engine_path("ee_resnet18", "seg3.engine")

    # ── 결과 디렉토리 ──
    if args.out_dir is None:
        args.out_dir = paths.eval_dir("benchmark")
    else:
        os.makedirs(args.out_dir, exist_ok=True)

    print("\n=== TRT 엔진 로드 ===")
    plain_engine = TRTEngine(args.plain)
    seg1         = TRTEngine(args.seg1)
    seg2         = TRTEngine(args.seg2)
    seg3         = TRTEngine(args.seg3)
    print("엔진 로드 완료\n")

    plain_stats, ee_stats, n = run_benchmark(
        plain_engine, seg1, seg2, seg3,
        threshold=args.threshold,
        num_samples=args.num_samples
    )

    print_comparison(plain_stats, ee_stats, args.threshold, n)

    base = os.path.join(args.out_dir, f'benchmark_thr{args.threshold:.2f}')
    plot_comparison(plain_stats, ee_stats, args.threshold, base + '.png')
    save_json(plain_stats, ee_stats, args.threshold, n, base + '.json')


if __name__ == '__main__':
    main()
