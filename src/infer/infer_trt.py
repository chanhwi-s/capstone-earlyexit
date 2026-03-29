"""
TensorRT Segmented Model Inference (pycuda-free, torch 기반)

Jetson AGX Orin에서 TRT engine을 동적으로 조율하여 early exit 추론 수행.
EE ResNet-18 (3-segment) 및 VEE ResNet-18 (2-segment) 지원.

TensorRT 10.x API 기준 (Orin JetPack 6.x)

사용법:
  # EE 3-segment CIFAR-10 sweep
  python infer_trt.py --seg1 ...seg1.engine --seg2 ...seg2.engine \
                      --seg3 ...seg3.engine --eval-cifar10 --sweep

  # VEE 2-segment CIFAR-10 sweep
  python infer_trt.py --vee-seg1 ...vee_seg1.engine \
                      --vee-seg2 ...vee_seg2.engine \
                      --eval-cifar10 --sweep-vee

  # EE + VEE 동시 sweep
  python infer_trt.py --seg1 ...  --seg2 ...  --seg3 ... \
                      --vee-seg1 ...  --vee-seg2 ... \
                      --eval-cifar10 --sweep --sweep-vee
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths


# ── TensorRT Engine (torch 기반) ─────────────────────────────────────────────

class TRTEngine:
    """
    TRT 10.x API + torch.cuda 기반 엔진 래퍼
    pycuda 불필요
    """
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 입력 / 출력 텐서 이름 분류
        self.input_names  = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"[TRT] 로드: {os.path.basename(engine_path)}")
        print(f"      입력:  {self.input_names}")
        print(f"      출력:  {self.output_names}")

    def infer(self, inputs):
        """
        Args:
            inputs: torch.Tensor (CUDA) 또는 dict {name: torch.Tensor}

        Returns:
            dict {name: torch.Tensor (CPU)}
        """
        # 단일 텐서면 dict로 변환
        if isinstance(inputs, torch.Tensor):
            inputs = {self.input_names[0]: inputs}

        # 입력 텐서를 CUDA float32로 변환 및 주소 등록
        input_tensors = {}
        for name, tensor in inputs.items():
            t = tensor.contiguous().cuda().float()
            self.context.set_input_shape(name, list(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors[name] = t  # GC 방지용 참조 유지

        # 출력 텐서 할당 및 주소 등록
        output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.zeros(shape, dtype=torch.float32, device='cuda')
            self.context.set_tensor_address(name, t.data_ptr())
            output_tensors[name] = t

        # 비동기 실행
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.synchronize()

        return {name: t.cpu() for name, t in output_tensors.items()}


# ── Dynamic Early Exit Orchestrator ─────────────────────────────────────────

class EarlyExitInference:
    def __init__(self, seg1: TRTEngine, seg2: TRTEngine, seg3: TRTEngine):
        self.seg1 = seg1
        self.seg2 = seg2
        self.seg3 = seg3

    def infer(self, image: torch.Tensor, threshold: float = 0.80):
        """
        Args:
            image: (1, 3, H, W) torch.Tensor
            threshold: confidence threshold

        Returns:
            dict {
                logits    : torch.Tensor (1, num_classes),
                exit_idx  : int (1, 2, 3),
                confidence: float,
                latency_ms: float
            }
        """
        t0 = time.perf_counter()

        # ── Segment 1: stem + layer1 + layer2 + exit1 ──
        out1        = self.seg1.infer(image)
        feat_layer2 = out1['feat_layer2']    # (1, 128, H', W')
        ee1_logits  = out1['ee1_logits']     # (1, num_classes)

        conf1 = F.softmax(ee1_logits, dim=1).max(dim=1).values.item()
        if conf1 >= threshold:
            return {
                'logits':     ee1_logits,
                'exit_idx':   1,
                'confidence': conf1,
                'latency_ms': (time.perf_counter() - t0) * 1000,
            }

        # ── Segment 2: layer3 + exit2 ──
        out2        = self.seg2.infer(feat_layer2)
        feat_layer3 = out2['feat_layer3']    # (1, 256, H'', W'')
        ee2_logits  = out2['ee2_logits']     # (1, num_classes)

        conf2 = F.softmax(ee2_logits, dim=1).max(dim=1).values.item()
        if conf2 >= threshold:
            return {
                'logits':     ee2_logits,
                'exit_idx':   2,
                'confidence': conf2,
                'latency_ms': (time.perf_counter() - t0) * 1000,
            }

        # ── Segment 3: layer4 + main_fc ──
        out3        = self.seg3.infer(feat_layer3)
        main_logits = out3['main_logits']    # (1, num_classes)

        conf3 = F.softmax(main_logits, dim=1).max(dim=1).values.item()
        return {
            'logits':     main_logits,
            'exit_idx':   3,
            'confidence': conf3,
            'latency_ms': (time.perf_counter() - t0) * 1000,
        }


# ── VEE 2-segment Inference ──────────────────────────────────────────────────

class VEEInference:
    """VEE ResNet-18: seg1(stem+layer1+exit1) → seg2(layer2~4+main_fc)"""
    def __init__(self, seg1: TRTEngine, seg2: TRTEngine):
        self.seg1 = seg1
        self.seg2 = seg2

    def infer(self, image: torch.Tensor, threshold: float = 0.80):
        t0 = time.perf_counter()

        # ── Segment 1: stem + layer1 + exit1 ──
        out1       = self.seg1.infer(image)
        feat       = out1.get('feat_layer1', list(out1.values())[0])
        ee1_logits = out1.get('ee1_logits',  list(out1.values())[-1])

        conf1 = F.softmax(ee1_logits, dim=1).max(dim=1).values.item()
        if conf1 >= threshold:
            return {
                'logits':     ee1_logits,
                'exit_idx':   1,
                'confidence': conf1,
                'latency_ms': (time.perf_counter() - t0) * 1000,
            }

        # ── Segment 2: layer2+3+4 + main_fc ──
        out2        = self.seg2.infer(feat)
        main_logits = list(out2.values())[0]
        conf2       = F.softmax(main_logits, dim=1).max(dim=1).values.item()
        return {
            'logits':     main_logits,
            'exit_idx':   2,
            'confidence': conf2,
            'latency_ms': (time.perf_counter() - t0) * 1000,
        }


# ── CIFAR-10 평가 ─────────────────────────────────────────────────────────────

def eval_cifar10(engine: EarlyExitInference, threshold: float, num_samples: int):
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

    correct      = 0
    exit_counts  = [0, 0, 0]
    latencies    = []

    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        label  = labels[0].item()
        result = engine.infer(images, threshold=threshold)

        pred = result['logits'].argmax(dim=1).item()
        if pred == label:
            correct += 1

        exit_counts[result['exit_idx'] - 1] += 1
        latencies.append(result['latency_ms'])

    n          = min(num_samples, i + 1)
    accuracy   = correct / n
    exit_rate  = [c / n * 100 for c in exit_counts]
    avg_lat    = float(np.mean(latencies))
    p50        = float(np.percentile(latencies, 50))
    p99        = float(np.percentile(latencies, 99))

    return {
        'accuracy':      accuracy,
        'exit_rate':     exit_rate,
        'avg_lat_ms':    avg_lat,
        'p50_lat_ms':    p50,
        'p99_lat_ms':    p99,
        'n':             n,
    }


def print_result(threshold, r):
    print(
        f"thr={threshold:.2f}  "
        f"acc={r['accuracy']:.4f}  "
        f"EE1={r['exit_rate'][0]:5.1f}%  "
        f"EE2={r['exit_rate'][1]:5.1f}%  "
        f"Main={r['exit_rate'][2]:5.1f}%  "
        f"avg={r['avg_lat_ms']:.2f}ms  "
        f"p50={r['p50_lat_ms']:.2f}ms  "
        f"p99={r['p99_lat_ms']:.2f}ms"
    )


def eval_cifar10_vee(engine: VEEInference, threshold: float, num_samples: int):
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

    correct     = 0
    exit_counts = [0, 0]   # [exit1, main]
    latencies   = []

    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break
        label  = labels[0].item()
        result = engine.infer(images, threshold=threshold)
        pred   = result['logits'].argmax(dim=1).item()
        if pred == label:
            correct += 1
        exit_counts[result['exit_idx'] - 1] += 1
        latencies.append(result['latency_ms'])

    n         = min(num_samples, i + 1)
    exit_rate = [c / n * 100 for c in exit_counts]
    return {
        'accuracy':   correct / n,
        'exit_rate':  exit_rate,   # [exit1%, main%]
        'avg_lat_ms': float(np.mean(latencies)),
        'p50_lat_ms': float(np.percentile(latencies, 50)),
        'p99_lat_ms': float(np.percentile(latencies, 99)),
        'n':          n,
    }


def print_result_vee(threshold, r):
    print(
        f"thr={threshold:.2f}  "
        f"acc={r['accuracy']:.4f}  "
        f"Exit1={r['exit_rate'][0]:5.1f}%  "
        f"Main={r['exit_rate'][1]:5.1f}%  "
        f"avg={r['avg_lat_ms']:.2f}ms  "
        f"p50={r['p50_lat_ms']:.2f}ms  "
        f"p99={r['p99_lat_ms']:.2f}ms"
    )


def plot_sweep_vee(all_results, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    thresholds = sorted(all_results.keys())
    acc        = [all_results[t]['accuracy']      for t in thresholds]
    ee1_rate   = [all_results[t]['exit_rate'][0]  for t in thresholds]
    main_rate  = [all_results[t]['exit_rate'][1]  for t in thresholds]
    avg_lat    = [all_results[t]['avg_lat_ms']    for t in thresholds]
    p50_lat    = [all_results[t]['p50_lat_ms']    for t in thresholds]
    p99_lat    = [all_results[t]['p99_lat_ms']    for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('VEE ResNet-18 — Threshold Sweep (Jetson AGX Orin)', fontsize=13)

    # 1) Exit Rate
    ax = axes[0]
    ax.plot(thresholds, ee1_rate,  marker='o', label='Exit1 (layer1)', color='royalblue')
    ax.plot(thresholds, main_rate, marker='^', label='Main  (layer4)', color='green')
    ax.set_title('Exit Distribution'); ax.set_xlabel('Threshold'); ax.set_ylabel('Exit Rate (%)')
    ax.set_ylim([0, 105]); ax.legend(); ax.grid(alpha=0.3)

    # 2) Overall Accuracy
    ax = axes[1]
    ax.plot(thresholds, acc, marker='D', color='black', linewidth=2)
    ax.set_title('Overall Accuracy'); ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim([0.80, 1.0]); ax.grid(alpha=0.3)

    # 3) Latency
    ax = axes[2]
    ax.plot(thresholds, avg_lat, marker='o', label='avg', color='steelblue')
    ax.plot(thresholds, p50_lat, marker='s', label='p50', color='seagreen')
    ax.plot(thresholds, p99_lat, marker='^', label='p99', color='tomato')
    ax.set_title('Latency'); ax.set_xlabel('Threshold'); ax.set_ylabel('ms')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n그래프 저장: {save_path}")


def plot_sweep(all_results, save_path):
    import matplotlib
    matplotlib.use('Agg')  # 디스플레이 없는 서버용
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    thresholds  = sorted(all_results.keys())
    acc         = [all_results[t]['accuracy']       for t in thresholds]
    ee1_rate    = [all_results[t]['exit_rate'][0]   for t in thresholds]
    ee2_rate    = [all_results[t]['exit_rate'][1]   for t in thresholds]
    main_rate   = [all_results[t]['exit_rate'][2]   for t in thresholds]
    avg_lat     = [all_results[t]['avg_lat_ms']     for t in thresholds]
    p50_lat     = [all_results[t]['p50_lat_ms']     for t in thresholds]
    p99_lat     = [all_results[t]['p99_lat_ms']     for t in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('TRT Early Exit — Threshold Sweep (Jetson AGX Orin)', fontsize=13)

    # 1) Exit Rate
    ax = axes[0]
    ax.plot(thresholds, ee1_rate,  marker='o', label='Exit1 (EE1)',  color='royalblue')
    ax.plot(thresholds, ee2_rate,  marker='s', label='Exit2 (EE2)',  color='darkorange')
    ax.plot(thresholds, main_rate, marker='^', label='Exit3 (Main)', color='green')
    ax.set_title('Exit Distribution'); ax.set_xlabel('Threshold'); ax.set_ylabel('Exit Rate (%)')
    ax.set_ylim([0, 105]); ax.legend(); ax.grid(alpha=0.3)

    # 2) Overall Accuracy
    ax = axes[1]
    ax.plot(thresholds, acc, marker='D', color='black', linewidth=2)
    ax.set_title('Overall Accuracy'); ax.set_xlabel('Threshold'); ax.set_ylabel('Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim([0.80, 1.0]); ax.grid(alpha=0.3)

    # 3) Latency
    ax = axes[2]
    ax.plot(thresholds, avg_lat, marker='o', label='avg',  color='steelblue')
    ax.plot(thresholds, p50_lat, marker='s', label='p50',  color='seagreen')
    ax.plot(thresholds, p99_lat, marker='^', label='p99',  color='tomato')
    ax.set_title('Latency'); ax.set_xlabel('Threshold'); ax.set_ylabel('ms')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n그래프 저장: {save_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # EE 3-segment 엔진
    parser.add_argument('--seg1',        type=str, default=None)
    parser.add_argument('--seg2',        type=str, default=None)
    parser.add_argument('--seg3',        type=str, default=None)
    # VEE 2-segment 엔진
    parser.add_argument('--vee-seg1',    type=str, default=None)
    parser.add_argument('--vee-seg2',    type=str, default=None)
    # 공통 옵션
    parser.add_argument('--threshold',    type=float, default=0.80)
    parser.add_argument('--num-samples',  type=int,   default=1000)
    parser.add_argument('--eval-cifar10', action='store_true',
                        help='CIFAR-10 테스트셋으로 평가')
    parser.add_argument('--sweep',        action='store_true',
                        help='EE threshold 0.50~0.95 sweep')
    parser.add_argument('--sweep-vee',    action='store_true',
                        help='VEE threshold 0.50~0.95 sweep')
    args = parser.parse_args()

    out_dir    = paths.eval_dir("trt_sweep")
    thresholds = np.arange(0.50, 1.00, 0.05)

    # ── EE 3-segment sweep ────────────────────────────────────────────────────
    if args.sweep and args.eval_cifar10:
        if not all([args.seg1, args.seg2, args.seg3]):
            # 경로 미지정 시 자동 선택
            import paths as _p
            args.seg1 = args.seg1 or _p.engine_path("ee_resnet18", "seg1.engine")
            args.seg2 = args.seg2 or _p.engine_path("ee_resnet18", "seg2.engine")
            args.seg3 = args.seg3 or _p.engine_path("ee_resnet18", "seg3.engine")

        print("\n=== EE TRT 엔진 로드 ===")
        seg1   = TRTEngine(args.seg1)
        seg2   = TRTEngine(args.seg2)
        seg3   = TRTEngine(args.seg3)
        engine = EarlyExitInference(seg1, seg2, seg3)
        print()

        all_results = {}
        print(f"{'thr':<8} {'acc':<8} {'EE1':>8} {'EE2':>8} {'Main':>8} {'avg_ms':>10} {'p50_ms':>10} {'p99_ms':>10}")
        print("=" * 80)
        for thr in thresholds:
            t = round(float(thr), 2)
            r = eval_cifar10(engine, t, args.num_samples)
            print_result(t, r)
            all_results[t] = r

        # PNG 저장
        save_path = os.path.join(out_dir, "ee_sweep_results.png")
        plot_sweep(all_results, save_path)

        # JSON 저장 (EE 전용)
        json_path = os.path.join(out_dir, "ee_sweep_results.json")
        json_data  = {}
        for t, r in all_results.items():
            json_data[str(t)] = {
                'threshold':   t,
                'accuracy':    r['accuracy'],
                'exit_rate':   r['exit_rate'],       # [ee1%, ee2%, main%]
                'avg_ms':      r['avg_ms'],
                'p50_ms':      r.get('p50_lat_ms'),
                'p99_ms':      r.get('p99_lat_ms'),
            }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"EE sweep JSON 저장: {json_path}")

    # ── VEE 2-segment sweep ───────────────────────────────────────────────────
    if args.sweep_vee and args.eval_cifar10:
        if not all([args.vee_seg1, args.vee_seg2]):
            args.vee_seg1 = args.vee_seg1 or paths.engine_path("vee_resnet18", "vee_seg1.engine")
            args.vee_seg2 = args.vee_seg2 or paths.engine_path("vee_resnet18", "vee_seg2.engine")

        print("\n=== VEE TRT 엔진 로드 ===")
        vee_seg1   = TRTEngine(args.vee_seg1)
        vee_seg2   = TRTEngine(args.vee_seg2)
        vee_engine = VEEInference(vee_seg1, vee_seg2)
        print()

        all_results_vee = {}
        print(f"{'thr':<8} {'acc':<8} {'Exit1':>8} {'Main':>8} {'avg_ms':>10} {'p50_ms':>10} {'p99_ms':>10}")
        print("=" * 75)
        for thr in thresholds:
            t = round(float(thr), 2)
            r = eval_cifar10_vee(vee_engine, t, args.num_samples)
            print_result_vee(t, r)
            all_results_vee[t] = r

        # PNG 저장 (VEE 전용)
        save_path = os.path.join(out_dir, "vee_sweep_results.png")
        plot_sweep_vee(all_results_vee, save_path)

        # JSON 저장 (VEE 전용)
        json_path = os.path.join(out_dir, "vee_sweep_results.json")
        json_data  = {}
        for t, r in all_results_vee.items():
            json_data[str(t)] = {
                'threshold':   t,
                'accuracy':    r['accuracy'],
                'exit_rate':   r['exit_rate'],       # [exit1%, main%]
                'avg_ms':      r['avg_ms'],
                'p50_ms':      r.get('p50_lat_ms'),
                'p99_ms':      r.get('p99_lat_ms'),
            }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"VEE sweep JSON 저장: {json_path}")

    # ── 단일 threshold 평가 (sweep 없을 때) ───────────────────────────────────
    if args.eval_cifar10 and not args.sweep and not args.sweep_vee:
        if args.seg1 and args.seg2 and args.seg3:
            print("\n=== EE TRT 엔진 로드 ===")
            engine = EarlyExitInference(
                TRTEngine(args.seg1), TRTEngine(args.seg2), TRTEngine(args.seg3))
            print()
            r = eval_cifar10(engine, args.threshold, args.num_samples)
            print_result(args.threshold, r)
            print(f"  Exit1 rate: {r['exit_rate'][0]:.2f}%  "
                  f"Exit2 rate: {r['exit_rate'][1]:.2f}%  "
                  f"Main rate: {r['exit_rate'][2]:.2f}%")

        if args.vee_seg1 and args.vee_seg2:
            print("\n=== VEE TRT 엔진 로드 ===")
            vee_engine = VEEInference(
                TRTEngine(args.vee_seg1), TRTEngine(args.vee_seg2))
            print()
            r = eval_cifar10_vee(vee_engine, args.threshold, args.num_samples)
            print_result_vee(args.threshold, r)

    if not args.eval_cifar10:
        print("--eval-cifar10 옵션을 지정하세요.")


if __name__ == '__main__':
    main()
