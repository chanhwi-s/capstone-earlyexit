"""
TensorRT Segmented Model Inference (pycuda-free, torch 기반)

Jetson AGX Orin에서 3개의 TRT engine을 동적으로 조율하여 early exit 추론 수행.
pycuda 없이 torch.cuda로 GPU 메모리 관리.

TensorRT 10.x API 기준 (Orin JetPack 6.x)

사용법:
  # CIFAR-10 평가
  python infer_trt.py --seg1 ../onnx/seg1.engine \
                      --seg2 ../onnx/seg2.engine \
                      --seg3 ../onnx/seg3.engine \
                      --eval-cifar10 --threshold 0.80 --num-samples 1000

  # 여러 threshold 일괄 평가
  python infer_trt.py --seg1 ../onnx/seg1.engine \
                      --seg2 ../onnx/seg2.engine \
                      --seg3 ../onnx/seg3.engine \
                      --eval-cifar10 --sweep
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt

sys.path.insert(0, os.path.dirname(__file__))


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


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg1',        type=str, required=True)
    parser.add_argument('--seg2',        type=str, required=True)
    parser.add_argument('--seg3',        type=str, required=True)
    parser.add_argument('--threshold',   type=float, default=0.80)
    parser.add_argument('--num-samples', type=int,   default=1000)
    parser.add_argument('--eval-cifar10', action='store_true',
                        help='CIFAR-10 테스트셋으로 평가')
    parser.add_argument('--sweep',       action='store_true',
                        help='threshold 0.50~0.95 전체 sweep')
    args = parser.parse_args()

    # ── 엔진 로드 ──
    print("\n=== TRT 엔진 로드 ===")
    seg1 = TRTEngine(args.seg1)
    seg2 = TRTEngine(args.seg2)
    seg3 = TRTEngine(args.seg3)
    engine = EarlyExitInference(seg1, seg2, seg3)
    print()

    if args.eval_cifar10:
        if args.sweep:
            # threshold sweep
            thresholds = np.arange(0.50, 1.00, 0.05)
            print(f"{'thr':<8} {'acc':<8} {'EE1':>8} {'EE2':>8} {'Main':>8} {'avg_ms':>10} {'p50_ms':>10} {'p99_ms':>10}")
            print("=" * 80)
            for thr in thresholds:
                r = eval_cifar10(engine, round(float(thr), 2), args.num_samples)
                print_result(round(float(thr), 2), r)
        else:
            # 단일 threshold
            print(f"=== CIFAR-10 평가 (threshold={args.threshold}, n={args.num_samples}) ===")
            r = eval_cifar10(engine, args.threshold, args.num_samples)
            print()
            print_result(args.threshold, r)
            print()
            print(f"  정확도:       {r['accuracy']:.4f}  ({r['accuracy']*100:.2f}%)")
            print(f"  Exit1 rate:   {r['exit_rate'][0]:.2f}%")
            print(f"  Exit2 rate:   {r['exit_rate'][1]:.2f}%")
            print(f"  Main  rate:   {r['exit_rate'][2]:.2f}%")
            print(f"  평균 레이턴시: {r['avg_lat_ms']:.3f} ms")
            print(f"  P50 레이턴시:  {r['p50_lat_ms']:.3f} ms")
            print(f"  P99 레이턴시:  {r['p99_lat_ms']:.3f} ms")
    else:
        print("--eval-cifar10 또는 --sweep 옵션을 지정하세요.")


if __name__ == '__main__':
    main()
