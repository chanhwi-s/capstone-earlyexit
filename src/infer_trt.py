"""
TensorRT Segmented Model Inference

Jetson AGX Orin에서 3개의 TRT engine을 동적으로 조율하여 early exit 추론을 수행.

구조:
  Seg1 (stem+layer1+layer2) → feat_layer2, ee1_logits
  Seg2 (layer3) → feat_layer3, ee2_logits (Seg1 출력 기반)
  Seg3 (layer4) → main_logits (Seg2 출력 기반)

사용법:
  python infer_trt.py --seg1 seg1.engine --seg2 seg2.engine --seg3 seg3.engine \\
                      --image test_image.jpg --threshold 0.8

또는 CIFAR-10 데이터셋으로 테스트:
  python infer_trt.py --seg1 seg1.engine --seg2 seg2.engine --seg3 seg3.engine \\
                      --eval-cifar10
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

sys.path.insert(0, os.path.dirname(__file__))


# ── TensorRT 헬퍼 ───────────────────────────────────────────────────────────

class TRTEngine:
    """TensorRT engine 로더 및 실행"""
    def __init__(self, engine_path, logger=None):
        if logger is None:
            logger = trt.Logger(trt.Logger.WARNING)
        self.logger = logger
        self.engine_path = engine_path

        # 엔진 로드
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.input_names = [self.engine.get_binding_name(0)]
        self.output_names = [self.engine.get_binding_name(i) for i in range(1, self.engine.num_bindings)]

        print(f"[TRT] 로드 완료: {engine_path}")
        print(f"      입력: {self.input_names}")
        print(f"      출력: {self.output_names}\n")

    def infer(self, inputs):
        """
        Args:
            inputs: dict or np.ndarray
                단일 입력: np.ndarray (C, H, W) or (B, C, H, W)
                다중 입력: dict {name: np.ndarray}

        Returns:
            outputs: dict {name: np.ndarray}
        """
        # 입력을 numpy array로 변환
        if isinstance(inputs, np.ndarray):
            input_data = inputs
        elif isinstance(inputs, dict):
            input_data = inputs[self.input_names[0]]
        else:
            raise TypeError("입력은 np.ndarray 또는 dict여야 함")

        # 배치 차원 확인
        if input_data.ndim == 3:  # (C, H, W)
            input_data = np.expand_dims(input_data, 0)  # (1, C, H, W)

        batch_size = input_data.shape[0]
        input_data = np.ascontiguousarray(input_data).astype(np.float32)

        # GPU 메모리 할당
        bindings = []
        for i in range(self.engine.num_bindings):
            if i < 1:  # 입력
                gpu_mem = cuda.mem_alloc(input_data.nbytes)
                cuda.memcpy_htod(gpu_mem, input_data)
                bindings.append(int(gpu_mem))
            else:  # 출력
                shape = self.context.get_binding_shape(i)
                shape[0] = batch_size  # 배치 크기 적용
                nbytes = np.prod(shape) * 4  # float32 = 4 bytes
                gpu_mem = cuda.mem_alloc(nbytes)
                bindings.append(int(gpu_mem))

        # 추론 실행
        self.context.set_binding_shape(0, list(input_data.shape))
        self.context.execute_v2(bindings)

        # 출력 회수
        outputs = {}
        for i, name in enumerate(self.output_names):
            shape = self.context.get_binding_shape(i + 1)
            output_gpu = bindings[i + 1]
            output_cpu = cuda.pagelocked_empty(shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_cpu, output_gpu)
            outputs[name] = output_cpu
            cuda.mem_free(output_gpu)

        cuda.mem_free(bindings[0])

        return outputs


# ── 데이터 전처리 ───────────────────────────────────────────────────────────

def load_image(image_path, size=32):
    """이미지 로드 및 전처리"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    img_np = np.array(img).astype(np.float32) / 255.0  # 정규화

    # CIFAR-10 평균/표준편차 정규화
    cifar10_mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    cifar10_std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    img_np = (img_np - cifar10_mean) / cifar10_std

    # (H, W, C) → (C, H, W)
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np


# ── 동적 Early Exit 추론 ─────────────────────────────────────────────────────

class DynamicEarlyExitInference:
    def __init__(self, seg1_engine, seg2_engine, seg3_engine):
        self.seg1 = seg1_engine
        self.seg2 = seg2_engine
        self.seg3 = seg3_engine

    def infer_with_threshold(self, image_input, threshold=0.8):
        """
        threshold 기반 동적 early exit 추론

        Args:
            image_input: np.ndarray (C, H, W) 또는 (B, C, H, W)
            threshold: confidence threshold

        Returns:
            dict {
                'logits': 최종 출력 logits,
                'exit_idx': 종료한 exit 위치 (1, 2, 3),
                'confidence': 해당 exit의 최대 confidence,
                'latency': 실행 시간
            }
        """
        import time
        t0 = time.time()

        # ── Segment 1 실행 ──
        outputs1 = self.seg1.infer(image_input)
        feat_layer2 = outputs1['feat_layer2']
        ee1_logits = outputs1['ee1_logits']

        # ee1 confidence 확인
        conf_ee1 = torch.from_numpy(ee1_logits)
        conf_ee1_max = F.softmax(conf_ee1, dim=1).max(dim=1).values.item()

        if conf_ee1_max >= threshold:
            t_elapsed = time.time() - t0
            return {
                'logits': ee1_logits[0],
                'exit_idx': 1,
                'confidence': conf_ee1_max,
                'latency_ms': t_elapsed * 1000,
            }

        # ── Segment 2 실행 ──
        outputs2 = self.seg2.infer(feat_layer2)
        feat_layer3 = outputs2['feat_layer3']
        ee2_logits = outputs2['ee2_logits']

        # ee2 confidence 확인
        conf_ee2 = torch.from_numpy(ee2_logits)
        conf_ee2_max = F.softmax(conf_ee2, dim=1).max(dim=1).values.item()

        if conf_ee2_max >= threshold:
            t_elapsed = time.time() - t0
            return {
                'logits': ee2_logits[0],
                'exit_idx': 2,
                'confidence': conf_ee2_max,
                'latency_ms': t_elapsed * 1000,
            }

        # ── Segment 3 실행 ──
        outputs3 = self.seg3.infer(feat_layer3)
        main_logits = outputs3['main_logits']

        t_elapsed = time.time() - t0
        conf_main = torch.from_numpy(main_logits)
        conf_main_max = F.softmax(conf_main, dim=1).max(dim=1).values.item()

        return {
            'logits': main_logits[0],
            'exit_idx': 3,
            'confidence': conf_main_max,
            'latency_ms': t_elapsed * 1000,
        }


# ── CIFAR-10 평가 ────────────────────────────────────────────────────────────

def eval_cifar10(infer_engine, threshold, num_samples=100):
    """
    CIFAR-10 테스트셋으로 성능 평가

    Returns:
        dict {
            'accuracy': 정확도,
            'exit_rate': [ee1_%, ee2_%, main_%],
            'avg_latency': 평균 레이턴시
        }
    """
    from datasets.dataloader import get_dataloader

    _, test_loader, _ = get_dataloader(
        dataset="cifar10",
        batch_size=1,
        data_root="../data",
        num_workers=0,
        seed=42
    )

    correct = 0
    exit_counts = [0, 0, 0]
    latencies = []

    for i, (images, labels) in enumerate(test_loader):
        if i >= num_samples:
            break

        image_np = images[0].cpu().numpy()  # (C, H, W)
        label = labels[0].item()

        result = infer_engine.infer_with_threshold(image_np, threshold=threshold)
        pred = np.argmax(result['logits'])

        if pred == label:
            correct += 1

        exit_counts[result['exit_idx'] - 1] += 1
        latencies.append(result['latency_ms'])

    accuracy = correct / num_samples
    exit_rate = [count / num_samples * 100 for count in exit_counts]
    avg_latency = np.mean(latencies)

    return {
        'accuracy': accuracy,
        'exit_rate': exit_rate,
        'avg_latency_ms': avg_latency,
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg1", type=str, required=True, help="seg1.engine 경로")
    parser.add_argument("--seg2", type=str, required=True, help="seg2.engine 경로")
    parser.add_argument("--seg3", type=str, required=True, help="seg3.engine 경로")
    parser.add_argument("--threshold", type=float, default=0.8, help="confidence threshold")
    parser.add_argument("--image", type=str, default=None, help="테스트 이미지 경로")
    parser.add_argument("--eval-cifar10", action="store_true", help="CIFAR-10 평가 모드")
    parser.add_argument("--num-samples", type=int, default=100, help="평가할 샘플 수")
    args = parser.parse_args()

    # ── TRT 엔진 로드 ──
    logger = trt.Logger(trt.Logger.WARNING)
    seg1 = TRTEngine(args.seg1, logger)
    seg2 = TRTEngine(args.seg2, logger)
    seg3 = TRTEngine(args.seg3, logger)

    infer = DynamicEarlyExitInference(seg1, seg2, seg3)

    # ── 단일 이미지 테스트 ──
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] 이미지 없음: {args.image}")
            sys.exit(1)

        print(f"이미지 로드: {args.image}")
        image_np = load_image(args.image)

        print(f"\n추론 실행 (threshold={args.threshold})...")
        result = infer.infer_with_threshold(image_np, threshold=args.threshold)

        print(f"  Exit 위치: {result['exit_idx']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Latency: {result['latency_ms']:.2f} ms")
        print(f"  Logits: {result['logits']}")
        print(f"  Prediction: {np.argmax(result['logits'])}")

    # ── CIFAR-10 평가 ──
    elif args.eval_cifar10:
        print(f"\nCIFAR-10 평가 (threshold={args.threshold}, samples={args.num_samples})...")
        eval_result = eval_cifar10(infer, args.threshold, args.num_samples)

        print(f"  정확도: {eval_result['accuracy']:.4f}")
        print(f"  Exit Rate: EE1={eval_result['exit_rate'][0]:.2f}%, "
              f"EE2={eval_result['exit_rate'][1]:.2f}%, "
              f"Main={eval_result['exit_rate'][2]:.2f}%")
        print(f"  평균 레이턴시: {eval_result['avg_latency_ms']:.2f} ms")

    else:
        print("[INFO] --image 또는 --eval-cifar10 옵션을 지정하세요")


if __name__ == "__main__":
    main()
