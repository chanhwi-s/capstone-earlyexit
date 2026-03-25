"""
TRT 엔진 Layer Fusion 분석 스크립트

사용법:
  cd ~/capstone-earlyexit/src
  python inspect_engines.py

출력:
  - 각 엔진의 레이어 수 (fusion 전/후 비교)
  - 레이어 타입별 분류 (Conv, Pooling, ElementWise 등)
  - I/O 텐서 정보
  - fusion 요약 (Conv+BN+ReLU 등 감지)
"""

import os
import re
from collections import Counter

import tensorrt as trt

HOME = os.environ['HOME']
ENGINES = {
    "Plain ResNet-18"              : f"{HOME}/capstone-earlyexit/onnx_plain/plain_resnet18.engine",
    "EE Seg1 (stem+layer1+2+EE1)" : f"{HOME}/capstone-earlyexit/onnx/seg1.engine",
    "EE Seg2 (layer3+EE2)"        : f"{HOME}/capstone-earlyexit/onnx/seg2.engine",
    "EE Seg3 (layer4+FC)"         : f"{HOME}/capstone-earlyexit/onnx/seg3.engine",
}

# ResNet-18 ONNX 원본 레이어 수 (참고용)
ONNX_LAYER_COUNTS = {
    "Plain ResNet-18"              : 60,   # Conv×20, BN×20, ReLU×17, Add×8, Pool×2, FC×1 등
    "EE Seg1 (stem+layer1+2+EE1)" : 32,
    "EE Seg2 (layer3+EE2)"        : 18,
    "EE Seg3 (layer4+FC)"         : 16,
}


def classify_layer(name: str) -> str:
    """레이어 이름으로 타입 분류."""
    n = name.lower()
    if 'myelin'    in n: return 'Myelin (GPU kernel fusion)'
    if 'reformat'  in n: return 'Reformat'
    if 'conv'      in n and 'relu' in n and ('bn' in n or 'norm' in n):
        return 'Conv+BN+ReLU (fused)'
    if 'conv'      in n and 'relu' in n: return 'Conv+ReLU (fused)'
    if 'conv'      in n and ('bn' in n or 'norm' in n): return 'Conv+BN (fused)'
    if 'conv'      in n: return 'Conv'
    if 'gemm'      in n or 'matmul' in n or 'linear' in n: return 'FC/GEMM'
    if 'pool'      in n: return 'Pooling'
    if 'add'       in n or 'residual' in n or 'eltwise' in n: return 'ElementWise/Residual'
    if 'relu'      in n or 'activation' in n: return 'Activation'
    if 'softmax'   in n: return 'Softmax'
    if 'flatten'   in n or 'reshape' in n: return 'Reshape/Flatten'
    if 'concat'    in n: return 'Concat'
    return 'Other'


def inspect_engine(label: str, path: str):
    if not os.path.exists(path):
        print(f"\n[SKIP] {label} — 파일 없음: {path}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    W = 65
    print(f"\n{'═'*W}")
    print(f"  {label}")
    print(f"{'═'*W}")

    # ── I/O 텐서 ──────────────────────────────────────────────
    print(f"\n  ▶ I/O Tensors  ({engine.num_io_tensors}개)")
    print(f"  {'방향':6s}  {'이름':35s}  {'shape':20s}  dtype")
    print(f"  {'-'*W}")
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        mode  = engine.get_tensor_mode(name)
        shape = list(engine.get_tensor_shape(name))
        dtype = str(engine.get_tensor_dtype(name)).split('.')[-1]
        tag   = 'INPUT ' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'
        print(f"  {tag}  {name:35s}  {str(shape):20s}  {dtype}")

    # ── 레이어 목록 ────────────────────────────────────────────
    num_layers = engine.num_layers
    onnx_ref   = ONNX_LAYER_COUNTS.get(label, '?')
    reduction  = (f"{onnx_ref} → {num_layers}  "
                  f"({(1 - num_layers/onnx_ref)*100:.0f}% 감소)"
                  if isinstance(onnx_ref, int) else str(num_layers))

    print(f"\n  ▶ 레이어 수: {reduction}  (ONNX 원본 기준)")
    print(f"\n  ▶ 레이어 목록")
    print(f"  {'#':>4}  {'분류':30s}  이름")
    print(f"  {'-'*W}")

    type_counter = Counter()
    for i in range(num_layers):
        layer_name = engine.get_layer(i)
        ltype      = classify_layer(layer_name)
        type_counter[ltype] += 1
        # 이름이 길면 앞 60자만 출력
        display = layer_name if len(layer_name) <= 55 else layer_name[:52] + '...'
        print(f"  {i:>4}  {ltype:30s}  {display}")

    # ── 타입별 집계 ────────────────────────────────────────────
    print(f"\n  ▶ 레이어 타입 집계")
    print(f"  {'-'*40}")
    for ltype, cnt in type_counter.most_common():
        bar = '█' * cnt
        print(f"  {ltype:30s}  {cnt:3d}  {bar}")

    # ── Fusion 하이라이트 ──────────────────────────────────────
    print(f"\n  ▶ Fusion 분석")
    fused = sum(v for k, v in type_counter.items()
                if 'fused' in k.lower() or 'myelin' in k.lower())
    if fused:
        print(f"  ✅ Fused 레이어: {fused}개")
        print(f"     - Conv+BN+ReLU fusion: TRT가 Conv→BN→ReLU를 단일 CUDA 커널로 합침")
        print(f"     - Myelin: NVIDIA 전용 딥러닝 컴파일러가 최적화한 커널")
    else:
        print(f"  ℹ️  레이어 이름만으로 fusion 타입 자동 감지 불가")
        print(f"     → trtexec --verbose 로 상세 로그 확인 권장")

    print()


def print_summary(results: dict):
    """전체 엔진 비교 요약."""
    print(f"\n{'═'*65}")
    print(f"  전체 엔진 비교 요약")
    print(f"{'═'*65}")
    print(f"  {'모델':35s}  {'ONNX':>6s}  {'TRT':>6s}  {'감소율':>8s}")
    print(f"  {'-'*65}")
    for label, num in results.items():
        ref = ONNX_LAYER_COUNTS.get(label, None)
        if ref and num:
            pct = (1 - num / ref) * 100
            print(f"  {label:35s}  {ref:>6d}  {num:>6d}  {pct:>7.0f}%")
    print()


if __name__ == '__main__':
    print("\n🔍 TRT Engine Layer Fusion 분석")
    print("   (ONNX 원본 레이어 수는 추정값, 실제와 다를 수 있음)\n")

    layer_counts = {}
    for label, path in ENGINES.items():
        inspect_engine(label, path)
        if os.path.exists(path):
            logger = trt.Logger(trt.Logger.WARNING)
            with open(path, "rb") as f:
                eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
            layer_counts[label] = eng.num_layers

    print_summary(layer_counts)
