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
import sys
import re
from collections import Counter

import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

ENGINES = {
    "Plain ResNet-18"              : paths.engine_path("plain_resnet18", "plain_resnet18.engine"),
    "EE Seg1 (stem+layer1+2+EE1)" : paths.engine_path("ee_resnet18",   "seg1.engine"),
    "EE Seg2 (layer3+EE2)"        : paths.engine_path("ee_resnet18",   "seg2.engine"),
    "EE Seg3 (layer4+FC)"         : paths.engine_path("ee_resnet18",   "seg3.engine"),
    "VEE Seg1 (stem+layer1+EE1)"  : paths.engine_path("vee_resnet18",  "vee_seg1.engine"),
    "VEE Seg2 (layer2+3+4+FC)"    : paths.engine_path("vee_resnet18",  "vee_seg2.engine"),
}

# ResNet-18 ONNX 원본 레이어 수 (참고용)
ONNX_LAYER_COUNTS = {
    "Plain ResNet-18"              : 60,   # Conv×20, BN×20, ReLU×17, Add×8, Pool×2, FC×1 등
    "EE Seg1 (stem+layer1+2+EE1)" : 32,
    "EE Seg2 (layer3+EE2)"        : 18,
    "EE Seg3 (layer4+FC)"         : 16,
    "VEE Seg1 (stem+layer1+EE1)"  : 20,   # stem(5) + layer1(2×BasicBlock=12) + exit1(3)
    "VEE Seg2 (layer2+3+4+FC)"    : 42,   # layer2+3+4(3×2×BasicBlock≈36) + main_fc(3) + pool(3)
}


def classify_layer_by_type(layer_type: str, name: str) -> str:
    """TRT LayerType 문자열로 분류 (IEngineInspector 용)."""
    t = layer_type.lower()
    n = name.lower()
    # TRT 10.x LayerType 값들
    if 'myelin'       in t or 'myelin'      in n: return 'Myelin (fused kernel)'
    if 'pointwise'    in t:                        return 'PointWise (fused)'
    if 'caskconv'     in t or 'caskconv'    in n: return 'CaskConv (fused)'
    if 'convolution'  in t:                        return 'Convolution'
    if 'activation'   in t:                        return 'Activation (ReLU 등)'
    if 'elementwise'  in t:                        return 'ElementWise (Add/Residual)'
    if 'pooling'      in t:                        return 'Pooling'
    if 'fullyconnect' in t or 'gemm' in t:         return 'FC / GEMM'
    if 'softmax'      in t:                        return 'Softmax'
    if 'shuffle'      in t or 'reshape' in t:      return 'Reshape/Shuffle'
    if 'reformat'     in t or 'reformat' in n:     return 'Reformat'
    if 'scale'        in t:                        return 'Scale (BN)'
    if 'concat'       in t:                        return 'Concat'
    if layer_type:                                  return layer_type  # 원본 그대로
    # layer_type 없으면 이름으로 fallback
    if 'myelin'    in n: return 'Myelin (fused kernel)'
    if 'reformat'  in n: return 'Reformat'
    if 'conv'      in n: return 'Conv (name 기반)'
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
    # ── IEngineInspector (TRT 10.x API) ───────────────────────
    inspector = engine.create_engine_inspector()
    context   = engine.create_execution_context()
    inspector.execution_context = context

    print(f"\n  ▶ 레이어 목록  (IEngineInspector, TRT 10.x)")
    print(f"  {'#':>4}  {'분류':30s}  이름")
    print(f"  {'-'*W}")

    type_counter = Counter()
    for i in range(num_layers):
        info = inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE)
        # ONELINE 포맷: "Name: xxx Type: yyy ..."
        name_m = re.search(r'Name:\s*([^,\n]+)',       info)
        type_m = re.search(r'LayerType:\s*([^,\n]+)',  info)
        layer_name = name_m.group(1).strip() if name_m else info[:60]
        layer_type = type_m.group(1).strip() if type_m else ''

        ltype   = classify_layer_by_type(layer_type, layer_name)
        type_counter[ltype] += 1
        display = layer_name if len(layer_name) <= 52 else layer_name[:49] + '...'
        print(f"  {i:>4}  {ltype:30s}  {display}")

    # ── 타입별 집계 ────────────────────────────────────────────
    print(f"\n  ▶ 레이어 타입 집계")
    print(f"  {'-'*50}")
    for ltype, cnt in type_counter.most_common():
        bar = '█' * min(cnt, 30)
        print(f"  {ltype:30s}  {cnt:3d}  {bar}")

    # ── Fusion 하이라이트 ──────────────────────────────────────
    print(f"\n  ▶ Fusion 분석")
    fused = sum(v for k, v in type_counter.items()
                if any(w in k.lower() for w in ('fused', 'myelin', 'pointwise')))
    if fused:
        print(f"  ✅ Fused/최적화 레이어: {fused}개")
        print(f"     - CaskConvolution / PointWise : Conv+BN+ReLU 등을 단일 커널로 합침")
        print(f"     - Myelin         : NVIDIA 전용 컴파일러로 추가 최적화된 커널")
    else:
        print(f"  ℹ️  레이어 타입 이름으로 fusion 자동 감지 어려움")
        print(f"     → 아래 raw JSON으로 직접 확인하세요")

    # ── Raw JSON 전체 파일 저장 ────────────────────────────────
    out_dir  = paths.engine_inspect_dir()
    safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    json_path  = os.path.abspath(os.path.join(out_dir, f"{safe_label}.json"))

    import json as _json
    layers_json = []
    for i in range(num_layers):
        raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
        try:
            layers_json.append(_json.loads(raw))
        except Exception:
            layers_json.append({"raw": raw})

    with open(json_path, 'w') as f:
        _json.dump({"engine": label, "num_layers": num_layers,
                    "layers": layers_json}, f, indent=2)
    print(f"\n  ▶ Raw JSON 저장됨: {json_path}")

    # ── Raw JSON 터미널 미리보기 (첫 3개 레이어) ──────────────
    print(f"\n  ▶ Raw Inspector JSON 미리보기 (첫 3개 레이어)")
    print(f"  {'-'*W}")
    for i in range(min(3, num_layers)):
        info = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
        preview = info.strip()[:300].replace('\n', ' ')
        print(f"  [{i}] {preview}")

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
