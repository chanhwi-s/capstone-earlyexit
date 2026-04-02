"""
ONNX Export Script for ResNet_EE (Early Exit ResNet-18)

두 가지 모드로 export:
  1. full  : 전체 모델을 하나의 ONNX로 export (3개 출력: ee1, ee2, main)
  2. seg   : 3개 세그먼트로 분리 export (TRT 기반 동적 추론용)

사용법:
  python export_onnx.py --mode full --ckpt experiments/.../checkpoints/best.pth
  python export_onnx.py --mode seg  --ckpt experiments/.../checkpoints/best.pth
  python export_onnx.py --mode both --ckpt experiments/.../checkpoints/best.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ee_resnet18 import build_model, BasicBlock, ExitHead
from utils import load_config
import paths


# ── 세그먼트 래퍼 모듈 ────────────────────────────────────────────────────────

class Segment1(nn.Module):
    """
    stem + layer1 + layer2 + exit1
    input : (B, 3, H, W)
    output: feat (B, 128, H/8, W/8),  ee1_logits (B, num_classes)
    """
    def __init__(self, backbone):
        super().__init__()
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.exit1   = backbone.exit1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        ee1 = self.exit1(x)
        return x, ee1          # feat, logits


class Segment2(nn.Module):
    """
    layer3 + exit2
    input : feat from Segment1 (B, 128, H/8, W/8)
    output: feat (B, 256, H/16, W/16),  ee2_logits (B, num_classes)
    """
    def __init__(self, backbone):
        super().__init__()
        self.layer3 = backbone.layer3
        self.exit2  = backbone.exit2

    def forward(self, x):
        x = self.layer3(x)
        ee2 = self.exit2(x)
        return x, ee2          # feat, logits


class Segment3(nn.Module):
    """
    layer4 + main_fc
    input : feat from Segment2 (B, 256, H/16, W/16)
    output: main_logits (B, num_classes)
    """
    def __init__(self, backbone):
        super().__init__()
        self.layer4  = backbone.layer4
        self.main_fc = backbone.main_fc

    def forward(self, x):
        x = self.layer4(x)
        return self.main_fc(x)


# ── Full model 래퍼 (ONNX는 list 출력 불가 → tuple로 변환) ──────────────────

class FullModelWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone

    def forward(self, x):
        outputs = self.model(x, threshold=None)   # list [ee1, ee2, main]
        return tuple(outputs)                      # tuple로 변환


# ── export 함수 ───────────────────────────────────────────────────────────────

def export_full(model, dummy_input, out_dir, num_classes):
    wrapper = FullModelWrapper(model)
    wrapper.eval()

    path = os.path.join(out_dir, "ee_resnet18_full.onnx")
    torch.onnx.export(
        wrapper,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["ee1_logits", "ee2_logits", "main_logits"],
        dynamic_axes={
            "input":       {0: "batch_size"},
            "ee1_logits":  {0: "batch_size"},
            "ee2_logits":  {0: "batch_size"},
            "main_logits": {0: "batch_size"},
        },
        opset_version=17,
        verbose=False,
    )
    print(f"[full]  saved → {path}")
    _verify(path, dummy_input, expected_outputs=3)


def export_seg(model, dummy_input, out_dir):
    seg1 = Segment1(model).eval()
    seg2 = Segment2(model).eval()
    seg3 = Segment3(model).eval()

    # seg1: 입력은 원본 이미지
    feat1, _ = seg1(dummy_input)
    feat2, _ = seg2(feat1)

    # ── Segment 1 ──
    path1 = os.path.join(out_dir, "seg1_stem_layer2.onnx")
    torch.onnx.export(
        seg1, dummy_input, path1,
        input_names=["image"],
        output_names=["feat_layer2", "ee1_logits"],
        dynamic_axes={
            "image":       {0: "batch_size"},
            "feat_layer2": {0: "batch_size"},
            "ee1_logits":  {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg1]  saved → {path1}")
    _verify(path1, dummy_input, expected_outputs=2)

    # ── Segment 2 ──
    path2 = os.path.join(out_dir, "seg2_layer3.onnx")
    torch.onnx.export(
        seg2, feat1.detach(), path2,
        input_names=["feat_layer2"],
        output_names=["feat_layer3", "ee2_logits"],
        dynamic_axes={
            "feat_layer2": {0: "batch_size"},
            "feat_layer3": {0: "batch_size"},
            "ee2_logits":  {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg2]  saved → {path2}")
    _verify(path2, feat1.detach(), expected_outputs=2)

    # ── Segment 3 ──
    path3 = os.path.join(out_dir, "seg3_layer4.onnx")
    torch.onnx.export(
        seg3, feat2.detach(), path3,
        input_names=["feat_layer3"],
        output_names=["main_logits"],
        dynamic_axes={
            "feat_layer3": {0: "batch_size"},
            "main_logits": {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg3]  saved → {path3}")
    _verify(path3, feat2.detach(), expected_outputs=1)


def _verify(onnx_path, dummy_input, expected_outputs):
    """onnx 파일 로드 및 shape 출력"""
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"        ✓ onnx.checker passed  (outputs: {expected_outputs})")
    except ImportError:
        print("        onnx 패키지 없음, 검증 스킵 (pip install onnx)")
    except Exception as e:
        print(f"        ✗ onnx.checker failed: {e}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="체크포인트 경로 (.pth). 미지정 시 가장 최근 실험의 best.pth 자동 선택")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["full", "seg", "both"],
                        help="export 모드 (default: both)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "imagenet"],
                        help="데이터셋 (cifar10→10 classes / imagenet→1000 classes). "
                             "미지정 시 configs/train.yaml 참조")
    parser.add_argument("--input-size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="입력 이미지 크기 (기본: cifar10→32 32, imagenet→224 224)")
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint("ee_resnet18")
        if args.ckpt is None:
            print("[ERROR] ee_resnet18 체크포인트 없음. --ckpt 직접 지정하세요.")
            sys.exit(1)
        print(f"자동 선택 체크포인트: {args.ckpt}")

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 파일 없음: {args.ckpt}")
        sys.exit(1)

    # ── dataset / num_classes / input_size 결정 ──
    cfg          = load_config("configs/train.yaml")
    dataset_name = (args.dataset or cfg["dataset"]["name"]).lower()
    num_classes  = 1000 if dataset_name == "imagenet" else 10
    if args.input_size is not None:
        H, W = args.input_size
    elif dataset_name == "imagenet":
        H, W = 224, 224
    else:
        H, W = 32, 32

    # ── 모델 로드 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    print(f"모델 로드 완료: {args.ckpt}  (dataset={dataset_name}, num_classes={num_classes}, device={device})")

    # ── 더미 입력 ──
    dummy_input = torch.randn(1, 3, H, W, device=device)
    print(f"더미 입력 크기: (1, 3, {H}, {W})\n")

    # ── 출력 디렉토리 ──
    out_dir = paths.onnx_dir("ee_resnet18")

    # ── Export ──
    with torch.no_grad():
        if args.mode in ("full", "both"):
            export_full(model, dummy_input, out_dir, num_classes)
        if args.mode in ("seg", "both"):
            export_seg(model, dummy_input, out_dir)

    print(f"\n모든 ONNX 파일 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
