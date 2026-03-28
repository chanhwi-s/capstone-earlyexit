"""
VEE-ResNet-18 ONNX Export (2-segment)

Segment 구조:
  seg1: stem + layer1 + exit1
    입력: image (B, 3, H, W)
    출력: feat_layer1 (B, 64, H', W'),  ee1_logits (B, num_classes)

  seg2: layer2 + layer3 + layer4 + main_fc
    입력: feat_layer1 (B, 64, H', W')
    출력: main_logits (B, num_classes)

사용법:
  python export_onnx_vee.py --mode both
  python export_onnx_vee.py --mode seg  --ckpt experiments/.../best.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from models.vee_resnet18 import build_model
from utils import load_config
import paths


# ── 세그먼트 래퍼 ─────────────────────────────────────────────────────────────

class VEE_Segment1(nn.Module):
    """stem + layer1 + exit1 → (feat, ee1_logits)"""
    def __init__(self, backbone):
        super().__init__()
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.exit1   = backbone.exit1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        ee1 = self.exit1(x)
        return x, ee1


class VEE_Segment2(nn.Module):
    """layer2 + layer3 + layer4 + main_fc → main_logits"""
    def __init__(self, backbone):
        super().__init__()
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4
        self.main_fc = backbone.main_fc

    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.main_fc(x)


# ── Full model 래퍼 ─────────────────────────────────────────────────────────

class VEE_FullWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone

    def forward(self, x):
        outputs = self.model(x, threshold=None)
        return tuple(outputs)


# ── export 함수 ──────────────────────────────────────────────────────────────

def export_full(model, dummy_input, out_dir, num_classes):
    wrapper = VEE_FullWrapper(model).eval()
    path = os.path.join(out_dir, "vee_resnet18_full.onnx")
    torch.onnx.export(
        wrapper, dummy_input, path,
        input_names=["input"],
        output_names=["ee1_logits", "main_logits"],
        dynamic_axes={
            "input":       {0: "batch_size"},
            "ee1_logits":  {0: "batch_size"},
            "main_logits": {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[full]  saved → {path}")
    _verify(path, 2)


def export_seg(model, dummy_input, out_dir):
    seg1 = VEE_Segment1(model).eval()
    seg2 = VEE_Segment2(model).eval()

    feat1, _ = seg1(dummy_input)

    # ── Segment 1 ──
    path1 = os.path.join(out_dir, "vee_seg1_stem_layer1.onnx")
    torch.onnx.export(
        seg1, dummy_input, path1,
        input_names=["image"],
        output_names=["feat_layer1", "ee1_logits"],
        dynamic_axes={
            "image":       {0: "batch_size"},
            "feat_layer1": {0: "batch_size"},
            "ee1_logits":  {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg1]  saved → {path1}")
    _verify(path1, 2)

    # ── Segment 2 ──
    path2 = os.path.join(out_dir, "vee_seg2_layer2to4.onnx")
    torch.onnx.export(
        seg2, feat1.detach(), path2,
        input_names=["feat_layer1"],
        output_names=["main_logits"],
        dynamic_axes={
            "feat_layer1": {0: "batch_size"},
            "main_logits": {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg2]  saved → {path2}")
    _verify(path2, 1)


def _verify(onnx_path, expected_outputs):
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"        ✓ onnx.checker passed  (outputs: {expected_outputs})")
    except ImportError:
        print("        onnx 패키지 없음, 검증 스킵")
    except Exception as e:
        print(f"        ✗ onnx.checker failed: {e}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["full", "seg", "both"])
    parser.add_argument("--input-size", type=int, nargs=2, default=[32, 32],
                        metavar=("H", "W"))
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint("vee_resnet18")
        if args.ckpt is None:
            print("[ERROR] vee_resnet18 체크포인트 없음. --ckpt 직접 지정하세요.")
            sys.exit(1)
        print(f"자동 선택 체크포인트: {args.ckpt}")

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 파일 없음: {args.ckpt}")
        sys.exit(1)

    cfg         = load_config("configs/train.yaml")
    num_classes = 10 if cfg["dataset"]["name"].lower() == "cifar10" else 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    print(f"모델 로드: {args.ckpt}  (num_classes={num_classes})")

    H, W        = args.input_size
    dummy_input = torch.randn(1, 3, H, W, device=device)
    print(f"더미 입력: (1, 3, {H}, {W})\n")

    out_dir = paths.onnx_dir("vee_resnet18")

    with torch.no_grad():
        if args.mode in ("full", "both"):
            export_full(model, dummy_input, out_dir, num_classes)
        if args.mode in ("seg", "both"):
            export_seg(model, dummy_input, out_dir)

    print(f"\nONNX 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
