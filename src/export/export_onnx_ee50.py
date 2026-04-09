"""
ONNX Export for EE ResNet-50 (4 exits → 4 segments)

TRT 추론을 위해 네트워크를 4개 세그먼트로 분리 export.
학습 후 train_log의 exit별 accuracy를 확인하고,
실제 TRT sweep에 사용할 exit 조합을 결정한 뒤 이 스크립트를 실행.

세그먼트 구조:
  Segment 1: stem + layer1 + exit1
    input : image          (B, 3, H, W)
    output: feat_layer1    (B, 256, H/4, W/4)
            ee1_logits     (B, num_classes)

  Segment 2: layer2 + exit2
    input : feat_layer1    (B, 256, H/4,  W/4 )
    output: feat_layer2    (B, 512, H/8,  W/8 )
            ee2_logits     (B, num_classes)

  Segment 3: layer3 + exit3
    input : feat_layer2    (B, 512,  H/8,  W/8 )
    output: feat_layer3    (B, 1024, H/16, W/16)
            ee3_logits     (B, num_classes)

  Segment 4: layer4 + main_fc
    input : feat_layer3    (B, 1024, H/16, W/16)
    output: main_logits    (B, num_classes)

사용법:
  cd src
  python export/export_onnx_ee50.py --dataset imagenet
  python export/export_onnx_ee50.py --ckpt experiments/.../best.pth --dataset imagenet
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ee_resnet50 import build_model
from utils import load_config
import paths


# ── 세그먼트 래퍼 모듈 ────────────────────────────────────────────────────────

class Segment1(nn.Module):
    """stem + layer1 + exit1 → (feat_layer1, ee1_logits)"""
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
        return x, ee1          # feat_layer1, logits


class Segment2(nn.Module):
    """layer2 + exit2 → (feat_layer2, ee2_logits)"""
    def __init__(self, backbone):
        super().__init__()
        self.layer2 = backbone.layer2
        self.exit2  = backbone.exit2

    def forward(self, x):
        x = self.layer2(x)
        ee2 = self.exit2(x)
        return x, ee2          # feat_layer2, logits


class Segment3(nn.Module):
    """layer3 + exit3 → (feat_layer3, ee3_logits)"""
    def __init__(self, backbone):
        super().__init__()
        self.layer3 = backbone.layer3
        self.exit3  = backbone.exit3

    def forward(self, x):
        x = self.layer3(x)
        ee3 = self.exit3(x)
        return x, ee3          # feat_layer3, logits


class Segment4(nn.Module):
    """layer4 + main_fc → main_logits"""
    def __init__(self, backbone):
        super().__init__()
        self.layer4  = backbone.layer4
        self.main_fc = backbone.main_fc

    def forward(self, x):
        x = self.layer4(x)
        return self.main_fc(x)


# ── 검증 헬퍼 ────────────────────────────────────────────────────────────────

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


# ── export 함수 ───────────────────────────────────────────────────────────────

def export_segments(model, dummy_input, out_dir):
    seg1 = Segment1(model).eval()
    seg2 = Segment2(model).eval()
    seg3 = Segment3(model).eval()
    seg4 = Segment4(model).eval()

    with torch.no_grad():
        feat1, _ = seg1(dummy_input)
        feat2, _ = seg2(feat1)
        feat3, _ = seg3(feat2)

    # ── Segment 1 ──
    path1 = os.path.join(out_dir, "ee50_seg1_stem_layer1.onnx")
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
    path2 = os.path.join(out_dir, "ee50_seg2_layer2.onnx")
    torch.onnx.export(
        seg2, feat1.detach(), path2,
        input_names=["feat_layer1"],
        output_names=["feat_layer2", "ee2_logits"],
        dynamic_axes={
            "feat_layer1": {0: "batch_size"},
            "feat_layer2": {0: "batch_size"},
            "ee2_logits":  {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg2]  saved → {path2}")
    _verify(path2, 2)

    # ── Segment 3 ──
    path3 = os.path.join(out_dir, "ee50_seg3_layer3.onnx")
    torch.onnx.export(
        seg3, feat2.detach(), path3,
        input_names=["feat_layer2"],
        output_names=["feat_layer3", "ee3_logits"],
        dynamic_axes={
            "feat_layer2": {0: "batch_size"},
            "feat_layer3": {0: "batch_size"},
            "ee3_logits":  {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg3]  saved → {path3}")
    _verify(path3, 2)

    # ── Segment 4 ──
    path4 = os.path.join(out_dir, "ee50_seg4_layer4.onnx")
    torch.onnx.export(
        seg4, feat3.detach(), path4,
        input_names=["feat_layer3"],
        output_names=["main_logits"],
        dynamic_axes={
            "feat_layer3": {0: "batch_size"},
            "main_logits": {0: "batch_size"},
        },
        opset_version=17, verbose=False,
    )
    print(f"[seg4]  saved → {path4}")
    _verify(path4, 1)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="체크포인트 경로. 미지정 시 자동 선택")
    parser.add_argument("--dataset",    type=str, default=None,
                        choices=["cifar10", "imagenet"])
    parser.add_argument("--input-size", type=int, nargs=2, default=None,
                        metavar=("H", "W"))
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint("ee_resnet50")
        if args.ckpt is None:
            print("[ERROR] ee_resnet50 체크포인트 없음. --ckpt 직접 지정하세요.")
            sys.exit(1)
        print(f"자동 선택 체크포인트: {args.ckpt}")

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 파일 없음: {args.ckpt}")
        sys.exit(1)

    cfg          = load_config("configs/train.yaml")
    dataset_name = (args.dataset or cfg["dataset"]["name"]).lower()
    num_classes  = 1000 if dataset_name == "imagenet" else 10
    if args.input_size is not None:
        H, W = args.input_size
    elif dataset_name == "imagenet":
        H, W = 224, 224
    else:
        H, W = 32, 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    print(f"모델 로드 완료: {args.ckpt}  (dataset={dataset_name}, num_classes={num_classes}, device={device})")

    dummy_input = torch.randn(1, 3, H, W, device=device)
    print(f"더미 입력 크기: (1, 3, {H}, {W})\n")

    out_dir = paths.onnx_dir("ee_resnet50")

    with torch.no_grad():
        export_segments(model, dummy_input, out_dir)

    print(f"\n모든 ONNX 파일 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
