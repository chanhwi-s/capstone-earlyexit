"""
Plain ResNet-50 ONNX Export

사용법:
  cd src
  python export/export_onnx_plain50.py                        # CIFAR-10 (기본값)
  python export/export_onnx_plain50.py --dataset imagenet     # ImageNet
  python export/export_onnx_plain50.py --ckpt experiments/.../best.pth --dataset imagenet
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.plain_resnet50 import build_model
from utils import load_config
import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       type=str, default=None)
    parser.add_argument("--dataset",    type=str, default=None,
                        choices=["cifar10", "imagenet"])
    parser.add_argument("--input-size", type=int, nargs=2, default=None,
                        metavar=("H", "W"))
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint("plain_resnet50")
        if args.ckpt is None:
            print("[ERROR] plain_resnet50 체크포인트 없음. --ckpt 직접 지정하세요.")
            sys.exit(1)
        print(f"자동 선택: {args.ckpt}")

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
    print(f"모델 로드: {args.ckpt}  (dataset={dataset_name}, num_classes={num_classes})")

    dummy_input = torch.randn(1, 3, H, W, device=device)
    print(f"더미 입력: (1, 3, {H}, {W})\n")

    out_dir   = paths.onnx_dir("plain_resnet50")
    save_path = os.path.join(out_dir, "plain_resnet50.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
            verbose=False,
        )

    print(f"ONNX 저장: {save_path}")

    try:
        import onnx
        m = onnx.load(save_path)
        onnx.checker.check_model(m)
        print("✓ onnx.checker passed")
    except ImportError:
        print("onnx 패키지 없음, 검증 스킵")
    except Exception as e:
        print(f"✗ onnx.checker failed: {e}")


if __name__ == "__main__":
    main()
