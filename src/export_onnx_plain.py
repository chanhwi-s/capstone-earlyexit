"""
Plain ResNet-18 ONNX Export

사용법:
  python export_onnx_plain.py --ckpt experiments_plain/.../best.pth
  python export_onnx_plain.py  (자동으로 최근 best.pth 선택)
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models.plain_resnet18 import build_model
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="체크포인트 경로. 미지정 시 자동 선택")
    parser.add_argument("--input-size", type=int, nargs=2, default=[32, 32],
                        metavar=("H", "W"),
                        help="입력 크기 (default: 32 32 for CIFAR-10)")
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    if args.ckpt is None:
        base = "experiments_plain"
        if not os.path.exists(base):
            print("[ERROR] experiments_plain/ 없음. 먼저 train_plain.py를 실행하세요.")
            sys.exit(1)
        dirs = sorted([
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ])
        if not dirs:
            print("[ERROR] experiments_plain/ 비어있음")
            sys.exit(1)
        args.ckpt = os.path.join(dirs[-1], "checkpoints", "best.pth")
        print(f"자동 선택: {args.ckpt}")

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 파일 없음: {args.ckpt}")
        sys.exit(1)

    # ── 모델 로드 ──
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

    # ── 출력 디렉토리 ──
    out_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), "onnx")
    os.makedirs(out_dir, exist_ok=True)

    # ── Export ──
    save_path = os.path.join(out_dir, "plain_resnet18.onnx")
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

    # ── 검증 ──
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
