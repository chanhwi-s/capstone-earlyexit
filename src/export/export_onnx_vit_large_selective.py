"""
export_onnx_vit_large_selective.py — ViT-L/16 2-exit ONNX export

생성 파일:
  {EXP_DIR}/onnx/ee_vit_large_2exit/seg1.onnx       (static, bs=BASELINE_BATCH_SIZE)
  {EXP_DIR}/onnx/plain_vit_large/plain_vit_large.onnx (static, bs=BASELINE_BATCH_SIZE)

  ※ seg2 static export는 별도로 실행:
     python export/export_onnx_seg2_static.py --model-variant large --batch-sizes 1 2 4 8 16 32 64

사용법 (5090에서):
  cd src
  python export/export_onnx_vit_large_selective.py
  python export/export_onnx_vit_large_selective.py --baseline-batch-size 16
  python export/export_onnx_vit_large_selective.py \\
      --ckpt /path/to/ee_vit_large_2exit/checkpoints/best.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import timm

os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/cap10/.cache/huggingface/hub')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.ee_vit_large_selective import build_model_large, SelectiveExitViTLarge


# ── 세그먼트 래퍼 (ViT-L 전용) ────────────────────────────────────────────────

class ViTLargeSeg1(nn.Module):
    """patch_embed + pos_embed + blocks[0:B1] + exit_heads[0] → (feat, ee_logits)"""

    def __init__(self, model: SelectiveExitViTLarge):
        super().__init__()
        B1 = model.exit_blocks[0]
        self.patch_embed = model.patch_embed
        self.cls_token   = model.cls_token
        self.pos_embed   = model.pos_embed
        self.pos_drop    = model.pos_drop
        self.blocks      = nn.Sequential(*list(model.blocks)[:B1])
        self.exit_head   = model.exit_heads[0]

    def forward(self, image: torch.Tensor):
        B = image.shape[0]
        x = self.patch_embed(image)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        return x, self.exit_head(x)


class PlainViTLargeWrapper(nn.Module):
    """timm pretrained ViT-L/16 — ONNX export 전용 래퍼"""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=True)

    def forward(self, image: torch.Tensor):
        return self.model(image)


# ── ONNX 검증 ─────────────────────────────────────────────────────────────────

def _verify(onnx_path: str):
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"        ✓ onnx.checker passed")
    except ImportError:
        pass
    except Exception as e:
        print(f"        ✗ onnx.checker failed: {e}")


# ── Export 함수 ───────────────────────────────────────────────────────────────

def export_seg1(model: SelectiveExitViTLarge, device: torch.device,
                out_dir: str, bsz: int):
    print(f"\n[ViT-L seg1] ONNX export (static batch={bsz}, exit_block={model.exit_blocks[0]}) ...")
    seg1 = ViTLargeSeg1(model).to(device).eval()
    dummy = torch.randn(bsz, 3, 224, 224, device=device)
    path  = os.path.join(out_dir, "seg1.onnx")

    with torch.no_grad():
        torch.onnx.export(
            seg1, dummy, path,
            input_names=["image"],
            output_names=["feat_out", "ee_logits"],
            opset_version=17,
            verbose=False,
        )
    print(f"  saved → {path}  (hidden={model.HIDDEN_DIM})")
    _verify(path)


def export_plain_large(device: torch.device, out_dir: str, bsz: int):
    print(f"\n[PlainViT-L] ONNX export (static batch={bsz}) ...")
    wrapper = PlainViTLargeWrapper().to(device).eval()
    dummy   = torch.randn(bsz, 3, 224, 224, device=device)
    path    = os.path.join(out_dir, "plain_vit_large.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            verbose=False,
        )
    print(f"  saved → {path}  (batch={bsz})")
    _verify(path)


# ── main ─────────────────────────────────────────────────────────────────────

def _find_ckpt(model_name: str) -> str | None:
    ckpt = paths.latest_checkpoint(model_name, "best.pth")
    if ckpt:
        return ckpt
    for exp_dir in reversed(paths.list_experiments()):
        p = os.path.join(exp_dir, "train", model_name, "checkpoints", "best.pth")
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(
        description="ViT-L/16 2-exit: seg1 + plain_vit_large ONNX export"
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="ee_vit_large_2exit 체크포인트 (미지정 시 최신 exp 자동 탐색)")
    parser.add_argument("--exit-blocks", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--baseline-batch-size", type=int, default=8,
                        help="seg1/plain 고정 batch 크기 (기본: 8)")
    parser.add_argument("--skip-plain", action="store_true",
                        help="plain_vit_large ONNX export 스킵")
    parser.add_argument("--out-exp", type=str, default=None,
                        help="출력 exp 디렉토리 (기본: 최신 exp 자동 선택)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bsz    = args.baseline_batch_size
    print(f"Device        : {device}")
    print(f"Exit blocks   : {args.exit_blocks}")
    print(f"Baseline batch: {bsz}\n")

    def _onnx_dir(model_name: str) -> str:
        if args.out_exp:
            d = os.path.join(args.out_exp, "onnx", model_name)
        else:
            d = paths.onnx_dir(model_name)
        os.makedirs(d, exist_ok=True)
        return d

    # ── seg1 export ──
    ckpt = args.ckpt or _find_ckpt("ee_vit_large_2exit")
    if not ckpt or not os.path.exists(ckpt):
        print("[ERROR] ee_vit_large_2exit 체크포인트 없음. --ckpt 로 지정하세요.")
        return
    print(f"체크포인트: {ckpt}")

    model = build_model_large(exit_blocks=args.exit_blocks, num_classes=1000)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    export_seg1(model, device, _onnx_dir("ee_vit_large_2exit"), bsz)
    del model

    # ── plain ViT-L export ──
    if not args.skip_plain:
        export_plain_large(device, _onnx_dir("plain_vit_large"), bsz)

    print(f"\nONNX export 완료  →  {paths.EXPERIMENTS_DIR}/onnx/")
    print("다음 단계: python export/export_onnx_seg2_static.py --model-variant large --batch-sizes 1 2 4 8 16 32 64")


if __name__ == "__main__":
    main()
