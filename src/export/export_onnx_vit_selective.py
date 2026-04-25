"""
SelectiveExitViT ONNX 세그먼트 Export (5090 실행)

세그먼트 구조:
  PlainViT:
    in:  image [1, 3, 224, 224]
    out: logits [1, 1000]

  2-exit (exit_blocks=[8, 12]):
    seg1: patch_embed + pos_embed + blocks[0:8]  + exit_heads[0]
          in:  image [1, 3, 224, 224]
          out: feat [1, 197, 768],  ee_logits [1, 1000]
    seg2: blocks[8:12] + exit_heads[1]
          in:  feat [1, 197, 768]
          out: ee_logits [1, 1000]

  3-exit (exit_blocks=[6, 9, 12]):
    seg1: patch_embed + pos_embed + blocks[0:6]  + exit_heads[0]
          in:  image [1, 3, 224, 224]
          out: feat [1, 197, 768],  ee_logits [1, 1000]
    seg2: blocks[6:9] + exit_heads[1]
          in:  feat [1, 197, 768]
          out: feat [1, 197, 768],  ee_logits [1, 1000]   (← 중간 세그먼트)
    seg3: blocks[9:12] + exit_heads[2]
          in:  feat [1, 197, 768]
          out: ee_logits [1, 1000]

사용법 (5090에서):
  cd src
  python export/export_onnx_vit_selective.py --model all
  python export/export_onnx_vit_selective.py --model plain
  python export/export_onnx_vit_selective.py --model 2exit --checkpoint /path/to/best.pth
  python export/export_onnx_vit_selective.py --model 3exit
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.ee_vit_selective import SelectiveExitViT, build_model


# ── 세그먼트 래퍼 ──────────────────────────────────────────────────────────────

class ViTSeg1(nn.Module):
    """patch_embed + pos_embed + blocks[0:B1] + exit_heads[0] → (feat, ee_logits)"""

    def __init__(self, model: SelectiveExitViT):
        super().__init__()
        B1 = model.exit_blocks[0]             # e.g. 8 or 6
        self.patch_embed = model.patch_embed
        self.cls_token   = model.cls_token    # nn.Parameter [1, 1, 768]
        self.pos_embed   = model.pos_embed    # nn.Parameter [1, 197, 768]
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


class ViTSegMid(nn.Module):
    """blocks[B_prev:B_next] + exit_heads[i] → (feat, ee_logits)  (중간 세그먼트)"""

    def __init__(self, model: SelectiveExitViT, seg_idx: int):
        super().__init__()
        B_prev = model.exit_blocks[seg_idx - 1]
        B_next = model.exit_blocks[seg_idx]
        self.blocks    = nn.Sequential(*list(model.blocks)[B_prev:B_next])
        self.exit_head = model.exit_heads[seg_idx]

    def forward(self, feat: torch.Tensor):
        x = self.blocks(feat)
        return x, self.exit_head(x)


class ViTSegLast(nn.Module):
    """blocks[B_prev:] + exit_heads[-1] → ee_logits only  (마지막 세그먼트)"""

    def __init__(self, model: SelectiveExitViT, seg_idx: int):
        super().__init__()
        B_prev = model.exit_blocks[seg_idx - 1]
        self.blocks    = nn.Sequential(*list(model.blocks)[B_prev:])
        self.exit_head = model.exit_heads[seg_idx]

    def forward(self, feat: torch.Tensor):
        x = self.blocks(feat)
        return self.exit_head(x)


class PlainViTWrapper(nn.Module):
    """timm pretrained ViT-B/16 — ONNX export 전용 래퍼"""

    def __init__(self):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)

    def forward(self, image: torch.Tensor):
        return self.model(image)


# ── ONNX 검증 ─────────────────────────────────────────────────────────────────

def _verify(onnx_path: str, n_outputs: int):
    try:
        import onnx
        m = onnx.load(onnx_path)
        onnx.checker.check_model(m)
        print(f"        ✓ onnx.checker passed  (outputs: {n_outputs})")
    except ImportError:
        print("        onnx 패키지 없음, 검증 스킵")
    except Exception as e:
        print(f"        ✗ onnx.checker failed: {e}")


# ── Export: PlainViT ──────────────────────────────────────────────────────────

def export_plain_vit(device: torch.device, out_dir: str):
    print("\n[PlainViT] ONNX export ...")
    wrapper = PlainViTWrapper().to(device).eval()
    dummy   = torch.randn(1, 3, 224, 224, device=device)
    path    = os.path.join(out_dir, "plain_vit.onnx")

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            verbose=False,
        )
    print(f"  saved → {path}")
    _verify(path, 1)


# ── Export: N-exit 세그먼트 ───────────────────────────────────────────────────

def export_selective_segs(model: SelectiveExitViT, device: torch.device,
                          out_dir: str):
    n_exits     = model.NUM_BLOCKS
    exit_blocks = model.exit_blocks
    model_tag   = f"{n_exits}exit ({'+'.join(f'B{b}' for b in exit_blocks)})"

    print(f"\n[{model_tag}] ONNX segment export ...")
    dummy_image = torch.randn(1, 3, 224, 224, device=device)

    # ── seg1 ──
    seg1 = ViTSeg1(model).to(device).eval()
    with torch.no_grad():
        feat_out, _ = seg1(dummy_image)

    path_seg1 = os.path.join(out_dir, "seg1.onnx")
    with torch.no_grad():
        torch.onnx.export(
            seg1, dummy_image, path_seg1,
            input_names=["image"],
            output_names=["feat", "ee_logits"],
            opset_version=17, verbose=False,
        )
    print(f"  seg1 saved → {path_seg1}")
    _verify(path_seg1, 2)

    feat_cur = feat_out.detach()

    # ── 중간 세그먼트 (3-exit 이상에서만 존재) ──
    for seg_idx in range(1, n_exits - 1):
        seg_mid = ViTSegMid(model, seg_idx).to(device).eval()
        with torch.no_grad():
            feat_next, _ = seg_mid(feat_cur)

        path_mid = os.path.join(out_dir, f"seg{seg_idx + 1}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                seg_mid, feat_cur, path_mid,
                input_names=["feat"],
                output_names=["feat", "ee_logits"],
                opset_version=17, verbose=False,
            )
        print(f"  seg{seg_idx + 1} (mid) saved → {path_mid}")
        _verify(path_mid, 2)
        feat_cur = feat_next.detach()

    # ── 마지막 세그먼트 ──
    last_idx = n_exits - 1
    seg_last = ViTSegLast(model, last_idx).to(device).eval()
    path_last = os.path.join(out_dir, f"seg{n_exits}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            seg_last, feat_cur, path_last,
            input_names=["feat"],
            output_names=["ee_logits"],
            opset_version=17, verbose=False,
        )
    print(f"  seg{n_exits} (last) saved → {path_last}")
    _verify(path_last, 1)

    print(f"  → {out_dir}/")


# ── 체크포인트 로드 ───────────────────────────────────────────────────────────

def load_selective_model(exit_blocks: list, ckpt: str,
                         device: torch.device) -> SelectiveExitViT:
    model = build_model(exit_blocks=exit_blocks, num_classes=1000)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SelectiveExitViT + PlainViT ONNX segment export"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["plain", "2exit", "3exit", "all"],
        help="export 대상 (기본: all)"
    )
    parser.add_argument("--ckpt-2exit", type=str, default=None,
                        help="2-exit 체크포인트 경로 (기본: 최신 ee_vit_2exit/best.pth)")
    parser.add_argument("--ckpt-3exit", type=str, default=None,
                        help="3-exit 체크포인트 경로 (기본: 최신 ee_vit_3exit/best.pth)")
    parser.add_argument("--exit-blocks-2", type=int, nargs="+", default=[8, 12],
                        help="2-exit 블록 번호 (기본: 8 12)")
    parser.add_argument("--exit-blocks-3", type=int, nargs="+", default=[6, 9, 12],
                        help="3-exit 블록 번호 (기본: 6 9 12)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Export target: {args.model}\n")

    do_plain = args.model in ("plain", "all")
    do_2exit = args.model in ("2exit", "all")
    do_3exit = args.model in ("3exit", "all")

    # ── PlainViT ──
    if do_plain:
        out_dir = paths.onnx_dir("plain_vit")
        export_plain_vit(device, out_dir)

    # ── 2-exit ──
    if do_2exit:
        ckpt = args.ckpt_2exit or paths.latest_checkpoint("ee_vit_2exit", "best.pth")
        if ckpt is None or not os.path.exists(ckpt):
            print(f"[ERROR] ee_vit_2exit 체크포인트 없음. --ckpt-2exit 로 지정하세요.")
            if not do_3exit:
                return
        else:
            print(f"2-exit 체크포인트: {ckpt}")
            model = load_selective_model(args.exit_blocks_2, ckpt, device)
            out_dir = paths.onnx_dir("ee_vit_2exit")
            export_selective_segs(model, device, out_dir)
            del model

    # ── 3-exit ──
    if do_3exit:
        ckpt = args.ckpt_3exit or paths.latest_checkpoint("ee_vit_3exit", "best.pth")
        if ckpt is None or not os.path.exists(ckpt):
            print(f"[ERROR] ee_vit_3exit 체크포인트 없음. --ckpt-3exit 로 지정하세요.")
            return
        print(f"3-exit 체크포인트: {ckpt}")
        model = load_selective_model(args.exit_blocks_3, ckpt, device)
        out_dir = paths.onnx_dir("ee_vit_3exit")
        export_selective_segs(model, device, out_dir)
        del model

    print("\n============================")
    print("  ONNX export 완료")
    print(f"  저장 위치: {paths.EXPERIMENTS_DIR}/onnx/")
    print("  다음 단계: Orin으로 전송 후 orin_vit_pipeline.sh 실행")
    print("============================")


if __name__ == "__main__":
    main()
