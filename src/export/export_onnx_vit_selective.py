"""
SelectiveExitViT ONNX 세그먼트 Export (5090 실행)

세그먼트 I/O 명명 규칙 (ONNX 입출력 이름 충돌 방지):
  seg1    : in="image"    → out=["feat_out", "ee_logits"]
  seg_mid : in="feat_in"  → out=["feat_out", "ee_logits"]
  seg_last: in="feat_in"  → out=["ee_logits"]
  plain   : in="image"    → out=["logits"]

  ※ input/output 동일 이름 사용 금지 (ONNX graph 유효성 검사 실패)

사용법 (5090에서):
  cd src
  python export/export_onnx_vit_selective.py --model all
  python export/export_onnx_vit_selective.py --model plain
  python export/export_onnx_vit_selective.py --model 2exit --ckpt-2exit /path/to/best.pth
  python export/export_onnx_vit_selective.py --model 3exit --ckpt-3exit /path/to/best.pth

  # 두 모델이 다른 exp 디렉토리에 있을 때
  python export/export_onnx_vit_selective.py --model all \\
      --ckpt-2exit experiments/exp_20260414_212957/train/ee_vit_2exit/checkpoints/best.pth \\
      --ckpt-3exit experiments/exp_20260414_213050/train/ee_vit_3exit/checkpoints/best.pth
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

    # ── seg1: in="image" → out=["feat_out", "ee_logits"] ──
    seg1 = ViTSeg1(model).to(device).eval()
    with torch.no_grad():
        feat_out, _ = seg1(dummy_image)

    path_seg1 = os.path.join(out_dir, "seg1.onnx")
    with torch.no_grad():
        torch.onnx.export(
            seg1, dummy_image, path_seg1,
            input_names=["image"],
            output_names=["feat_out", "ee_logits"],   # "feat" ≠ input name → 충돌 없음
            opset_version=17, verbose=False,
        )
    print(f"  seg1 saved → {path_seg1}")
    _verify(path_seg1, 2)

    feat_cur = feat_out.detach()

    # ── 중간 세그먼트: in="feat_in" → out=["feat_out", "ee_logits"] (3-exit 이상) ──
    for seg_idx in range(1, n_exits - 1):
        seg_mid = ViTSegMid(model, seg_idx).to(device).eval()
        with torch.no_grad():
            feat_next, _ = seg_mid(feat_cur)

        path_mid = os.path.join(out_dir, f"seg{seg_idx + 1}.onnx")
        with torch.no_grad():
            torch.onnx.export(
                seg_mid, feat_cur, path_mid,
                input_names=["feat_in"],               # 입력: feat_in
                output_names=["feat_out", "ee_logits"],# 출력: feat_out (이름 충돌 없음)
                opset_version=17, verbose=False,
            )
        print(f"  seg{seg_idx + 1} (mid) saved → {path_mid}")
        _verify(path_mid, 2)
        feat_cur = feat_next.detach()

    # ── 마지막 세그먼트: in="feat_in" → out=["ee_logits"] ──
    last_idx = n_exits - 1
    seg_last = ViTSegLast(model, last_idx).to(device).eval()
    path_last = os.path.join(out_dir, f"seg{n_exits}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            seg_last, feat_cur, path_last,
            input_names=["feat_in"],
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


def find_checkpoint_any_exp(model_name: str, filename: str = "best.pth") -> str | None:
    """모든 exp_* 디렉토리를 최신순으로 탐색해 체크포인트 반환."""
    for exp_dir in reversed(paths.list_experiments()):
        ckpt = os.path.join(exp_dir, "train", model_name, "checkpoints", filename)
        if os.path.exists(ckpt):
            return ckpt
    return None


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
                        help="2-exit 체크포인트 경로 (다른 exp에 있을 때 직접 지정)")
    parser.add_argument("--ckpt-3exit", type=str, default=None,
                        help="3-exit 체크포인트 경로 (다른 exp에 있을 때 직접 지정)")
    parser.add_argument("--exit-blocks-2", type=int, nargs="+", default=[8, 12],
                        help="2-exit 블록 번호 (기본: 8 12)")
    parser.add_argument("--exit-blocks-3", type=int, nargs="+", default=[6, 9, 12],
                        help="3-exit 블록 번호 (기본: 6 9 12)")
    parser.add_argument("--out-exp", type=str, default=None,
                        help="ONNX 저장 exp 디렉토리 (기본: 최신 exp_* 자동 선택)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Export target: {args.model}\n")

    # ONNX 출력 exp 디렉토리 — 명시하지 않으면 최신 exp (paths.EXPERIMENTS_DIR)
    if args.out_exp:
        out_exp = os.path.realpath(args.out_exp)
        def _onnx_dir(model_name):
            d = os.path.join(out_exp, "onnx", model_name)
            os.makedirs(d, exist_ok=True)
            return d
    else:
        _onnx_dir = paths.onnx_dir

    do_plain = args.model in ("plain", "all")
    do_2exit = args.model in ("2exit", "all")
    do_3exit = args.model in ("3exit", "all")

    # ── PlainViT ──
    if do_plain:
        export_plain_vit(device, _onnx_dir("plain_vit"))

    # ── 2-exit ──
    if do_2exit:
        # 명시 → 최신 exp → 전체 exp 탐색 순서로 체크포인트 결정
        ckpt = (args.ckpt_2exit
                or paths.latest_checkpoint("ee_vit_2exit", "best.pth")
                or find_checkpoint_any_exp("ee_vit_2exit", "best.pth"))
        if ckpt is None or not os.path.exists(ckpt):
            print("[ERROR] ee_vit_2exit 체크포인트 없음. --ckpt-2exit 로 지정하세요.")
            print("  예) --ckpt-2exit experiments/exp_20260414_212957/train/ee_vit_2exit/checkpoints/best.pth")
        else:
            print(f"2-exit 체크포인트: {ckpt}")
            model = load_selective_model(args.exit_blocks_2, ckpt, device)
            export_selective_segs(model, device, _onnx_dir("ee_vit_2exit"))
            del model

    # ── 3-exit ──
    if do_3exit:
        ckpt = (args.ckpt_3exit
                or paths.latest_checkpoint("ee_vit_3exit", "best.pth")
                or find_checkpoint_any_exp("ee_vit_3exit", "best.pth"))
        if ckpt is None or not os.path.exists(ckpt):
            print("[ERROR] ee_vit_3exit 체크포인트 없음. --ckpt-3exit 로 지정하세요.")
            print("  예) --ckpt-3exit experiments/exp_20260414_213050/train/ee_vit_3exit/checkpoints/best.pth")
        else:
            print(f"3-exit 체크포인트: {ckpt}")
            model = load_selective_model(args.exit_blocks_3, ckpt, device)
            export_selective_segs(model, device, _onnx_dir("ee_vit_3exit"))
            del model

    print(f"\n  ONNX export 완료  →  {paths.EXPERIMENTS_DIR}/onnx/")
    print("  다음 단계: Orin으로 전송 후 orin_vit_pipeline.sh 실행")


if __name__ == "__main__":
    main()
