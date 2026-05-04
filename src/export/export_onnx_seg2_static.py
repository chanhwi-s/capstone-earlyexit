"""
export_onnx_seg2_static.py — 2-exit seg2 static batch ONNX export (goodput benchmark용)

각 bs2마다 seg2_bs{N}.onnx 별도 export.
ViT-B/16 (기본) 및 ViT-L/16 (--model-variant large) 모두 지원.

사용법:
  cd src
  # ViT-B/16 (기본)
  python export/export_onnx_seg2_static.py --batch-sizes 1 2 4 8 16 32 64
  python export/export_onnx_seg2_static.py --batch-sizes 1 2 4 8 16 32 64 \\
      --ckpt-2exit /path/to/ee_vit_2exit/best.pth

  # ViT-L/16
  python export/export_onnx_seg2_static.py --model-variant large \\
      --batch-sizes 1 2 4 8 16 32 64 \\
      --ckpt-2exit /path/to/ee_vit_large_2exit/best.pth
"""

import os, sys, argparse
import torch

os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/cap10/.cache/huggingface/hub')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src/export/

import paths
from export_onnx_vit_selective import ViTSegLast, find_checkpoint_any_exp


# ── 모델 variant별 로더 ────────────────────────────────────────────────────────

def _load_model(variant: str, exit_blocks: list, ckpt: str, device: torch.device):
    if variant == 'large':
        from models.ee_vit_large_selective import build_model_large
        model = build_model_large(exit_blocks=exit_blocks, num_classes=1000)
    else:
        from models.ee_vit_selective import build_model
        model = build_model(exit_blocks=exit_blocks, num_classes=1000)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ── Export ────────────────────────────────────────────────────────────────────

def export_seg2_static(model, device: torch.device, out_dir: str, bs2: int) -> bool:
    """seg2 (last segment) static batch=bs2 export.
    hidden_dim은 model.HIDDEN_DIM에서 자동 추론 (B: 768, L: 1024).
    OOM 발생 시 스킵하고 False 반환.
    """
    hidden_dim = model.HIDDEN_DIM
    path       = os.path.join(out_dir, f'seg2_bs{bs2}.onnx')

    try:
        seg_last = ViTSegLast(model, seg_idx=1).to(device).eval()
        dummy    = torch.randn(bs2, 197, hidden_dim, device=device)

        with torch.no_grad():
            torch.onnx.export(
                seg_last, dummy, path,
                input_names=['feat_in'],
                output_names=['ee_logits'],
                opset_version=17,
                verbose=False,
            )
        print(f'  seg2_bs{bs2:>4}  →  {path}  (static batch={bs2}, hidden={hidden_dim})')

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f'  seg2_bs{bs2:>4}  →  [SKIP] OOM (bs={bs2}, hidden={hidden_dim})')
        return False
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            print(f'  seg2_bs{bs2:>4}  →  [SKIP] OOM (bs={bs2}): {e}')
            return False
        raise

    try:
        import onnx
        m = onnx.load(path)
        onnx.checker.check_model(m)
        print(f'             ✓ onnx.checker passed')
    except ImportError:
        pass
    except Exception as e:
        print(f'             ✗ onnx.checker failed: {e}')

    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='2-exit seg2 static batch ONNX export (goodput benchmark용)'
    )
    parser.add_argument('--model-variant', type=str, default='base',
                        choices=['base', 'large'],
                        help='backbone variant: base=ViT-B/16, large=ViT-L/16 (기본: base)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64],
                        help='seg2 고정 batch 크기 목록 (기본: 8 16 32 64)')
    parser.add_argument('--ckpt-2exit', type=str, default=None,
                        help='체크포인트 경로 (미지정 시 최신 exp 자동 탐색)')
    parser.add_argument('--exit-blocks-2', type=int, nargs='+', default=None,
                        help='exit block 번호 (기본: base=[8,12], large=[12,24])')
    parser.add_argument('--out-exp', type=str, default=None,
                        help='출력 exp 디렉토리 (기본: 최신 exp 자동 선택)')
    args = parser.parse_args()

    # variant별 기본값
    variant_is_large = (args.model_variant == 'large')
    if variant_is_large:
        model_dir_name  = 'ee_vit_large_2exit'
        default_blocks  = [12, 24]
    else:
        model_dir_name  = 'ee_vit_2exit'
        default_blocks  = [8, 12]

    exit_blocks = args.exit_blocks_2 or default_blocks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device        : {device}')
    print(f'Model variant : ViT-{"L" if variant_is_large else "B"}/16  ({model_dir_name})')
    print(f'Exit blocks   : {exit_blocks}')
    print(f'Batch sizes   : {args.batch_sizes}')

    ckpt = (args.ckpt_2exit
            or paths.latest_checkpoint(model_dir_name, 'best.pth')
            or find_checkpoint_any_exp(model_dir_name, 'best.pth'))
    if not ckpt or not os.path.exists(ckpt):
        print(f'[ERROR] {model_dir_name} 체크포인트 없음. --ckpt-2exit 로 지정하세요.')
        return

    print(f'체크포인트    : {ckpt}')
    model = _load_model(args.model_variant, exit_blocks, ckpt, device)
    print(f'Hidden dim    : {model.HIDDEN_DIM}')

    if args.out_exp:
        out_dir = os.path.join(args.out_exp, 'onnx', model_dir_name)
    else:
        out_dir = paths.onnx_dir(model_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n[seg2 static export]  →  {out_dir}/')
    ok, skipped = [], []
    for bs2 in args.batch_sizes:
        if export_seg2_static(model, device, out_dir, bs2):
            ok.append(bs2)
        else:
            skipped.append(bs2)

    print(f'\n완료. 성공: {[f"seg2_bs{b}.onnx" for b in ok]}')
    if skipped:
        print(f'  스킵(OOM): bs={skipped}')
    print('다음 단계: benchmark_hybrid_2exit_goodput_5090.sh 실행')


if __name__ == '__main__':
    main()
