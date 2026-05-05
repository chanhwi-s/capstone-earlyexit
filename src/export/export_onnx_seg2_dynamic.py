"""
export_onnx_seg2_dynamic.py — seg2 dynamic batch ONNX export (dynamic benchmark용)

static export와 달리 batch axis를 dynamic으로 설정.
timeout-based dynamic batching 시뮬레이션에서 가변 batch 실행에 사용.

사용법:
  cd src
  python export/export_onnx_seg2_dynamic.py
  python export/export_onnx_seg2_dynamic.py --model-variant large
  python export/export_onnx_seg2_dynamic.py --model-variant large \\
      --ckpt-2exit /path/to/ee_vit_large_2exit/best.pth
"""

import os, sys, argparse
import torch

os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/cap10/.cache/huggingface/hub')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
from export_onnx_vit_selective import ViTSegLast, find_checkpoint_any_exp


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


def export_seg2_dynamic(model, device: torch.device, out_dir: str) -> str:
    hidden_dim = model.HIDDEN_DIM
    path       = os.path.join(out_dir, 'seg2_dynamic.onnx')

    seg_last = ViTSegLast(model, seg_idx=1).to(device).eval()
    dummy    = torch.randn(1, 197, hidden_dim, device=device)

    with torch.no_grad():
        torch.onnx.export(
            seg_last, dummy, path,
            input_names=['feat_in'],
            output_names=['ee_logits'],
            dynamic_axes={'feat_in': {0: 'batch'}, 'ee_logits': {0: 'batch'}},
            opset_version=17,
            verbose=False,
        )
    print(f'  seg2_dynamic  →  {path}  (dynamic batch, hidden={hidden_dim})')

    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print(f'               ✓ onnx.checker passed')
    except ImportError:
        pass
    except Exception as e:
        print(f'               ✗ onnx.checker failed: {e}')

    return path


def main():
    parser = argparse.ArgumentParser(
        description='seg2 dynamic batch ONNX export (dynamic benchmark용)'
    )
    parser.add_argument('--model-variant', type=str, default='base',
                        choices=['base', 'large'])
    parser.add_argument('--ckpt-2exit',   type=str, default=None)
    parser.add_argument('--exit-blocks',  type=int, nargs='+', default=None)
    parser.add_argument('--out-exp',      type=str, default=None)
    args = parser.parse_args()

    variant_is_large = (args.model_variant == 'large')
    model_dir_name   = 'ee_vit_large_2exit' if variant_is_large else 'ee_vit_2exit'
    default_blocks   = [12, 24]             if variant_is_large else [8, 12]
    exit_blocks      = args.exit_blocks or default_blocks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device        : {device}')
    print(f'Model variant : ViT-{"L" if variant_is_large else "B"}/16  ({model_dir_name})')
    print(f'Exit blocks   : {exit_blocks}')

    ckpt = (args.ckpt_2exit
            or paths.latest_checkpoint(model_dir_name, 'best.pth')
            or find_checkpoint_any_exp(model_dir_name, 'best.pth'))
    if not ckpt or not os.path.exists(ckpt):
        print(f'[ERROR] {model_dir_name} 체크포인트 없음.')
        return

    print(f'Checkpoint    : {ckpt}')
    model = _load_model(args.model_variant, exit_blocks, ckpt, device)

    out_dir = (os.path.join(args.out_exp, 'onnx', model_dir_name)
               if args.out_exp else paths.onnx_dir(model_dir_name))
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n[seg2 dynamic export]  →  {out_dir}/')
    export_seg2_dynamic(model, device, out_dir)
    print('\n완료. 다음 단계: dynamic_benchmark.sh 실행')


if __name__ == '__main__':
    main()
