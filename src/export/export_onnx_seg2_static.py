"""
export_onnx_seg2_static.py — 2-exit seg2 static batch ONNX export (goodput benchmark용)

각 bs2마다 seg2_bs{N}.onnx 별도 export.
seg1.onnx / plain_vit.onnx는 기존 export_onnx_vit_selective.py로 생성된 것 재사용.

사용법:
  cd src
  python export/export_onnx_seg2_static.py --batch-sizes 8 16 32 64
  python export/export_onnx_seg2_static.py --batch-sizes 8 16 32 64 \\
      --ckpt-2exit /path/to/best.pth
"""

import os, sys, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # src/export/

import paths
from export_onnx_vit_selective import (
    ViTSegLast, load_selective_model, find_checkpoint_any_exp,
)


def export_seg2_static(model, device: torch.device, out_dir: str, bs2: int):
    """2-exit seg2 (last segment) static batch=bs2 export."""
    seg_last = ViTSegLast(model, seg_idx=1).to(device).eval()
    dummy    = torch.randn(bs2, 197, 768, device=device)
    path     = os.path.join(out_dir, f'seg2_bs{bs2}.onnx')

    with torch.no_grad():
        torch.onnx.export(
            seg_last, dummy, path,
            input_names=['feat_in'],
            output_names=['ee_logits'],
            opset_version=17,
            verbose=False,
        )
    print(f'  seg2_bs{bs2:>4}  →  {path}  (static batch={bs2})')

    try:
        import onnx
        m = onnx.load(path)
        onnx.checker.check_model(m)
        print(f'             ✓ onnx.checker passed')
    except ImportError:
        pass
    except Exception as e:
        print(f'             ✗ onnx.checker failed: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='2-exit seg2 static batch ONNX export (goodput benchmark용)'
    )
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64],
                        help='seg2 고정 batch 크기 목록 (기본: 8 16 32 64)')
    parser.add_argument('--ckpt-2exit', type=str, default=None,
                        help='ee_vit_2exit 체크포인트 경로')
    parser.add_argument('--exit-blocks-2', type=int, nargs='+', default=[8, 12],
                        help='2-exit 블록 번호 (기본: 8 12)')
    parser.add_argument('--out-exp', type=str, default=None,
                        help='출력 exp 디렉토리 (기본: 최신 exp 자동 선택)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device       : {device}')
    print(f'Batch sizes  : {args.batch_sizes}')

    ckpt = (args.ckpt_2exit
            or paths.latest_checkpoint('ee_vit_2exit', 'best.pth')
            or find_checkpoint_any_exp('ee_vit_2exit', 'best.pth'))
    if not ckpt or not os.path.exists(ckpt):
        print('[ERROR] ee_vit_2exit 체크포인트 없음. --ckpt-2exit 로 지정하세요.')
        return

    print(f'체크포인트   : {ckpt}')
    model = load_selective_model(args.exit_blocks_2, ckpt, device)

    if args.out_exp:
        out_dir = os.path.join(args.out_exp, 'onnx', 'ee_vit_2exit')
    else:
        out_dir = paths.onnx_dir('ee_vit_2exit')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n[seg2 static export]  →  {out_dir}/')
    for bs2 in args.batch_sizes:
        export_seg2_static(model, device, out_dir, bs2)

    print(f'\n완료. 생성 파일: {[f"seg2_bs{b}.onnx" for b in args.batch_sizes]}')
    print('다음 단계: benchmark_hybrid_2exit_goodput_5090.sh 실행')


if __name__ == '__main__':
    main()
