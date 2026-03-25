"""
Exit별 샘플 시각화 스크립트

각 exit point(EE1, EE2, Main)에서 어떤 샘플들이 조기 종료됐는지 시각화.
올바르게 분류된 샘플 / 틀린 샘플을 구분해서 보여줌.

사용법:
  python visualize_exit_samples.py --ckpt experiments/.../best.pth --threshold 0.8
  python visualize_exit_samples.py --threshold 0.8  (자동 ckpt 선택)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models.ee_resnet18 import build_model
from datasets.dataloader import get_dataloader
from utils import load_config


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 역정규화용
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010])


def denormalize(img_tensor):
    """(C, H, W) tensor → (H, W, C) numpy, 0~1 범위"""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    img = img * CIFAR10_STD + CIFAR10_MEAN
    return np.clip(img, 0, 1)


def collect_exit_samples(model, test_loader, threshold, device, max_per_exit=20):
    """
    threshold 기반으로 각 exit에서 샘플 수집

    Returns:
        exit_samples: dict {
            1: [{'image': tensor, 'label': int, 'pred': int, 'conf': float}, ...],
            2: [...],
            3: [...]
        }
    """
    model.eval()
    exit_samples = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 3개 출력 모두 얻기
            out_ee1, out_ee2, out_main = model(images, threshold=None)

            B = images.size(0)
            for i in range(B):
                img    = images[i]
                label  = labels[i].item()

                # EE1 confidence
                conf1 = F.softmax(out_ee1[i:i+1], dim=1).max().item()
                pred1 = out_ee1[i].argmax().item()

                if conf1 >= threshold:
                    exit_idx = 1
                    pred = pred1
                    conf = conf1
                else:
                    # EE2 confidence
                    conf2 = F.softmax(out_ee2[i:i+1], dim=1).max().item()
                    pred2 = out_ee2[i].argmax().item()

                    if conf2 >= threshold:
                        exit_idx = 2
                        pred = pred2
                        conf = conf2
                    else:
                        # Main
                        conf3 = F.softmax(out_main[i:i+1], dim=1).max().item()
                        pred3 = out_main[i].argmax().item()
                        exit_idx = 3
                        pred = pred3
                        conf = conf3

                if len(exit_samples[exit_idx]) < max_per_exit:
                    exit_samples[exit_idx].append({
                        'image': img.cpu(),
                        'label': label,
                        'pred':  pred,
                        'conf':  conf,
                    })

            # 모든 exit에 충분히 수집되면 종료
            if all(len(v) >= max_per_exit for v in exit_samples.values()):
                break

    return exit_samples


def plot_exit_samples(exit_samples, threshold, out_dir, cols=10):
    """
    각 exit별 샘플 그리드 시각화
    - 초록 테두리: 정답 맞춤
    - 빨간 테두리: 오답
    """
    exit_labels = {
        1: 'Exit 1 (EE1) — after layer2',
        2: 'Exit 2 (EE2) — after layer3',
        3: 'Exit 3 (Main) — after layer4',
    }
    exit_colors = {1: '#4C8EFF', 2: '#FF8C42', 3: '#4CAF50'}

    # 최대 샘플 수 기준으로 행 결정
    max_samples = max(len(v) for v in exit_samples.values())
    rows_per_exit = (max_samples + cols - 1) // cols

    fig_rows = 3 * (rows_per_exit + 1)  # +1은 헤더용
    fig, axes = plt.subplots(
        3 * rows_per_exit + 3,
        cols,
        figsize=(cols * 1.6, 3 * rows_per_exit * 1.8 + 3)
    )
    fig.suptitle(f'Exit별 샘플 시각화  (threshold={threshold})', fontsize=14, y=1.01)

    ax_row = 0
    for exit_idx in [1, 2, 3]:
        samples = exit_samples[exit_idx]
        color   = exit_colors[exit_idx]
        label   = exit_labels[exit_idx]

        # 헤더 행
        header_ax = axes[ax_row, 0]
        for c in range(cols):
            axes[ax_row, c].axis('off')
        header_ax.text(
            0, 0.5,
            f'{label}\n({len(samples)} samples)',
            fontsize=11, fontweight='bold', color=color,
            va='center', ha='left', transform=header_ax.transAxes
        )
        ax_row += 1

        # 샘플 행
        for sample_idx in range(rows_per_exit * cols):
            row_in_block = sample_idx // cols
            col_in_block = sample_idx % cols
            ax = axes[ax_row + row_in_block, col_in_block]
            ax.axis('off')

            if sample_idx >= len(samples):
                continue

            s     = samples[sample_idx]
            img   = denormalize(s['image'])
            label_name = CIFAR10_CLASSES[s['label']]
            pred_name  = CIFAR10_CLASSES[s['pred']]
            is_correct = (s['label'] == s['pred'])
            border_color = '#00C853' if is_correct else '#FF1744'

            ax.imshow(img, interpolation='nearest')

            # 테두리
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

            # 제목: pred / conf
            title_color = '#00C853' if is_correct else '#FF1744'
            ax.set_title(
                f'{pred_name}\n{s["conf"]:.2f}',
                fontsize=6.5, color=title_color, pad=2
            )

            # 실제 정답을 틀린 경우 아래에 표시
            if not is_correct:
                ax.set_xlabel(f'GT:{label_name}', fontsize=5.5, color='gray', labelpad=1)

        ax_row += rows_per_exit

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    save_path = os.path.join(out_dir, f'exit_samples_thr{threshold:.2f}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"저장 완료: {save_path}")
    plt.show()


def print_exit_summary(exit_samples, threshold):
    print(f"\n{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}")
    for exit_idx in [1, 2, 3]:
        samples = exit_samples[exit_idx]
        if not samples:
            print(f"Exit {exit_idx}: 0 samples")
            continue
        correct = sum(1 for s in samples if s['label'] == s['pred'])
        avg_conf = np.mean([s['conf'] for s in samples])
        print(
            f"Exit {exit_idx}: {len(samples):>3}개  "
            f"정확도={correct/len(samples):.3f}  "
            f"평균 confidence={avg_conf:.3f}"
        )
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None,
                        help='체크포인트 경로. 미지정 시 자동 선택')
    parser.add_argument('--threshold', type=float, default=0.80,
                        help='confidence threshold (default: 0.80)')
    parser.add_argument('--samples', type=int, default=20,
                        help='exit별 최대 샘플 수 (default: 20)')
    parser.add_argument('--cols', type=int, default=10,
                        help='그리드 열 수 (default: 10)')
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    if args.ckpt is None:
        base = 'experiments'
        if os.path.exists(base):
            dirs = sorted([
                os.path.join(base, d) for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d))
            ])
            args.ckpt = os.path.join(dirs[-1], 'checkpoints', 'best.pth')
            print(f"자동 선택: {args.ckpt}")
        else:
            print("[ERROR] --ckpt 로 체크포인트 경로를 지정하세요")
            sys.exit(1)

    # ── 설정 / 모델 로드 ──
    cfg = load_config('configs/train.yaml')
    num_classes = 10 if cfg['dataset']['name'].lower() == 'cifar10' else 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    print(f"모델 로드: {args.ckpt}  device={device}")

    # ── 데이터로더 (batch_size=32로 빠르게 수집) ──
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'],
        batch_size=32,
        data_root=cfg['dataset']['data_root'],
        num_workers=0,
        seed=cfg['train']['seed']
    )

    # ── 샘플 수집 ──
    print(f"\n샘플 수집 중... (threshold={args.threshold}, max={args.samples}개/exit)")
    exit_samples = collect_exit_samples(
        model, test_loader, args.threshold, device, max_per_exit=args.samples
    )

    # ── 요약 출력 ──
    print_exit_summary(exit_samples, args.threshold)

    # ── 시각화 ──
    out_dir = os.path.dirname(os.path.dirname(args.ckpt))
    plot_exit_samples(exit_samples, args.threshold, out_dir, cols=args.cols)


if __name__ == '__main__':
    main()
