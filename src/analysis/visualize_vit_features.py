"""
visualize_vit_features.py  —  ViT-B/16 encoder block별 feature map 시각화

각 Transformer block의 출력 feature map [B, 197, 768]에서
patch token [196, 768]을 14×14 공간 그리드로 복원해 시각화.

시각화 방법 2가지:
  1. Mean Activation Map  : 768채널 평균 → [14,14] heatmap
  2. PCA RGB Map          : PCA로 768→3차원 축소 → [14,14,3] RGB

출력 (--out-dir 지정 디렉토리):
  feature_mean_all_blocks.png   ← 12블록 mean heatmap (+ 원본 이미지)
  feature_pca_all_blocks.png    ← 12블록 PCA RGB map
  feature_block_{i:02d}_detail.png  ← 각 블록 상세 (mean + PCA + diff)
  feature_shapes.txt            ← block별 in/out shape 텍스트 요약

사용법:
  cd src
  python analysis/visualize_vit_features.py
  python analysis/visualize_vit_features.py --num-samples 5
  python analysis/visualize_vit_features.py --image /path/to/image.jpg
  python analysis/visualize_vit_features.py --model ee_vit --checkpoint /path/to/best.pth
  python analysis/visualize_vit_features.py --model plain_vit --num-samples 8 --out-dir /tmp/vit_vis

인자:
  --image        시각화할 단일 이미지 경로 (지정 시 --num-samples 무시)
  --num-samples  ImageNet val에서 로드할 샘플 수 (기본: 1)
                 각 샘플은 sample_001/, sample_002/ ... 서브디렉토리에 저장
  --model        plain_vit | ee_vit (기본: plain_vit)
  --checkpoint   EE-ViT 체크포인트 경로 (--model ee_vit 일 때만 사용)
  --data-root    ImageNet 루트 경로 (기본: /home2)
  --out-dir      결과 저장 디렉토리 (기본: /tmp/vit_features_YYYYMMDD_HHMMSS)
  --blocks       시각화할 블록 번호 0-indexed (기본: 0~11 전체)
                 예: --blocks 0 3 6 9 11
"""

import os
import sys
import argparse
from datetime import datetime

# HuggingFace 캐시 경로 (공유 캐시 권한 오류 방지) — timm import 전에 설정
os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.plain_vit import build_model as build_plain
from models.ee_vit import EEViT, build_model as build_ee
import paths

# ImageNet 정규화 역변환 상수
_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

PATCH_GRID = 14   # 224 / 16 = 14
N_BLOCKS   = 12


# ── 이미지 전처리 ──────────────────────────────────────────────────────────────

def preprocess_image(path: str) -> tuple[torch.Tensor, np.ndarray]:
    """
    이미지 로드 → 224×224 리사이즈 → ImageNet 정규화.
    Returns:
        tensor : [1, 3, 224, 224]  (모델 입력용)
        orig   : [224, 224, 3]     (시각화용 uint8)
    """
    from torchvision import transforms
    img  = Image.open(path).convert('RGB')
    tfm  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    tensor = tfm(img).unsqueeze(0)  # [1, 3, 224, 224]

    # 시각화용 원본 (정규화 역변환)
    arr  = tensor[0].permute(1, 2, 0).numpy()
    arr  = arr * _STD + _MEAN
    orig = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return tensor, orig


def load_imagenet_samples(num_samples: int,
                          data_root: str = None,
                          num_workers: int = 4) -> list[tuple[torch.Tensor, np.ndarray]]:
    """
    ImageNet val에서 num_samples개 로드.
    Returns: list of (tensor [1,3,224,224], orig [224,224,3] uint8)
    """
    from utils import load_config
    from datasets.dataloader import get_dataloader

    cfg = load_config('configs/train.yaml')
    if data_root is None:
        data_root = '/home2'   # 서버 기본 경로

    _, loader, _ = get_dataloader(
        dataset='imagenet', batch_size=1, data_root=data_root,
        num_workers=num_workers, seed=42,
    )

    samples = []
    for i, (img, lbl) in enumerate(loader):
        if i >= num_samples:
            break
        arr  = img[0].permute(1, 2, 0).numpy()
        arr  = arr * _STD + _MEAN
        orig = np.clip(arr * 255, 0, 255).astype(np.uint8)
        samples.append((img, orig, lbl[0].item()))

    print(f"  ImageNet val 로드 완료: {len(samples)}개 샘플")
    return samples


# ── Feature Hook ──────────────────────────────────────────────────────────────

class BlockFeatureHook:
    """
    nn.Module의 forward hook으로 각 block 입력/출력 텐서를 캡처.

    captures[i] = {
        'input':  Tensor [1, 197, 768]  (block[i] 입력)
        'output': Tensor [1, 197, 768]  (block[i] 출력)
    }
    """

    def __init__(self):
        self.captures: list[dict] = []
        self._handles: list       = []

    def register(self, blocks):
        """blocks: nn.Sequential (timm ViT의 self.blocks)"""
        for i, block in enumerate(blocks):
            idx = i  # 클로저 캡처용

            def make_hook(block_idx):
                def hook(module, inp, out):
                    # inp은 tuple, inp[0]이 실제 텐서
                    self.captures.append({
                        'block_idx': block_idx,
                        'input':     inp[0].detach().cpu(),   # [1, 197, 768]
                        'output':    out.detach().cpu(),       # [1, 197, 768]
                    })
                return hook

            h = block.register_forward_hook(make_hook(idx))
            self._handles.append(h)

    def reset(self):
        """캡처 목록만 초기화 (hook은 유지). 샘플 간 재사용 시 호출."""
        self.captures.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Feature → 2D 변환 ─────────────────────────────────────────────────────────

def to_spatial(feat: torch.Tensor) -> np.ndarray:
    """
    [1, 197, 768] → [14, 14, 768]
    CLS 토큰(index 0) 제거 후 patch token 196개를 14×14로 reshape.
    """
    x = feat[0, 1:, :].numpy()         # [196, 768]  (CLS 제거)
    return x.reshape(PATCH_GRID, PATCH_GRID, -1)   # [14, 14, 768]


def mean_map(spatial: np.ndarray) -> np.ndarray:
    """[14, 14, 768] → [14, 14]  (채널 평균)"""
    return spatial.mean(axis=-1)


def pca_rgb_map(spatial: np.ndarray, pca=None):
    """
    [14, 14, 768] → [14, 14, 3]  (PCA 3채널)
    pca: 이미 fit된 PCA 객체를 재사용할 경우 전달.
    Returns: rgb [14, 14, 3] uint8, pca 객체
    """
    flat = spatial.reshape(-1, spatial.shape[-1])   # [196, 768]
    if pca is None:
        pca = PCA(n_components=3)
        components = pca.fit_transform(flat)
    else:
        components = pca.transform(flat)            # [196, 3]

    rgb = components.reshape(PATCH_GRID, PATCH_GRID, 3)

    # 각 채널 독립적으로 [0, 1] 정규화
    for c in range(3):
        ch = rgb[:, :, c]
        mn, mx = ch.min(), ch.max()
        rgb[:, :, c] = (ch - mn) / (mx - mn + 1e-8)

    return (rgb * 255).astype(np.uint8), pca


def normalize_map(m: np.ndarray) -> np.ndarray:
    """[14, 14] → [0, 1] 정규화 (시각화용)"""
    mn, mx = m.min(), m.max()
    return (m - mn) / (mx - mn + 1e-8)


# ── Plot 1: 전체 12블록 Mean Heatmap ─────────────────────────────────────────

def plot_mean_all_blocks(captures: list, orig: np.ndarray,
                         target_blocks: list, save_path: str):
    """
    원본 이미지 + 각 블록 mean activation heatmap 그리드.
    """
    n   = len(target_blocks)
    cols = min(6, n + 1)
    rows = ((n + 1) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.0))
    fig.suptitle('ViT-B/16  —  Mean Activation Map per Block\n'
                 '(avg 768 channels over 196 patch tokens → 14×14 heatmap)',
                 fontsize=12)

    axes = np.array(axes).flatten()

    # 원본 이미지
    axes[0].imshow(orig)
    axes[0].set_title('Input Image\n(224×224)', fontsize=9)
    axes[0].axis('off')

    # 각 블록 mean map
    for plot_i, block_i in enumerate(target_blocks):
        cap    = next(c for c in captures if c['block_idx'] == block_i)
        sp_out = to_spatial(cap['output'])
        mmap   = normalize_map(mean_map(sp_out))

        ax = axes[plot_i + 1]
        im = ax.imshow(mmap, cmap='viridis', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(f'Block {block_i + 1}\n(out: [14,14] mean)', fontsize=8)
        ax.axis('off')
        # 격자선 (14×14 패치 경계)
        for k in range(1, PATCH_GRID):
            ax.axhline(k - 0.5, color='white', linewidth=0.3, alpha=0.4)
            ax.axvline(k - 0.5, color='white', linewidth=0.3, alpha=0.4)

    # 남은 subplot 숨기기
    for j in range(len(target_blocks) + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Mean heatmap 저장: {save_path}")


# ── Plot 2: 전체 12블록 PCA RGB ──────────────────────────────────────────────

def plot_pca_all_blocks(captures: list, orig: np.ndarray,
                        target_blocks: list, save_path: str):
    """
    원본 이미지 + 각 블록 PCA RGB map 그리드.
    각 블록은 독립적인 PCA (블록마다 feature 특성이 다르므로).
    """
    n    = len(target_blocks)
    cols = min(6, n + 1)
    rows = ((n + 1) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.0))
    fig.suptitle('ViT-B/16  —  PCA RGB Feature Map per Block\n'
                 '(768-dim → PCA 3ch → RGB, per-block independent PCA)',
                 fontsize=12)

    axes = np.array(axes).flatten()

    axes[0].imshow(orig)
    axes[0].set_title('Input Image\n(224×224)', fontsize=9)
    axes[0].axis('off')

    for plot_i, block_i in enumerate(target_blocks):
        cap    = next(c for c in captures if c['block_idx'] == block_i)
        sp_out = to_spatial(cap['output'])
        rgb, _ = pca_rgb_map(sp_out)

        ax = axes[plot_i + 1]
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(f'Block {block_i + 1}\n(PCA RGB [14,14,3])', fontsize=8)
        ax.axis('off')
        for k in range(1, PATCH_GRID):
            ax.axhline(k - 0.5, color='white', linewidth=0.3, alpha=0.4)
            ax.axvline(k - 0.5, color='white', linewidth=0.3, alpha=0.4)

    for j in range(len(target_blocks) + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PCA RGB map 저장: {save_path}")


# ── Plot 3: 블록별 상세 (in / out / diff) ────────────────────────────────────

def plot_block_detail(captures: list, orig: np.ndarray,
                      block_i: int, save_path: str):
    """
    block_i에 대해:
      col 1: 입력 mean map
      col 2: 출력 mean map
      col 3: 출력 - 입력 차이 (변화량)
      col 4: 입력 PCA RGB
      col 5: 출력 PCA RGB
    """
    cap    = next(c for c in captures if c['block_idx'] == block_i)
    sp_in  = to_spatial(cap['input'])    # [14, 14, 768]
    sp_out = to_spatial(cap['output'])   # [14, 14, 768]

    mmap_in  = mean_map(sp_in)
    mmap_out = mean_map(sp_out)
    diff_raw = mmap_out - mmap_in

    rgb_in,  pca_obj = pca_rgb_map(sp_in)
    rgb_out, _       = pca_rgb_map(sp_out)   # 각 블록 독립 PCA

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
    fig.suptitle(
        f'ViT-B/16  Block {block_i + 1} / 12  —  In → Out Feature Map Detail\n'
        f'in/out: [1, 197, 768]  →  patch tokens [14, 14, 768]  →  visualized',
        fontsize=10)

    def show(ax, img, title, cmap='viridis', vmin=None, vmax=None, norm=None):
        if img.ndim == 3:
            ax.imshow(img, interpolation='nearest')
        else:
            im = ax.imshow(img, cmap=cmap, interpolation='nearest',
                           vmin=vmin, vmax=vmax, norm=norm)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        for k in range(1, PATCH_GRID):
            ax.axhline(k - 0.5, color='white', linewidth=0.3, alpha=0.5)
            ax.axvline(k - 0.5, color='white', linewidth=0.3, alpha=0.5)

    show(axes[0], normalize_map(mmap_in),
         f'Input Mean\n[14,14] (avg 768ch)')
    show(axes[1], normalize_map(mmap_out),
         f'Output Mean\n[14,14] (avg 768ch)')

    # diff: 양/음 모두 표현 (RdBu)
    abs_max = max(abs(diff_raw.min()), abs(diff_raw.max())) + 1e-8
    show(axes[2], diff_raw,
         f'Out - In (diff)\n[14,14]',
         cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)

    show(axes[3], rgb_in,
         f'Input PCA RGB\n[14,14,3]')
    show(axes[4], rgb_out,
         f'Output PCA RGB\n[14,14,3]')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Shape 요약 텍스트 출력 ────────────────────────────────────────────────────

def print_and_save_shapes(captures: list, save_path: str):
    lines = []
    lines.append("=" * 60)
    lines.append("  ViT-B/16 Block별 Feature Map Shape 요약")
    lines.append("=" * 60)
    lines.append(f"  {'Block':>7}  {'Input Shape':>20}  {'Output Shape':>20}")
    lines.append(f"  {'-'*55}")

    for cap in sorted(captures, key=lambda c: c['block_idx']):
        i    = cap['block_idx']
        s_in  = tuple(cap['input'].shape)
        s_out = tuple(cap['output'].shape)
        lines.append(f"  Block {i+1:>2}/12  {str(s_in):>20}  {str(s_out):>20}")

    lines.append(f"  {'-'*55}")
    lines.append("  ※ 모든 블록의 in/out shape이 동일합니다:")
    lines.append("    [B, 197, 768]")
    lines.append("      197 = 1(CLS) + 196(patch) = 14×14 공간 격자")
    lines.append("      768 = hidden_dim (ViT-B/16 고정)")
    lines.append("")
    lines.append("  2D 시각화 변환:")
    lines.append("    [1, 197, 768]")
    lines.append("      → patch token [1:197] = [196, 768]")
    lines.append("      → reshape → [14, 14, 768]  (14×14 공간 복원)")
    lines.append("      → mean(axis=-1) → [14, 14]  heatmap")
    lines.append("      → PCA(n=3) → [14, 14, 3]    RGB 이미지")
    lines.append("=" * 60)

    text = "\n".join(lines)
    print(text)
    with open(save_path, 'w') as f:
        f.write(text + "\n")
    print(f"\n  shape 요약 저장: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def run_one_sample(wrapper, blocks, tensor: torch.Tensor, orig: np.ndarray,
                   label: int, sample_dir: str, target_blocks: list,
                   device: torch.device, hook: 'BlockFeatureHook',
                   save_shapes: bool = False):
    """단일 샘플에 대해 hook 캡처 → 그래프 저장."""
    os.makedirs(sample_dir, exist_ok=True)

    hook.reset()
    with torch.no_grad():
        out = wrapper(tensor.to(device))

    # plain_vit: logits tensor, ee_vit(threshold=None): list of logits
    if isinstance(out, list):
        pred = out[-1].argmax(dim=1).item()
    else:
        pred = out.argmax(dim=1).item()

    captures = sorted(hook.captures, key=lambda c: c['block_idx'])

    if save_shapes:
        print_and_save_shapes(captures,
                              os.path.join(os.path.dirname(sample_dir),
                                           'feature_shapes.txt'))

    print(f"  그래프 생성 중... (label={label}, pred={pred})")
    plot_mean_all_blocks(captures, orig, target_blocks,
                         os.path.join(sample_dir, 'feature_mean_all_blocks.png'))
    plot_pca_all_blocks(captures, orig, target_blocks,
                        os.path.join(sample_dir, 'feature_pca_all_blocks.png'))
    for block_i in target_blocks:
        detail_path = os.path.join(sample_dir,
                                   f'feature_block_{block_i+1:02d}_detail.png')
        plot_block_detail(captures, orig, block_i, detail_path)

    print(f"    저장 완료: {sample_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='ViT-B/16 encoder block별 feature map 2D 시각화')
    parser.add_argument('--image',       type=str,   default=None,
                        help='단일 이미지 경로 (지정 시 --num-samples 무시)')
    parser.add_argument('--num-samples', type=int,   default=1,
                        help='ImageNet val에서 로드할 샘플 수 (기본: 1)')
    parser.add_argument('--model',       type=str,   default='plain_vit',
                        choices=['plain_vit', 'ee_vit'],
                        help='plain_vit | ee_vit (기본: plain_vit)')
    parser.add_argument('--checkpoint',  type=str,   default=None,
                        help='EE-ViT 체크포인트 경로 (--model ee_vit 시 사용)')
    parser.add_argument('--data-root',   type=str,   default='/home2',
                        help='ImageNet 루트 경로 (기본: /home2)')
    parser.add_argument('--out-dir',     type=str,   default=None,
                        help='결과 저장 디렉토리')
    parser.add_argument('--blocks',      type=int,   nargs='+', default=None,
                        help='시각화할 블록 번호 0-indexed (기본: 0~11 전체)\n'
                             '예: --blocks 0 3 6 9 11')
    args = parser.parse_args()

    # ── device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # ── 출력 디렉토리 ────────────────────────────────────────────────────────
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or f'/tmp/vit_features_{ts}'
    os.makedirs(out_dir, exist_ok=True)
    print(f"출력 디렉토리: {out_dir}")

    target_blocks = args.blocks if args.blocks else list(range(N_BLOCKS))
    target_blocks = [b for b in target_blocks if 0 <= b < N_BLOCKS]

    # ── 샘플 준비 ────────────────────────────────────────────────────────────
    if args.image:
        print(f"\n이미지 로드: {args.image}")
        tensor, orig = preprocess_image(args.image)
        samples = [(tensor, orig, -1)]   # label 모름
    else:
        print(f"\nImageNet val 샘플 {args.num_samples}개 로드 중...")
        samples = load_imagenet_samples(args.num_samples, args.data_root)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print(f"\n모델 로드 중: {args.model}")
    if args.model == 'plain_vit':
        wrapper = build_plain(num_classes=1000).to(device)
        blocks  = wrapper.model.blocks
    else:
        ckpt = args.checkpoint or paths.latest_checkpoint('ee_vit', 'best.pth')
        if ckpt is None or not os.path.exists(ckpt):
            print(f"[ERROR] EE-ViT 체크포인트를 찾을 수 없습니다. --checkpoint 로 지정하세요.")
            sys.exit(1)
        wrapper = build_ee(num_classes=1000).to(device)
        state   = torch.load(ckpt, map_location=device, weights_only=True)
        wrapper.load_state_dict(state)
        blocks  = wrapper.blocks

    wrapper.eval()
    print(f"  모델 로드 완료")

    # ── Hook 등록 (모델 로드 후 한 번만) ─────────────────────────────────────
    hook = BlockFeatureHook()
    hook.register(blocks)

    # ── 샘플별 시각화 ────────────────────────────────────────────────────────
    n = len(samples)
    for idx, item in enumerate(samples):
        tensor, orig, label = item
        # 샘플이 1개면 루트에, 여러 개면 sample_NNN/ 서브디렉토리에 저장
        if n == 1:
            sample_dir = out_dir
        else:
            sample_dir = os.path.join(out_dir, f'sample_{idx+1:03d}')

        print(f"\n[{idx+1}/{n}] 샘플 처리 중... ", end='')
        run_one_sample(wrapper, blocks, tensor, orig, label,
                       sample_dir, target_blocks, device, hook,
                       save_shapes=(idx == 0))

    hook.remove()

    print(f"\n{'='*55}")
    print(f"완료! 결과 위치: {out_dir}")
    if n > 1:
        print(f"  sample_001/ ~ sample_{n:03d}/  각 샘플별 서브디렉토리")
    print(f"  feature_shapes.txt             — block별 shape 요약")
    print(f"  feature_mean_all_blocks.png    — 12블록 mean activation heatmap")
    print(f"  feature_pca_all_blocks.png     — 12블록 PCA RGB feature map")
    print(f"  feature_block_XX_detail.png    — 각 블록 in/out/diff 상세")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
