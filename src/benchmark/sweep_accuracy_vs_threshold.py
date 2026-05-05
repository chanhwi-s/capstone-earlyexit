"""
sweep_accuracy_vs_threshold.py — EE threshold별 accuracy 분석

목적:
  earlyexit으로 인한 accuracy overhead 정량화.
  plain 대신 "all-exit2" (모든 샘플을 seg2까지 통과)를 기준선으로 사용.
  이유: exit head가 30 epoch finetuning 된 모델이므로, plain보다 신뢰도 높은 비교.

방법:
  1. threshold=None 포워드 → 모든 샘플에 대해 (logit1, logit2) 동시 획득 (GPU 1회 패스)
  2. "all-exit2" 기준선 accuracy: logit2만 사용
  3. threshold T 스윕: conf(logit1) >= T → logit1 사용, else → logit2 사용

출력:
  accuracy_vs_threshold.png   — accuracy + exit rate vs threshold (3-panel)
  accuracy_sweep.csv          — threshold별 수치
  accuracy_sweep.json         — 원시 결과

사용법:
  cd src
  python benchmark/sweep_accuracy_vs_threshold.py
  python benchmark/sweep_accuracy_vs_threshold.py --n-samples 5000 --batch-size 64
  python benchmark/sweep_accuracy_vs_threshold.py --ckpt /path/to/best.pth
  python benchmark/sweep_accuracy_vs_threshold.py \\
      --thresholds 0.50 0.60 0.70 0.80 0.90 0.95 0.99
"""

import os, sys, csv, json, argparse, random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/cap10/.cache/huggingface/hub')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.ee_vit_large_selective import build_model_large


# ── 데이터 ────────────────────────────────────────────────────────────────────

def build_dataset(data_root: str):
    tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(os.path.join(data_root, 'val'), transform=tf)


def sample_data(dataset, n: int, seed: int):
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n)
    return torch.utils.data.Subset(dataset, indices)


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, exit_blocks: list, device: torch.device):
    model = build_model_large(exit_blocks=exit_blocks, num_classes=1000)
    ckpt  = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ── 추론 ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(model, loader, device):
    """
    모든 샘플에 대해 [logit1, logit2] 수집.
    Returns:
        logits1 : (N, C) float32
        logits2 : (N, C) float32
        labels  : (N,)   int64
    """
    all_l1, all_l2, all_lbl = [], [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        outs = model(imgs, threshold=None)   # [logit1, logit2]
        all_l1.append(outs[0].cpu())
        all_l2.append(outs[1].cpu())
        all_lbl.append(lbls)
        print(f'  collected {sum(t.shape[0] for t in all_lbl):>5} / ?', end='\r')
    print()
    return (torch.cat(all_l1),
            torch.cat(all_l2),
            torch.cat(all_lbl))


# ── 분석 ──────────────────────────────────────────────────────────────────────

def sweep(logits1: torch.Tensor, logits2: torch.Tensor,
          labels: torch.Tensor, thresholds: list):
    """
    각 threshold마다 accuracy / exit1_rate 계산.

    Returns: list of dict {threshold, accuracy_pct, exit1_rate_pct,
                           n_exit1, n_exit2, n_total}
    """
    confs1 = F.softmax(logits1, dim=1).max(dim=1).values   # (N,)
    preds1 = logits1.argmax(dim=1)
    preds2 = logits2.argmax(dim=1)
    N      = labels.shape[0]

    rows = []
    for thr in thresholds:
        mask_exit1 = confs1 >= thr            # True → exit1 사용
        preds      = torch.where(mask_exit1, preds1, preds2)
        acc        = (preds == labels).float().mean().item() * 100
        n_exit1    = mask_exit1.sum().item()
        rows.append({
            'threshold':       round(float(thr), 4),
            'accuracy_pct':    round(acc, 4),
            'exit1_rate_pct':  round(n_exit1 / N * 100, 4),
            'n_exit1':         int(n_exit1),
            'n_exit2':         N - int(n_exit1),
            'n_total':         N,
        })
    return rows


def baseline_accuracy(logits2: torch.Tensor, labels: torch.Tensor) -> float:
    """all-exit2: 모든 샘플이 seg2를 통과한 경우의 accuracy."""
    preds = logits2.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100


# ── 플롯 ──────────────────────────────────────────────────────────────────────

def plot_sweep(rows: list, baseline_acc: float, out_path: str,
               device_label: str = ''):
    thrs  = [r['threshold']      for r in rows]
    accs  = [r['accuracy_pct']   for r in rows]
    exits = [r['exit1_rate_pct'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title = 'Accuracy vs Threshold — 2-exit ViT-L/16'
    if device_label:
        title += f'  ({device_label})'
    fig.suptitle(title, fontsize=13)

    # ── 좌: accuracy vs threshold ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(thrs, accs, 'o-', color='#4c72b0', linewidth=2, markersize=5,
            label='Hybrid EE accuracy')
    ax.axhline(baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'All-exit2 baseline ({baseline_acc:.2f}%)')
    ax.set_xlabel('Confidence Threshold', fontsize=11)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy vs Threshold', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(min(thrs) - 0.01, max(thrs) + 0.01)

    # ── 중: exit1 rate vs threshold ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(thrs, exits, 's-', color='#dd8452', linewidth=2, markersize=5,
            label='Exit1 rate')
    ax.set_xlabel('Confidence Threshold', fontsize=11)
    ax.set_ylabel('Exit1 Rate (%)', fontsize=11)
    ax.set_title('Exit1 Rate vs Threshold', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(min(thrs) - 0.01, max(thrs) + 0.01)

    # ── 우: accuracy vs exit1 rate (효율 프론티어) ─────────────────────────────
    ax = axes[2]
    sc = ax.scatter(exits, accs, c=thrs, cmap='viridis', s=60, zorder=3)
    ax.plot(exits, accs, '-', color='gray', linewidth=1, alpha=0.5)
    ax.axhline(baseline_acc, color='black', linestyle='--', linewidth=2,
               label=f'All-exit2 baseline ({baseline_acc:.2f}%)')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Threshold', fontsize=9)
    # threshold 몇 개 annotate
    step = max(1, len(rows) // 5)
    for r in rows[::step]:
        ax.annotate(f'{r["threshold"]:.2f}',
                    (r['exit1_rate_pct'], r['accuracy_pct']),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    ax.set_xlabel('Exit1 Rate (%)', fontsize=11)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy vs Exit Rate (Efficiency Frontier)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


# ── 저장 ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list, baseline_acc: float, path: str):
    fields = ['threshold', 'accuracy_pct', 'accuracy_drop_pct',
              'exit1_rate_pct', 'n_exit1', 'n_exit2', 'n_total']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'accuracy_drop_pct': round(baseline_acc - r['accuracy_pct'], 4)})
    print(f'  CSV : {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='EE threshold별 accuracy sweep (all-exit2 기준선 대비)'
    )
    parser.add_argument('--data-root',   type=str, default='/home2/imagenet')
    parser.add_argument('--n-samples',   type=int, default=5000)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--batch-size',  type=int, default=64)
    parser.add_argument('--ckpt',        type=str, default=None,
                        help='ee_vit_large_2exit 체크포인트. 미지정 시 자동 탐색.')
    parser.add_argument('--exit-blocks', type=int, nargs='+', default=[12, 24])
    parser.add_argument('--thresholds',  type=float, nargs='+', default=None,
                        help='스윕할 threshold 목록. 미지정 시 0.50~0.99 자동 생성.')
    parser.add_argument('--out-dir',     type=str, default=None)
    parser.add_argument('--device-label', type=str, default='RTX 5090 (ViT-L)')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'accuracy_sweep_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    thresholds = args.thresholds or list(np.round(np.arange(0.50, 1.00, 0.02), 4))

    ckpt_path = args.ckpt or paths.latest_checkpoint('ee_vit_large_2exit')
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'[ERROR] 체크포인트를 찾을 수 없습니다: {ckpt_path}')
        print('  --ckpt 로 직접 지정하거나, 학습을 먼저 실행하세요.')
        return

    print(f'Device       : {device}  ({args.device_label})')
    print(f'Checkpoint   : {ckpt_path}')
    print(f'N-Samples    : {args.n_samples}  (seed={args.seed})')
    print(f'Batch size   : {args.batch_size}')
    print(f'Exit blocks  : {args.exit_blocks}')
    print(f'Thresholds   : {thresholds}')
    print(f'Output       : {out_dir}\n')

    # ── 모델 로드 ──
    print('모델 로드 ...')
    model = load_model(ckpt_path, args.exit_blocks, device)
    print(f'  exit_blocks = {model.exit_blocks}\n')

    # ── 데이터 로드 ──
    print(f'데이터 로드 ({args.n_samples}개) ...')
    dataset = build_dataset(args.data_root)
    subset  = sample_data(dataset, args.n_samples, args.seed)
    loader  = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    print(f'  {len(subset)} samples  ({len(loader)} batches)\n')

    # ── 전체 logit 수집 (GPU 1회 패스) ──
    print('전체 샘플 추론 (threshold=None, logit1+logit2 동시 수집) ...')
    logits1, logits2, labels = collect_logits(model, loader, device)
    print(f'  logits1: {tuple(logits1.shape)}  logits2: {tuple(logits2.shape)}\n')

    # ── all-exit2 기준선 ──
    base_acc = baseline_accuracy(logits2, labels)
    print(f'All-exit2 baseline accuracy : {base_acc:.4f}%\n')

    # ── threshold sweep ──
    print(f'Threshold sweep ({len(thresholds)} 포인트) ...')
    rows = sweep(logits1, logits2, labels, thresholds)
    for r in rows:
        drop = base_acc - r['accuracy_pct']
        print(f'  thr={r["threshold"]:.2f}  acc={r["accuracy_pct"]:.2f}%  '
              f'drop={drop:+.2f}%  exit1={r["exit1_rate_pct"]:.1f}%')

    # ── 저장 ──
    save_csv(rows, base_acc, os.path.join(out_dir, 'accuracy_sweep.csv'))
    result = {
        'baseline_acc_pct': round(base_acc, 4),
        'n_samples':        len(labels),
        'exit_blocks':      args.exit_blocks,
        'checkpoint':       ckpt_path,
        'sweep':            rows,
    }
    with open(os.path.join(out_dir, 'accuracy_sweep.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  JSON: {os.path.join(out_dir, "accuracy_sweep.json")}')

    # ── 플롯 ──
    plot_sweep(rows, base_acc,
               os.path.join(out_dir, 'accuracy_vs_threshold.png'),
               args.device_label)

    print(f'\nDone! → {out_dir}')


if __name__ == '__main__':
    main()
