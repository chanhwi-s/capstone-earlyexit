"""
Exit Rate / Accuracy 평가 스크립트 — EE (3-exit) / VEE (2-exit) 지원

threshold 범위를 sweep 하며:
  - 각 threshold에서 exit별 비율 (exit rate)
  - exit별 / 전체 accuracy
  - 시각화 + JSON 저장

사용법:
  # 프로젝트 루트에서
  python src/analysis/eval_exit_rate.py --model ee
  python src/analysis/eval_exit_rate.py --model vee
  python src/analysis/eval_exit_rate.py --model vee --ckpt /path/to/best.pth
  python src/analysis/eval_exit_rate.py --model vee --threshold-range 0.60 0.95 0.05
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ee_resnet18 import build_model as build_ee
from models.vee_resnet18 import build_model as build_vee
from datasets.dataloader import get_dataloader
from utils import load_config
import paths

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── EE (3-exit) 평가 ──────────────────────────────────────────────────────────

def evaluate_ee(model, test_loader, thresholds, device):
    """
    Returns:
        results: {threshold: {exit_rate: [r1,r2,r3], acc_exit1, acc_exit2, acc_exit3, acc_overall}}
        total_samples: int
    """
    model.eval()
    results = {t: {'exit1_count': 0, 'exit2_count': 0, 'exit3_count': 0,
                   'exit1_correct': 0, 'exit2_correct': 0, 'exit3_correct': 0}
               for t in thresholds}
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += inputs.size(0)

            logits_ee1, logits_ee2, logits_main = model(inputs, threshold=None)

            for t in thresholds:
                r = results[t]
                conf1 = F.softmax(logits_ee1, dim=1).max(dim=1).values
                pred1 = logits_ee1.argmax(dim=1)

                m1 = conf1 >= t
                r['exit1_count']   += m1.sum().item()
                r['exit1_correct'] += (pred1[m1] == labels[m1]).sum().item()

                m_cont1 = ~m1
                if m_cont1.sum() > 0:
                    conf2 = F.softmax(logits_ee2[m_cont1], dim=1).max(dim=1).values
                    pred2 = logits_ee2[m_cont1].argmax(dim=1)
                    lbl_cont1 = labels[m_cont1]

                    m2 = conf2 >= t
                    r['exit2_count']   += m2.sum().item()
                    r['exit2_correct'] += (pred2[m2] == lbl_cont1[m2]).sum().item()

                    m_cont2 = ~m2
                    if m_cont2.sum() > 0:
                        pred3 = logits_main[m_cont1][m_cont2].argmax(dim=1)
                        r['exit3_count']   += m_cont2.sum().item()
                        r['exit3_correct'] += (pred3 == lbl_cont1[m_cont2]).sum().item()

    # 정확도 / exit rate 계산
    final = {}
    for t in thresholds:
        r = results[t]
        n1, n2, n3 = r['exit1_count'], r['exit2_count'], r['exit3_count']
        final[t] = {
            'exit_rate':  [n1 / total_samples * 100,
                           n2 / total_samples * 100,
                           n3 / total_samples * 100],
            'acc_exit1':  r['exit1_correct'] / n1 if n1 > 0 else 0.0,
            'acc_exit2':  r['exit2_correct'] / n2 if n2 > 0 else 0.0,
            'acc_exit3':  r['exit3_correct'] / n3 if n3 > 0 else 0.0,
            'acc_overall': (r['exit1_correct'] + r['exit2_correct'] + r['exit3_correct']) / total_samples,
        }
    return final, total_samples


# ── VEE (2-exit) 평가 ─────────────────────────────────────────────────────────

def evaluate_vee(model, test_loader, thresholds, device):
    """
    Returns:
        results: {threshold: {exit_rate: [r1, r_main], acc_exit1, acc_main, acc_overall}}
        total_samples: int
    """
    model.eval()
    results = {t: {'exit1_count': 0, 'main_count': 0,
                   'exit1_correct': 0, 'main_correct': 0}
               for t in thresholds}
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += inputs.size(0)

            out_ee1, out_main = model(inputs, threshold=None)

            for t in thresholds:
                r = results[t]
                conf1 = F.softmax(out_ee1, dim=1).max(dim=1).values
                pred1 = out_ee1.argmax(dim=1)

                m1 = conf1 >= t
                r['exit1_count']   += m1.sum().item()
                r['exit1_correct'] += (pred1[m1] == labels[m1]).sum().item()

                m_main = ~m1
                if m_main.sum() > 0:
                    pred_main = out_main[m_main].argmax(dim=1)
                    r['main_count']   += m_main.sum().item()
                    r['main_correct'] += (pred_main == labels[m_main]).sum().item()

    final = {}
    for t in thresholds:
        r = results[t]
        n1, nm = r['exit1_count'], r['main_count']
        final[t] = {
            'exit_rate':  [n1 / total_samples * 100,
                           nm / total_samples * 100],
            'acc_exit1':  r['exit1_correct'] / n1 if n1 > 0 else 0.0,
            'acc_main':   r['main_correct']  / nm if nm > 0 else 0.0,
            'acc_overall': (r['exit1_correct'] + r['main_correct']) / total_samples,
        }
    return final, total_samples


# ── 시각화 ────────────────────────────────────────────────────────────────────

def plot_results(results, out_dir, model_name):
    thresholds = sorted(results.keys())
    is_vee = model_name == 'vee'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Exit Rate & Accuracy — {model_name.upper()} ResNet-18', fontsize=13)

    # ── 1) Exit Rate ──
    ax = axes[0]
    if is_vee:
        ax.plot(thresholds, [results[t]['exit_rate'][0] for t in thresholds],
                marker='o', label='Exit1 (VEE1)', color='royalblue')
        ax.plot(thresholds, [results[t]['exit_rate'][1] for t in thresholds],
                marker='^', label='Main', color='green')
    else:
        ax.plot(thresholds, [results[t]['exit_rate'][0] for t in thresholds],
                marker='o', label='Exit1 (EE1)', color='royalblue')
        ax.plot(thresholds, [results[t]['exit_rate'][1] for t in thresholds],
                marker='s', label='Exit2 (EE2)', color='darkorange')
        ax.plot(thresholds, [results[t]['exit_rate'][2] for t in thresholds],
                marker='^', label='Exit3 (Main)', color='green')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Exit Rate (%)')
    ax.set_title('Exit Distribution by Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])

    # ── 2) Accuracy ──
    ax = axes[1]
    if is_vee:
        ax.plot(thresholds, [results[t]['acc_exit1'] for t in thresholds],
                marker='o', label='Exit1 Accuracy', color='royalblue', linestyle='--')
        ax.plot(thresholds, [results[t]['acc_main'] for t in thresholds],
                marker='^', label='Main Accuracy', color='green', linestyle='--')
    else:
        ax.plot(thresholds, [results[t]['acc_exit1'] for t in thresholds],
                marker='o', label='Exit1 Accuracy', color='royalblue', linestyle='--')
        ax.plot(thresholds, [results[t]['acc_exit2'] for t in thresholds],
                marker='s', label='Exit2 Accuracy', color='darkorange', linestyle='--')
        ax.plot(thresholds, [results[t]['acc_exit3'] for t in thresholds],
                marker='^', label='Exit3 Accuracy', color='green', linestyle='--')
    ax.plot(thresholds, [results[t]['acc_overall'] for t in thresholds],
            marker='D', label='Overall Accuracy', color='black', linewidth=2)
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"exit_rate_analysis_{model_name}.png")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"시각화 저장: {save_path}")


# ── 터미널 출력 ────────────────────────────────────────────────────────────────

def print_results_table(results, model_name):
    thresholds = sorted(results.keys())
    is_vee = model_name == 'vee'

    if is_vee:
        header = f"{'Threshold':<12} {'Exit1 Rate':<14} {'Main Rate':<14} {'Overall Acc':<14} {'Exit1 Acc':<14} {'Main Acc':<14}"
        sep = "=" * 82
    else:
        header = f"{'Threshold':<12} {'Exit1 Rate':<14} {'Exit2 Rate':<14} {'Exit3 Rate':<14} {'Overall Acc':<14} {'Exit1 Acc':<14} {'Exit2 Acc':<14} {'Exit3 Acc':<14}"
        sep = "=" * 110

    print(f"\n{sep}\n{header}\n{sep}")
    for t in thresholds:
        r = results[t]
        if is_vee:
            print(f"{t:<12.2f} "
                  f"{r['exit_rate'][0]:<14.1f} "
                  f"{r['exit_rate'][1]:<14.1f} "
                  f"{r['acc_overall']:<14.4f} "
                  f"{r['acc_exit1']:<14.4f} "
                  f"{r['acc_main']:<14.4f}")
        else:
            print(f"{t:<12.2f} "
                  f"{r['exit_rate'][0]:<14.1f} "
                  f"{r['exit_rate'][1]:<14.1f} "
                  f"{r['exit_rate'][2]:<14.1f} "
                  f"{r['acc_overall']:<14.4f} "
                  f"{r['acc_exit1']:<14.4f} "
                  f"{r['acc_exit2']:<14.4f} "
                  f"{r['acc_exit3']:<14.4f}")
    print(sep)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ee', 'vee'], default='ee',
                        help='평가할 모델 종류 (ee | vee)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='체크포인트 경로. 미지정 시 자동 선택')
    parser.add_argument('--threshold-range', type=float, nargs=3,
                        default=[0.50, 0.95, 0.05],
                        metavar=('MIN', 'MAX', 'STEP'))
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    ckpt_key = 'ee_resnet18' if args.model == 'ee' else 'vee_resnet18'
    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint(ckpt_key)
        if args.ckpt is None:
            print(f"[ERROR] {ckpt_key} 체크포인트 없음. --ckpt 로 직접 지정하세요.")
            sys.exit(1)
        print(f"자동 선택: {args.ckpt}")

    # ── 설정 / 디바이스 ──
    cfg = load_config(os.path.join(_SRC_DIR, 'configs/train.yaml'))
    num_classes = 10 if cfg['dataset']['name'].lower() == 'cifar10' else 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 모델 로드 ──
    print(f"{args.model.upper()} 모델 로드: {args.ckpt}")
    model = (build_ee if args.model == 'ee' else build_vee)(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    # ── 데이터로더 ──
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'],
        batch_size=64,
        data_root=cfg['dataset']['data_root'],
        num_workers=2,
        seed=cfg['train']['seed'],
    )

    # ── Threshold 범위 ──
    min_t, max_t, step_t = args.threshold_range
    thresholds = list(np.round(np.arange(min_t, max_t + step_t * 0.5, step_t), 4))
    print(f"Threshold 범위: {thresholds}")

    # ── 평가 ──
    print("\n평가 중...")
    if args.model == 'ee':
        results, total = evaluate_ee(model, test_loader, thresholds, device)
    else:
        results, total = evaluate_vee(model, test_loader, thresholds, device)

    # ── 출력 / 저장 ──
    print_results_table(results, args.model)

    out_dir = paths.eval_dir("exit_rate")
    plot_results(results, out_dir, args.model)

    json_path = os.path.join(out_dir, f"exit_rate_results_{args.model}.json")
    with open(json_path, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"JSON 저장: {json_path}")


if __name__ == '__main__':
    main()
