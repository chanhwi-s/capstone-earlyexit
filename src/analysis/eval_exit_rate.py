"""
Exit Rate / Accuracy 평가 스크립트 — EE (3-exit) / VEE (2-exit) / EE50 (4-exit) 지원

threshold 범위를 sweep 하며:
  - 각 threshold에서 exit별 비율 (exit rate)
  - exit별 / 전체 accuracy
  - 시각화 + JSON 저장

PyTorch만으로 5090에서 실행 가능 (TRT 불필요).
학습 완료 직후 체크포인트로 바로 측정 가능.

사용법:
  cd src
  python analysis/eval_exit_rate.py --model ee
  python analysis/eval_exit_rate.py --model vee
  python analysis/eval_exit_rate.py --model ee50
  python analysis/eval_exit_rate.py --model ee50 --ckpt experiments/.../best.pth
  python analysis/eval_exit_rate.py --model ee50 --dataset imagenet --threshold-range 0.60 0.95 0.05
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
from models.ee_resnet50 import build_model as build_ee50
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


# ── EE50 (4-exit) 평가 ───────────────────────────────────────────────────────

def evaluate_ee50(model, test_loader, thresholds, device):
    """
    EE ResNet-50 (4-exit) 평가.
    학습된 모델의 forward()에서 모든 logit을 한 번에 얻은 뒤,
    threshold별로 exit 분기를 시뮬레이션 (배치 단위, 빠름).

    Returns:
        results: {threshold: {exit_rate:[r1,r2,r3,r4], acc_exit1..4, acc_overall}}
        total_samples: int
    """
    model.eval()
    results = {t: {'exit1_count': 0, 'exit1_correct': 0,
                   'exit2_count': 0, 'exit2_correct': 0,
                   'exit3_count': 0, 'exit3_correct': 0,
                   'main_count':  0, 'main_correct':  0}
               for t in thresholds}
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            total_samples += inputs.size(0)

            # 학습 모드 forward → 4개 logit 동시 획득 (threshold 없이)
            out1, out2, out3, out_main = model(inputs, threshold=None)

            conf1 = F.softmax(out1,      dim=1).max(dim=1).values
            conf2 = F.softmax(out2,      dim=1).max(dim=1).values
            conf3 = F.softmax(out3,      dim=1).max(dim=1).values
            pred1 = out1.argmax(dim=1)
            pred2 = out2.argmax(dim=1)
            pred3 = out3.argmax(dim=1)
            pred4 = out_main.argmax(dim=1)

            for t in thresholds:
                r = results[t]
                # exit1
                m1 = conf1 >= t
                r['exit1_count']   += m1.sum().item()
                r['exit1_correct'] += (pred1[m1] == labels[m1]).sum().item()

                # exit2: exit1 통과 못한 것 중
                m_no1 = ~m1
                m2 = m_no1 & (conf2 >= t)
                r['exit2_count']   += m2.sum().item()
                r['exit2_correct'] += (pred2[m2] == labels[m2]).sum().item()

                # exit3: exit1,2 통과 못한 것 중
                m_no2 = m_no1 & ~m2
                m3 = m_no2 & (conf3 >= t)
                r['exit3_count']   += m3.sum().item()
                r['exit3_correct'] += (pred3[m3] == labels[m3]).sum().item()

                # main: 나머지 전부
                m4 = m_no2 & ~m3
                r['main_count']   += m4.sum().item()
                r['main_correct'] += (pred4[m4] == labels[m4]).sum().item()

    final = {}
    for t in thresholds:
        r = results[t]
        n1 = r['exit1_count']; n2 = r['exit2_count']
        n3 = r['exit3_count']; n4 = r['main_count']
        total_correct = r['exit1_correct'] + r['exit2_correct'] + r['exit3_correct'] + r['main_correct']
        final[t] = {
            'exit_rate': [n1 / total_samples * 100,
                          n2 / total_samples * 100,
                          n3 / total_samples * 100,
                          n4 / total_samples * 100],
            'acc_exit1':   r['exit1_correct'] / n1 if n1 > 0 else 0.0,
            'acc_exit2':   r['exit2_correct'] / n2 if n2 > 0 else 0.0,
            'acc_exit3':   r['exit3_correct'] / n3 if n3 > 0 else 0.0,
            'acc_main':    r['main_correct']  / n4 if n4 > 0 else 0.0,
            'acc_overall': total_correct / total_samples,
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

    # 모델별 exit 레이블 / 색상 정의
    if model_name == 'vee':
        exit_labels  = ['Exit1 (layer1)', 'Main (layer4)']
        acc_keys     = ['acc_exit1', 'acc_main']
        colors       = ['royalblue', 'green']
        markers      = ['o', '^']
        title_suffix = 'VEE ResNet-18'
    elif model_name == 'ee50':
        exit_labels  = ['Exit1 (layer1)', 'Exit2 (layer2)', 'Exit3 (layer3)', 'Main (layer4)']
        acc_keys     = ['acc_exit1', 'acc_exit2', 'acc_exit3', 'acc_main']
        colors       = ['royalblue', 'darkorange', 'purple', 'green']
        markers      = ['o', 's', 'P', '^']
        title_suffix = 'EE ResNet-50 (4-exit)'
    else:  # ee
        exit_labels  = ['Exit1 (layer2)', 'Exit2 (layer3)', 'Main (layer4)']
        acc_keys     = ['acc_exit1', 'acc_exit2', 'acc_exit3']
        colors       = ['royalblue', 'darkorange', 'green']
        markers      = ['o', 's', '^']
        title_suffix = 'EE ResNet-18'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Exit Rate & Accuracy — {title_suffix}', fontsize=13)

    # ── 1) Exit Rate ──
    ax = axes[0]
    for i, (label, color, marker) in enumerate(zip(exit_labels, colors, markers)):
        ax.plot(thresholds, [results[t]['exit_rate'][i] for t in thresholds],
                marker=marker, label=label, color=color)
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Exit Rate (%)')
    ax.set_title('Exit Distribution by Threshold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])

    # ── 2) Accuracy ──
    ax = axes[1]
    for key, label, color, marker in zip(acc_keys, exit_labels, colors, markers):
        ax.plot(thresholds, [results[t][key] for t in thresholds],
                marker=marker, label=label, color=color, linestyle='--')
    ax.plot(thresholds, [results[t]['acc_overall'] for t in thresholds],
            marker='D', label='Overall', color='black', linewidth=2)
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Threshold')
    ax.legend(fontsize=8)
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

    if model_name == 'vee':
        header = (f"{'Thr':<8} {'E1%':<8} {'Main%':<8} "
                  f"{'Overall':<10} {'E1 Acc':<10} {'Main Acc':<10}")
        sep = "=" * 56
        def row(t, r):
            return (f"{t:<8.2f} {r['exit_rate'][0]:<8.1f} {r['exit_rate'][1]:<8.1f} "
                    f"{r['acc_overall']:<10.4f} {r['acc_exit1']:<10.4f} {r['acc_main']:<10.4f}")
    elif model_name == 'ee50':
        header = (f"{'Thr':<8} {'E1%':<8} {'E2%':<8} {'E3%':<8} {'Main%':<8} "
                  f"{'Overall':<10} {'E1 Acc':<10} {'E2 Acc':<10} {'E3 Acc':<10} {'Main Acc':<10}")
        sep = "=" * 90
        def row(t, r):
            return (f"{t:<8.2f} "
                    f"{r['exit_rate'][0]:<8.1f} {r['exit_rate'][1]:<8.1f} "
                    f"{r['exit_rate'][2]:<8.1f} {r['exit_rate'][3]:<8.1f} "
                    f"{r['acc_overall']:<10.4f} {r['acc_exit1']:<10.4f} "
                    f"{r['acc_exit2']:<10.4f} {r['acc_exit3']:<10.4f} {r['acc_main']:<10.4f}")
    else:  # ee
        header = (f"{'Thr':<8} {'E1%':<8} {'E2%':<8} {'Main%':<8} "
                  f"{'Overall':<10} {'E1 Acc':<10} {'E2 Acc':<10} {'Main Acc':<10}")
        sep = "=" * 74
        def row(t, r):
            return (f"{t:<8.2f} "
                    f"{r['exit_rate'][0]:<8.1f} {r['exit_rate'][1]:<8.1f} {r['exit_rate'][2]:<8.1f} "
                    f"{r['acc_overall']:<10.4f} {r['acc_exit1']:<10.4f} "
                    f"{r['acc_exit2']:<10.4f} {r['acc_exit3']:<10.4f}")

    print(f"\n{sep}\n{header}\n{sep}")
    for t in thresholds:
        print(row(t, results[t]))
    print(sep)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ee', 'vee', 'ee50'], default='ee',
                        help='평가할 모델 종류 (ee | vee | ee50)')
    parser.add_argument('--ckpt',    type=str, default=None,
                        help='체크포인트 경로. 미지정 시 자동 선택')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['cifar10', 'imagenet'],
                        help='데이터셋. 미지정 시 configs/train.yaml 참조')
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threshold-range', type=float, nargs=3,
                        default=[0.50, 0.95, 0.05],
                        metavar=('MIN', 'MAX', 'STEP'))
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    ckpt_map = {'ee': 'ee_resnet18', 'vee': 'vee_resnet18', 'ee50': 'ee_resnet50'}
    ckpt_key = ckpt_map[args.model]
    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint(ckpt_key)
        if args.ckpt is None:
            print(f"[ERROR] {ckpt_key} 체크포인트 없음. --ckpt 로 직접 지정하세요.")
            sys.exit(1)
        print(f"자동 선택: {args.ckpt}")

    # ── 설정 / 디바이스 ──
    cfg = load_config(os.path.join(_SRC_DIR, 'configs/train.yaml'))
    dataset_name = (args.dataset or cfg['dataset']['name']).lower()
    data_root    = args.data_root
    if data_root is None:
        if dataset_name == 'imagenet' and 'imagenet' in cfg:
            data_root = cfg['imagenet']['data_root']
        else:
            data_root = cfg['dataset']['data_root']
    num_classes  = 1000 if dataset_name == 'imagenet' else 10
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 모델 로드 ──
    print(f"{args.model.upper()} 모델 로드: {args.ckpt}  (dataset={dataset_name})")
    build_fn = {'ee': build_ee, 'vee': build_vee, 'ee50': build_ee50}[args.model]
    model = build_fn(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    # ── 데이터로더 ──
    _, test_loader, _ = get_dataloader(
        dataset=dataset_name,
        batch_size=args.batch_size,
        data_root=data_root,
        num_workers=4,
        seed=cfg['train']['seed'],
    )

    # ── Threshold 범위 ──
    min_t, max_t, step_t = args.threshold_range
    thresholds = list(np.round(np.arange(min_t, max_t + step_t * 0.5, step_t), 4))
    print(f"Threshold 범위: {thresholds}")

    # ── 평가 ──
    print("\n평가 중...")
    eval_fn = {'ee': evaluate_ee, 'vee': evaluate_vee, 'ee50': evaluate_ee50}[args.model]
    results, total = eval_fn(model, test_loader, thresholds, device)
    print(f"총 샘플 수: {total}")

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
