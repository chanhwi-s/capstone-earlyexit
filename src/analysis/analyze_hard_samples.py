"""
Hard Sample 분석 — EE / VEE 모델의 마지막 exit 샘플이 Plain에서도 어려운지 검증

분석 흐름:
  1. EE/VEE ResNet-18에서 threshold 적용 → 마지막 exit(main)으로 나가는 샘플 추출
  2. 해당 샘플들만 모아서 Plain ResNet-18으로 추론
  3. 두 모델의 accuracy/confidence 비교
  4. 시각화: 마지막 exit 샘플의 클래스별 분포, confidence 분포

의미:
  "마지막 exit으로 가는 샘플 = 어려운 샘플"이라는 가설 검증.
  Plain에서도 accuracy가 낮다면, EE/VEE 구조의 문제가 아니라
  본질적으로 어려운 샘플임을 증명할 수 있음.

사용법:
  cd src
  python analysis/analyze_hard_samples.py --model ee --threshold 0.80
  python analysis/analyze_hard_samples.py --model vee --threshold 0.80
  python analysis/analyze_hard_samples.py --model vee --vee-ckpt ... --plain-ckpt ... --threshold 0.80
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ee_resnet18 import build_model as build_ee
from models.vee_resnet18 import build_model as build_vee
from models.plain_resnet18 import build_model as build_plain
from datasets.dataloader import get_dataloader
from utils import load_config
import paths


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def extract_hard_samples_ee(ee_model, test_loader, threshold, device):
    """
    EE 모델(3-exit)에서 마지막 exit(exit3/main)까지 통과하는 샘플 추출.

    Returns:
        hard_samples: list of dicts
        total: int
        exit_counts: [exit1, exit2, exit3]
    """
    ee_model.eval()
    hard_samples = []
    exit_counts = [0, 0, 0]
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            out_ee1, out_ee2, out_main = ee_model(images, threshold=None)

            B = images.size(0)
            for i in range(B):
                total += 1

                conf1 = F.softmax(out_ee1[i:i+1], dim=1).max().item()
                conf2 = F.softmax(out_ee2[i:i+1], dim=1).max().item()
                conf3 = F.softmax(out_main[i:i+1], dim=1).max().item()

                if conf1 >= threshold:
                    exit_counts[0] += 1
                    continue
                if conf2 >= threshold:
                    exit_counts[1] += 1
                    continue

                exit_counts[2] += 1
                pred3 = out_main[i].argmax().item()

                hard_samples.append({
                    'image':    images[i].cpu(),
                    'label':    labels[i].item(),
                    'ee_pred':  pred3,
                    'ee_conf':  conf3,
                    'exit_idx': 3,
                    'exit_confs': [conf1, conf2, conf3],
                })

    return hard_samples, total, exit_counts


def extract_hard_samples_vee(vee_model, test_loader, threshold, device):
    """
    VEE 모델(2-exit)에서 마지막 exit(main)까지 통과하는 샘플 추출.

    Returns:
        hard_samples: list of dicts
        total: int
        exit_counts: [exit1, main]
    """
    vee_model.eval()
    hard_samples = []
    exit_counts = [0, 0]
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            out_ee1, out_main = vee_model(images, threshold=None)

            B = images.size(0)
            for i in range(B):
                total += 1

                conf1 = F.softmax(out_ee1[i:i+1], dim=1).max().item()
                conf_main = F.softmax(out_main[i:i+1], dim=1).max().item()

                if conf1 >= threshold:
                    exit_counts[0] += 1
                    continue

                exit_counts[1] += 1
                pred_main = out_main[i].argmax().item()

                hard_samples.append({
                    'image':    images[i].cpu(),
                    'label':    labels[i].item(),
                    'ee_pred':  pred_main,
                    'ee_conf':  conf_main,
                    'exit_idx': 2,
                    'exit_confs': [conf1, conf_main],
                })

    return hard_samples, total, exit_counts


def eval_plain_on_hard(plain_model, hard_samples, device):
    """Plain ResNet-18으로 hard sample들 추론."""
    plain_model.eval()
    results = []

    with torch.no_grad():
        for s in hard_samples:
            img = s['image'].unsqueeze(0).to(device)
            logits = plain_model(img)
            conf   = F.softmax(logits, dim=1).max().item()
            pred   = logits.argmax(dim=1).item()
            results.append({
                'label':       s['label'],
                'ee_pred':     s['ee_pred'],
                'ee_conf':     s['ee_conf'],
                'plain_pred':  pred,
                'plain_conf':  conf,
                'exit_confs':  s['exit_confs'],
            })

    return results


def analyze_and_report(results, exit_counts, total, out_dir, threshold, model_name):
    """분석 결과를 출력하고 시각화."""
    n_hard = len(results)
    if n_hard == 0:
        print("Hard sample 0개 — 모든 샘플이 조기 종료됨.")
        return

    ee_correct    = sum(1 for r in results if r['ee_pred']    == r['label'])
    plain_correct = sum(1 for r in results if r['plain_pred'] == r['label'])

    ee_acc    = ee_correct    / n_hard
    plain_acc = plain_correct / n_hard

    both_wrong = sum(
        1 for r in results
        if r['ee_pred'] != r['label'] and r['plain_pred'] != r['label']
    )
    both_wrong_rate = both_wrong / n_hard
    ee_only    = sum(1 for r in results if r['ee_pred'] == r['label'] and r['plain_pred'] != r['label'])
    plain_only = sum(1 for r in results if r['ee_pred'] != r['label'] and r['plain_pred'] == r['label'])

    print(f"\n{'='*60}")
    print(f"  Hard Sample Analysis  ({model_name.upper()}, threshold={threshold})")
    print(f"{'='*60}")
    print(f"  전체 테스트 샘플     : {total}")

    if len(exit_counts) == 3:
        labels_exit = ["Exit1 (EE1)", "Exit2 (EE2)", "Exit3 (Main=Hard)"]
    else:
        labels_exit = ["Exit1 (VEE1)", "Main (Hard)"]

    for lbl, cnt in zip(labels_exit, exit_counts):
        print(f"  {lbl:<20} : {cnt} ({cnt/total*100:.1f}%)")

    print()
    print(f"  Hard 샘플 수         : {n_hard}")
    print(f"  {model_name.upper():<5} Accuracy (Hard): {ee_acc:.4f}  ({ee_correct}/{n_hard})")
    print(f"  Plain Accuracy (Hard): {plain_acc:.4f}  ({plain_correct}/{n_hard})")
    print()
    print(f"  둘 다 틀림           : {both_wrong} ({both_wrong_rate*100:.1f}%)")
    print(f"  {model_name.upper()}만 맞춤          : {ee_only}")
    print(f"  Plain만 맞춤         : {plain_only}")
    print(f"{'='*60}")

    # ── JSON 저장 ──
    report = {
        'model':              model_name,
        'threshold':          threshold,
        'total_samples':      total,
        'exit_counts':        exit_counts,
        'hard_count':         n_hard,
        'ee_accuracy_hard':   ee_acc,
        'plain_accuracy_hard': plain_acc,
        'both_wrong':         both_wrong,
        'both_wrong_rate':    both_wrong_rate,
        'ee_only_correct':    ee_only,
        'plain_only_correct': plain_only,
    }
    json_path = os.path.join(out_dir, f"hard_sample_analysis_{model_name}_thr{threshold:.2f}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON 저장: {json_path}")

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f'Hard Sample Analysis  ({model_name.upper()}, threshold={threshold}, n_hard={n_hard})',
        fontsize=13
    )

    last_exit_label = "exit3 (Main)" if len(exit_counts) == 3 else "main"

    # 1) accuracy 비교
    ax = axes[0]
    bars = ax.bar([f'{model_name.upper()} ({last_exit_label})', 'Plain'],
                  [ee_acc * 100, plain_acc * 100],
                  color=['tomato', 'steelblue'], alpha=0.8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Hard Sample Accuracy')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis='y')

    # 2) confidence 분포
    ax = axes[1]
    ee_confs    = [r['ee_conf']    for r in results]
    plain_confs = [r['plain_conf'] for r in results]
    ax.hist(ee_confs,    bins=30, alpha=0.6, label=f'{model_name.upper()} ({last_exit_label})', color='tomato')
    ax.hist(plain_confs, bins=30, alpha=0.6, label='Plain', color='steelblue')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution (Hard Samples)')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) 클래스별 hard sample 수
    ax = axes[2]
    class_counts = {}
    for r in results:
        c = r['label']
        class_counts[c] = class_counts.get(c, 0) + 1

    classes = sorted(class_counts.keys())
    counts  = [class_counts[c] for c in classes]
    names   = [CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else str(c) for c in classes]

    ax.bar(names, counts, color='gray', alpha=0.6)
    ax.set_xlabel('Class')
    ax.set_ylabel('Hard Sample Count')
    ax.set_title('Hard Samples per Class')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"hard_sample_analysis_{model_name}_thr{threshold:.2f}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"그래프 저장: {png_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      type=str, choices=['ee', 'vee'], default='ee',
                        help='분석할 모델 종류 (ee | vee)')
    parser.add_argument('--ee-ckpt',    type=str, default=None)
    parser.add_argument('--vee-ckpt',   type=str, default=None)
    parser.add_argument('--plain-ckpt', type=str, default=None)
    parser.add_argument('--threshold',  type=float, default=0.80)
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    if args.model == 'ee':
        ckpt_key = 'ee_resnet18'
        ckpt_arg = args.ee_ckpt
    else:
        ckpt_key = 'vee_resnet18'
        ckpt_arg = args.vee_ckpt

    if ckpt_arg is None:
        ckpt_arg = paths.latest_checkpoint(ckpt_key)
        if ckpt_arg is None:
            print(f"[ERROR] {ckpt_key} 체크포인트 없음.")
            sys.exit(1)
        print(f"{args.model.upper()} 자동 선택: {ckpt_arg}")

    if args.plain_ckpt is None:
        args.plain_ckpt = paths.latest_checkpoint("plain_resnet18")
        if args.plain_ckpt is None:
            print("[ERROR] plain_resnet18 체크포인트 없음.")
            sys.exit(1)
        print(f"Plain 자동 선택: {args.plain_ckpt}")

    # ── 설정 / 디바이스 ──
    cfg = load_config('configs/train.yaml')
    num_classes = 10 if cfg['dataset']['name'].lower() == 'cifar10' else 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 모델 로드 ──
    print(f"\n{args.model.upper()} 모델 로드: {ckpt_arg}")
    if args.model == 'ee':
        model = build_ee(num_classes=num_classes)
    else:
        model = build_vee(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_arg, map_location=device))
    model.to(device).eval()

    print(f"Plain 모델 로드: {args.plain_ckpt}")
    plain_model = build_plain(num_classes=num_classes)
    plain_model.load_state_dict(torch.load(args.plain_ckpt, map_location=device))
    plain_model.to(device).eval()

    # ── 데이터 로더 ──
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'],
        batch_size=32,
        data_root=cfg['dataset']['data_root'],
        num_workers=2,
        seed=cfg['train']['seed'],
    )

    # ── 1단계: hard sample 추출 ──
    print(f"\nHard sample 추출 중... (threshold={args.threshold})")
    if args.model == 'ee':
        hard_samples, total, exit_counts = extract_hard_samples_ee(
            model, test_loader, args.threshold, device
        )
    else:
        hard_samples, total, exit_counts = extract_hard_samples_vee(
            model, test_loader, args.threshold, device
        )
    print(f"  추출 완료: {len(hard_samples)}개 / 전체 {total}개")

    # ── 2단계: Plain으로 평가 ──
    print("Plain 모델로 hard sample 추론 중...")
    results = eval_plain_on_hard(plain_model, hard_samples, device)

    # ── 3단계: 분석 & 시각화 ──
    out_dir = paths.eval_dir(f"hard_sample_analysis_{args.model}")
    analyze_and_report(results, exit_counts, total, out_dir, args.threshold, args.model)

    # ── 전체 테스트셋 Plain accuracy ──
    print("\n[참고] 전체 테스트셋 Plain accuracy 계산 중...")
    plain_model.eval()
    correct_all = total_all = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = plain_model(images)
            correct_all += logits.argmax(1).eq(labels).sum().item()
            total_all   += labels.size(0)
    full_acc = correct_all / total_all
    print(f"  Plain 전체 정확도: {full_acc:.4f}  ({correct_all}/{total_all})")
    print(f"  → Hard 샘플 subset 정확도와 비교하면 차이가 명확합니다.")


if __name__ == '__main__':
    main()
