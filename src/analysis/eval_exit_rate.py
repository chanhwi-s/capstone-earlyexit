"""
Exit Rate 평가 스크립트

학습된 모델을 여러 threshold 값으로 테스트해서:
- 각 threshold에서 샘플별 exit 위치 분포
- threshold별 exit rate (%)
- 조기 종료 시 정확도 변화

사용법:
  python eval_exit_rate.py --ckpt experiments/.../best.pth --threshold-range 0.5 0.99 0.05
  python eval_exit_rate.py  (자동으로 최근 best.pth 선택)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ee_resnet18 import build_model
from datasets.dataloader import get_dataloader
from utils import load_config
import paths


def evaluate_with_threshold(model, test_loader, thresholds, device, num_classes):
    """
    여러 threshold로 테스트해서 exit rate 및 정확도 측정

    Returns:
        results: dict
            {
                threshold: {
                    'exit_rate': [%, exit1, exit2, exit3],  # exit1: ee1에서 나간 비율
                    'acc_exit1': ee1에서 나간 샘플들의 정확도,
                    'acc_exit2': ee2에서 나간 샘플들의 정확도,
                    'acc_exit3': main에서 나간 샘플들의 정확도,
                    'acc_overall': 전체 정확도 (모든 exit 통합)
                }
            }
    """
    model.eval()
    results = {}

    total_samples = 0
    total_correct = 0

    # 각 exit별로 정보 저장 (모든 threshold 통합)
    exit1_correct = 0
    exit1_total = 0
    exit2_correct = 0
    exit2_total = 0
    exit3_correct = 0
    exit3_total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            # threshold=None: 학습 모드 (3개 출력 모두)
            logits_ee1, logits_ee2, logits_main = model(inputs, threshold=None)

            # 각 threshold별로 처리
            for threshold in thresholds:
                if threshold not in results:
                    results[threshold] = {
                        'exit1_count': 0,
                        'exit2_count': 0,
                        'exit3_count': 0,
                        'exit1_correct': 0,
                        'exit2_correct': 0,
                        'exit3_correct': 0,
                    }

                # exit1 confidence 확인
                conf_ee1 = F.softmax(logits_ee1, dim=1).max(dim=1).values  # (B,)
                pred_ee1 = logits_ee1.argmax(dim=1)

                mask_exit1 = (conf_ee1 >= threshold)  # (B,)
                results[threshold]['exit1_count'] += mask_exit1.sum().item()
                results[threshold]['exit1_correct'] += (pred_ee1[mask_exit1] == labels[mask_exit1]).sum().item()

                # exit1에서 나가지 않은 샘플들만 exit2 검사
                mask_continue1 = ~mask_exit1
                if mask_continue1.sum() > 0:
                    conf_ee2 = F.softmax(logits_ee2[mask_continue1], dim=1).max(dim=1).values
                    pred_ee2 = logits_ee2[mask_continue1].argmax(dim=1)

                    mask_exit2 = (conf_ee2 >= threshold)
                    results[threshold]['exit2_count'] += mask_exit2.sum().item()
                    results[threshold]['exit2_correct'] += (pred_ee2[mask_exit2] == labels[mask_continue1][mask_exit2]).sum().item()

                    # exit2에서도 나가지 않은 샘플들만 exit3
                    mask_continue2 = ~mask_exit2
                    if mask_continue2.sum() > 0:
                        pred_main = logits_main[mask_continue1][mask_continue2].argmax(dim=1)
                        results[threshold]['exit3_count'] += mask_continue2.sum().item()
                        results[threshold]['exit3_correct'] += (pred_main == labels[mask_continue1][mask_continue2]).sum().item()
                else:
                    # 모두 exit1에서 나감
                    results[threshold]['exit2_count'] += 0
                    results[threshold]['exit3_count'] += 0

    # 정확도 계산
    for threshold in thresholds:
        r = results[threshold]

        # 각 exit별 정확도
        acc_exit1 = r['exit1_correct'] / r['exit1_count'] if r['exit1_count'] > 0 else 0.0
        acc_exit2 = r['exit2_correct'] / r['exit2_count'] if r['exit2_count'] > 0 else 0.0
        acc_exit3 = r['exit3_correct'] / r['exit3_count'] if r['exit3_count'] > 0 else 0.0

        # 전체 정확도
        total_correct = r['exit1_correct'] + r['exit2_correct'] + r['exit3_correct']
        acc_overall = total_correct / total_samples

        # Exit rate (%)
        exit_rate_1 = r['exit1_count'] / total_samples * 100
        exit_rate_2 = r['exit2_count'] / total_samples * 100
        exit_rate_3 = r['exit3_count'] / total_samples * 100

        results[threshold] = {
            'exit_rate': [exit_rate_1, exit_rate_2, exit_rate_3],
            'acc_exit1': acc_exit1,
            'acc_exit2': acc_exit2,
            'acc_exit3': acc_exit3,
            'acc_overall': acc_overall,
        }

    return results, total_samples


def plot_results(results, out_dir):
    """시각화"""
    thresholds = sorted(results.keys())

    exit_rates_1 = [results[t]['exit_rate'][0] for t in thresholds]
    exit_rates_2 = [results[t]['exit_rate'][1] for t in thresholds]
    exit_rates_3 = [results[t]['exit_rate'][2] for t in thresholds]

    acc_exit1 = [results[t]['acc_exit1'] for t in thresholds]
    acc_exit2 = [results[t]['acc_exit2'] for t in thresholds]
    acc_exit3 = [results[t]['acc_exit3'] for t in thresholds]
    acc_overall = [results[t]['acc_overall'] for t in thresholds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Exit Rate vs Threshold
    ax = axes[0]
    ax.plot(thresholds, exit_rates_1, marker='o', label='Exit1 (EE1)', color='royalblue')
    ax.plot(thresholds, exit_rates_2, marker='s', label='Exit2 (EE2)', color='darkorange')
    ax.plot(thresholds, exit_rates_3, marker='^', label='Exit3 (Main)', color='green')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Exit Rate (%)')
    ax.set_title('Exit Distribution by Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])

    # 2) Accuracy by Exit & Overall
    ax = axes[1]
    ax.plot(thresholds, acc_exit1, marker='o', label='Exit1 Accuracy', color='royalblue', linestyle='--')
    ax.plot(thresholds, acc_exit2, marker='s', label='Exit2 Accuracy', color='darkorange', linestyle='--')
    ax.plot(thresholds, acc_exit3, marker='^', label='Exit3 Accuracy', color='green', linestyle='--')
    ax.plot(thresholds, acc_overall, marker='D', label='Overall Accuracy', color='black', linewidth=2)
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    save_path = os.path.join(out_dir, "exit_rate_analysis.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n시각화 저장: {save_path}")
    plt.show()


def print_results_table(results, total_samples):
    """테이블 출력"""
    print("\n" + "="*120)
    print(f"{'Threshold':<12} {'Exit1 Rate':<15} {'Exit2 Rate':<15} {'Exit3 Rate':<15} {'Overall Acc':<15} {'Exit1 Acc':<15} {'Exit2 Acc':<15} {'Exit3 Acc':<15}")
    print("="*120)

    for threshold in sorted(results.keys()):
        r = results[threshold]
        print(
            f"{threshold:<12.2f} "
            f"{r['exit_rate'][0]:<15.2f}% "
            f"{r['exit_rate'][1]:<15.2f}% "
            f"{r['exit_rate'][2]:<15.2f}% "
            f"{r['acc_overall']:<15.4f} "
            f"{r['acc_exit1']:<15.4f} "
            f"{r['acc_exit2']:<15.4f} "
            f"{r['acc_exit3']:<15.4f}"
        )
    print("="*120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="체크포인트 경로. 미지정 시 자동 선택")
    parser.add_argument("--threshold-range", type=float, nargs=3,
                        default=[0.50, 0.95, 0.05],
                        metavar=("MIN", "MAX", "STEP"),
                        help="threshold 범위 (default: 0.50 0.95 0.05)")
    args = parser.parse_args()

    # ── 체크포인트 자동 선택 ──
    if args.ckpt is None:
        args.ckpt = paths.latest_checkpoint("ee_resnet18")
        if args.ckpt is None:
            print("[ERROR] ee_resnet18 체크포인트 없음.")
            print("--ckpt로 직접 지정하세요: python eval_exit_rate.py --ckpt /path/to/best.pth")
            sys.exit(1)
        print(f"자동 선택 체크포인트: {args.ckpt}")

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] 파일 없음: {args.ckpt}")
        sys.exit(1)

    # ── 설정 및 모델 로드 ──
    cfg = load_config("configs/train.yaml")
    num_classes = 10 if cfg["dataset"]["name"].lower() == "cifar10" else 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()
    print(f"모델 로드 완료: {args.ckpt}")

    # ── 데이터로더 ──
    _, test_loader, _ = get_dataloader(
        dataset=cfg["dataset"]["name"],
        batch_size=1,  # batch_size=1로 샘플별 exit 추적
        data_root=cfg["dataset"]["data_root"],
        num_workers=0,
        seed=cfg["train"]["seed"]
    )

    # ── Threshold 범위 생성 ──
    min_t, max_t, step_t = args.threshold_range
    thresholds = np.arange(min_t, max_t + step_t, step_t)
    print(f"\nThreshold 범위: {thresholds}")

    # ── 평가 ──
    print("\n평가 중...")
    results, total_samples = evaluate_with_threshold(model, test_loader, thresholds, device, num_classes)

    # ── 결과 출력 ──
    print_results_table(results, total_samples)

    # ── 시각화 ──
    out_dir = paths.eval_dir("exit_rate")
    plot_results(results, out_dir)

    # ── JSON 저장 (나중 분석용) ──
    import json
    result_path = os.path.join(out_dir, "exit_rate_results.json")
    json.dump(
        {str(k): {
            'exit_rate': v['exit_rate'],
            'acc_exit1': v['acc_exit1'],
            'acc_exit2': v['acc_exit2'],
            'acc_exit3': v['acc_exit3'],
            'acc_overall': v['acc_overall'],
        } for k, v in results.items()},
        open(result_path, 'w'),
        indent=2
    )
    print(f"결과 JSON 저장: {result_path}")


if __name__ == "__main__":
    main()
