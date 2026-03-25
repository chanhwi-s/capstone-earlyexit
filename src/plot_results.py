"""
학습 결과 시각화 스크립트
EE 모델 / Plain 모델 CSV 컬럼 자동 감지

사용법:
  python plot_results.py                                      # 최근 EE 실험 자동 선택
  python plot_results.py experiments/2026-03-24_14-44-14     # EE 실험 지정
  python plot_results.py --plain                             # 최근 plain 실험 자동 선택
  python plot_results.py experiments_plain/2026-03-25_...   # plain 실험 지정
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(log_path):
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def is_ee_csv(rows):
    """CSV 컬럼 보고 EE 모델 여부 자동 판단"""
    return "train_acc_ee1" in rows[0]


def plot(exp_dir):
    log_path = os.path.join(exp_dir, "train_log.csv")
    if not os.path.exists(log_path):
        print(f"[ERROR] 로그 파일 없음: {log_path}")
        sys.exit(1)

    rows   = load_csv(log_path)
    epochs = [r["epoch"] for r in rows]
    is_ee  = is_ee_csv(rows)

    train_loss = [r["train_loss"] for r in rows]
    test_loss  = [r["test_loss"]  for r in rows]
    lr         = [r["lr"]         for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_type = "EE ResNet-18" if is_ee else "Plain ResNet-18"
    fig.suptitle(f"Training Results — {model_type}\n{os.path.basename(exp_dir)}", fontsize=13)

    # 1) Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train Loss", color="steelblue")
    ax.plot(epochs, test_loss,  label="Test Loss",  color="tomato")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) Train Accuracy
    ax = axes[0, 1]
    if is_ee:
        ax.plot(epochs, [r["train_acc_ee1"]  for r in rows], label="Train EE1",  color="royalblue",  linestyle="--")
        ax.plot(epochs, [r["train_acc_ee2"]  for r in rows], label="Train EE2",  color="darkorange", linestyle="--")
        ax.plot(epochs, [r["train_acc_main"] for r in rows], label="Train Main", color="green")
    else:
        ax.plot(epochs, [r["train_acc"] for r in rows], label="Train Acc", color="steelblue")
    ax.set_title("Train Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(); ax.grid(alpha=0.3)

    # 3) Test Accuracy
    ax = axes[1, 0]
    if is_ee:
        ax.plot(epochs, [r["test_acc_ee1"]  for r in rows], label="Test EE1",  color="royalblue",  linestyle="--")
        ax.plot(epochs, [r["test_acc_ee2"]  for r in rows], label="Test EE2",  color="darkorange", linestyle="--")
        ax.plot(epochs, [r["test_acc_main"] for r in rows], label="Test Main", color="green")
    else:
        ax.plot(epochs, [r["test_acc"] for r in rows], label="Test Acc", color="tomato")
    ax.set_title("Test Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(); ax.grid(alpha=0.3)

    # 4) Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, lr, color="purple")
    ax.set_title("Learning Rate (CosineAnnealingLR)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(exp_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"저장 완료: {save_path}")
    plt.show()


def latest_exp(base):
    if not os.path.exists(base):
        return None
    dirs = sorted([
        os.path.join(base, d) for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])
    return dirs[-1] if dirs else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", nargs="?", default=None,
                        help="실험 디렉토리 경로 (미지정 시 자동 선택)")
    parser.add_argument("--plain", action="store_true",
                        help="experiments_plain/ 에서 자동 선택")
    args = parser.parse_args()

    if args.exp_dir:
        exp_dir = args.exp_dir
    elif args.plain:
        exp_dir = latest_exp("experiments_plain")
        if exp_dir is None:
            print("[ERROR] experiments_plain/ 없음. 먼저 train_plain.py를 실행하세요.")
            sys.exit(1)
        print(f"자동 선택 (plain): {exp_dir}")
    else:
        exp_dir = latest_exp("experiments")
        if exp_dir is None:
            print("[ERROR] experiments/ 없음.")
            sys.exit(1)
        print(f"자동 선택 (EE): {exp_dir}")

    plot(exp_dir)
