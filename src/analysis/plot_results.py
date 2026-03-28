"""
학습 결과 시각화 스크립트
EE / Plain / VEE 모델 CSV 컬럼 자동 감지

사용법:
  python plot_results.py                                         # 최근 EE 실험 자동 선택
  python plot_results.py /절대경로/run_20260328_120955          # 직접 경로 지정
  python plot_results.py --plain                                # 최근 plain 실험 자동 선택
  python plot_results.py --vee                                  # 최근 VEE 실험 자동 선택
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths


def load_csv(log_path):
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def detect_model_type(rows):
    """CSV 컬럼으로 모델 종류 자동 판단: 'ee' / 'vee' / 'plain'"""
    cols = rows[0].keys()
    if "train_acc_ee2" in cols:
        return "ee"    # EE: ee1 + ee2 + main
    elif "train_acc_ee1" in cols:
        return "vee"   # VEE: ee1 + main (ee2 없음)
    else:
        return "plain"


def plot(exp_dir):
    log_path = os.path.join(exp_dir, "train_log.csv")
    if not os.path.exists(log_path):
        print(f"[ERROR] 로그 파일 없음: {log_path}")
        sys.exit(1)

    rows       = load_csv(log_path)
    epochs     = [r["epoch"] for r in rows]
    model_type = detect_model_type(rows)

    train_loss = [r["train_loss"] for r in rows]
    test_loss  = [r["test_loss"]  for r in rows]
    lr         = [r["lr"]         for r in rows]

    titles = {"ee": "EE ResNet-18", "vee": "VEE ResNet-18", "plain": "Plain ResNet-18"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Results — {titles[model_type]}\n{os.path.basename(exp_dir)}", fontsize=13)

    # 1) Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train Loss", color="steelblue")
    ax.plot(epochs, test_loss,  label="Test Loss",  color="tomato")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) Train Accuracy
    ax = axes[0, 1]
    if model_type == "ee":
        ax.plot(epochs, [r["train_acc_ee1"]  for r in rows], label="Train EE1",  color="royalblue",  linestyle="--")
        ax.plot(epochs, [r["train_acc_ee2"]  for r in rows], label="Train EE2",  color="darkorange", linestyle="--")
        ax.plot(epochs, [r["train_acc_main"] for r in rows], label="Train Main", color="green")
    elif model_type == "vee":
        ax.plot(epochs, [r["train_acc_ee1"]  for r in rows], label="Train EE1 (layer1)", color="royalblue", linestyle="--")
        ax.plot(epochs, [r["train_acc_main"] for r in rows], label="Train Main",          color="green")
    else:
        ax.plot(epochs, [r["train_acc"] for r in rows], label="Train Acc", color="steelblue")
    ax.set_title("Train Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(); ax.grid(alpha=0.3)

    # 3) Test Accuracy
    ax = axes[1, 0]
    if model_type == "ee":
        ax.plot(epochs, [r["test_acc_ee1"]  for r in rows], label="Test EE1",  color="royalblue",  linestyle="--")
        ax.plot(epochs, [r["test_acc_ee2"]  for r in rows], label="Test EE2",  color="darkorange", linestyle="--")
        ax.plot(epochs, [r["test_acc_main"] for r in rows], label="Test Main", color="green")
    elif model_type == "vee":
        ax.plot(epochs, [r["test_acc_ee1"]  for r in rows], label="Test EE1 (layer1)", color="royalblue", linestyle="--")
        ax.plot(epochs, [r["test_acc_main"] for r in rows], label="Test Main",          color="green")
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
                        help="실험 디렉토리 경로 (절대경로 권장)")
    parser.add_argument("--plain", action="store_true",
                        help="최근 plain_resnet18 실험 자동 선택")
    parser.add_argument("--vee", action="store_true",
                        help="최근 vee_resnet18 실험 자동 선택")
    args = parser.parse_args()

    if args.exp_dir:
        exp_dir = args.exp_dir
    elif args.plain:
        exp_dir = paths.latest_train_dir("plain_resnet18")
        if exp_dir is None:
            print("[ERROR] plain_resnet18 학습 디렉토리 없음. 먼저 train_plain.py를 실행하세요.")
            sys.exit(1)
        print(f"자동 선택 (plain): {exp_dir}")
    elif args.vee:
        exp_dir = paths.latest_train_dir("vee_resnet18")
        if exp_dir is None:
            print("[ERROR] vee_resnet18 학습 디렉토리 없음. 먼저 train_vee.py를 실행하세요.")
            sys.exit(1)
        print(f"자동 선택 (VEE): {exp_dir}")
    else:
        exp_dir = paths.latest_train_dir("ee_resnet18")
        if exp_dir is None:
            print("[ERROR] ee_resnet18 학습 디렉토리 없음.")
            sys.exit(1)
        print(f"자동 선택 (EE): {exp_dir}")

    plot(exp_dir)
