"""
학습 결과 시각화 스크립트
사용법: python plot_results.py [exp_dir]
예시:  python plot_results.py experiments/2026-03-24_14-44-14
       python plot_results.py  (자동으로 가장 최근 실험 선택)
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(log_path):
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot(exp_dir):
    log_path = os.path.join(exp_dir, "train_log.csv")
    if not os.path.exists(log_path):
        print(f"[ERROR] 로그 파일 없음: {log_path}")
        sys.exit(1)

    rows  = load_csv(log_path)
    epochs = [r["epoch"] for r in rows]

    # ── 데이터 추출 ──────────────────────────────────────────────
    train_loss     = [r["train_loss"]     for r in rows]
    test_loss      = [r["test_loss"]      for r in rows]

    train_ee1      = [r["train_acc_ee1"]  for r in rows]
    train_ee2      = [r["train_acc_ee2"]  for r in rows]
    train_main     = [r["train_acc_main"] for r in rows]

    test_ee1       = [r["test_acc_ee1"]   for r in rows]
    test_ee2       = [r["test_acc_ee2"]   for r in rows]
    test_main      = [r["test_acc_main"]  for r in rows]

    lr             = [r["lr"]             for r in rows]

    # ── 플롯 ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Results\n{os.path.basename(exp_dir)}", fontsize=13)

    # 1) Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="Train Loss", color="steelblue")
    ax.plot(epochs, test_loss,  label="Test Loss",  color="tomato")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) Train Accuracy (3 exits)
    ax = axes[0, 1]
    ax.plot(epochs, train_ee1,  label="Train EE1",  color="royalblue",  linestyle="--")
    ax.plot(epochs, train_ee2,  label="Train EE2",  color="darkorange", linestyle="--")
    ax.plot(epochs, train_main, label="Train Main", color="green")
    ax.set_title("Train Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(); ax.grid(alpha=0.3)

    # 3) Test Accuracy (3 exits)
    ax = axes[1, 0]
    ax.plot(epochs, test_ee1,  label="Test EE1",  color="royalblue",  linestyle="--")
    ax.plot(epochs, test_ee2,  label="Test EE2",  color="darkorange", linestyle="--")
    ax.plot(epochs, test_main, label="Test Main", color="green")
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        # 가장 최근 실험 자동 선택
        base = "experiments"
        dirs = sorted([
            os.path.join(base, d) for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ])
        if not dirs:
            print("[ERROR] experiments/ 디렉토리가 비어있음")
            sys.exit(1)
        exp_dir = dirs[-1]
        print(f"자동 선택된 실험: {exp_dir}")

    plot(exp_dir)
