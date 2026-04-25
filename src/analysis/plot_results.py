"""
SelectiveExitViT 학습 결과 시각화

사용법:
  cd src
  python analysis/plot_results.py                              # 최신 ee_vit_2exit 자동 선택
  python analysis/plot_results.py --model ee_vit_3exit        # 3-exit
  python analysis/plot_results.py --model plain_vit           # PlainViT
  python analysis/plot_results.py /path/to/train_dir          # 직접 경로 지정
"""

import os
import sys
import csv
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths


def load_csv(log_path: str) -> list[dict]:
    with open(log_path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def detect_n_exits(rows: list[dict]) -> int:
    cols = rows[0].keys()
    if "train_acc_exit2" in cols:
        return 3
    if "train_acc_exit1" in cols:
        return 2
    return 0  # plain


def plot(train_dir: str):
    log_path = os.path.join(train_dir, "train_log.csv")
    if not os.path.exists(log_path):
        print(f"[ERROR] 로그 파일 없음: {log_path}")
        sys.exit(1)

    rows   = load_csv(log_path)
    epochs = [r["epoch"] for r in rows]
    n_exits = detect_n_exits(rows)

    model_label = {0: "PlainViT", 2: "SelectiveExitViT (2-exit)", 3: "SelectiveExitViT (3-exit)"}[n_exits]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Results — {model_label}\n{os.path.basename(train_dir)}", fontsize=13)

    # 1) Loss
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in rows], label="Train Loss", color="steelblue")
    ax.plot(epochs, [r["test_loss"]  for r in rows], label="Test Loss",  color="tomato")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) Train Accuracy
    ax = axes[0, 1]
    colors = ["royalblue", "darkorange", "green"]
    if n_exits >= 2:
        for i in range(n_exits - 1):
            ax.plot(epochs, [r[f"train_acc_exit{i+1}"] for r in rows],
                    label=f"Train Exit{i+1}", color=colors[i], linestyle="--")
    key_last = "train_acc" if n_exits == 0 else "train_acc_main"
    ax.plot(epochs, [r[key_last] for r in rows], label="Train Main", color=colors[n_exits - 1] if n_exits else "steelblue")
    ax.set_title("Train Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(); ax.grid(alpha=0.3)

    # 3) Test Accuracy
    ax = axes[1, 0]
    if n_exits >= 2:
        for i in range(n_exits - 1):
            ax.plot(epochs, [r[f"test_acc_exit{i+1}"] for r in rows],
                    label=f"Test Exit{i+1}", color=colors[i], linestyle="--")
    key_last = "test_acc" if n_exits == 0 else "test_acc_main"
    ax.plot(epochs, [r[key_last] for r in rows], label="Test Main", color=colors[n_exits - 1] if n_exits else "tomato")
    ax.set_title("Test Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(); ax.grid(alpha=0.3)

    # 4) Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, [r["lr"] for r in rows], color="purple")
    ax.set_title("Learning Rate"); ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(train_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장 완료: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", nargs="?", default=None,
                        help="train 디렉토리 경로 (기본: 최신 실험 자동 선택)")
    parser.add_argument("--model", type=str, default="ee_vit_2exit",
                        choices=["ee_vit_2exit", "ee_vit_3exit", "plain_vit"],
                        help="자동 선택할 모델명 (기본: ee_vit_2exit)")
    args = parser.parse_args()

    if args.train_dir:
        train_dir = args.train_dir
    else:
        train_dir = paths.latest_train_dir(args.model)
        if train_dir is None:
            print(f"[ERROR] {args.model} 학습 디렉토리 없음.")
            sys.exit(1)
        print(f"자동 선택: {train_dir}")

    plot(train_dir)
