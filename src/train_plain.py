"""
Plain ResNet-18 학습 스크립트
EE ResNet-18과 동일한 하이퍼파라미터로 학습 (공정한 비교를 위해)

사용법:
  cd src
  python train_plain.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from models.plain_resnet18 import build_model
from datasets.dataloader import get_dataloader
from engine.trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import shutil
from datetime import datetime


# ── config 로드 ───────────────────────────────────────────────────────────────
config_path = "configs/train.yaml"
cfg = load_config(config_path)

dataset     = cfg["dataset"]["name"]
data_root   = cfg["dataset"]["data_root"]
num_workers = cfg["dataset"]["num_workers"]

batch_size  = cfg["train"]["batch_size"]
epochs      = cfg["train"]["epochs"]
seed        = cfg["train"]["seed"]

lr           = float(cfg["optimizer"]["lr"])
momentum     = float(cfg["optimizer"]["momentum"])
weight_decay = float(cfg["optimizer"]["weight_decay"])

T_max   = cfg["scheduler"]["T_max"]
eta_min = float(cfg["scheduler"]["eta_min"])


def create_exp_dir(config_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir   = os.path.join("experiments_plain", timestamp)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
    return exp_dir


def train():
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir  = create_exp_dir(config_path)
    log_path = os.path.join(exp_dir, "train_log.csv")
    print(f"Experiment dir : {exp_dir}")
    print(f"Log file       : {log_path}\n")

    train_loader, test_loader, num_classes = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed
    )

    model     = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    best_test_acc = 0.0

    for epoch in range(epochs):

        # plain 모델은 단일 출력 → weights 불필요
        train_loss, _, _, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_loss, _, _, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[{epoch+1:3d}/{epochs}] lr={current_lr:.5f} | "
            f"train loss={train_loss:.4f}  acc={train_acc:.4f} | "
            f"test  loss={test_loss:.4f}  acc={test_acc:.4f}"
        )

        log_to_csv(log_path, epoch + 1, {
            "lr":         current_lr,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "test_loss":  test_loss,
            "test_acc":   test_acc,
        })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                model.state_dict(),
                os.path.join(exp_dir, "checkpoints", "best.pth")
            )

        torch.save(
            model.state_dict(),
            os.path.join(exp_dir, "checkpoints", f"epoch_{epoch+1}.pth")
        )

    torch.save(
        model.state_dict(),
        os.path.join(exp_dir, "checkpoints", "final.pth")
    )

    print(f"\n학습 완료. best test_acc={best_test_acc:.4f}")
    print(f"로그: cat {log_path}")


if __name__ == "__main__":
    train()
