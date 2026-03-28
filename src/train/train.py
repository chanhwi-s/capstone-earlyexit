"""
Early-Exit ResNet-18 학습 스크립트

사용법:
  cd src
  python train.py

결과 저장 위치:
  experiments/train/ee_resnet18/run_YYYYMMDD_HHMMSS/
    checkpoints/  best.pth  final.pth  epoch_N.pth
    config.yaml
    train_log.csv
"""

import os
import sys
import shutil
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
import torch.optim as optim

from models.ee_resnet18 import build_model
from datasets.dataloader import get_dataloader
from engine.ee_trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import paths


# ── config 로드 ──────────────────────────────────────────────────────────────
config_path = "configs/train.yaml"
cfg = load_config(config_path)

dataset     = cfg["dataset"]["name"]
data_root   = cfg["dataset"]["data_root"]
num_workers = cfg["dataset"]["num_workers"]

batch_size = cfg["train"]["batch_size"]
epochs     = cfg["train"]["epochs"]
seed       = cfg["train"]["seed"]
weights    = (cfg["train"]["w1"], cfg["train"]["w2"], cfg["train"]["w3"])

lr           = float(cfg["optimizer"]["lr"])
momentum     = float(cfg["optimizer"]["momentum"])
weight_decay = float(cfg["optimizer"]["weight_decay"])

T_max   = cfg["scheduler"]["T_max"]
eta_min = float(cfg["scheduler"]["eta_min"])


def train():
    set_seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── 실험 디렉토리 (paths.py 중앙화) ──
    exp_dir  = paths.new_train_dir("ee_resnet18")
    log_path = os.path.join(exp_dir, "train_log.csv")
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
    print(f"Experiment dir : {exp_dir}")
    print(f"Log file       : {log_path}\n")

    train_loader, test_loader, num_classes = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed,
    )

    model     = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min)

    best_test_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc1, train_acc2, train_acc3 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, weights=weights
        )
        test_loss, test_acc1, test_acc2, test_acc_main = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[{epoch+1:3d}/{epochs}] lr={current_lr:.5f} | "
            f"train loss={train_loss:.4f}  ee1={train_acc1:.4f} ee2={train_acc2:.4f} main={train_acc3:.4f} | "
            f"test  loss={test_loss:.4f}  ee1={test_acc1:.4f}  ee2={test_acc2:.4f}  main={test_acc_main:.4f}"
        )

        log_to_csv(log_path, epoch + 1, {
            "lr":             current_lr,
            "train_loss":     train_loss,
            "train_acc_ee1":  train_acc1,
            "train_acc_ee2":  train_acc2,
            "train_acc_main": train_acc3,
            "test_loss":      test_loss,
            "test_acc_ee1":   test_acc1,
            "test_acc_ee2":   test_acc2,
            "test_acc_main":  test_acc_main,
        })

        if test_acc_main > best_test_acc:
            best_test_acc = test_acc_main
            torch.save(
                model.state_dict(),
                os.path.join(exp_dir, "checkpoints", "best.pth"),
            )

        torch.save(
            model.state_dict(),
            os.path.join(exp_dir, "checkpoints", f"epoch_{epoch+1}.pth"),
        )

    torch.save(
        model.state_dict(),
        os.path.join(exp_dir, "checkpoints", "final.pth"),
    )

    print(f"\n학습 완료. best test_acc_main={best_test_acc:.4f}")
    print(f"로그 확인: cat {log_path}")


if __name__ == "__main__":
    train()
