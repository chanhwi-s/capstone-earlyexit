"""
Plain ResNet-50 학습 스크립트

사용법:
  cd src
  python train/train_plain50.py                          # CIFAR-10 (기본값)
  python train/train_plain50.py --dataset imagenet       # ImageNet
  python train/train_plain50.py --dataset imagenet --epochs 90 --batch-size 256

주요 인자:
  --dataset     cifar10 | imagenet  (기본: cifar10)
  --data-root   데이터 루트 경로     (기본: configs/train.yaml 값)
  --epochs      학습 에포크 수       (기본: dataset별 프리셋)
  --batch-size  배치 크기            (기본: dataset별 프리셋)
  --lr          초기 learning rate   (기본: dataset별 프리셋)
  --seed        랜덤 시드

결과 저장 위치:
  experiments/exp_.../train/plain_resnet50/
    checkpoints/  best.pth  final.pth  epoch_N.pth
    config.yaml
    train_log.csv
"""

import os
import sys
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.plain_resnet50 import build_model
from datasets.dataloader import get_dataloader
from engine.plain_trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import paths


def build_config(args):
    cfg = load_config("configs/train.yaml")

    dataset = args.dataset.lower()
    if dataset == "imagenet" and "imagenet" in cfg:
        preset = cfg["imagenet"]
        cfg["dataset"]["name"]        = "imagenet"
        cfg["dataset"]["data_root"]   = preset.get("data_root",   cfg["dataset"]["data_root"])
        cfg["dataset"]["num_workers"] = preset.get("num_workers", cfg["dataset"]["num_workers"])
        cfg["train"].update(preset.get("train",     {}))
        cfg["optimizer"].update(preset.get("optimizer", {}))
        cfg["scheduler"].update(preset.get("scheduler", {}))
    else:
        cfg["dataset"]["name"] = "cifar10"

    if args.data_root  is not None: cfg["dataset"]["data_root"] = args.data_root
    if args.epochs     is not None: cfg["train"]["epochs"]      = args.epochs
    if args.batch_size is not None: cfg["train"]["batch_size"]  = args.batch_size
    if args.lr         is not None: cfg["optimizer"]["lr"]      = args.lr
    if args.seed       is not None: cfg["train"]["seed"]        = args.seed

    cfg["scheduler"]["T_max"] = cfg["train"]["epochs"]
    return cfg


def train(cfg):
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset      = cfg["dataset"]["name"]
    data_root    = cfg["dataset"]["data_root"]
    num_workers  = cfg["dataset"]["num_workers"]
    batch_size   = cfg["train"]["batch_size"]
    epochs       = cfg["train"]["epochs"]
    seed         = cfg["train"]["seed"]
    lr           = float(cfg["optimizer"]["lr"])
    momentum     = float(cfg["optimizer"]["momentum"])
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    T_max        = cfg["scheduler"]["T_max"]
    eta_min      = float(cfg["scheduler"]["eta_min"])

    print(f"\n  Dataset    : {dataset}")
    print(f"  Data root  : {data_root}")
    print(f"  Batch size : {batch_size}  Epochs: {epochs}  LR: {lr}\n")

    exp_dir  = paths.new_train_dir("plain_resnet50")
    log_path = os.path.join(exp_dir, "train_log.csv")
    shutil.copy("configs/train.yaml", os.path.join(exp_dir, "config.yaml"))
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
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
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

    print(f"\n학습 완료. best test_acc={best_test_acc:.4f}")
    print(f"로그: cat {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plain ResNet-50 학습")
    parser.add_argument("--dataset",    type=str, default="cifar10",
                        choices=["cifar10", "imagenet"])
    parser.add_argument("--data-root",  type=str, default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    args = parser.parse_args()

    cfg = build_config(args)
    train(cfg)
