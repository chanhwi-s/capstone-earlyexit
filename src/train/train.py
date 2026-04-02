"""
Early-Exit ResNet-18 학습 스크립트

사용법:
  cd src
  python train/train.py                          # CIFAR-10 (기본값)
  python train/train.py --dataset imagenet       # ImageNet (프리셋 자동 적용)
  python train/train.py --dataset imagenet --epochs 60 --batch-size 128   # 개별 오버라이드

주요 인자:
  --dataset     cifar10 | imagenet  (기본: cifar10)
  --data-root   데이터 루트 경로     (기본: configs/train.yaml 값)
  --epochs      학습 에포크 수       (기본: dataset별 프리셋)
  --batch-size  배치 크기            (기본: dataset별 프리셋)
  --lr          초기 learning rate   (기본: dataset별 프리셋)
  --seed        랜덤 시드

결과 저장 위치:
  experiments/exp_.../train/ee_resnet18/
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

from models.ee_resnet18 import build_model
from datasets.dataloader import get_dataloader
from engine.ee_trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import paths


# ── config 로드 + dataset별 프리셋 병합 ──────────────────────────────────────

def build_config(args):
    """
    train.yaml 로드 후 dataset 프리셋 병합 → CLI 인자로 최종 오버라이드.
    우선순위: CLI 인자 > dataset 프리셋 > train.yaml 기본값
    """
    cfg = load_config("configs/train.yaml")

    # dataset 프리셋 병합 (imagenet 선택 시)
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

    # CLI 인자 오버라이드 (지정된 항목만)
    if args.data_root  is not None: cfg["dataset"]["data_root"]   = args.data_root
    if args.epochs     is not None: cfg["train"]["epochs"]         = args.epochs
    if args.batch_size is not None: cfg["train"]["batch_size"]     = args.batch_size
    if args.lr         is not None: cfg["optimizer"]["lr"]         = args.lr
    if args.seed       is not None: cfg["train"]["seed"]           = args.seed
    if args.w1         is not None: cfg["train"]["w1"]             = args.w1
    if args.w2         is not None: cfg["train"]["w2"]             = args.w2
    if args.w3         is not None: cfg["train"]["w3"]             = args.w3

    # T_max를 epochs와 동기화 (scheduler가 cosine이면)
    cfg["scheduler"]["T_max"] = cfg["train"]["epochs"]

    return cfg


def train(cfg):
    set_seed(cfg["train"]["seed"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dataset      = cfg["dataset"]["name"]
    data_root    = cfg["dataset"]["data_root"]
    num_workers  = cfg["dataset"]["num_workers"]
    batch_size   = cfg["train"]["batch_size"]
    epochs       = cfg["train"]["epochs"]
    seed         = cfg["train"]["seed"]
    weights      = (cfg["train"]["w1"], cfg["train"]["w2"], cfg["train"]["w3"])
    lr           = float(cfg["optimizer"]["lr"])
    momentum     = float(cfg["optimizer"]["momentum"])
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    T_max        = cfg["scheduler"]["T_max"]
    eta_min      = float(cfg["scheduler"]["eta_min"])

    print(f"\n  Dataset    : {dataset}")
    print(f"  Data root  : {data_root}")
    print(f"  Batch size : {batch_size}  Epochs: {epochs}  LR: {lr}")
    print(f"  Loss weights: w1={weights[0]}, w2={weights[1]}, w3={weights[2]}\n")

    # ── 실험 디렉토리 ──
    exp_dir  = paths.new_train_dir("ee_resnet18")
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
    parser = argparse.ArgumentParser(description="EE ResNet-18 학습")
    parser.add_argument("--dataset",    type=str, default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="학습 데이터셋 (기본: cifar10)")
    parser.add_argument("--data-root",  type=str, default=None,
                        help="데이터 루트 경로 (기본: configs/train.yaml 값)")
    parser.add_argument("--epochs",     type=int,   default=None,
                        help="학습 에포크 수 (기본: dataset 프리셋)")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="배치 크기 (기본: dataset 프리셋)")
    parser.add_argument("--lr",         type=float, default=None,
                        help="초기 learning rate (기본: dataset 프리셋)")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="랜덤 시드")
    parser.add_argument("--w1",         type=float, default=None,
                        help="EE exit1 loss 가중치")
    parser.add_argument("--w2",         type=float, default=None,
                        help="EE exit2 loss 가중치")
    parser.add_argument("--w3",         type=float, default=None,
                        help="main loss 가중치")
    args = parser.parse_args()

    cfg = build_config(args)
    train(cfg)
