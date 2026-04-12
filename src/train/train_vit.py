"""
EE-ViT-B/16 exit head fine-tuning 스크립트

Pretrained backbone은 frozen 상태로 유지하고 exit head(12개)만 학습.
데이터셋: ImageNet (224×224)

사용법:
  cd src
  python train/train_vit.py                                     # 기본값
  python train/train_vit.py --epochs 20 --batch-size 32        # 오버라이드
  python train/train_vit.py --weight-mode linear                # 선형 가중치
  python train/train_vit.py --lr 5e-4 --weight-decay 0.01

주요 인자:
  --data-root    ImageNet 루트 경로  (기본: configs/train.yaml vit.data_root)
  --epochs       학습 에포크 수      (기본: 30)
  --batch-size   배치 크기           (기본: 64)
  --lr           learning rate       (기본: 1e-3, AdamW)
  --weight-decay AdamW weight decay  (기본: 0.05)
  --weight-mode  equal | linear      (기본: equal)
  --seed         랜덤 시드

결과 저장 위치:
  experiments/exp_.../train/ee_vit/
    checkpoints/  best.pth  final.pth
    config.yaml
    train_log.csv        ← epoch, lr, loss, acc_ee1~ee12 (train/test)
"""

import os
import sys
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ee_vit import build_model, print_trainable_params
from datasets.dataloader import get_dataloader
from engine.vit_trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import paths


# ── config 로드 ───────────────────────────────────────────────────────────────

def build_config(args):
    """
    train.yaml의 vit 프리셋 로드 → CLI 인자로 최종 오버라이드.
    우선순위: CLI 인자 > vit 프리셋 > train.yaml 기본값
    """
    cfg = load_config("configs/train.yaml")

    # vit 프리셋 병합
    vit_preset = cfg.get("vit", {})
    merged = {
        "dataset": {
            "name":        "imagenet",
            "data_root":   vit_preset.get("data_root",   cfg["dataset"]["data_root"]),
            "num_workers": vit_preset.get("num_workers", cfg["dataset"]["num_workers"]),
        },
        "train": {
            "batch_size":  64,
            "epochs":      30,
            "seed":        42,
            "weight_mode": "equal",
            **vit_preset.get("train", {}),
        },
        "optimizer": {
            "lr":           1e-3,
            "weight_decay": 0.05,
            **vit_preset.get("optimizer", {}),
        },
        "scheduler": {
            "T_max":   30,
            "eta_min": 1e-5,
            **vit_preset.get("scheduler", {}),
        },
    }

    # CLI 오버라이드
    if args.data_root    is not None: merged["dataset"]["data_root"]    = args.data_root
    if args.epochs       is not None: merged["train"]["epochs"]         = args.epochs
    if args.batch_size   is not None: merged["train"]["batch_size"]     = args.batch_size
    if args.lr           is not None: merged["optimizer"]["lr"]         = args.lr
    if args.weight_decay is not None: merged["optimizer"]["weight_decay"] = args.weight_decay
    if args.weight_mode  is not None: merged["train"]["weight_mode"]    = args.weight_mode
    if args.seed         is not None: merged["train"]["seed"]           = args.seed

    merged["scheduler"]["T_max"] = merged["train"]["epochs"]
    return merged


# ── 학습 루프 ─────────────────────────────────────────────────────────────────

def train(cfg):
    set_seed(cfg["train"]["seed"])

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    data_root    = cfg["dataset"]["data_root"]
    num_workers  = cfg["dataset"]["num_workers"]
    batch_size   = cfg["train"]["batch_size"]
    epochs       = cfg["train"]["epochs"]
    seed         = cfg["train"]["seed"]
    weight_mode  = cfg["train"]["weight_mode"]
    lr           = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    T_max        = cfg["scheduler"]["T_max"]
    eta_min      = float(cfg["scheduler"]["eta_min"])

    print(f"\n  Dataset    : ImageNet")
    print(f"  Data root  : {data_root}")
    print(f"  Batch size : {batch_size}  Epochs: {epochs}")
    print(f"  LR (AdamW) : {lr}  weight_decay: {weight_decay}")
    print(f"  Weight mode: {weight_mode}\n")

    # ── 실험 디렉토리 ──
    exp_dir  = paths.new_train_dir("ee_vit")
    log_path = os.path.join(exp_dir, "train_log.csv")
    shutil.copy("configs/train.yaml", os.path.join(exp_dir, "config.yaml"))
    print(f"Experiment dir : {exp_dir}")
    print(f"Log file       : {log_path}\n")

    # ── 데이터 로더 (ImageNet) ──
    train_loader, test_loader, num_classes = get_dataloader(
        dataset="imagenet",
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed,
    )

    # ── 모델 ──
    model = build_model(num_classes=num_classes).to(device)
    print_trainable_params(model)   # exit head만 학습되는지 확인

    # ── Optimizer: exit head 파라미터만 포함 ──
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )

    criterion     = nn.CrossEntropyLoss()
    best_test_acc = 0.0   # 마지막 exit(exit 12) 기준

    for epoch in range(epochs):
        train_loss, train_accs = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            weight_mode=weight_mode,
        )
        test_loss, test_accs = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── 콘솔 출력: ee1 / ee6 / ee12만 표시 (12개 모두 출력하면 너무 길어짐) ──
        print(
            f"[{epoch+1:3d}/{epochs}] lr={current_lr:.5f} | "
            f"train loss={train_loss:.4f}  "
            f"ee1={train_accs[0]:.4f} ee6={train_accs[5]:.4f} ee12={train_accs[11]:.4f} | "
            f"test  loss={test_loss:.4f}  "
            f"ee1={test_accs[0]:.4f}  ee6={test_accs[5]:.4f}  ee12={test_accs[11]:.4f}"
        )

        # ── CSV 로그: 12개 전부 기록 ──
        row = {
            "lr":         current_lr,
            "train_loss": train_loss,
            "test_loss":  test_loss,
        }
        for i in range(model.NUM_BLOCKS):
            row[f"train_acc_ee{i+1}"]  = train_accs[i]
            row[f"test_acc_ee{i+1}"]   = test_accs[i]

        log_to_csv(log_path, epoch + 1, row)

        # ── 체크포인트: best (ee12 기준) + final만 저장 ──
        # ViT state dict는 ~350MB이므로 per-epoch 저장 생략
        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]
            torch.save(
                model.state_dict(),
                os.path.join(exp_dir, "checkpoints", "best.pth"),
            )

    torch.save(
        model.state_dict(),
        os.path.join(exp_dir, "checkpoints", "final.pth"),
    )

    print(f"\n학습 완료. best test_acc (ee12) = {best_test_acc:.4f}")
    print(f"로그 확인: cat {log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EE-ViT-B/16 exit head fine-tuning")
    parser.add_argument("--data-root",    type=str,   default=None,
                        help="ImageNet 루트 경로")
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="학습 에포크 수 (기본: 30)")
    parser.add_argument("--batch-size",   type=int,   default=None,
                        help="배치 크기 (기본: 64)")
    parser.add_argument("--lr",           type=float, default=None,
                        help="AdamW learning rate (기본: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="AdamW weight decay (기본: 0.05)")
    parser.add_argument("--weight-mode",  type=str,   default=None,
                        choices=["equal", "linear"],
                        help="exit head loss 가중치 모드 (기본: equal)")
    parser.add_argument("--seed",         type=int,   default=None,
                        help="랜덤 시드")
    args = parser.parse_args()

    cfg = build_config(args)
    train(cfg)
