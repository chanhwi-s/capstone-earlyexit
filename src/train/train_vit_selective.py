"""
SelectiveExitViT-B/16 exit head fine-tuning 스크립트

Pretrained backbone은 frozen 상태로 유지하고 선택된 블록의 exit head만 학습.
데이터셋: ImageNet (224×224)

사용법:
  cd src
  python train/train_vit_selective.py --exit-blocks 8 12
  python train/train_vit_selective.py --exit-blocks 6 9 12
  python train/train_vit_selective.py --exit-blocks 8 12 --epochs 20 --batch-size 32
  python train/train_vit_selective.py --exit-blocks 6 9 12 --weight-mode linear

주요 인자:
  --exit-blocks  exit head를 붙일 블록 번호 목록 (1-indexed, 마지막은 반드시 12)
  --data-root    ImageNet 루트 경로  (기본: configs/train.yaml vit.data_root)
  --epochs       학습 에포크 수      (기본: 30)
  --batch-size   배치 크기           (기본: 64)
  --lr           learning rate       (기본: 1e-3, AdamW)
  --weight-decay AdamW weight decay  (기본: 0.05)
  --weight-mode  equal | linear      (기본: equal)
  --seed         랜덤 시드

결과 저장 위치:
  experiments/exp_.../train/ee_vit_2exit/   (exit-blocks=[8,12])
  experiments/exp_.../train/ee_vit_3exit/   (exit-blocks=[6,9,12])
    checkpoints/  best.pth  final.pth
    config.yaml
    train_log.csv        ← epoch, lr, loss, acc_ee{block}... (train/test)
"""

import os
import sys
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ee_vit_selective import build_model, print_trainable_params
from datasets.dataloader import get_dataloader
from engine.vit_selective_trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import paths


# ── config 로드 ───────────────────────────────────────────────────────────────

def build_config(args):
    cfg = load_config("configs/train.yaml")

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

    if args.data_root    is not None: merged["dataset"]["data_root"]      = args.data_root
    if args.epochs       is not None: merged["train"]["epochs"]           = args.epochs
    if args.batch_size   is not None: merged["train"]["batch_size"]       = args.batch_size
    if args.lr           is not None: merged["optimizer"]["lr"]           = args.lr
    if args.weight_decay is not None: merged["optimizer"]["weight_decay"] = args.weight_decay
    if args.weight_mode  is not None: merged["train"]["weight_mode"]      = args.weight_mode
    if args.seed         is not None: merged["train"]["seed"]             = args.seed

    merged["scheduler"]["T_max"] = merged["train"]["epochs"]
    merged["exit_blocks"] = args.exit_blocks
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

    exit_blocks  = cfg["exit_blocks"]
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

    print(f"\n  Exit blocks : {exit_blocks}")
    print(f"  Dataset     : ImageNet")
    print(f"  Data root   : {data_root}")
    print(f"  Batch size  : {batch_size}  Epochs: {epochs}")
    print(f"  LR (AdamW)  : {lr}  weight_decay: {weight_decay}")
    print(f"  Weight mode : {weight_mode}\n")

    # ── 모델 이름: ee_vit_2exit / ee_vit_3exit ──
    model_name = f"ee_vit_{len(exit_blocks)}exit"
    exp_dir    = paths.new_train_dir(model_name)
    log_path   = os.path.join(exp_dir, "train_log.csv")
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
    model = build_model(exit_blocks=exit_blocks, num_classes=num_classes).to(device)
    print_trainable_params(model)

    # ── Optimizer: exit head 파라미터만 포함 ──
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )

    criterion     = nn.CrossEntropyLoss()
    best_test_acc = 0.0   # 마지막 exit 기준
    n_exits       = model.NUM_BLOCKS
    labels_str    = model.exit_block_labels   # ['B8','B12'] or ['B6','B9','B12']

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

        # ── 콘솔 출력: 모든 exit 표시 (2~3개라 괜찮음) ──
        exit_train_str = "  ".join(
            f"{labels_str[i]}={train_accs[i]:.4f}" for i in range(n_exits)
        )
        exit_test_str  = "  ".join(
            f"{labels_str[i]}={test_accs[i]:.4f}"  for i in range(n_exits)
        )
        print(
            f"[{epoch+1:3d}/{epochs}] lr={current_lr:.5f} | "
            f"train loss={train_loss:.4f}  {exit_train_str} | "
            f"test  loss={test_loss:.4f}  {exit_test_str}"
        )

        # ── CSV 로그 ──
        row = {
            "lr":         current_lr,
            "train_loss": train_loss,
            "test_loss":  test_loss,
        }
        for i, lbl in enumerate(labels_str):
            row[f"train_acc_{lbl.lower()}"] = train_accs[i]
            row[f"test_acc_{lbl.lower()}"]  = test_accs[i]

        log_to_csv(log_path, epoch + 1, row)

        # ── 체크포인트: best (마지막 exit 기준) + final ──
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

    last_label = labels_str[-1]
    print(f"\nTraining complete. best test_acc ({last_label}) = {best_test_acc:.4f}")
    print(f"Log: cat {log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SelectiveExitViT-B/16 exit head fine-tuning"
    )
    parser.add_argument("--exit-blocks",   type=int,   nargs="+", required=True,
                        help="Exit block positions (1-indexed, must end with 12). "
                             "Example: --exit-blocks 8 12  or  --exit-blocks 6 9 12")
    parser.add_argument("--data-root",    type=str,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--weight-mode",  type=str,   default=None,
                        choices=["equal", "linear"])
    parser.add_argument("--seed",         type=int,   default=None)
    args = parser.parse_args()

    # 기본 검증
    if args.exit_blocks[-1] != 12:
        parser.error("Last exit block must be 12 (final ViT block).")
    if args.exit_blocks != sorted(args.exit_blocks):
        parser.error("--exit-blocks must be in ascending order.")

    cfg = build_config(args)
    train(cfg)
