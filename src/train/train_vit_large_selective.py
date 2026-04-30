"""
SelectiveExitViT-L/16 exit head fine-tuning

ViT-L/16 backbone frozen, exit heads only 학습.
exit_blocks=[12, 24] 고정 (2-exit).
하이퍼파라미터는 ViT-B/16 학습과 동일.

사용법:
  cd src
  python train/train_vit_large_selective.py
  python train/train_vit_large_selective.py --epochs 20 --batch-size 32
  python train/train_vit_large_selective.py --weight-mode linear

결과:
  experiments/exp_.../train/ee_vit_large_2exit/
    checkpoints/best.pth  final.pth
    config.yaml
    train_log.csv
"""

import os, sys, shutil, argparse
import torch
import torch.nn as nn
import torch.optim as optim

os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/cap10/.cache/huggingface/hub')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ee_vit_large_selective import build_model_large, print_trainable_params
from datasets.dataloader import get_dataloader
from engine.vit_selective_trainer import train_one_epoch, evaluate
from utils import load_config, set_seed, log_to_csv
import paths


EXIT_BLOCKS = [12, 24]   # ViT-L/16 2-exit 고정


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
    return merged


def train(cfg):
    set_seed(cfg["train"]["seed"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
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

    print(f"\n  Model       : ViT-L/16  (vit_large_patch16_224)")
    print(f"  Exit blocks : {EXIT_BLOCKS}  (2-exit)")
    print(f"  Dataset     : ImageNet")
    print(f"  Data root   : {data_root}")
    print(f"  Batch size  : {batch_size}  Epochs: {epochs}")
    print(f"  LR (AdamW)  : {lr}  weight_decay: {weight_decay}")
    print(f"  Weight mode : {weight_mode}\n")

    model_name = "ee_vit_large_2exit"
    exp_dir    = paths.new_train_dir(model_name)
    log_path   = os.path.join(exp_dir, "train_log.csv")
    shutil.copy("configs/train.yaml", os.path.join(exp_dir, "config.yaml"))
    print(f"Experiment dir : {exp_dir}")
    print(f"Log file       : {log_path}\n")

    train_loader, test_loader, num_classes = get_dataloader(
        dataset="imagenet",
        batch_size=batch_size,
        data_root=data_root,
        num_workers=num_workers,
        seed=seed,
    )

    model = build_model_large(exit_blocks=EXIT_BLOCKS, num_classes=num_classes).to(device)
    print_trainable_params(model)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max, eta_min=eta_min
    )

    criterion     = nn.CrossEntropyLoss()
    best_test_acc = 0.0
    n_exits       = model.NUM_BLOCKS
    labels_str    = model.exit_block_labels   # ['B12', 'B24']

    for epoch in range(epochs):
        train_loss, train_accs = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            weight_mode=weight_mode,
        )
        test_loss, test_accs = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        exit_train_str = "  ".join(f"{labels_str[i]}={train_accs[i]:.4f}" for i in range(n_exits))
        exit_test_str  = "  ".join(f"{labels_str[i]}={test_accs[i]:.4f}"  for i in range(n_exits))
        print(
            f"[{epoch+1:3d}/{epochs}] lr={current_lr:.5f} | "
            f"train loss={train_loss:.4f}  {exit_train_str} | "
            f"test  loss={test_loss:.4f}  {exit_test_str}"
        )

        row = {"lr": current_lr, "train_loss": train_loss, "test_loss": test_loss}
        for i, lbl in enumerate(labels_str):
            row[f"train_acc_{lbl.lower()}"] = train_accs[i]
            row[f"test_acc_{lbl.lower()}"]  = test_accs[i]
        log_to_csv(log_path, epoch + 1, row)

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
    print(f"\n학습 완료. best test_acc (B24) = {best_test_acc:.4f}")
    print(f"체크포인트: {exp_dir}/checkpoints/best.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SelectiveExitViT-L/16 exit head fine-tuning (2-exit, exit_blocks=[12,24])"
    )
    parser.add_argument("--data-root",    type=str,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--weight-mode",  type=str,   default=None,
                        choices=["equal", "linear"])
    parser.add_argument("--seed",         type=int,   default=None)
    args = parser.parse_args()

    cfg = build_config(args)
    train(cfg)
