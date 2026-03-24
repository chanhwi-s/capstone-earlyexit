import os
import csv
import random
import shutil
import yaml
import numpy as np
import torch
from datetime import datetime


# ── Config ──────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Seed ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Experiment ──────────────────────────────────────────────────────────────

def create_experiment_dir(config_path):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
    return exp_dir


# ── CSV Logger ───────────────────────────────────────────────────────────────

def log_to_csv(log_path: str, epoch: int, metrics: dict):
    """
    매 epoch 지표를 CSV에 한 줄씩 append.

    - 파일이 없으면 헤더를 먼저 작성
    - 학습 중 실시간 확인:  tail -f <log_path>
    - 학습 후 전체 확인:    cat <log_path>
    """
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow({"epoch": epoch, **metrics})


# ── Metrics ─────────────────────────────────────────────────────────────────

def accuracy(output, target):
    _, pred = torch.max(output, 1)
    correct = (pred == target).sum().item()
    return correct / target.size(0)
