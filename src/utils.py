import os
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
    os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)
    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))
    return exp_dir


# ── Metrics ─────────────────────────────────────────────────────────────────

def accuracy(output, target):
    _, pred = torch.max(output, 1)
    correct = (pred == target).sum().item()
    return correct / target.size(0)
