import os
import shutil
from datetime import datetime


def create_experiment_dir(config_path):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    exp_dir = os.path.join("experiments", timestamp)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)

    shutil.copy(config_path, os.path.join(exp_dir, "config.yaml"))

    return exp_dir
