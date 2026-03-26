"""
중앙화된 경로 관리 모듈
모든 스크립트는 이 모듈에서 경로를 가져옵니다.

디렉토리 구조:
  experiments/
  ├── train/
  │   ├── ee_resnet18/run_YYYYMMDD_HHMMSS/
  │   │   ├── checkpoints/ (best.pth, final.pth, epoch_*.pth)
  │   │   ├── config.yaml
  │   │   ├── train_log.csv
  │   │   └── training_curves.png
  │   └── plain_resnet18/run_YYYYMMDD_HHMMSS/
  ├── onnx/
  │   ├── ee_resnet18/   (seg1.onnx, seg2.onnx, seg3.onnx, full.onnx)
  │   └── plain_resnet18/ (plain_resnet18.onnx)
  ├── trt_engines/
  │   ├── ee_resnet18/   (seg1.engine, seg2.engine, seg3.engine)
  │   └── plain_resnet18/ (plain_resnet18.engine)
  └── eval/
      ├── exit_rate/
      ├── trt_sweep/
      ├── exit_samples/
      ├── benchmark/
      └── engine_inspect/
"""

import os
from datetime import datetime

# ── 루트 경로 ──────────────────────────────────────────────────────────────────
# paths.py 가 src/ 에 있으므로 PROJECT_ROOT = src/../ = 프로젝트 루트
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")


# ── 학습 경로 ──────────────────────────────────────────────────────────────────

def new_train_dir(model_name: str) -> str:
    """새 학습 실험 디렉토리 생성 후 반환.
    experiments/train/{model_name}/run_YYYYMMDD_HHMMSS/
    """
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    d   = os.path.join(EXPERIMENTS_DIR, "train", model_name, f"run_{ts}")
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    return d


def latest_train_dir(model_name: str) -> str | None:
    """가장 최근 학습 run 디렉토리 반환. 없으면 None."""
    base = os.path.join(EXPERIMENTS_DIR, "train", model_name)
    if not os.path.isdir(base):
        return None
    runs = sorted(d for d in os.listdir(base) if d.startswith("run_"))
    return os.path.join(base, runs[-1]) if runs else None


def latest_checkpoint(model_name: str, name: str = "best.pth") -> str | None:
    """가장 최근 run의 checkpoint 경로 반환. 없으면 None."""
    run = latest_train_dir(model_name)
    if run is None:
        return None
    ckpt = os.path.join(run, "checkpoints", name)
    return ckpt if os.path.exists(ckpt) else None


# ── ONNX 경로 ──────────────────────────────────────────────────────────────────

def onnx_dir(model_name: str) -> str:
    """experiments/onnx/{model_name}/ 생성 및 반환."""
    d = os.path.join(EXPERIMENTS_DIR, "onnx", model_name)
    os.makedirs(d, exist_ok=True)
    return d


# ── TRT Engine 경로 ────────────────────────────────────────────────────────────

def engine_dir(model_name: str) -> str:
    """experiments/trt_engines/{model_name}/ 생성 및 반환."""
    d = os.path.join(EXPERIMENTS_DIR, "trt_engines", model_name)
    os.makedirs(d, exist_ok=True)
    return d


def engine_path(model_name: str, filename: str) -> str:
    """experiments/trt_engines/{model_name}/{filename} 경로 반환."""
    return os.path.join(engine_dir(model_name), filename)


# ── 평가 결과 경로 ─────────────────────────────────────────────────────────────

def eval_dir(eval_type: str) -> str:
    """experiments/eval/{eval_type}/ 생성 및 반환.

    eval_type 예: 'exit_rate', 'trt_sweep', 'exit_samples', 'benchmark', 'engine_inspect'
    """
    d = os.path.join(EXPERIMENTS_DIR, "eval", eval_type)
    os.makedirs(d, exist_ok=True)
    return d
