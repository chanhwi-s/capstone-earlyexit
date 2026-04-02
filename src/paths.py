"""
중앙화된 경로 관리 모듈
모든 스크립트는 이 모듈에서 경로를 가져옵니다.

━━━ 실험 디렉토리 구조 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  experiments/
  └── exp_YYYYMMDD_HHMMSS/        ← 실험마다 독립된 디렉토리 (EXP_DIR)
      ├── train/
      │   ├── ee_resnet18/
      │   │   ├── checkpoints/ (best.pth, final.pth)
      │   │   ├── config.yaml
      │   │   ├── train_log.csv
      │   │   └── training_curves.png
      │   ├── plain_resnet18/
      │   └── vee_resnet18/
      ├── onnx/
      │   ├── ee_resnet18/   (seg1.onnx, seg2.onnx, seg3.onnx, full.onnx)
      │   ├── plain_resnet18/ (plain_resnet18.onnx)
      │   └── vee_resnet18/  (vee_seg1.onnx, vee_seg2.onnx, full.onnx)
      ├── trt_engines/
      │   ├── ee_resnet18/   (seg1.engine, seg2.engine, seg3.engine)
      │   ├── plain_resnet18/ (plain_resnet18.engine)
      │   └── vee_resnet18/  (vee_seg1.engine, vee_seg2.engine)
      ├── engine_inspect/           ← 엔진이 동일하면 불변 → eval 밖에 저장
      │   ├── Plain_ResNet-18.json
      │   ├── EE_Seg1_....json
      │   └── ...
      └── eval/
          ├── exit_rate/            ← eval_dir() — 모델 기반, 런 타임스탬프 불필요
          ├── exit_samples/
          ├── hard_sample_analysis/
          └── run_YYYYMMDD_HHMMSS/  ← run_eval_dir() — 실행마다 독립된 결과
              ├── benchmark_comparison/
              ├── hybrid_grid/
              └── trt_sweep/

━━━ EXP_DIR 결정 우선순위 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 환경변수 EXP_DIR 이 설정되어 있으면 그것을 사용
     (train_pipeline.sh / orin_pipeline.sh 에서 export)
  2. 없으면 experiments/ 내 가장 최신 exp_* 디렉토리 자동 선택
  3. exp_* 디렉토리가 아예 없으면 experiments/ 직접 사용 (하위 호환 fallback)

━━━ RUN_TIMESTAMP 결정 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  run_eval_dir() 는 RUN_TIMESTAMP 환경변수를 사용해 실행별 하위 디렉토리 생성.
  orin_pipeline.sh 에서 export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) 로 설정됨.
  환경변수 없으면 모듈 임포트 시점의 시간을 자동 사용 (스크립트 직접 실행 시).
"""

import os
from datetime import datetime

# ── 실행 타임스탬프 (run_eval_dir 용) ──────────────────────────────────────────
# orin_pipeline.sh 에서 export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S) 로 주입.
# 환경변수 없으면 모듈 임포트 시점의 시간을 사용 (스크립트 단독 실행 시 fallback).
RUN_TIMESTAMP: str = (
    os.environ.get("RUN_TIMESTAMP", "").strip()
    or datetime.now().strftime("run_%Y%m%d_%H%M%S")
)

# ── 루트 경로 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXPERIMENTS_BASE = os.path.join(PROJECT_ROOT, "experiments")


def _resolve_exp_dir() -> str:
    """EXP_DIR 환경변수 → 최신 exp_* 자동 탐지 → fallback 순으로 결정."""
    # 1. 환경변수 우선
    env = os.environ.get("EXP_DIR", "").strip()
    if env:
        return os.path.abspath(env)

    # 2. experiments/ 내 exp_* 디렉토리 중 가장 최신
    if os.path.isdir(_EXPERIMENTS_BASE):
        candidates = sorted(
            d for d in os.listdir(_EXPERIMENTS_BASE)
            if d.startswith("exp_") and
            os.path.isdir(os.path.join(_EXPERIMENTS_BASE, d))
        )
        if candidates:
            return os.path.join(_EXPERIMENTS_BASE, candidates[-1])

    # 3. fallback: experiments/ 자체 (하위 호환)
    return _EXPERIMENTS_BASE


# 모듈 임포트 시점에 한 번 결정 (스크립트 실행 중에는 고정)
EXPERIMENTS_DIR = _resolve_exp_dir()
EXP_NAME        = os.path.basename(EXPERIMENTS_DIR)  # 예: exp_20260401_120000


# ── 학습 경로 ──────────────────────────────────────────────────────────────────

def new_train_dir(model_name: str) -> str:
    """새 학습 결과 디렉토리 생성 후 반환.
    {EXP_DIR}/train/{model_name}/
    """
    d = os.path.join(EXPERIMENTS_DIR, "train", model_name)
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    return d


def latest_train_dir(model_name: str) -> str | None:
    """현재 실험의 train/{model_name}/ 반환. 없으면 None."""
    d = os.path.join(EXPERIMENTS_DIR, "train", model_name)
    return d if os.path.isdir(d) else None


def latest_checkpoint(model_name: str, name: str = "best.pth") -> str | None:
    """현재 실험의 checkpoint 경로 반환. 없으면 None."""
    run = latest_train_dir(model_name)
    if run is None:
        return None
    ckpt = os.path.join(run, "checkpoints", name)
    return ckpt if os.path.exists(ckpt) else None


# ── ONNX 경로 ──────────────────────────────────────────────────────────────────

def onnx_dir(model_name: str) -> str:
    """{EXP_DIR}/onnx/{model_name}/ 생성 및 반환."""
    d = os.path.join(EXPERIMENTS_DIR, "onnx", model_name)
    os.makedirs(d, exist_ok=True)
    return d


# ── TRT Engine 경로 ────────────────────────────────────────────────────────────

def engine_dir(model_name: str) -> str:
    """{EXP_DIR}/trt_engines/{model_name}/ 생성 및 반환."""
    d = os.path.join(EXPERIMENTS_DIR, "trt_engines", model_name)
    os.makedirs(d, exist_ok=True)
    return d


def engine_path(model_name: str, filename: str) -> str:
    """{EXP_DIR}/trt_engines/{model_name}/{filename} 경로 반환."""
    return os.path.join(engine_dir(model_name), filename)


# ── 평가 결과 경로 ─────────────────────────────────────────────────────────────

def eval_dir(eval_type: str) -> str:
    """{EXP_DIR}/eval/{eval_type}/ 생성 및 반환.

    모델 종속 결과(exit_rate, exit_samples, hard_sample_analysis 등)에 사용.
    런마다 달라지지 않으므로 타임스탬프 없음.
    """
    d = os.path.join(EXPERIMENTS_DIR, "eval", eval_type)
    os.makedirs(d, exist_ok=True)
    return d


def run_eval_dir(eval_type: str) -> str:
    """{EXP_DIR}/eval/{RUN_TIMESTAMP}/{eval_type}/ 생성 및 반환.

    실행마다 달라지는 벤치마크 결과에 사용 (덮어쓰기 방지).
      eval_type 예: 'benchmark_comparison', 'hybrid_grid', 'trt_sweep'

    RUN_TIMESTAMP 는 모듈 임포트 시 결정됨:
      - 환경변수 RUN_TIMESTAMP 가 있으면 그 값 사용
      - 없으면 모듈 임포트 시점 시간 자동 사용
    """
    d = os.path.join(EXPERIMENTS_DIR, "eval", RUN_TIMESTAMP, eval_type)
    os.makedirs(d, exist_ok=True)
    return d


def engine_inspect_dir() -> str:
    """{EXP_DIR}/engine_inspect/ 생성 및 반환.

    TRT 엔진 레이어 fusion 분석 결과 저장.
    엔진이 동일하면 결과가 변하지 않으므로 eval/ 밖에 독립 디렉토리로 저장.
    """
    d = os.path.join(EXPERIMENTS_DIR, "engine_inspect")
    os.makedirs(d, exist_ok=True)
    return d


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def exp_base_dir() -> str:
    """experiments/ 최상위 경로 반환 (실험 목록 열람용)."""
    return _EXPERIMENTS_BASE


def list_experiments() -> list[str]:
    """experiments/ 내 모든 exp_* 디렉토리를 시간순 정렬하여 반환."""
    if not os.path.isdir(_EXPERIMENTS_BASE):
        return []
    return sorted(
        os.path.join(_EXPERIMENTS_BASE, d)
        for d in os.listdir(_EXPERIMENTS_BASE)
        if d.startswith("exp_") and
        os.path.isdir(os.path.join(_EXPERIMENTS_BASE, d))
    )
