# Early-Exit ResNet-18 for Efficient Edge Inference on Jetson AGX Orin

CIFAR-10 분류 태스크에서 Early Exit을 적용한 ResNet-18과 일반 ResNet-18을 학습·비교하고,
TensorRT FP16으로 최적화하여 Jetson AGX Orin 위에서 레이턴시/전력/에너지를 측정한 캡스톤 프로젝트.

---

## 디렉토리 구조

```text
capstonedesign/
├── src/
│   ├── configs/
│   │   └── train.yaml              # 학습 하이퍼파라미터
│   ├── datasets/
│   │   └── dataloader.py
│   ├── engine/
│   │   ├── ee_trainer.py           # EE 모델 전용 train/eval (출력 3개)
│   │   └── plain_trainer.py        # Plain 모델 전용 train/eval (출력 1개)
│   ├── models/
│   │   ├── ee_resnet18.py          # Early-Exit ResNet-18 (EE1@layer2, EE2@layer3)
│   │   └── plain_resnet18.py       # 표준 ResNet-18
│   ├── paths.py                    # ★ 중앙화 경로 관리 (모든 스크립트가 참조)
│   ├── utils.py
│   ├── train.py                    # EE 모델 학습
│   ├── train_plain.py              # Plain 모델 학습
│   ├── export_onnx.py              # EE 모델 ONNX 변환 (full / seg 모드)
│   ├── export_onnx_plain.py        # Plain 모델 ONNX 변환
│   ├── eval_exit_rate.py           # threshold별 exit rate 평가
│   ├── infer_trt.py                # TRT 기반 Early Exit 추론 + sweep
│   ├── benchmark_trt.py            # Plain vs EE 레이턴시/전력 비교
│   ├── inspect_engines.py          # TRT 엔진 레이어 fusion 분석
│   ├── plot_results.py             # 학습 곡선 시각화
│   └── visualize_exit_samples.py   # exit별 샘플 시각화
│
├── scripts/
│   ├── train_pipeline.sh           # ★ 서버(5090): 학습 + ONNX 변환 자동화
│   └── orin_pipeline.sh            # ★ Orin: TRT 빌드 + 벤치마크 자동화
│
├── experiments/                    # ← .gitignore 됨 (결과물 저장)
│   ├── train/
│   │   ├── ee_resnet18/run_YYYYMMDD_HHMMSS/
│   │   │   ├── checkpoints/  (best.pth, final.pth, epoch_N.pth)
│   │   │   ├── config.yaml
│   │   │   └── train_log.csv
│   │   └── plain_resnet18/run_YYYYMMDD_HHMMSS/
│   ├── onnx/
│   │   ├── ee_resnet18/    (seg1_stem_layer2.onnx, seg2_layer3.onnx, seg3_layer4.onnx, ee_resnet18_full.onnx)
│   │   └── plain_resnet18/ (plain_resnet18.onnx)
│   ├── trt_engines/
│   │   ├── ee_resnet18/    (seg1.engine, seg2.engine, seg3.engine)
│   │   └── plain_resnet18/ (plain_resnet18.engine)
│   └── eval/
│       ├── exit_rate/      (exit_rate_analysis.png, exit_rate_results.json)
│       ├── trt_sweep/      (trt_sweep_results.png)
│       ├── exit_samples/   (exit1/2/3_samples_thrX.XX.png)
│       ├── benchmark/      (benchmark_thr0.80.png, .json)
│       └── engine_inspect/ (Plain_ResNet-18.json, EE_Seg1_*.json, ...)
│
├── data/                           # ← .gitignore 됨 (CIFAR-10 자동 다운로드)
├── requirements.txt
└── README.md
```

---

## 환경 설정

```bash
# Python 가상환경 (서버 / Orin 공통)
python3.10 -m venv cap10
source cap10/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 학습 (RTX 5090 서버)

### 한 번에 실행 (권장)

```bash
cd capstonedesign
bash scripts/train_pipeline.sh
```

학습 완료 후 `experiments/onnx/` 에 ONNX 파일이 생성됩니다.

### 개별 실행

```bash
cd src

# EE ResNet-18 학습
python train.py

# Plain ResNet-18 학습
python train_plain.py

# 학습 곡선 시각화 (최근 실험 자동 선택)
python plot_results.py          # EE 모델
python plot_results.py --plain  # Plain 모델

# threshold별 exit rate 분석
python eval_exit_rate.py

# exit별 샘플 시각화 (threshold=0.80)
python visualize_exit_samples.py --threshold 0.80
```

### ONNX 변환

```bash
cd src

# EE 모델 (full + seg 모두 변환)
python export_onnx.py --mode both

# Plain 모델
python export_onnx_plain.py
```

---

## TRT 빌드 + 벤치마크 (Jetson AGX Orin)

### 사전 준비

서버에서 생성된 ONNX 파일을 Orin으로 복사합니다.

```bash
# 서버 → Orin (예시)
scp -r experiments/onnx/ orin:~/capstonedesign/experiments/
```

### 한 번에 실행 (권장)

```bash
cd capstonedesign
bash scripts/orin_pipeline.sh
```

### 개별 실행

```bash
# TRT 엔진 빌드 (FP16)
trtexec --onnx=experiments/onnx/ee_resnet18/seg1_stem_layer2.onnx \
        --saveEngine=experiments/trt_engines/ee_resnet18/seg1.engine \
        --fp16 --workspace=1024

trtexec --onnx=experiments/onnx/ee_resnet18/seg2_layer3.onnx \
        --saveEngine=experiments/trt_engines/ee_resnet18/seg2.engine \
        --fp16 --workspace=1024

trtexec --onnx=experiments/onnx/ee_resnet18/seg3_layer4.onnx \
        --saveEngine=experiments/trt_engines/ee_resnet18/seg3.engine \
        --fp16 --workspace=1024

trtexec --onnx=experiments/onnx/plain_resnet18/plain_resnet18.onnx \
        --saveEngine=experiments/trt_engines/plain_resnet18/plain_resnet18.engine \
        --fp16 --workspace=1024

cd src

# Plain vs EE 벤치마크 (경로 자동 선택)
python benchmark_trt.py --threshold 0.80 --num-samples 1000

# TRT threshold sweep (0.50 ~ 0.95)
python infer_trt.py \
    --seg1 ../experiments/trt_engines/ee_resnet18/seg1.engine \
    --seg2 ../experiments/trt_engines/ee_resnet18/seg2.engine \
    --seg3 ../experiments/trt_engines/ee_resnet18/seg3.engine \
    --eval-cifar10 --sweep --num-samples 1000

# 엔진 레이어 fusion 분석
python inspect_engines.py
```

---

## 모델 구조

| 구분 | EE ResNet-18 | Plain ResNet-18 |
|------|-------------|----------------|
| 출력 수 | 3개 (EE1, EE2, Main) | 1개 |
| Exit 위치 | layer2 후 (EE1), layer3 후 (EE2) | — |
| 학습 손실 | 0.3·L(EE1) + 0.3·L(EE2) + 1.0·L(Main) | CrossEntropy |
| TRT 엔진 | 3개 분리 (seg1, seg2, seg3) | 1개 |

### EE 추론 흐름 (TRT)

```
입력 이미지
  └→ seg1 (stem+layer1+layer2+EE1) → EE1 conf ≥ thr? → 즉시 반환
      └→ seg2 (layer3+EE2)          → EE2 conf ≥ thr? → 즉시 반환
          └→ seg3 (layer4+FC)        → Main 출력 반환
```

---

## 주요 실험 결과 (CIFAR-10, threshold=0.80)

| 지표 | Plain ResNet-18 | EE ResNet-18 |
|------|----------------|-------------|
| 정확도 | 87.10% | 88.60% |
| 평균 레이턴시 | — | 1.40× 빠름 |
| P50 레이턴시 | — | 1.81× 빠름 |
| Exit 분포 | — | EE1 83.2% / EE2 3.7% / Main 13.1% |
| TRT 레이어 수 감소 | 60 → N | 66 → N (합산) |

---

## paths.py — 경로 중앙화 구조

모든 스크립트는 `src/paths.py`를 통해 경로를 결정합니다. 하드코딩된 절대경로 없이 실행 가능합니다.

```python
import paths

# 새 학습 디렉토리 생성
exp_dir = paths.new_train_dir("ee_resnet18")

# 최근 체크포인트 자동 선택
ckpt = paths.latest_checkpoint("ee_resnet18")  # best.pth

# ONNX / TRT 엔진 경로
onnx_out = paths.onnx_dir("ee_resnet18")
engine   = paths.engine_path("ee_resnet18", "seg1.engine")

# 평가 결과 저장 디렉토리
out_dir  = paths.eval_dir("benchmark")   # experiments/eval/benchmark/
```
