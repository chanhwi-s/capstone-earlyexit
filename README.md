# Early-Exit ResNet-18 for Efficient Edge Inference on Jetson AGX Orin

CIFAR-10 분류 태스크에서 Early Exit을 적용한 ResNet-18 변형들을 학습·비교하고,
TensorRT FP16으로 최적화하여 Jetson AGX Orin 위에서 레이턴시/전력/에너지를 측정하는 캡스톤 프로젝트.

---

## 모델 라인업

| 모델 | Exit 위치 | 출력 수 | 특징 |
|------|-----------|--------|------|
| **Plain ResNet-18** | 없음 | 1 | 베이스라인 |
| **EE ResNet-18** | layer2, layer3 | 3 | 기존 Early Exit |
| **VEE ResNet-18** | layer1 (최초!) | 2 | Very Early Exit — 쉬운 샘플을 훨씬 빠르게 처리 |

VEE 설계 근거: EE 실험에서 layer3 뒤 exit(EE2) 비율이 ~3.7%로 매우 낮음 → exit head가 2개까지 필요 없음. 대신 layer1 직후에 exit을 배치해 쉬운 샘플은 최대한 빨리 종료.

---

## 디렉토리 구조

```text
capstonedesign/
├── src/
│   ├── configs/
│   │   └── train.yaml
│   ├── datasets/
│   │   └── dataloader.py              # CIFAR-10 / ImageNet 지원
│   ├── engine/
│   │   ├── ee_trainer.py              # EE 전용 (3출력, 가중 손실)
│   │   ├── plain_trainer.py           # Plain 전용 (1출력)
│   │   └── vee_trainer.py             # VEE 전용 (2출력, 가중 손실)
│   ├── models/
│   │   ├── ee_resnet18.py             # EE ResNet-18 (exit @ layer2, layer3)
│   │   ├── plain_resnet18.py          # 표준 ResNet-18
│   │   └── vee_resnet18.py            # VEE ResNet-18 (exit @ layer1 only)
│   ├── paths.py                       # ★ 중앙화 경로 관리
│   ├── profiling_utils.py             # ★ P50/P90/P95/P99 + Goodput 계산
│   ├── utils.py
│   │
│   ├── train.py                       # EE 학습
│   ├── train_plain.py                 # Plain 학습
│   ├── train_vee.py                   # VEE 학습
│   │
│   ├── export_onnx.py                 # EE → ONNX (full + 3-seg)
│   ├── export_onnx_plain.py           # Plain → ONNX
│   ├── export_onnx_vee.py             # VEE → ONNX (full + 2-seg)
│   │
│   ├── eval_exit_rate.py              # threshold별 exit rate 평가
│   ├── infer_trt.py                   # EE 3-seg TRT 추론 + sweep
│   ├── infer_trt_hybrid.py            # ★ Hybrid 런타임 (VEE + batched plain fallback)
│   ├── benchmark_trt.py               # Plain vs EE 벤치마크
│   ├── benchmark_trt_hybrid.py        # ★ 4-Way 비교 벤치마크 (Plain/EE/VEE/Hybrid)
│   ├── analyze_hard_samples.py        # ★ Hard sample 분석 (EE→Plain 정확도 비교)
│   ├── inspect_engines.py             # TRT 레이어 fusion 분석
│   ├── plot_results.py                # 학습 곡선 시각화
│   └── visualize_exit_samples.py      # exit별 샘플 시각화
│
├── scripts/
│   ├── train_pipeline.sh              # ★ 서버: EE+Plain+VEE 학습 + ONNX 변환
│   └── orin_pipeline.sh               # ★ Orin: TRT 빌드 + 전체 벤치마크
│
├── experiments/                       # ← .gitignore (결과물)
│   ├── train/{ee,plain,vee}_resnet18/run_YYYYMMDD_HHMMSS/
│   ├── onnx/{ee,plain,vee}_resnet18/
│   ├── trt_engines/{ee,plain,vee}_resnet18/
│   └── eval/
│       ├── exit_rate/
│       ├── trt_sweep/
│       ├── exit_samples/
│       ├── benchmark/
│       ├── benchmark_comparison/      # 4-Way 비교 결과
│       ├── hybrid_runtime/            # Hybrid grid search 결과
│       ├── hard_sample_analysis/      # Hard sample 분석 결과
│       └── engine_inspect/
│
├── data/                              # ← .gitignore (CIFAR-10 자동 다운로드)
├── requirements.txt
└── README.md
```

---

## 환경 설정

```bash
python3.10 -m venv cap10
source cap10/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 1단계: 학습 + ONNX 변환 (RTX 5090 서버)

### 한 번에 실행 (권장)

```bash
cd capstonedesign
bash scripts/train_pipeline.sh

# 일부 스킵
SKIP_EE=1   bash scripts/train_pipeline.sh   # Plain + VEE만 학습
SKIP_VEE=1  bash scripts/train_pipeline.sh   # EE + Plain만 학습
```

### 개별 실행

```bash
cd src

# 학습
python train.py           # EE ResNet-18
python train_plain.py     # Plain ResNet-18
python train_vee.py       # VEE ResNet-18

# 학습 곡선 시각화
python plot_results.py            # EE (자동 선택)
python plot_results.py --plain    # Plain
python plot_results.py experiments/train/vee_resnet18/run_.../  # VEE (직접 지정)

# ONNX 변환
python export_onnx.py --mode both      # EE (full + 3-seg)
python export_onnx_plain.py            # Plain
python export_onnx_vee.py --mode both  # VEE (full + 2-seg)

# Exit rate 분석 (PyTorch, 서버에서)
python eval_exit_rate.py --threshold 0.80
python analyze_hard_samples.py --threshold 0.80
```

---

## 2단계: TRT 빌드 + 벤치마크 (Jetson AGX Orin)

### 사전 준비

```bash
# 서버 → Orin 파일 전송
scp -r experiments/onnx/ orin:~/capstonedesign/experiments/
```

### 한 번에 실행 (권장)

```bash
cd capstonedesign
bash scripts/orin_pipeline.sh

# 옵션
SKIP_BUILD=1    bash scripts/orin_pipeline.sh  # 빌드 스킵
THRESHOLD=0.85  bash scripts/orin_pipeline.sh  # threshold 변경
N_SAMPLES=500   bash scripts/orin_pipeline.sh  # 샘플 수 조절
```

### TRT 엔진 빌드 (수동)

```bash
# EE (3-segment)
trtexec --onnx=experiments/onnx/ee_resnet18/seg1_stem_layer2.onnx \
        --saveEngine=experiments/trt_engines/ee_resnet18/seg1.engine \
        --fp16 --workspace=1024

trtexec --onnx=experiments/onnx/ee_resnet18/seg2_layer3.onnx \
        --saveEngine=experiments/trt_engines/ee_resnet18/seg2.engine --fp16 --workspace=1024

trtexec --onnx=experiments/onnx/ee_resnet18/seg3_layer4.onnx \
        --saveEngine=experiments/trt_engines/ee_resnet18/seg3.engine --fp16 --workspace=1024

# Plain
trtexec --onnx=experiments/onnx/plain_resnet18/plain_resnet18.onnx \
        --saveEngine=experiments/trt_engines/plain_resnet18/plain_resnet18.engine \
        --fp16 --workspace=1024

# VEE (2-segment)
trtexec --onnx=experiments/onnx/vee_resnet18/vee_seg1_stem_layer1.onnx \
        --saveEngine=experiments/trt_engines/vee_resnet18/vee_seg1.engine --fp16 --workspace=1024

trtexec --onnx=experiments/onnx/vee_resnet18/vee_seg2_layer2to4.onnx \
        --saveEngine=experiments/trt_engines/vee_resnet18/vee_seg2.engine --fp16 --workspace=1024
```

### 벤치마크 (수동)

```bash
cd src

# Plain vs EE (기존)
python benchmark_trt.py --threshold 0.80

# 4-Way 비교: Plain / EE-3seg / VEE-2seg / Hybrid
python benchmark_trt_hybrid.py --threshold 0.80 --hybrid-bs 8 --hybrid-to-ms 10

# Hybrid grid search (batch_size × timeout 최적값 탐색)
python infer_trt_hybrid.py --grid-search \
    --batch-sizes 1 2 4 8 16 32 \
    --timeouts    2 5 10 20 50

# EE threshold sweep
python infer_trt.py \
    --seg1 ../experiments/trt_engines/ee_resnet18/seg1.engine \
    --seg2 ../experiments/trt_engines/ee_resnet18/seg2.engine \
    --seg3 ../experiments/trt_engines/ee_resnet18/seg3.engine \
    --eval-cifar10 --sweep

# Hard sample 분석
python analyze_hard_samples.py --threshold 0.80
```

---

## 프로파일링 지표 (`profiling_utils.py`)

모든 벤치마크 스크립트에서 표준 지표를 자동 출력합니다.

| 지표 | 설명 |
|------|------|
| P50 / P90 / P95 / P99 / P99.9 | Latency percentiles |
| Avg Throughput | 1000 / avg_ms (inf/sec) |
| **P90 Goodput** | 1000 / P90_ms — 90% 보장 처리량 (엣지 SLO 핵심 지표) |
| **P95 / P99 Goodput** | 95% / 99% 보장 처리량 |
| Tail Ratio (P99/P50) | tail latency 안정성 — 1에 가까울수록 균일 |
| IQR | Interquartile Range (jitter 지표) |

**Goodput 개념:** "P90 Goodput = 500 inf/s"는 "90%의 요청이 2ms 이내에 처리되는 조건에서 지속 가능한 최대 처리량"을 의미. 평균 throughput보다 엣지 실시간 추론의 실질적 성능을 잘 반영.

```python
from profiling_utils import compute_latency_stats, print_latency_report

stats = compute_latency_stats(latencies_ms)
print_latency_report(stats, "My Model")
```

---

## Hybrid Runtime 설계 (`infer_trt_hybrid.py`)

```
입력 스트림 (sample 순차 도착)
  └→ VEE seg1 (stem+layer1+exit1, ~매우 빠름)
       ├─ confidence ≥ threshold? → 즉시 반환 (Exit1, 최소 레이턴시)
       └─ confidence < threshold? → fallback queue 추가
                                      │
                    ┌─────────────────┘
                    ▼
              [batch_size 충족] 또는 [timeout_ms 경과]
                    │
                    ▼
           Plain ResNet-18 batched inference
           (queue에 모인 sample을 zero-pad로 batch 구성 → 일괄 처리)
                    │
                    ▼
           유효 sample 결과만 반환 (padding sample 결과 무시)
```

**핵심 파라미터:**
- `batch_size`: fallback batch 크기 (grid search로 최적값 탐색)
- `timeout_ms`: 최대 대기 시간 (이 시간이 지나면 partial batch로 즉시 처리)
- 예: plain 1회 실행 10ms, seg 이후 5ms → batch ≥ 2이어야 이득

---

## Hard Sample 분석 (`analyze_hard_samples.py`)

**가설:** "EE 마지막 exit까지 통과하는 샘플 = 본질적으로 어려운 샘플 (EE 구조 문제 아님)"

**검증 방법:**
1. EE 모델에서 threshold 적용 → exit3(main)으로 나가는 샘플 추출
2. 해당 샘플만 Plain ResNet-18으로 추론
3. EE와 Plain의 accuracy / confidence 비교

**기대 결과:** Hard sample subset에서 Plain도 accuracy가 낮으면 → EE 구조가 아닌 샘플 난이도 문제임을 실증

```bash
python analyze_hard_samples.py --threshold 0.80
# → experiments/eval/hard_sample_analysis/hard_sample_analysis_thr0.80.json
# → experiments/eval/hard_sample_analysis/hard_sample_analysis_thr0.80.png
```

---

## paths.py — 경로 중앙화

```python
import paths

exp_dir  = paths.new_train_dir("vee_resnet18")          # 새 학습 디렉토리
ckpt     = paths.latest_checkpoint("ee_resnet18")        # 최근 best.pth
onnx_out = paths.onnx_dir("vee_resnet18")               # ONNX 출력 디렉토리
engine   = paths.engine_path("vee_resnet18", "vee_seg1.engine")
out_dir  = paths.eval_dir("benchmark_comparison")       # 평가 결과 디렉토리
```

---

## ImageNet-1K로 스케일업 (향후 계획)

`dataloader.py`는 이미 ImageNet 지원. `train.yaml`에서 dataset 변경:

```yaml
dataset:
  name: imagenet        # cifar10 → imagenet
  data_root: /data/imagenet
```

모든 모델·학습·벤치마크 스크립트가 자동 대응 (num_classes=1000 자동 설정).
