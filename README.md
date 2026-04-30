# Early Exit ViT — SLO-based Goodput Optimization

ViT-B/16 · ViT-L/16 기반 2-exit 모델에 Static Batching을 결합하여
SLO(Service Level Objective) 기반 **Goodput**을 Plain ViT 대비 향상시키는 추론 시스템.

---

## 모델 구조

### 2-exit SelectiveExitViT

```
seg1: blocks[0:B1] + exit_head  →  feat + ee_logits
      confidence ≥ τ  →  즉시 반환  (rt = seg1_ms)
      confidence < τ  →  큐 적재

seg2: blocks[B1:] + exit_head   →  ee_logits
      bs2개 non-exit 샘플이 모이면 일괄 실행
      rt = seg1_ms + queue_wait + seg2_ms
```

| | ViT-B/16 | ViT-L/16 |
|--|--|--|
| Total blocks | 12 | 24 |
| exit_blocks | [8, 12] | [12, 24] |
| Hidden dim | 768 | 1024 |
| Backbone | Frozen | Frozen |
| 학습 대상 | exit heads only | exit heads only |

---

## 핵심 지표: Goodput

```
goodput(SLO) = count(rt ≤ SLO_ms) / total_wall_time
```

- SLO를 만족한 샘플만 유효 처리로 인정
- Plain ViT도 동일 SLO 필터 적용 (공정 비교)
- `overall_tput = n / total_wall_time` 도 병행 측정

---

## 디렉토리 구조

```
src/
├── models/
│   ├── ee_vit_selective.py          # ViT-B/16 2-exit / 3-exit
│   └── ee_vit_large_selective.py    # ViT-L/16 2-exit
├── train/
│   ├── train_vit_selective.py       # ViT-B/16 학습
│   └── train_vit_large_selective.py # ViT-L/16 학습
├── export/
│   ├── export_onnx_vit_selective.py # seg1(static) + seg2(dynamic) + plain export
│   └── export_onnx_seg2_static.py   # seg2 static bs별 export (goodput용)
└── benchmark/
    ├── hybrid_vit_2exit_goodput.py      # ★ goodput 핵심 벤치마크
    └── hybrid_vit_2exit_onnx_realrun.py # dynamic batch 비교용

scripts/
├── train_vit_large_5090.sh
├── benchmark_hybrid_2exit_goodput_5090.sh
└── benchmark_hybrid_2exit_onnx_realrun_5090.sh

experiments/exp_YYYYMMDD_HHMMSS/
├── train/ee_vit_{large_}2exit/checkpoints/best.pth
├── onnx/ee_vit_{large_}2exit/seg1.onnx, seg2_bs{N}.onnx
└── eval/hybrid_2exit_goodput_*/
    ├── goodput_summary.csv
    ├── goodput_vs_slo.png
    ├── latency_cdf.png
    └── latency_decomposition.png
```

---

## 실행 방법

### 1. 환경 설정

```bash
python3.10 -m venv cap10
source cap10/bin/activate
pip install -r requirements.txt
```

### 2. 학습 (RTX 5090)

```bash
# ViT-L/16 2-exit (backbone frozen, exit heads 학습)
nohup bash scripts/train_vit_large_5090.sh > train_large.log 2>&1 &

# ViT-B/16 2-exit
cd src
python train/train_vit_selective.py --exit-blocks 8 12
```

### 3. ONNX Export

```bash
cd src

# seg1(dynamic) + plain
python export/export_onnx_vit_selective.py --model large-2exit   # ViT-L
python export/export_onnx_vit_selective.py --model 2exit         # ViT-B

# seg2 static (goodput 벤치마크용, bs별 별도 파일)
python export/export_onnx_seg2_static.py --model-variant large --batch-sizes 1 2 4 8 16 32 64
python export/export_onnx_seg2_static.py                         --batch-sizes 1 2 4 8 16 32 64
```

### 4. Goodput 벤치마크

```bash
# ViT-B/16
BATCH_SIZES="1 2 4 8 16 32 64" \
nohup bash scripts/benchmark_hybrid_2exit_goodput_5090.sh --threshold 0.80 \
    > goodput_b_thr0.8.log 2>&1 &

# ViT-L/16 (onnx dir 변경 필요)
ONNX_DIR=".../onnx/ee_vit_large_2exit" \
BATCH_SIZES="1 2 4 8 16 32 64" \
nohup bash scripts/benchmark_hybrid_2exit_goodput_5090.sh --threshold 0.80 \
    > goodput_l_thr0.8.log 2>&1 &
```

### 결과 해석

| 파일 | 설명 |
|--|--|
| `goodput_vs_slo.png` | SLO별 goodput 절대값 + hybrid/plain 비율 |
| `latency_cdf.png` | 응답시간 CDF (log-scale), SLO 기준선 표시 |
| `latency_decomposition.png` | bs2별 seg1·queue_wait·seg2 분해 바 차트 |
| `goodput_summary.csv` | bs2 × SLO 조합 전체 수치 |
| `records_bs{N}.json` | per-sample rt 원본 (SLO 재분석 가능) |

---

## 하이퍼파라미터 (학습)

| 항목 | 값 |
|--|--|
| Optimizer | AdamW |
| LR | 1e-3 |
| Epochs | 30 |
| Batch size | 64 |
| Scheduler | CosineAnnealingLR |
| Dataset | ImageNet-1K |

## 실험 설정 (벤치마크)

| 항목 | 값 |
|--|--|
| seg1 batch size (bs1) | 8 고정 |
| seg2 batch size (bs2) | 1 · 2 · 4 · 8 · 16 · 32 · 64 |
| Threshold τ | 0.70 · 0.80 |
| SLO sweep | 10 · 20 · 30 · 50 · 75 · 100 ms |
| 평가 샘플 | ImageNet val 5,000장 |
| Runtime | ONNX Runtime CUDA EP |
| GPU | RTX 5090 |
