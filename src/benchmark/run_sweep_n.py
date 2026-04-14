"""
run_sweep_n.py  —  EE + VEE + EE50 threshold sweep N회 반복 + 단일 파일 취합

기존 방식(step1_sweep.sh에서 infer_trt.py를 N번 호출 → run_* 디렉토리 N개 생성)을
다음과 같이 개선합니다:
  ✓ 엔진 / 데이터를 한 번만 로드
  ✓ N회 실행 결과를 하나의 디렉토리에 단일 JSON + CSV로 취합
  ✓ 실행마다 그래프 생성하지 않음 → 취합 후 한 번만 그래프 생성
  ✓ cifar10 / imagenet 모두 지원

생성되는 출력:
  {EXP_DIR}/eval/sweep_N{N}_YYYYMMDD_HHMMSS/
    ee_sweep_raw.json       ← EE ResNet-18 (3-seg)
    ee_sweep_summary.csv
    vee_sweep_raw.json      ← VEE ResNet-18 (2-seg)
    vee_sweep_summary.csv
    ee50_sweep_raw.json     ← EE ResNet-50 (4-seg)
    ee50_sweep_summary.csv
    *_sweep_dist.png        ← KDE overlay
    *_sweep_summary.png     ← accuracy / exit rate / p99 요약

사용법:
  cd src
  python benchmark/run_sweep_n.py --n 20
  python benchmark/run_sweep_n.py --n 20 --dataset imagenet
  python benchmark/run_sweep_n.py --n 20 --num-samples 500
  python benchmark/run_sweep_n.py --n 20 --thresholds 0.70 0.75 0.80 0.85 0.90

인자:
  --n              반복 횟수 (기본: 10)
  --dataset        cifar10 | imagenet (기본: cifar10)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     탐색 threshold 목록 (기본: 0.50~0.95, 0.05 step)
  --seg1/2/3       EE ResNet-18 세그먼트 엔진 경로
  --vee-seg1/2     VEE ResNet-18 세그먼트 엔진 경로
  --ee50-seg1/2/3/4  EE ResNet-50 세그먼트 엔진 경로
  --out-dir        결과 저장 디렉토리
  --no-ee          EE ResNet-18 sweep 스킵
  --no-vee         VEE ResNet-18 sweep 스킵
  --no-ee50        EE ResNet-50 sweep 스킵
"""

import os
import sys
import json
import csv
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths


# ── TRT Engine ────────────────────────────────────────────────────────────────

class TRTEngine:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names  = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inputs):
        if isinstance(inputs, torch.Tensor):
            inputs = {self.input_names[0]: inputs}
        input_tensors = {}
        for name, tensor in inputs.items():
            t = tensor.contiguous().cuda().float()
            self.context.set_input_shape(name, list(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())
            input_tensors[name] = t
        output_tensors = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.zeros(shape, dtype=torch.float32, device='cuda')
            self.context.set_tensor_address(name, t.data_ptr())
            output_tensors[name] = t
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        torch.cuda.synchronize()
        return {name: t.cpu() for name, t in output_tensors.items()}


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_test_data(dataset: str, num_samples: int, data_root: str = None):
    from datasets.dataloader import get_dataloader
    from utils import load_config
    cfg = load_config('configs/train.yaml')
    if data_root is None:
        if dataset == 'imagenet' and 'imagenet' in cfg:
            data_root = cfg['imagenet']['data_root']
        else:
            data_root = cfg['dataset']['data_root']

    _, test_loader, _ = get_dataloader(
        dataset=dataset,
        batch_size=1,
        data_root=data_root,
        num_workers=0,
        seed=cfg['train']['seed'],
    )
    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images.append(img)
        labels.append(lbl[0].item())
    return images, labels


# ── EE 3-segment 단일 threshold 실행 ─────────────────────────────────────────

def run_ee_sweep_once(seg1: TRTEngine, seg2: TRTEngine, seg3: TRTEngine,
                      images, labels, threshold: float):
    """
    EE 3-segment 추론 1회 실행.
    Returns: {accuracy, exit_rate:[ee1%,ee2%,main%], latencies_ms:[...], stats}
    """
    correct      = 0
    exit_counts  = [0, 0, 0]
    latencies    = []

    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()

        out1        = seg1.infer(img)
        ee1_logits  = out1.get('ee1_logits',  list(out1.values())[-1])
        feat_layer2 = out1.get('feat_layer2', list(out1.values())[0])

        conf = F.softmax(ee1_logits, dim=1).max().item()
        if conf >= threshold:
            pred = ee1_logits.argmax(dim=1).item()
            exit_counts[0] += 1
        else:
            out2        = seg2.infer(feat_layer2)
            ee2_logits  = out2.get('ee2_logits',  list(out2.values())[-1])
            feat_layer3 = out2.get('feat_layer3', list(out2.values())[0])

            conf = F.softmax(ee2_logits, dim=1).max().item()
            if conf >= threshold:
                pred = ee2_logits.argmax(dim=1).item()
                exit_counts[1] += 1
            else:
                out3        = seg3.infer(feat_layer3)
                main_logits = list(out3.values())[0]
                pred        = main_logits.argmax(dim=1).item()
                exit_counts[2] += 1

        latencies.append((time.perf_counter() - t0) * 1000)
        if pred == lbl:
            correct += 1

    n = len(labels)
    return {
        'accuracy':   correct / n,
        'exit_rate':  [c / n * 100 for c in exit_counts],
        'latencies_ms': latencies,
        'avg_ms':     float(np.mean(latencies)),
        'p50_ms':     float(np.percentile(latencies, 50)),
        'p99_ms':     float(np.percentile(latencies, 99)),
    }


# ── EE50 4-segment 단일 threshold 실행 ───────────────────────────────────────

def run_ee50_sweep_once(seg1: TRTEngine, seg2: TRTEngine,
                        seg3: TRTEngine, seg4: TRTEngine,
                        images, labels, threshold: float):
    """
    EE ResNet-50 4-segment 추론 1회 실행.
    텐서 이름 규칙:
      seg1: image → (feat_layer1, ee1_logits)
      seg2: feat_layer1 → (feat_layer2, ee2_logits)
      seg3: feat_layer2 → (feat_layer3, ee3_logits)
      seg4: feat_layer3 → main_logits
    Returns: {accuracy, exit_rate:[ee1%,ee2%,ee3%,main%], latencies_ms:[...], stats}
    """
    correct     = 0
    exit_counts = [0, 0, 0, 0]
    latencies   = []

    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()

        out1        = seg1.infer(img)
        ee1_logits  = out1.get('ee1_logits',  list(out1.values())[-1])
        feat_layer1 = out1.get('feat_layer1', list(out1.values())[0])

        conf = F.softmax(ee1_logits, dim=1).max().item()
        if conf >= threshold:
            pred = ee1_logits.argmax(dim=1).item()
            exit_counts[0] += 1
        else:
            out2        = seg2.infer(feat_layer1)
            ee2_logits  = out2.get('ee2_logits',  list(out2.values())[-1])
            feat_layer2 = out2.get('feat_layer2', list(out2.values())[0])

            conf = F.softmax(ee2_logits, dim=1).max().item()
            if conf >= threshold:
                pred = ee2_logits.argmax(dim=1).item()
                exit_counts[1] += 1
            else:
                out3        = seg3.infer(feat_layer2)
                ee3_logits  = out3.get('ee3_logits',  list(out3.values())[-1])
                feat_layer3 = out3.get('feat_layer3', list(out3.values())[0])

                conf = F.softmax(ee3_logits, dim=1).max().item()
                if conf >= threshold:
                    pred = ee3_logits.argmax(dim=1).item()
                    exit_counts[2] += 1
                else:
                    out4        = seg4.infer(feat_layer3)
                    main_logits = list(out4.values())[0]
                    pred        = main_logits.argmax(dim=1).item()
                    exit_counts[3] += 1

        latencies.append((time.perf_counter() - t0) * 1000)
        if pred == lbl:
            correct += 1

    n = len(labels)
    return {
        'accuracy':     correct / n,
        'exit_rate':    [c / n * 100 for c in exit_counts],
        'latencies_ms': latencies,
        'avg_ms':       float(np.mean(latencies)),
        'p50_ms':       float(np.percentile(latencies, 50)),
        'p99_ms':       float(np.percentile(latencies, 99)),
    }


# ── VEE 2-segment 단일 threshold 실행 ────────────────────────────────────────

def run_vee_sweep_once(seg1: TRTEngine, seg2: TRTEngine,
                       images, labels, threshold: float):
    """
    VEE 2-segment 추론 1회 실행.
    Returns: {accuracy, exit_rate:[exit1%,main%], latencies_ms:[...], stats}
    """
    correct     = 0
    exit_counts = [0, 0]
    latencies   = []

    for img, lbl in zip(images, labels):
        t0 = time.perf_counter()

        out1       = seg1.infer(img)
        ee1_logits = out1.get('ee1_logits',  list(out1.values())[-1])
        feat       = out1.get('feat_layer1', list(out1.values())[0])

        conf = F.softmax(ee1_logits, dim=1).max().item()
        if conf >= threshold:
            pred = ee1_logits.argmax(dim=1).item()
            exit_counts[0] += 1
        else:
            out2        = seg2.infer(feat)
            main_logits = list(out2.values())[0]
            pred        = main_logits.argmax(dim=1).item()
            exit_counts[1] += 1

        latencies.append((time.perf_counter() - t0) * 1000)
        if pred == lbl:
            correct += 1

    n = len(labels)
    return {
        'accuracy':   correct / n,
        'exit_rate':  [c / n * 100 for c in exit_counts],
        'latencies_ms': latencies,
        'avg_ms':     float(np.mean(latencies)),
        'p50_ms':     float(np.percentile(latencies, 50)),
        'p99_ms':     float(np.percentile(latencies, 99)),
    }


# ── N회 sweep 실행 ────────────────────────────────────────────────────────────

def run_n_sweeps(run_fn, engines: tuple, images, labels,
                 thresholds: list, N: int, model_label: str):
    """
    run_fn에 대해 N회 반복 sweep 실행.
    Returns: {threshold_str: {'accuracy', 'exit_rate', 'runs': [per_run_stats]}}
    """
    results = {str(round(t, 2)): {
        'threshold': round(t, 2),
        'accuracy':  None,   # run 0에서 설정 (매번 동일)
        'exit_rate': None,   # run 0에서 설정
        'runs': [],          # N개의 per-run stats
    } for t in thresholds}

    total = N * len(thresholds)
    done  = 0

    print(f"\n  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}")
    print(f"  {'-'*45}")

    for run_idx in range(N):
        for thr in thresholds:
            key = str(round(thr, 2))
            r = run_fn(*engines, images, labels, thr)

            # accuracy/exit_rate는 첫 번째 run에서만 저장 (이후 동일)
            if results[key]['accuracy'] is None:
                results[key]['accuracy']  = r['accuracy']
                results[key]['exit_rate'] = r['exit_rate']

            results[key]['runs'].append({
                'run_idx':     run_idx,
                'avg_ms':      r['avg_ms'],
                'p50_ms':      r['p50_ms'],
                'p99_ms':      r['p99_ms'],
                'latencies_ms': r['latencies_ms'],
            })

            done += 1
            print(f"  [{model_label}] {thr:.2f}  run {run_idx+1:>3}/{N}  "
                  f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
                  f"[{done}/{total}]")

    # threshold별 N회 집계 통계 추가
    for key in results:
        runs   = results[key]['runs']
        p99s   = [r['p99_ms'] for r in runs]
        avgs   = [r['avg_ms'] for r in runs]
        p50s   = [r['p50_ms'] for r in runs]
        results[key]['summary'] = {
            'p99_mean': float(np.mean(p99s)),
            'p99_std':  float(np.std(p99s)),
            'p99_min':  float(np.min(p99s)),
            'p99_max':  float(np.max(p99s)),
            'avg_mean': float(np.mean(avgs)),
            'avg_std':  float(np.std(avgs)),
            'p50_mean': float(np.mean(p50s)),
            'p50_std':  float(np.std(p50s)),
        }

    return results


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_summary_csv(results: dict, out_path: str, model: str):
    """threshold별 통계를 CSV로 저장."""
    fieldnames = [
        'model', 'threshold', 'accuracy', 'exit_rate_str', 'n_runs',
        'p99_mean', 'p99_std', 'p99_min', 'p99_max',
        'avg_mean', 'avg_std', 'p50_mean', 'p50_std',
    ]
    rows = []
    for key, data in sorted(results.items(), key=lambda x: float(x[0])):
        s = data['summary']
        er = data['exit_rate']
        er_str = '|'.join(f'{v:.1f}%' for v in er) if er else 'N/A'
        rows.append({
            'model':        model,
            'threshold':    data['threshold'],
            'accuracy':     round(data['accuracy'], 6) if data['accuracy'] else None,
            'exit_rate_str': er_str,
            'n_runs':       len(data['runs']),
            'p99_mean':     round(s['p99_mean'], 4),
            'p99_std':      round(s['p99_std'],  4),
            'p99_min':      round(s['p99_min'],  4),
            'p99_max':      round(s['p99_max'],  4),
            'avg_mean':     round(s['avg_mean'], 4),
            'avg_std':      round(s['avg_std'],  4),
            'p50_mean':     round(s['p50_mean'], 4),
            'p50_std':      round(s['p50_std'],  4),
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV 저장: {out_path}")


# ── Plot: latency distribution overlay ────────────────────────────────────────

def plot_latency_dist(results: dict, title: str, save_path: str):
    """
    threshold별 latency distribution (KDE) overlay.
    각 threshold마다 N회 × num_samples의 모든 latency를 합쳐서 KDE 추정.
    """
    thresholds = sorted(results.keys(), key=float)
    cmap       = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{title}  —  Latency Distribution by Threshold  (N={len(list(results.values())[0]["runs"])} runs)', fontsize=13)

    # ── subplot 1: KDE overlay ────────────────────────────────────────────────
    ax = axes[0]
    all_lats = []
    for key, color in zip(thresholds, cmap):
        runs = results[key]['runs']
        lats = []
        for r in runs:
            lats.extend(r['latencies_ms'])
        all_lats.extend(lats)
        arr = np.array(lats)

        # KDE 추정
        try:
            kde = gaussian_kde(arr, bw_method='scott')
            x_range = np.linspace(arr.min(), np.percentile(arr, 99.5), 300)
            ax.plot(x_range, kde(x_range), color=color, linewidth=1.8,
                    label=f'thr={float(key):.2f}')
            # 중앙값 수직선
            ax.axvline(np.median(arr), color=color, linestyle='--',
                       linewidth=0.8, alpha=0.6)
        except Exception:
            pass

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('KDE per Threshold  (dashed = median)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # ── subplot 2: p99 error bar (mean ± std) ────────────────────────────────
    ax = axes[1]
    thr_vals  = [float(k) for k in thresholds]
    p99_means = [results[k]['summary']['p99_mean'] for k in thresholds]
    p99_stds  = [results[k]['summary']['p99_std']  for k in thresholds]
    ax.errorbar(thr_vals, p99_means, yerr=p99_stds,
                fmt='o-', color='tomato', linewidth=2, markersize=6,
                capsize=5, capthick=1.5, label='P99 mean ± std')
    ax.fill_between(thr_vals,
                    [m - s for m, s in zip(p99_means, p99_stds)],
                    [m + s for m, s in zip(p99_means, p99_stds)],
                    alpha=0.2, color='tomato')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('P99 Latency (ms)')
    ax.set_title('P99 Latency  mean ± std  across N runs')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  KDE plot 저장: {save_path}")


# ── Plot: accuracy / exit rate / latency 요약 ─────────────────────────────────

def plot_summary(results: dict, title: str, save_path: str, exit_labels: list):
    """
    threshold별 accuracy / exit rate / p99 latency 요약 3-subplot.
    """
    thresholds = sorted(results.keys(), key=float)
    thr_vals   = [float(k) for k in thresholds]
    n_runs     = len(list(results.values())[0]['runs'])

    acc      = [results[k]['accuracy']        for k in thresholds]
    exit_rates = [results[k]['exit_rate']     for k in thresholds]
    p99_means  = [results[k]['summary']['p99_mean'] for k in thresholds]
    p99_stds   = [results[k]['summary']['p99_std']  for k in thresholds]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{title}  —  Summary  (N={n_runs} runs)', fontsize=13)

    # 1) Accuracy
    ax = axes[0]
    ax.plot(thr_vals, acc, marker='D', color='black', linewidth=2)
    ax.set_title('Accuracy (constant across runs)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim([max(0.5, min(acc) - 0.05), 1.0])
    ax.grid(alpha=0.3)

    # 2) Exit Rate (stacked bar)
    ax = axes[1]
    n_exits = len(exit_labels)
    exit_colors = plt.cm.Set2(np.linspace(0, 1, n_exits))
    bottom = np.zeros(len(thr_vals))
    for ei, (label, color) in enumerate(zip(exit_labels, exit_colors)):
        vals = [exit_rates[ti][ei] for ti in range(len(thresholds))]
        ax.bar(thr_vals, vals, 0.035, bottom=bottom, label=label, color=color, alpha=0.85)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 3:
                ax.text(thr_vals[xi], b + v / 2, f'{v:.0f}%',
                        ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        bottom += np.array(vals)
    ax.set_title('Exit Rate Distribution')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Exit Rate (%)')
    ax.set_ylim([0, 110])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # 3) P99 latency error bar
    ax = axes[2]
    ax.errorbar(thr_vals, p99_means, yerr=p99_stds,
                fmt='o-', color='tomato', linewidth=2, markersize=6,
                capsize=5, capthick=1.5)
    ax.fill_between(thr_vals,
                    [m - s for m, s in zip(p99_means, p99_stds)],
                    [m + s for m, s in zip(p99_means, p99_stds)],
                    alpha=0.2, color='tomato')
    ax.set_title('P99 Latency (mean ± std)')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('ms')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Summary plot 저장: {save_path}")


# ── JSON 저장 헬퍼 ────────────────────────────────────────────────────────────

def save_raw_json(results: dict, out_path: str, metadata: dict):
    """raw latency 포함 전체 데이터를 JSON으로 저장."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # latencies_ms는 리스트 → JSON 직렬화 가능
    data = {'metadata': metadata, 'results': results}
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  JSON 저장: {out_path}")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='EE + VEE Threshold Sweep N회 반복')
    parser.add_argument('--n',           type=int,   default=10,
                        help='반복 횟수 (기본: 10)')
    parser.add_argument('--dataset',     type=str,   default='cifar10',
                        choices=['cifar10', 'imagenet'],
                        help='데이터셋 (기본: cifar10)')
    parser.add_argument('--data-root',   type=str,   default=None,
                        help='데이터 루트 경로')
    parser.add_argument('--num-samples', type=int,   default=1000,
                        help='샘플 수 (기본: 1000)')
    parser.add_argument('--thresholds',  type=float, nargs='+',
                        default=list(np.round(np.arange(0.50, 1.00, 0.05), 2)),
                        help='탐색할 threshold 목록 (기본: 0.50~0.95)')
    # EE ResNet-18 엔진 경로
    parser.add_argument('--seg1',      type=str, default=None)
    parser.add_argument('--seg2',      type=str, default=None)
    parser.add_argument('--seg3',      type=str, default=None)
    # VEE ResNet-18 엔진 경로
    parser.add_argument('--vee-seg1',  type=str, default=None)
    parser.add_argument('--vee-seg2',  type=str, default=None)
    # EE ResNet-50 엔진 경로 (4-segment)
    parser.add_argument('--ee50-seg1', type=str, default=None)
    parser.add_argument('--ee50-seg2', type=str, default=None)
    parser.add_argument('--ee50-seg3', type=str, default=None)
    parser.add_argument('--ee50-seg4', type=str, default=None)
    # 모드 선택
    parser.add_argument('--no-ee',   action='store_true', help='EE ResNet-18 sweep 스킵')
    parser.add_argument('--no-vee',  action='store_true', help='VEE ResNet-18 sweep 스킵')
    parser.add_argument('--no-ee50', action='store_true', help='EE ResNet-50 sweep 스킵')
    # 출력
    parser.add_argument('--out-dir', type=str, default=None,
                        help='결과 저장 디렉토리 (기본: {EXP_DIR}/eval/sweep_N{n}_YYYYMMDD/)')
    args = parser.parse_args()

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'sweep_N{args.n}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    # 엔진 경로 자동 선택
    ee_engine_dir   = paths.engine_dir('ee_resnet18')
    vee_engine_dir  = paths.engine_dir('vee_resnet18')
    ee50_engine_dir = paths.engine_dir('ee_resnet50')
    seg1      = args.seg1      or os.path.join(ee_engine_dir,   'seg1.engine')
    seg2      = args.seg2      or os.path.join(ee_engine_dir,   'seg2.engine')
    seg3      = args.seg3      or os.path.join(ee_engine_dir,   'seg3.engine')
    vee_seg1  = args.vee_seg1  or os.path.join(vee_engine_dir,  'vee_seg1.engine')
    vee_seg2  = args.vee_seg2  or os.path.join(vee_engine_dir,  'vee_seg2.engine')
    ee50_seg1 = args.ee50_seg1 or os.path.join(ee50_engine_dir, 'ee50_seg1.engine')
    ee50_seg2 = args.ee50_seg2 or os.path.join(ee50_engine_dir, 'ee50_seg2.engine')
    ee50_seg3 = args.ee50_seg3 or os.path.join(ee50_engine_dir, 'ee50_seg3.engine')
    ee50_seg4 = args.ee50_seg4 or os.path.join(ee50_engine_dir, 'ee50_seg4.engine')

    thresholds = sorted(set(round(t, 2) for t in args.thresholds))

    print('=' * 60)
    print(f'  EE + VEE Threshold Sweep  ×  {args.n}회')
    print(f'  Dataset     : {args.dataset}')
    print(f'  Samples     : {args.num_samples}')
    print(f'  Thresholds  : {thresholds}')
    print(f'  출력 디렉토리: {out_dir}')
    print('=' * 60)

    # ── 데이터 로드 ──
    print(f'\n데이터 로드 중 (n={args.num_samples})...')
    images, labels = load_test_data(args.dataset, args.num_samples, args.data_root)
    print(f'  로드 완료: {len(images)}개\n')

    metadata = {
        'n':          args.n,
        'dataset':    args.dataset,
        'num_samples': args.num_samples,
        'thresholds': thresholds,
        'timestamp':  ts,
    }

    # ── EE Sweep ──────────────────────────────────────────────────────────────
    if not args.no_ee:
        if not (os.path.exists(seg1) and os.path.exists(seg2) and os.path.exists(seg3)):
            print(f'[WARN] EE 엔진 파일 없음, EE sweep 스킵')
        else:
            print('\n=== EE TRT 엔진 로드 ===')
            ee_seg1 = TRTEngine(seg1)
            ee_seg2 = TRTEngine(seg2)
            ee_seg3 = TRTEngine(seg3)

            print(f'\n[EE] {args.n}회 sweep 시작...')
            ee_results = run_n_sweeps(
                run_ee_sweep_once,
                (ee_seg1, ee_seg2, ee_seg3),
                images, labels, thresholds, args.n, 'EE',
            )

            # 저장
            save_raw_json(ee_results,
                          os.path.join(out_dir, 'ee_sweep_raw.json'),
                          {**metadata, 'model': 'ee'})
            save_summary_csv(ee_results,
                             os.path.join(out_dir, 'ee_sweep_summary.csv'),
                             'EE-3Seg')

            # 그래프
            plot_latency_dist(ee_results, 'EE ResNet-18',
                              os.path.join(out_dir, 'ee_sweep_dist.png'))
            plot_summary(ee_results, 'EE ResNet-18',
                         os.path.join(out_dir, 'ee_sweep_summary.png'),
                         ['Exit1 (layer2)', 'Exit2 (layer3)', 'Main (layer4)'])

    # ── VEE Sweep ─────────────────────────────────────────────────────────────
    if not args.no_vee:
        if not (os.path.exists(vee_seg1) and os.path.exists(vee_seg2)):
            print(f'[WARN] VEE 엔진 파일 없음, VEE sweep 스킵')
        else:
            print('\n=== VEE TRT 엔진 로드 ===')
            vseg1 = TRTEngine(vee_seg1)
            vseg2 = TRTEngine(vee_seg2)

            print(f'\n[VEE] {args.n}회 sweep 시작...')
            vee_results = run_n_sweeps(
                run_vee_sweep_once,
                (vseg1, vseg2),
                images, labels, thresholds, args.n, 'VEE',
            )

            save_raw_json(vee_results,
                          os.path.join(out_dir, 'vee_sweep_raw.json'),
                          {**metadata, 'model': 'vee'})
            save_summary_csv(vee_results,
                             os.path.join(out_dir, 'vee_sweep_summary.csv'),
                             'VEE-2Seg')

            plot_latency_dist(vee_results, 'VEE ResNet-18',
                              os.path.join(out_dir, 'vee_sweep_dist.png'))
            plot_summary(vee_results, 'VEE ResNet-18',
                         os.path.join(out_dir, 'vee_sweep_summary.png'),
                         ['Exit1 (layer1)', 'Main (layer4)'])

    # ── EE50 4-segment Sweep ──────────────────────────────────────────────────
    if not args.no_ee50:
        ee50_engines = [ee50_seg1, ee50_seg2, ee50_seg3, ee50_seg4]
        if not all(os.path.exists(p) for p in ee50_engines):
            print(f'[WARN] EE50 엔진 파일 없음, EE50 sweep 스킵')
        else:
            print('\n=== EE ResNet-50 TRT 엔진 로드 ===')
            s1 = TRTEngine(ee50_seg1)
            s2 = TRTEngine(ee50_seg2)
            s3 = TRTEngine(ee50_seg3)
            s4 = TRTEngine(ee50_seg4)

            print(f'\n[EE50] {args.n}회 sweep 시작...')
            ee50_results = run_n_sweeps(
                run_ee50_sweep_once,
                (s1, s2, s3, s4),
                images, labels, thresholds, args.n, 'EE50',
            )

            save_raw_json(ee50_results,
                          os.path.join(out_dir, 'ee50_sweep_raw.json'),
                          {**metadata, 'model': 'ee50'})
            save_summary_csv(ee50_results,
                             os.path.join(out_dir, 'ee50_sweep_summary.csv'),
                             'EE50-4Seg')

            plot_latency_dist(ee50_results, 'EE ResNet-50',
                              os.path.join(out_dir, 'ee50_sweep_dist.png'))
            plot_summary(ee50_results, 'EE ResNet-50',
                         os.path.join(out_dir, 'ee50_sweep_summary.png'),
                         ['Exit1 (layer1)', 'Exit2 (layer2)',
                          'Exit3 (layer3)', 'Main (layer4)'])

    print(f'\n{"=" * 60}')
    print(f'  Sweep 완료!')
    print(f'  결과 저장: {out_dir}/')
    print(f'  다음 단계: threshold 확인 후')
    print(f'    bash scripts/step2_benchmark.sh <N> <THRESHOLD>')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
