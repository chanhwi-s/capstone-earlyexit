"""
run_vit_trt_sweep.py  —  SelectiveExitViT TRT threshold sweep  (Jetson AGX Orin)

run_vit_selective_sweep.py 의 TRT 버전.
PyTorch 대신 TRT 세그먼트 엔진으로 추론하며, 동일한 분석·플롯을 생성한다.

생성 파일 ({EXP_DIR}/eval/vit_trt_sel{N}exit_sweep_N{N}_YYYYMMDD/):
  trt_sel_sweep_raw.json
  trt_sel_sweep_summary.csv
  trt_sel_sweep_exit_heatmap.png
  trt_sel_sweep_acc_heatmap.png
  trt_sel_sweep_latency_dist.png
  trt_sel_sweep_summary.png

사용법 (Orin):
  cd src
  python benchmark/run_vit_trt_sweep.py --exit-blocks 8 12 --n 10
  python benchmark/run_vit_trt_sweep.py --exit-blocks 6 9 12 --n 10
  python benchmark/run_vit_trt_sweep.py --exit-blocks 8 12 --n 5 --latency-only

인자:
  --exit-blocks    exit 블록 번호 (필수, 예: 8 12 또는 6 9 12)
  --n              반복 횟수 (기본: 10)
  --num-samples    샘플 수 (기본: 1000)
  --thresholds     threshold 목록 (기본: 0.1~0.99)
  --data-root      ImageNet val 루트 (없으면 --latency-only 자동 전환)
  --latency-only   랜덤 노이즈로 latency만 측정 (accuracy 생략)
  --warmup         warmup 샘플 수 (기본: 20)
  --out-dir        결과 저장 디렉토리
"""

import os
import sys
import json
import csv
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from infer.infer_trt_vit_selective import SelectiveViTTRT, load_selective_vit_trt

# 플롯 함수는 PyTorch sweep 스크립트에서 재사용
from benchmark.run_vit_selective_sweep import (
    save_raw_json,
    save_summary_csv,
    plot_exit_heatmap,
    plot_acc_heatmap,
    plot_latency_dist,
    plot_summary,
    print_result_table,
)

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_imagenet_samples(num_samples: int, data_root: str, num_workers: int = 2):
    from datasets.dataloader import get_dataloader
    from utils import load_config

    cfg = load_config('configs/train.yaml')
    _, test_loader, _ = get_dataloader(
        dataset='imagenet',
        batch_size=1,
        data_root=data_root,
        num_workers=num_workers,
        seed=cfg['train']['seed'],
    )

    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images.append(img)
        labels.append(lbl[0].item())

    print(f"  ImageNet 데이터 로드: {len(images)} 샘플")
    return images, labels


def make_dummy_samples(num_samples: int):
    images = [torch.randn(1, 3, 224, 224) for _ in range(num_samples)]
    labels = [0] * num_samples   # 더미 레이블 (accuracy 무의미)
    print(f"  랜덤 더미 데이터: {num_samples} 샘플  (latency-only 모드)")
    return images, labels


# ── 단일 threshold 1회 실행 (TRT) ────────────────────────────────────────────

def run_sweep_once_trt(engine: SelectiveViTTRT,
                       images, labels,
                       threshold: float,
                       warmup: int = 20,
                       latency_only: bool = False):
    exit_blocks = engine.exit_blocks
    n_exits     = engine.n_segs
    exit_counts      = [0] * n_exits
    correct_per_exit = [0] * n_exits
    latencies        = []
    correct          = 0

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_cuda = img.cuda()

        logits, exit_block, lat_ms = engine.infer(img_cuda, threshold=threshold)

        if i >= warmup:
            latencies.append(lat_ms)

        head_idx = exit_blocks.index(exit_block)
        exit_counts[head_idx] += 1

        if not latency_only:
            pred = logits.argmax(dim=1).item()
            if pred == lbl:
                correct += 1
                correct_per_exit[head_idx] += 1

    n   = len(labels)
    lat = np.array(latencies) if latencies else np.array([0.0])

    accuracy = (correct / n) if not latency_only else None
    acc_per_exit = [
        (correct_per_exit[i] / exit_counts[i]) if (exit_counts[i] > 0 and not latency_only) else None
        for i in range(n_exits)
    ]
    avg_exit_block = sum(exit_blocks[i] * exit_counts[i] for i in range(n_exits)) / n

    return {
        'accuracy':       accuracy,
        'exit_counts':    exit_counts,
        'exit_rate':      [c / n * 100 for c in exit_counts],
        'acc_per_exit':   acc_per_exit,
        'avg_exit_block': avg_exit_block,
        'latencies_ms':   latencies,
        'avg_ms':  float(np.mean(lat)),
        'p50_ms':  float(np.percentile(lat, 50)),
        'p99_ms':  float(np.percentile(lat, 99)),
    }


# ── N회 반복 sweep ────────────────────────────────────────────────────────────

def run_n_sweeps_trt(engine: SelectiveViTTRT,
                     images, labels,
                     thresholds: list, N: int,
                     warmup: int = 20,
                     latency_only: bool = False):
    exit_blocks  = engine.exit_blocks
    labels_str   = [f'B{b}' for b in exit_blocks]

    results = {str(round(t, 2)): {
        'threshold':      round(t, 2),
        'accuracy':       None,
        'exit_rate':      None,
        'acc_per_exit':   None,
        'avg_exit_block': None,
        'runs': [],
    } for t in thresholds}

    total = N * len(thresholds)
    done  = 0

    print(f"\n  {'thr':>6}  {'run':>4}  {'avg_ms':>8}  {'p50_ms':>8}  {'p99_ms':>8}  {'avg_block':>10}")
    print(f"  {'-'*60}")

    for run_idx in range(N):
        for thr in thresholds:
            key = str(round(thr, 2))
            r   = run_sweep_once_trt(engine, images, labels, thr,
                                     warmup=warmup, latency_only=latency_only)

            if results[key]['accuracy'] is None and not latency_only:
                results[key]['accuracy']       = r['accuracy']
                results[key]['exit_rate']      = r['exit_rate']
                results[key]['acc_per_exit']   = r['acc_per_exit']
                results[key]['avg_exit_block'] = r['avg_exit_block']

            if results[key]['exit_rate'] is None:
                results[key]['exit_rate']      = r['exit_rate']
                results[key]['avg_exit_block'] = r['avg_exit_block']

            results[key]['runs'].append({
                'run_idx':      run_idx,
                'avg_ms':       r['avg_ms'],
                'p50_ms':       r['p50_ms'],
                'p99_ms':       r['p99_ms'],
                'latencies_ms': r['latencies_ms'],
            })

            done += 1
            exit_str = "  ".join(
                f"{labels_str[i]}={r['exit_rate'][i]:.0f}%"
                for i in range(engine.n_segs)
            )
            print(f"  {thr:.2f}  run {run_idx+1:>3}/{N}  "
                  f"{r['avg_ms']:>8.2f}  {r['p50_ms']:>8.2f}  {r['p99_ms']:>8.2f}  "
                  f"avg={r['avg_exit_block']:>5.1f}  [{done}/{total}]  {exit_str}")

    for key in results:
        runs  = results[key]['runs']
        p99s  = [r['p99_ms'] for r in runs]
        avgs  = [r['avg_ms'] for r in runs]
        p50s  = [r['p50_ms'] for r in runs]
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT TRT Threshold Sweep (Orin)'
    )
    parser.add_argument('--exit-blocks',  type=int, nargs='+', required=True)
    parser.add_argument('--n',            type=int, default=10)
    parser.add_argument('--num-samples',  type=int, default=1000)
    parser.add_argument('--thresholds',   type=float, nargs='+',
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument('--data-root',    type=str,   default=None)
    parser.add_argument('--latency-only', action='store_true',
                        help='랜덤 노이즈로 latency만 측정 (ImageNet 없을 때)')
    parser.add_argument('--num-workers',  type=int,   default=2)
    parser.add_argument('--warmup',       type=int,   default=20)
    parser.add_argument('--out-dir',      type=str,   default=None)
    args = parser.parse_args()

    exit_blocks = args.exit_blocks
    if exit_blocks[-1] != 12:
        parser.error("Last exit block must be 12.")
    if exit_blocks != sorted(exit_blocks):
        parser.error("--exit-blocks must be ascending.")

    n_exits    = len(exit_blocks)
    model_name = f"ee_vit_{n_exits}exit"
    model_tag  = f"SelectiveExitViT-TRT ({'+'.join(f'B{b}' for b in exit_blocks)})"

    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval',
        f'vit_trt_sel{n_exits}exit_sweep_N{args.n}_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model   : {model_tag}")
    print(f"N       : {args.n} runs  samples={args.num_samples}  warmup={args.warmup}")
    print(f"Output  : {out_dir}")

    # ── 데이터 로드 ──
    latency_only = args.latency_only
    if not latency_only and args.data_root is None:
        from utils import load_config
        cfg = load_config('configs/train.yaml')
        dr = cfg.get('vit', {}).get('data_root',
             cfg.get('imagenet', {}).get('data_root', None))
        if dr is None or not os.path.isdir(dr):
            print("[WARN] data_root 없음 → latency-only 모드로 전환")
            latency_only = True
        else:
            args.data_root = dr

    if latency_only:
        images, labels = make_dummy_samples(args.num_samples)
    else:
        images, labels = load_imagenet_samples(
            args.num_samples, args.data_root, args.num_workers)

    # ── TRT 엔진 로드 ──
    print("\nLoading TRT engines ...")
    engine = load_selective_vit_trt(exit_blocks)

    # ── Sweep ──
    thresholds = sorted(set(args.thresholds))
    print(f"\nSweep start ({args.n} runs × {len(thresholds)} thresholds × {len(images)} samples) ...")
    results = run_n_sweeps_trt(engine, images, labels, thresholds,
                                args.n, warmup=args.warmup,
                                latency_only=latency_only)

    # ── 저장 ──
    print("\nSaving results ...")
    save_raw_json(results, len(images), "orin_trt", exit_blocks, thresholds, args.n,
                  os.path.join(out_dir, 'trt_sel_sweep_raw.json'))
    save_summary_csv(results, exit_blocks,
                     os.path.join(out_dir, 'trt_sel_sweep_summary.csv'))

    print("Generating plots ...")
    plot_exit_heatmap(results, exit_blocks, model_tag,
                      os.path.join(out_dir, 'trt_sel_sweep_exit_heatmap.png'))
    if not latency_only:
        plot_acc_heatmap(results, exit_blocks, model_tag,
                         os.path.join(out_dir, 'trt_sel_sweep_acc_heatmap.png'))
    plot_latency_dist(results, args.n, model_tag,
                      os.path.join(out_dir, 'trt_sel_sweep_latency_dist.png'))
    plot_summary(results, exit_blocks, model_tag,
                 os.path.join(out_dir, 'trt_sel_sweep_summary.png'))

    if not latency_only:
        print_result_table(results, exit_blocks)

    print(f"\nDone! Results at:\n  {out_dir}")


if __name__ == '__main__':
    main()
