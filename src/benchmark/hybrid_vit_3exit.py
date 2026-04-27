"""
hybrid_vit_3exit.py — SelectiveExitViT 3-exit 하이브리드 런타임 벤치마크 (3가지 variant)

3-exit 모델 (B6, B9, B12) 에서 "batch 처리 포인트"를 어디에 두느냐에 따라 3가지 variant:

  Variant A — single_batch (첫 exit 이후 한 번만 배치):
    Seg1: sample-by-sample → 비탈출 → 큐 적재
    큐 flush: → Seg2 batch → 비탈출은 즉시 Seg3 batch (같은 묶음으로 연속 처리)
    배치 경계: Seg1→Seg2 1곳만.

  Variant B — last_only (마지막 segment만 배치):
    Seg1: sample-by-sample (배치 없음)
    Seg2: sample-by-sample (배치 없음)
    Seg2 비탈출 → 큐 적재 → flush → Seg3 batch
    배치 경계: Seg2→Seg3 1곳만.

  Variant C — cascade (각 segment 경계마다 독립 배치):
    큐1: Seg1 비탈출 누적 → flush → Seg2 batch
    큐2: Seg2 비탈출 누적 → flush → Seg3 batch
    큐1/큐2 각각 동일한 (batch_size, timeout_ms) 파라미터 적용.
    배치 경계: Seg1→Seg2, Seg2→Seg3 두 곳.

Grid search:
  batch_size × timeout_ms 조합별 시뮬레이션 (사전계산 + LUT 기반, GPU 재실행 없음).

출력 ({EXP_DIR}/eval/hybrid_3exit_YYYYMMDD/):
  hybrid_3exit_plain.json
  hybrid_3exit_{variant}_grid.csv
  hybrid_3exit_{variant}_grid_avg_heatmap.png
  hybrid_3exit_{variant}_grid_p99_heatmap.png
  hybrid_3exit_variant_comparison.png  — 3 variant 최적 조합 비교

사용법:
  cd src
  python benchmark/hybrid_vit_3exit.py --threshold 0.80
  python benchmark/hybrid_vit_3exit.py --threshold 0.80 \\
      --batch-sizes 1 4 8 16 32 --timeout-ms 1 2 5 10 20

인자:
  --threshold      early exit threshold (필수)
  --data-root      ImageNet val 루트 (기본: /home2/imagenet)
  --exit-blocks    3-exit 블록 번호 (기본: 6 9 12)
  --batch-sizes    grid search batch_size 목록 (기본: 1 4 8 16 32)
  --timeout-ms     grid search timeout 목록 ms (기본: 1 2 5 10 20)
  --variants       실행할 variant 목록 (기본: A B C)
  --warmup         warmup 샘플 수 (기본: 200)
  --num-workers    DataLoader 워커 수 (기본: 8)
  --lut-reps       LUT 측정 반복 횟수 (기본: 50)
  --out-dir        결과 저장 디렉토리 (기본: auto)
  --device-label   플롯 디바이스 이름 (기본: RTX 5090)
  --skip-plain     PlainViT 기준선 측정 스킵
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.plain_vit import build_model as build_plain
from models.ee_vit_selective import build_model as build_selective
from benchmark.hybrid_vit_utils import (
    precompute_all_3exit, measure_seg_lut,
    lut_lookup, bench_plain, lat_stats, throughput_stats,
)
from benchmark.benchmark_pytorch_vit import build_val_loader, load_checkpoint


# ── Variant A: single_batch ───────────────────────────────────────────────────

def simulate_A(precomp, seg2_data, seg3_preds_of_s2_nonexit,
               seg2_lut, seg3_lut,
               threshold, batch_size, timeout_ms,
               eb1, eb2, eb3) -> dict:
    """
    Variant A — 첫 번째 exit 이후 한 번만 배치.
    큐 flush 시 → Seg2 batch → 비탈출은 즉시 Seg3 batch (새 큐 없음).
    """
    confs_s1   = precomp['confs']
    preds_s1   = precomp['preds']
    seg1_times = precomp['seg1_times']
    labels     = precomp['labels']
    N          = len(labels)

    confs_s2   = seg2_data['confs']    # 인덱스: Seg1 비탈출 중 j번째
    preds_s2   = seg2_data['preds']

    # Seg1 비탈출 → j번째 샘플에 대해 (seg2_conf, seg2_pred, seg3_pred) 매핑
    ne1_mask  = confs_s1 < threshold
    ne1_idxs  = np.where(ne1_mask)[0]   # 원본 샘플 인덱스
    # seg2_data는 ne1_idxs 순서로 indexed
    # ne2_mask: ne1 내에서 seg2도 비탈출
    ne2_mask_in_ne1 = confs_s2 < threshold
    ne2_pos_in_ne1  = np.where(ne2_mask_in_ne1)[0]  # ne1 내 위치
    # seg3_preds는 ne2_pos_in_ne1 순서
    seg3_pred_map = {}  # ne1 내 위치 → seg3 pred
    for k, pos in enumerate(ne2_pos_in_ne1):
        seg3_pred_map[int(pos)] = int(seg3_preds_of_s2_nonexit[k])

    response_times = []
    throughputs    = []  # per-sample 유효 처리량 (samples/ms)
    exit_at        = []
    correct        = []

    q_t_s1_start = []
    q_t_entry    = []
    q_ne1_pos    = []   # ne1 내 위치 (seg2_data 인덱스)

    t = 0.0

    def flush():
        nonlocal t
        if not q_ne1_pos:
            return
        bs_q    = len(q_ne1_pos)
        seg2_ms = lut_lookup(seg2_lut, bs_q)
        t      += seg2_ms
        seg2_tput = bs_q / seg2_ms  # Seg2 배치 처리량

        # Seg2 판단 → exit or 즉시 Seg3
        seg3_batch_pos    = []  # ne1 내 위치 (Seg3 입력)
        seg3_batch_starts = []  # 해당 샘플들의 seg1 시작 시각

        for k, pos in enumerate(q_ne1_pos):
            if confs_s2[pos] >= threshold:  # Seg2에서 탈출
                response_times.append(t - q_t_s1_start[k])
                throughputs.append(seg2_tput)
                exit_at.append(eb2)
                orig_idx = int(ne1_idxs[pos])
                correct.append(int(preds_s2[pos] == labels[orig_idx]))
            else:                           # Seg2도 탈출 실패 → Seg3로
                seg3_batch_pos.append(pos)
                seg3_batch_starts.append(q_t_s1_start[k])

        if seg3_batch_pos:
            n_seg3   = len(seg3_batch_pos)
            seg3_ms  = lut_lookup(seg3_lut, n_seg3)
            t       += seg3_ms
            seg3_tput = n_seg3 / seg3_ms
            for k, pos in enumerate(seg3_batch_pos):
                response_times.append(t - seg3_batch_starts[k])
                throughputs.append(seg3_tput)
                exit_at.append(eb3)
                orig_idx = int(ne1_idxs[pos])
                s3_pred  = seg3_pred_map.get(pos, -1)
                correct.append(int(s3_pred == labels[orig_idx]))

        q_t_s1_start.clear()
        q_t_entry.clear()
        q_ne1_pos.clear()

    ne1_counter = 0   # ne1_idxs 내 순서 추적

    for i in range(N):
        t_start = t
        t      += seg1_times[i]

        if confs_s1[i] >= threshold:
            response_times.append(seg1_times[i])
            throughputs.append(1.0 / seg1_times[i])
            exit_at.append(eb1)
            correct.append(int(preds_s1[i] == labels[i]))
        else:
            pos = ne1_counter
            ne1_counter += 1
            q_t_s1_start.append(t_start)
            q_t_entry.append(t)
            q_ne1_pos.append(pos)

            oldest_wait = t - q_t_entry[0]
            if len(q_ne1_pos) >= batch_size or oldest_wait >= timeout_ms:
                flush()

    flush()

    total_sim_time = t
    n       = len(response_times)
    n_e1    = sum(1 for b in exit_at if b == eb1)
    n_e2    = sum(1 for b in exit_at if b == eb2)
    return {
        'variant':         'A_single_batch',
        'batch_size':      batch_size,
        'timeout_ms':      timeout_ms,
        'threshold':       threshold,
        'accuracy':        sum(correct) / n,
        'accuracy_pct':    sum(correct) / n * 100,
        'exit_rate_b1':    n_e1 / n * 100,
        'exit_rate_b2':    n_e2 / n * 100,
        'exit_rate_b3':    (n - n_e1 - n_e2) / n * 100,
        'n_samples':       n,
        'overall_tput':    n / total_sim_time,
        **lat_stats(response_times),
        **throughput_stats(throughputs),
    }


# ── Variant B: last_only ──────────────────────────────────────────────────────

def simulate_B(precomp, seg2_data, seg3_preds_of_s2_nonexit,
               seg2_lut, seg3_lut,
               threshold, batch_size, timeout_ms,
               eb1, eb2, eb3) -> dict:
    """
    Variant B — 마지막 segment만 배치.
    Seg1, Seg2는 sample-by-sample (LUT bs=1 시간). Seg3만 batch 큐.
    """
    confs_s1   = precomp['confs']
    preds_s1   = precomp['preds']
    seg1_times = precomp['seg1_times']
    labels     = precomp['labels']
    N          = len(labels)

    confs_s2   = seg2_data['confs']
    preds_s2   = seg2_data['preds']

    ne1_mask    = confs_s1 < threshold
    ne1_idxs    = np.where(ne1_mask)[0]
    ne2_mask_in = confs_s2 < threshold
    ne2_pos     = np.where(ne2_mask_in)[0]
    seg3_pred_map = {int(p): int(seg3_preds_of_s2_nonexit[k])
                     for k, p in enumerate(ne2_pos)}

    seg2_single_ms = lut_lookup(seg2_lut, 1)
    seg2_single_tput = 1.0 / seg2_single_ms  # bs=1 처리량

    response_times = []
    throughputs    = []  # per-sample 유효 처리량 (samples/ms)
    exit_at        = []
    correct        = []

    q_t_s1_start = []
    q_t_entry    = []
    q_ne1_pos    = []

    t           = 0.0
    ne1_counter = 0

    def flush_seg3():
        nonlocal t
        if not q_ne1_pos:
            return
        bs_q    = len(q_ne1_pos)
        seg3_ms = lut_lookup(seg3_lut, bs_q)
        t      += seg3_ms
        seg3_tput = bs_q / seg3_ms
        for k, pos in enumerate(q_ne1_pos):
            response_times.append(t - q_t_s1_start[k])
            throughputs.append(seg3_tput)
            exit_at.append(eb3)
            orig_idx = int(ne1_idxs[pos])
            s3_pred  = seg3_pred_map.get(pos, -1)
            correct.append(int(s3_pred == labels[orig_idx]))
        q_t_s1_start.clear()
        q_t_entry.clear()
        q_ne1_pos.clear()

    for i in range(N):
        t_start = t
        t      += seg1_times[i]

        if confs_s1[i] >= threshold:
            response_times.append(t - t_start)
            throughputs.append(1.0 / seg1_times[i])
            exit_at.append(eb1)
            correct.append(int(preds_s1[i] == labels[i]))
        else:
            pos = ne1_counter
            ne1_counter += 1
            # Seg2 sample-by-sample (no batching)
            t += seg2_single_ms
            if confs_s2[pos] >= threshold:
                response_times.append(t - t_start)
                throughputs.append(seg2_single_tput)
                exit_at.append(eb2)
                orig_idx = int(ne1_idxs[pos])
                correct.append(int(preds_s2[pos] == labels[orig_idx]))
            else:
                # Seg3 큐 적재
                q_t_s1_start.append(t_start)
                q_t_entry.append(t)
                q_ne1_pos.append(pos)

                oldest_wait = t - q_t_entry[0]
                if len(q_ne1_pos) >= batch_size or oldest_wait >= timeout_ms:
                    flush_seg3()

    flush_seg3()

    total_sim_time = t
    n    = len(response_times)
    n_e1 = sum(1 for b in exit_at if b == eb1)
    n_e2 = sum(1 for b in exit_at if b == eb2)
    return {
        'variant':      'B_last_only',
        'batch_size':   batch_size,
        'timeout_ms':   timeout_ms,
        'threshold':    threshold,
        'accuracy':     sum(correct) / n,
        'accuracy_pct': sum(correct) / n * 100,
        'exit_rate_b1': n_e1 / n * 100,
        'exit_rate_b2': n_e2 / n * 100,
        'exit_rate_b3': (n - n_e1 - n_e2) / n * 100,
        'n_samples':    n,
        'overall_tput': n / total_sim_time,
        **lat_stats(response_times),
        **throughput_stats(throughputs),
    }


# ── Variant C: cascade ────────────────────────────────────────────────────────

def simulate_C(precomp, seg2_data, seg3_preds_of_s2_nonexit,
               seg2_lut, seg3_lut,
               threshold, batch_size, timeout_ms,
               eb1, eb2, eb3) -> dict:
    """
    Variant C — 각 segment 경계마다 독립 배치 큐.
    큐1: Seg1→Seg2 / 큐2: Seg2→Seg3
    큐1 flush 후 Seg2 비탈출이 즉시 큐2에 합류.
    큐2는 이후 seg1 루프 중 조건 충족 시 flush.
    """
    confs_s1   = precomp['confs']
    preds_s1   = precomp['preds']
    seg1_times = precomp['seg1_times']
    labels     = precomp['labels']
    N          = len(labels)

    confs_s2   = seg2_data['confs']
    preds_s2   = seg2_data['preds']

    ne1_mask    = confs_s1 < threshold
    ne1_idxs    = np.where(ne1_mask)[0]
    ne2_mask_in = confs_s2 < threshold
    ne2_pos     = np.where(ne2_mask_in)[0]
    seg3_pred_map = {int(p): int(seg3_preds_of_s2_nonexit[k])
                     for k, p in enumerate(ne2_pos)}

    response_times = []
    throughputs    = []  # per-sample 유효 처리량 (samples/ms)
    exit_at        = []
    correct        = []

    # 큐1 (Seg1→Seg2)
    q1_t_start = []
    q1_t_entry = []
    q1_ne1_pos = []

    # 큐2 (Seg2→Seg3)
    q2_t_start = []
    q2_t_entry = []
    q2_ne1_pos = []

    t           = 0.0
    ne1_counter = 0

    def flush_q2():
        nonlocal t
        if not q2_ne1_pos:
            return
        bs_q    = len(q2_ne1_pos)
        seg3_ms = lut_lookup(seg3_lut, bs_q)
        t      += seg3_ms
        seg3_tput = bs_q / seg3_ms
        for k, pos in enumerate(q2_ne1_pos):
            response_times.append(t - q2_t_start[k])
            throughputs.append(seg3_tput)
            exit_at.append(eb3)
            orig_idx = int(ne1_idxs[pos])
            s3_pred  = seg3_pred_map.get(pos, -1)
            correct.append(int(s3_pred == labels[orig_idx]))
        q2_t_start.clear()
        q2_t_entry.clear()
        q2_ne1_pos.clear()

    def flush_q1():
        nonlocal t
        if not q1_ne1_pos:
            return
        bs_q    = len(q1_ne1_pos)
        seg2_ms = lut_lookup(seg2_lut, bs_q)
        t      += seg2_ms
        seg2_tput = bs_q / seg2_ms

        for k, pos in enumerate(q1_ne1_pos):
            orig_idx = int(ne1_idxs[pos])
            if confs_s2[pos] >= threshold:
                response_times.append(t - q1_t_start[k])
                throughputs.append(seg2_tput)
                exit_at.append(eb2)
                correct.append(int(preds_s2[pos] == labels[orig_idx]))
            else:
                # 큐2 적재 (throughput은 flush_q2에서 할당)
                q2_t_start.append(q1_t_start[k])
                q2_t_entry.append(t)
                q2_ne1_pos.append(pos)

        q1_t_start.clear()
        q1_t_entry.clear()
        q1_ne1_pos.clear()

        # 큐1 flush 직후 큐2 flush 조건 체크
        if q2_ne1_pos:
            oldest_q2 = t - q2_t_entry[0]
            if len(q2_ne1_pos) >= batch_size or oldest_q2 >= timeout_ms:
                flush_q2()

    for i in range(N):
        t_start = t
        t      += seg1_times[i]

        if confs_s1[i] >= threshold:
            response_times.append(t - t_start)
            throughputs.append(1.0 / seg1_times[i])
            exit_at.append(eb1)
            correct.append(int(preds_s1[i] == labels[i]))
        else:
            pos = ne1_counter
            ne1_counter += 1
            q1_t_start.append(t_start)
            q1_t_entry.append(t)
            q1_ne1_pos.append(pos)

            # 큐1 flush 조건
            oldest_q1 = t - q1_t_entry[0]
            if len(q1_ne1_pos) >= batch_size or oldest_q1 >= timeout_ms:
                flush_q1()

            # 큐2 flush 조건 (Seg1 루프 중에도 체크)
            if q2_ne1_pos:
                oldest_q2 = t - q2_t_entry[0]
                if len(q2_ne1_pos) >= batch_size or oldest_q2 >= timeout_ms:
                    flush_q2()

    flush_q1()
    flush_q2()

    total_sim_time = t
    n    = len(response_times)
    n_e1 = sum(1 for b in exit_at if b == eb1)
    n_e2 = sum(1 for b in exit_at if b == eb2)
    return {
        'variant':      'C_cascade',
        'batch_size':   batch_size,
        'timeout_ms':   timeout_ms,
        'threshold':    threshold,
        'accuracy':     sum(correct) / n,
        'accuracy_pct': sum(correct) / n * 100,
        'exit_rate_b1': n_e1 / n * 100,
        'exit_rate_b2': n_e2 / n * 100,
        'exit_rate_b3': (n - n_e1 - n_e2) / n * 100,
        'n_samples':    n,
        'overall_tput': n / total_sim_time,
        **lat_stats(response_times),
        **throughput_stats(throughputs),
    }


# ── Grid Search ───────────────────────────────────────────────────────────────

def run_grid(sim_fn, label, precomp, seg2_data, seg3_preds,
             seg2_lut, seg3_lut, threshold,
             batch_sizes, timeout_ms_list, eb1, eb2, eb3) -> list:
    results = []
    total = len(batch_sizes) * len(timeout_ms_list)
    print(f"\n  [Variant {label}] {total} combos ...")
    idx = 0
    for bs in batch_sizes:
        for tms in timeout_ms_list:
            st = sim_fn(precomp, seg2_data, seg3_preds, seg2_lut, seg3_lut,
                        threshold, bs, tms, eb1, eb2, eb3)
            results.append(st)
            idx += 1
            print(f"    [{idx:>3}/{total}] bs={bs:>3} tms={tms:>5.1f}ms  "
                  f"acc={st['accuracy_pct']:.2f}%  avg={st['avg_ms']:.2f}ms  "
                  f"p99={st['p99_ms']:.2f}ms  "
                  f"exit=[{st['exit_rate_b1']:.0f}%/{st['exit_rate_b2']:.0f}%/{st['exit_rate_b3']:.0f}%]")
    return results


# ── Save ─────────────────────────────────────────────────────────────────────

def save_csv(rows, path):
    fields = ['variant', 'batch_size', 'timeout_ms', 'threshold', 'accuracy_pct',
              'exit_rate_b1', 'exit_rate_b2', 'exit_rate_b3',
              'avg_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'std_ms',
              'overall_tput',
              'avg_tput', 'p50_tput', 'p90_tput', 'p95_tput', 'p99_tput', 'std_tput',
              'n_samples']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _grid_mat(results, batch_sizes, timeout_ms_list, metric):
    mat = np.full((len(batch_sizes), len(timeout_ms_list)), np.nan)
    for st in results:
        bi = batch_sizes.index(st['batch_size'])
        ti = timeout_ms_list.index(st['timeout_ms'])
        mat[bi, ti] = st[metric]
    return mat


def plot_heatmap(results, plain_val, batch_sizes, timeout_ms_list,
                 metric, metric_label, out_path, device_label, variant_label,
                 higher_better=False, raw_fmt='.2f', raw_unit='ms'):
    mat_raw = _grid_mat(results, batch_sizes, timeout_ms_list, metric)
    if higher_better:
        mat = (mat_raw - plain_val) / abs(plain_val) * 100
    else:
        mat = (plain_val - mat_raw) / plain_val * 100

    fig, ax = plt.subplots(figsize=(max(6, len(timeout_ms_list) + 1),
                                    max(4, len(batch_sizes) + 1)))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto',
                   vmin=min(-5, np.nanmin(mat)), vmax=max(5, np.nanmax(mat)))
    plt.colorbar(im, ax=ax, label='Improvement over PlainViT (%)')
    ax.set_xticks(range(len(timeout_ms_list)))
    ax.set_xticklabels([f'{t}ms' for t in timeout_ms_list])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([f'bs={b}' for b in batch_sizes])
    ax.set_xlabel('Timeout')
    ax.set_ylabel('Batch Size')
    ax.set_title(f'{metric_label} Improvement — Variant {variant_label}\n'
                 f'(thr={results[0]["threshold"]:.2f}, {device_label})')
    for bi in range(len(batch_sizes)):
        for ti in range(len(timeout_ms_list)):
            v = mat[bi, ti]
            raw = mat_raw[bi, ti]
            if not np.isnan(v):
                ax.text(ti, bi, f'{v:+.1f}%\n({raw:{raw_fmt}}{raw_unit})',
                        ha='center', va='center', fontsize=7,
                        color='black' if abs(v) < 15 else 'white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  heatmap ({variant_label}): {out_path}")


def plot_variant_comparison(bests: dict, plain_st, out_path, device_label, threshold):
    """각 variant의 best 조합을 PlainViT와 함께 비교하는 bar chart."""
    names = ['PlainViT'] + [f"Var {k}\nbs={v['batch_size']} t={v['timeout_ms']}ms"
                             for k, v in bests.items()]
    sts   = [plain_st] + list(bests.values())
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple']
    metrics = ['avg_ms', 'p90_ms', 'p95_ms', 'p99_ms']
    labels  = ['avg', 'p90', 'p95', 'p99']

    x     = np.arange(len(metrics))
    width = 0.18
    n     = len(names)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ni, (name, st) in enumerate(zip(names, sts)):
        vals = [st[m] for m in metrics]
        offset = (ni - n / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width, label=name, color=colors[ni % len(colors)], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'Latency Comparison  ({device_label})')
    ax1.legend(fontsize=7)
    ax1.grid(axis='y', alpha=0.3)

    # Accuracy 비교
    accs = [plain_st['accuracy'] * 100] + [v['accuracy_pct'] for v in bests.values()]
    short_names = ['Plain'] + [f'Var {k}' for k in bests.keys()]
    ax2.bar(short_names, accs, color=colors[:len(short_names)], alpha=0.85)
    for i, (c, v) in enumerate(zip(short_names, accs)):
        ax2.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontsize=8)
    ax2.set_ylabel('Top-1 Accuracy (%)')
    ax2.set_title(f'Accuracy  (thr={threshold:.2f})')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f'PlainViT vs 3-exit Hybrid Variants  ({device_label})', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  variant comparison: {out_path}")


def print_table(results, plain_st, variant_label, device_label):
    print(f"\n{'='*100}")
    print(f"  Hybrid 3-exit Variant {variant_label} — {device_label}  "
          f"(thr={results[0]['threshold']:.2f})")
    print(f"  PlainViT: acc={plain_st['accuracy']*100:.2f}%  "
          f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms")
    print(f"{'='*100}")
    hdr = (f"  {'bs':>4} {'tms':>7}  {'acc':>8}  {'avg':>7}  {'p90':>7}  "
           f"{'p99':>7}  {'B6%':>7}  {'B9%':>7}  {'B12%':>7}")
    print(hdr)
    print(f"  {'-'*96}")
    for r in sorted(results, key=lambda x: x['avg_ms']):
        print(f"  {r['batch_size']:>4} {r['timeout_ms']:>6.1f}ms  "
              f"{r['accuracy_pct']:>7.2f}%  "
              f"{r['avg_ms']:>7.2f}  {r['p90_ms']:>7.2f}  {r['p99_ms']:>7.2f}  "
              f"{r['exit_rate_b1']:>6.1f}%  {r['exit_rate_b2']:>6.1f}%  {r['exit_rate_b3']:>6.1f}%")
    print(f"{'='*100}")


# ── Main ─────────────────────────────────────────────────────────────────────

VARIANT_FNS = {
    'A': (simulate_A, 'A_single_batch'),
    'B': (simulate_B, 'B_last_only'),
    'C': (simulate_C, 'C_cascade'),
}


def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT 3-exit Hybrid Runtime Benchmark (Variants A/B/C)'
    )
    parser.add_argument('--threshold',    type=float, required=True)
    parser.add_argument('--data-root',    type=str,   default='/home2/imagenet')
    parser.add_argument('--exit-blocks',  type=int,   nargs='+', default=[6, 9, 12])
    parser.add_argument('--batch-sizes',  type=int,   nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--timeout-ms',   type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument('--variants',     type=str,   nargs='+', default=['A', 'B', 'C'])
    parser.add_argument('--warmup',       type=int,   default=200)
    parser.add_argument('--num-workers',  type=int,   default=8)
    parser.add_argument('--lut-reps',     type=int,   default=50)
    parser.add_argument('--out-dir',      type=str,   default=None)
    parser.add_argument('--device-label', type=str,   default='RTX 5090')
    parser.add_argument('--skip-plain',   action='store_true')
    args = parser.parse_args()

    eb1, eb2, eb3 = args.exit_blocks[0], args.exit_blocks[1], args.exit_blocks[2]

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'hybrid_3exit_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device    : {device}  ({args.device_label})")
    print(f"Threshold : {args.threshold}")
    print(f"ExitBlocks: {args.exit_blocks}")
    print(f"Variants  : {args.variants}")
    print(f"BatchSizes: {args.batch_sizes}")
    print(f"TimeoutMs : {args.timeout_ms}")
    print(f"Output    : {out_dir}\n")

    # ── 모델 / 데이터 ──
    ckpt = paths.latest_checkpoint('ee_vit_3exit')
    assert ckpt is not None, "ee_vit_3exit checkpoint 없음"
    print(f"Checkpoint: {ckpt}")
    model = build_selective(args.exit_blocks).to(device)
    load_checkpoint(model, ckpt)
    model.eval()

    loader = build_val_loader(args.data_root, args.num_workers)

    # ── Step 1: 단일 패스 사전계산 (피처 미저장, 스칼라만 ~1 MB) ──
    print(f"\n[Step 1] Single-pass precompute  ({len(loader):,} samples) ...")
    data    = precompute_all_3exit(model, loader, device, eb1, eb2, eb3,
                                   args.threshold, args.warmup)
    precomp = {
        'confs':      data['confs_s1'],
        'preds':      data['preds_s1'],
        'seg1_times': data['seg1_times'],
        'labels':     data['labels'],
    }
    seg2_data  = {'confs': data['confs_s2'], 'preds': data['preds_s2']}
    seg3_preds = data['preds_s3']
    N = len(precomp['labels'])
    ne1_count = int((precomp['confs'] < args.threshold).sum())
    ne2_count = int((seg2_data['confs'] < args.threshold).sum())
    print(f"  → {N:,} samples  (ne1: {ne1_count:,}, ne2: {ne2_count:,})")

    # ── Step 2: LUT 측정 ──
    print(f"\n[Step 2] Seg2 latency LUT ...")
    seg2_lut = measure_seg_lut(model, device, eb1, eb2, 1,
                                args.batch_sizes, args.lut_reps)
    print(f"\n[Step 2] Seg3 latency LUT ...")
    seg3_lut = measure_seg_lut(model, device, eb2, eb3, 2,
                                args.batch_sizes, args.lut_reps)

    del model
    torch.cuda.empty_cache()

    # ── Step 3: PlainViT 기준선 ──
    plain_st = None
    if not args.skip_plain:
        print(f"\n[Step 3] PlainViT baseline ...")
        plain_model = build_plain().to(device)
        plain_lats, plain_correct = bench_plain(plain_model, loader, device, args.warmup)
        del plain_model
        torch.cuda.empty_cache()
        plain_throughputs = [1.0 / l for l in plain_lats]
        plain_st = {
            'accuracy':     sum(plain_correct) / len(plain_correct),
            'accuracy_pct': sum(plain_correct) / len(plain_correct) * 100,
            'overall_tput': len(plain_lats) / sum(plain_lats),
            **lat_stats(plain_lats),
            **throughput_stats(plain_throughputs),
        }
        print(f"  PlainViT: acc={plain_st['accuracy_pct']:.2f}%  "
              f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms  "
              f"avg_tput={plain_st['avg_tput']:.4f}/ms")
        with open(os.path.join(out_dir, 'hybrid_3exit_plain.json'), 'w') as f:
            json.dump(plain_st, f, indent=2)

    # ── Step 4: Grid Search (variant별) ──
    all_grids = {}
    bests     = {}
    dl        = args.device_label

    for vkey in args.variants:
        if vkey not in VARIANT_FNS:
            print(f"[WARN] 알 수 없는 variant: {vkey}  (A/B/C만 지원)")
            continue
        sim_fn, vlabel = VARIANT_FNS[vkey]
        grid = run_grid(sim_fn, vkey, precomp, seg2_data, seg3_preds,
                        seg2_lut, seg3_lut, args.threshold,
                        args.batch_sizes, args.timeout_ms, eb1, eb2, eb3)
        all_grids[vkey] = grid

        save_csv(grid, os.path.join(out_dir, f'hybrid_3exit_{vlabel}_grid.csv'))
        with open(os.path.join(out_dir, f'hybrid_3exit_{vlabel}_grid.json'), 'w') as f:
            json.dump(grid, f, indent=2)

        if plain_st:
            print_table(grid, plain_st, vkey, dl)
            plot_heatmap(grid, plain_st['avg_ms'], args.batch_sizes, args.timeout_ms,
                         'avg_ms', 'Avg Latency',
                         os.path.join(out_dir, f'hybrid_3exit_{vkey}_grid_avg_heatmap.png'),
                         dl, vkey)
            plot_heatmap(grid, plain_st['p99_ms'], args.batch_sizes, args.timeout_ms,
                         'p99_ms', 'P99 Latency',
                         os.path.join(out_dir, f'hybrid_3exit_{vkey}_grid_p99_heatmap.png'),
                         dl, vkey)
            plot_heatmap(grid, plain_st['avg_tput'], args.batch_sizes, args.timeout_ms,
                         'avg_tput', 'Avg Throughput',
                         os.path.join(out_dir, f'hybrid_3exit_{vkey}_grid_avg_tput_heatmap.png'),
                         dl, vkey, higher_better=True, raw_fmt='.4f', raw_unit='/ms')
            plot_heatmap(grid, plain_st['p99_tput'], args.batch_sizes, args.timeout_ms,
                         'p99_tput', 'P99 Throughput',
                         os.path.join(out_dir, f'hybrid_3exit_{vkey}_grid_p99_tput_heatmap.png'),
                         dl, vkey, higher_better=True, raw_fmt='.4f', raw_unit='/ms')

        best = min(grid, key=lambda x: x['avg_ms'])
        bests[vkey] = best
        print(f"\n  Variant {vkey} Best: bs={best['batch_size']} tms={best['timeout_ms']}ms  "
              f"avg={best['avg_ms']:.2f}ms  p99={best['p99_ms']:.2f}ms  "
              f"acc={best['accuracy_pct']:.2f}%")

    # ── 3-variant 비교 플롯 ──
    if plain_st and bests:
        plot_variant_comparison(
            bests, plain_st,
            os.path.join(out_dir, 'hybrid_3exit_variant_comparison.png'),
            dl, args.threshold
        )

    print(f"\nDone! → {out_dir}")


if __name__ == '__main__':
    main()
