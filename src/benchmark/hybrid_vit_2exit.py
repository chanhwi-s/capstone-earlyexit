"""
hybrid_vit_2exit.py — SelectiveExitViT 2-exit 하이브리드 런타임 벤치마크

동작 방식:
  Seg1(B1-B8)을 sample-by-sample로 실행.
  confidence >= threshold → 즉시 early exit.
  confidence <  threshold → 배치 큐에 적재.
  큐가 batch_size에 도달하거나 oldest 샘플의 대기 시간이 timeout_ms 초과 →
  Seg2(B9-B12) batch 실행 후 결과 반환.

grid search:
  batch_size × timeout_ms 조합별로 시뮬레이션 실행 → heatmap 출력.
  시뮬레이션은 사전계산된 seg1 결과 + LUT 기반으로 GPU 재실행 없이 고속 수행.

출력 ({EXP_DIR}/eval/hybrid_2exit_YYYYMMDD/):
  hybrid_2exit_plain.json           — PlainViT 기준선
  hybrid_2exit_grid.csv             — grid search 전체 결과
  hybrid_2exit_grid_avg_heatmap.png — avg latency 개선율 heatmap
  hybrid_2exit_grid_p99_heatmap.png — p99 latency 개선율 heatmap
  hybrid_2exit_best_comparison.png  — best 조합 vs PlainViT 비교

사용법:
  cd src
  python benchmark/hybrid_vit_2exit.py --threshold 0.80
  python benchmark/hybrid_vit_2exit.py --threshold 0.80 \\
      --batch-sizes 1 4 8 16 32 --timeout-ms 1 2 5 10 20

인자:
  --threshold      early exit threshold (필수)
  --data-root      ImageNet val 루트 (기본: /home2/imagenet)
  --exit-blocks    2-exit 블록 번호 (기본: 8 12)
  --batch-sizes    grid search batch_size 목록 (기본: 1 4 8 16 32)
  --timeout-ms     grid search timeout 목록 ms (기본: 1 2 5 10 20)
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
    precompute_all_2exit, measure_seg_lut,
    lut_lookup, bench_plain, lat_stats,
)
from benchmark.benchmark_pytorch_vit import build_val_loader, load_checkpoint


# ── 시뮬레이션 ────────────────────────────────────────────────────────────────

def simulate(precomp: dict, seg2_preds: np.ndarray, seg2_lut: dict,
             threshold: float, batch_size: int, timeout_ms: float,
             exit_block_1: int, exit_block_2: int) -> dict:
    """
    2-exit 하이브리드 런타임 시뮬레이션.

    타임라인(누적 GPU 시간 기준):
      - seg1 개별 실행 → exit or 큐 적재
      - 큐 flush 조건: len >= batch_size OR (oldest 대기 >= timeout_ms)
      - flush 시: seg2 LUT 레이턴시 추가, 응답 시간 기록

    precomp:    precompute_all_2exit 결과 (confs/preds/seg1_times/labels)
    seg2_preds: 비탈출 샘플에 대한 seg2 예측 (threshold 기준 비탈출 순서)
    seg2_lut:   {batch_size: latency_ms}
    """
    confs      = precomp['confs']
    preds_s1   = precomp['preds']
    seg1_times = precomp['seg1_times']
    labels     = precomp['labels']
    N          = len(labels)

    # 전체 비탈출 샘플의 seg2 pred 매핑
    non_exit_idxs = np.where(confs < threshold)[0]
    seg2_pred_map = {int(idx): int(seg2_preds[j])
                     for j, idx in enumerate(non_exit_idxs)}

    response_times = []
    exit_at        = []
    correct        = []

    # 큐 상태
    q_t_s1_start = []  # seg1 시작 시각 (각 샘플)
    q_t_entry    = []  # 큐 진입 시각 (seg1 완료 후)
    q_idxs       = []  # 원본 샘플 인덱스

    t = 0.0  # 누적 시뮬레이션 시간(ms)

    def flush():
        nonlocal t
        if not q_idxs:
            return
        bs     = len(q_idxs)
        bat_ms = lut_lookup(seg2_lut, bs)
        t     += bat_ms
        for k in range(bs):
            response_times.append(t - q_t_s1_start[k])
            exit_at.append(exit_block_2)
            sidx = q_idxs[k]
            correct.append(int(seg2_pred_map[sidx] == labels[sidx]))
        q_t_s1_start.clear()
        q_t_entry.clear()
        q_idxs.clear()

    for i in range(N):
        t_start = t
        t      += seg1_times[i]

        if confs[i] >= threshold:
            response_times.append(seg1_times[i])
            exit_at.append(exit_block_1)
            correct.append(int(preds_s1[i] == labels[i]))
        else:
            q_t_s1_start.append(t_start)
            q_t_entry.append(t)
            q_idxs.append(i)

            oldest_wait = t - q_t_entry[0]
            if len(q_idxs) >= batch_size or oldest_wait >= timeout_ms:
                flush()

    flush()  # 잔여 샘플 처리

    n       = len(response_times)
    n_early = sum(1 for b in exit_at if b == exit_block_1)

    return {
        'batch_size':      batch_size,
        'timeout_ms':      timeout_ms,
        'threshold':       threshold,
        'accuracy':        sum(correct) / n,
        'accuracy_pct':    sum(correct) / n * 100,
        'early_exit_rate': n_early / n * 100,
        'n_samples':       n,
        **lat_stats(response_times),
    }


# ── Grid Search ───────────────────────────────────────────────────────────────

def run_grid(precomp, seg2_preds, seg2_lut, threshold,
             batch_sizes, timeout_ms_list,
             exit_block_1, exit_block_2) -> list:
    results = []
    total = len(batch_sizes) * len(timeout_ms_list)
    idx   = 0
    for bs in batch_sizes:
        for tms in timeout_ms_list:
            st = simulate(precomp, seg2_preds, seg2_lut, threshold,
                          bs, tms, exit_block_1, exit_block_2)
            results.append(st)
            idx += 1
            print(f"  [{idx:>3}/{total}] bs={bs:>3} tms={tms:>5.1f}ms  "
                  f"acc={st['accuracy_pct']:.2f}%  avg={st['avg_ms']:.2f}ms  "
                  f"p99={st['p99_ms']:.2f}ms  early={st['early_exit_rate']:.1f}%")
    return results


# ── Save ─────────────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str, extra_fields: list = None):
    base = ['batch_size', 'timeout_ms', 'threshold', 'accuracy_pct',
            'early_exit_rate', 'avg_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms',
            'std_ms', 'n_samples']
    fields = base + (extra_fields or [])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _make_grid_matrix(results, batch_sizes, timeout_ms_list, metric):
    """grid 결과를 [len(batch_sizes), len(timeout_ms_list)] 행렬로 변환."""
    mat = np.full((len(batch_sizes), len(timeout_ms_list)), np.nan)
    for st in results:
        bi  = batch_sizes.index(st['batch_size'])
        ti  = timeout_ms_list.index(st['timeout_ms'])
        mat[bi, ti] = st[metric]
    return mat


def plot_grid_heatmap(results, plain_val, batch_sizes, timeout_ms_list,
                      metric, metric_label, out_path, device_label, higher_better=False):
    """
    grid search 결과 heatmap.
    셀 값 = (plain_val - hybrid_val) / plain_val * 100  (레이턴시 개선율 %)
    higher_better=True 이면 hybrid가 클수록 좋은 지표(accuracy 등).
    """
    mat_raw = _make_grid_matrix(results, batch_sizes, timeout_ms_list, metric)
    if higher_better:
        mat = (mat_raw - plain_val) / abs(plain_val) * 100   # positive = better
    else:
        mat = (plain_val - mat_raw) / plain_val * 100         # positive = improvement

    fig, ax = plt.subplots(figsize=(max(6, len(timeout_ms_list) + 1), max(4, len(batch_sizes) + 1)))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto',
                   vmin=min(-5, np.nanmin(mat)), vmax=max(5, np.nanmax(mat)))
    plt.colorbar(im, ax=ax, label='Improvement over PlainViT (%)')
    ax.set_xticks(range(len(timeout_ms_list)))
    ax.set_xticklabels([f'{t}ms' for t in timeout_ms_list])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([f'bs={b}' for b in batch_sizes])
    ax.set_xlabel('Timeout')
    ax.set_ylabel('Batch Size')
    ax.set_title(f'{metric_label} Improvement vs PlainViT\n'
                 f'(thr={results[0]["threshold"]:.2f}, {device_label})')
    for bi in range(len(batch_sizes)):
        for ti in range(len(timeout_ms_list)):
            v = mat[bi, ti]
            raw = mat_raw[bi, ti]
            if not np.isnan(v):
                txt = f'{v:+.1f}%\n({raw:.2f}ms)'
                ax.text(ti, bi, txt, ha='center', va='center', fontsize=7,
                        color='black' if abs(v) < 15 else 'white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  heatmap: {out_path}")


def plot_comparison(best_st, plain_st, out_path, device_label):
    """Best 조합 vs PlainViT bar chart."""
    models  = ['PlainViT', f"Hybrid 2-exit\nbs={best_st['batch_size']} t={best_st['timeout_ms']}ms"]
    metrics = ['avg_ms', 'p90_ms', 'p95_ms', 'p99_ms']
    labels  = ['avg', 'p90', 'p95', 'p99']
    colors  = ['steelblue', 'darkorange']
    x       = np.arange(len(metrics))
    width   = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Latency
    for mi, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = [plain_st[m], best_st[m]]
        bars = ax1.bar(x + (mi - 1.5) * width / 2, vals, width / 2,
                       label=lbl, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_ms', '') for m in metrics])
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'Latency Comparison  ({device_label})')
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Accuracy + Early exit rate
    cats = ['PlainViT', 'Hybrid 2-exit']
    accs = [plain_st['accuracy'] * 100, best_st['accuracy_pct']]
    ax2.bar(cats, accs, color=colors, alpha=0.85)
    for i, (c, v) in enumerate(zip(cats, accs)):
        ax2.text(i, v + 0.2, f'{v:.2f}%', ha='center', fontsize=9)
    ax2.set_ylabel('Top-1 Accuracy (%)')
    ax2.set_title(f"Accuracy (early exit={best_st['early_exit_rate']:.1f}%)")
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(f"PlainViT vs Hybrid 2-exit  (thr={best_st['threshold']:.2f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  comparison: {out_path}")


def print_table(results, plain_st, device_label):
    print(f"\n{'='*95}")
    print(f"  Hybrid 2-exit Grid Search — {device_label}  (thr={results[0]['threshold']:.2f})")
    print(f"  PlainViT baseline: acc={plain_st['accuracy']*100:.2f}%  "
          f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms")
    print(f"{'='*95}")
    hdr = f"  {'bs':>4} {'tms':>7}  {'acc':>8}  {'avg':>7}  {'p90':>7}  {'p99':>7}  {'early%':>8}"
    print(hdr)
    print(f"  {'-'*91}")
    for r in sorted(results, key=lambda x: x['avg_ms']):
        print(f"  {r['batch_size']:>4} {r['timeout_ms']:>6.1f}ms  "
              f"{r['accuracy_pct']:>7.2f}%  "
              f"{r['avg_ms']:>7.2f}  {r['p90_ms']:>7.2f}  {r['p99_ms']:>7.2f}  "
              f"{r['early_exit_rate']:>7.1f}%")
    print(f"{'='*95}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT 2-exit Hybrid Runtime Benchmark'
    )
    parser.add_argument('--threshold',    type=float, required=True)
    parser.add_argument('--data-root',    type=str,   default='/home2/imagenet')
    parser.add_argument('--exit-blocks',  type=int,   nargs='+', default=[8, 12])
    parser.add_argument('--batch-sizes',  type=int,   nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--timeout-ms',   type=float, nargs='+', default=[1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument('--warmup',       type=int,   default=200)
    parser.add_argument('--num-workers',  type=int,   default=8)
    parser.add_argument('--lut-reps',     type=int,   default=50)
    parser.add_argument('--out-dir',      type=str,   default=None)
    parser.add_argument('--device-label', type=str,   default='RTX 5090')
    parser.add_argument('--skip-plain',   action='store_true')
    args = parser.parse_args()

    eb1, eb2 = args.exit_blocks[0], args.exit_blocks[-1]

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'hybrid_2exit_{ts}'
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device    : {device}  ({args.device_label})")
    print(f"Threshold : {args.threshold}")
    print(f"ExitBlocks: {args.exit_blocks}")
    print(f"BatchSizes: {args.batch_sizes}")
    print(f"TimeoutMs : {args.timeout_ms}")
    print(f"Output    : {out_dir}\n")

    # ── 모델 / 데이터 ──
    ckpt = paths.latest_checkpoint('ee_vit_2exit')
    assert ckpt is not None, "ee_vit_2exit checkpoint 없음"
    print(f"Checkpoint: {ckpt}")
    model = build_selective(args.exit_blocks).to(device)
    load_checkpoint(model, ckpt)
    model.eval()

    loader = build_val_loader(args.data_root, args.num_workers)

    # ── Step 1: 단일 패스 사전계산 (피처 미저장, 스칼라만 ~1 MB) ──
    print(f"\n[Step 1] Single-pass precompute  ({len(loader):,} samples) ...")
    data    = precompute_all_2exit(model, loader, device, eb1, eb2,
                                   args.threshold, args.warmup)
    precomp = {
        'confs':      data['confs_s1'],
        'preds':      data['preds_s1'],
        'seg1_times': data['seg1_times'],
        'labels':     data['labels'],
    }
    seg2_preds = data['preds_s2']
    N = len(precomp['labels'])
    ne_count = int((precomp['confs'] < args.threshold).sum())
    print(f"  → {N:,} samples  (non-exiters: {ne_count:,})")

    # ── Step 2: Seg2 LUT 측정 ──
    print(f"\n[Step 2] Seg2 latency LUT ...")
    seg2_lut = measure_seg_lut(model, device,
                                start_block=eb1, end_block=eb2, head_idx=1,
                                batch_sizes=args.batch_sizes, n_reps=args.lut_reps)

    del model
    torch.cuda.empty_cache()

    # ── Step 3: PlainViT 기준선 ──
    plain_st = None
    if not args.skip_plain:
        print(f"\n[Step 4] PlainViT baseline ...")
        plain_model = build_plain().to(device)
        plain_lats, plain_correct = bench_plain(plain_model, loader, device, args.warmup)
        del plain_model
        torch.cuda.empty_cache()
        plain_st = {
            'accuracy':     sum(plain_correct) / len(plain_correct),
            'accuracy_pct': sum(plain_correct) / len(plain_correct) * 100,
            **lat_stats(plain_lats),
        }
        print(f"  PlainViT: acc={plain_st['accuracy_pct']:.2f}%  "
              f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms")
        with open(os.path.join(out_dir, 'hybrid_2exit_plain.json'), 'w') as f:
            json.dump(plain_st, f, indent=2)

    # ── Step 4: Grid Search ──
    print(f"\n[Step 5] Grid search  ({len(args.batch_sizes)*len(args.timeout_ms)} combos) ...")
    grid = run_grid(precomp, seg2_preds, seg2_lut, args.threshold,
                    args.batch_sizes, args.timeout_ms, eb1, eb2)

    # ── Save ──
    save_csv(grid, os.path.join(out_dir, 'hybrid_2exit_grid.csv'))
    with open(os.path.join(out_dir, 'hybrid_2exit_grid.json'), 'w') as f:
        json.dump(grid, f, indent=2)

    # ── Plots ──
    dl = args.device_label
    if plain_st:
        print_table(grid, plain_st, dl)
        plot_grid_heatmap(grid, plain_st['avg_ms'], args.batch_sizes, args.timeout_ms,
                          'avg_ms', 'Avg Latency',
                          os.path.join(out_dir, 'hybrid_2exit_grid_avg_heatmap.png'), dl)
        plot_grid_heatmap(grid, plain_st['p99_ms'], args.batch_sizes, args.timeout_ms,
                          'p99_ms', 'P99 Latency',
                          os.path.join(out_dir, 'hybrid_2exit_grid_p99_heatmap.png'), dl)
        # Best = min avg_ms
        best = min(grid, key=lambda x: x['avg_ms'])
        plot_comparison(best, plain_st,
                        os.path.join(out_dir, 'hybrid_2exit_best_comparison.png'), dl)
        print(f"\nBest: bs={best['batch_size']} tms={best['timeout_ms']}ms  "
              f"avg={best['avg_ms']:.2f}ms  p99={best['p99_ms']:.2f}ms  "
              f"acc={best['accuracy_pct']:.2f}%")

    print(f"\nDone! → {out_dir}")


if __name__ == '__main__':
    main()
