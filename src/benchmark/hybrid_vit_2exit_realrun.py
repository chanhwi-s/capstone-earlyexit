"""
hybrid_vit_2exit_realrun.py — 실제 GPU 실행 기반 2-exit 하이브리드 벤치마크

LUT 시뮬레이션이 아닌 실제 GPU 실행으로 latency/throughput 측정.
  - seg1: sample-by-sample 실제 GPU 타이밍
  - seg2: flush마다 실제 GPU 배치 실행 → 편차(thermal 등) 반영

ImageNet val에서 N개 랜덤 샘플링 → batch_size × timeout_ms grid search.
각 (bs, timeout) 조합마다 동일한 N개 샘플을 독립적으로 실행.

사용법:
  cd src
  python benchmark/hybrid_vit_2exit_realrun.py --threshold 0.80 --n-samples 1000
  python benchmark/hybrid_vit_2exit_realrun.py --threshold 0.80 --n-samples 2000 \\
      --batch-sizes 1 4 8 16 --timeout-ms 1 2 5 10
"""

import os, sys, json, csv, argparse, random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from models.plain_vit import build_model as build_plain
from models.ee_vit_selective import build_model as build_selective
from benchmark.hybrid_vit_utils import lat_stats, throughput_stats
from benchmark.benchmark_pytorch_vit import load_checkpoint


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def build_val_dataset(data_root):
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(os.path.join(data_root, 'val'), transform=transform)


def sample_data(dataset, n, seed=42):
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), n)
    return [dataset[i] for i in idxs]  # list of (tensor[3,224,224], int)


# ── GPU 실행 헬퍼 ─────────────────────────────────────────────────────────────

def run_seg1(model, img, device, eb1):
    """단일 샘플 seg1 실행. Returns (feat, conf, pred, elapsed_ms)."""
    x = img.unsqueeze(0).to(device, non_blocking=True)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        s.record()
        feat = model._embed(x)
        for bi in range(eb1):
            feat = model.blocks[bi](feat)
        logits = model.exit_heads[0](feat)
        e.record()
    torch.cuda.synchronize()
    conf = F.softmax(logits, dim=1).max(1).values.item()
    pred = logits.argmax(1).item()
    return feat.detach(), conf, pred, s.elapsed_time(e)


def run_seg_batch(model, feats, start_block, end_block, head_idx):
    """feats: list of [1,197,768] GPU tensor. Returns (feat_out, confs, preds, ms)."""
    batch = torch.cat(feats, dim=0)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        s.record()
        x = batch
        for bi in range(start_block, end_block):
            x = model.blocks[bi](x)
        logits = model.exit_heads[head_idx](x)
        e.record()
    torch.cuda.synchronize()
    confs = F.softmax(logits, dim=1).max(1).values.tolist()
    preds = logits.argmax(1).tolist()
    return x.detach(), confs, preds, s.elapsed_time(e)


def _result(bs, tms, thr, rts, tputs, correct, exit_at, total_t, eb1, eb2):
    n = len(rts)
    ne1 = sum(1 for b in exit_at if b == eb1)
    return {
        'batch_size':   bs,
        'timeout_ms':   tms,
        'threshold':    thr,
        'accuracy_pct': sum(correct) / n * 100,
        'exit_rate_b1': ne1 / n * 100,
        'exit_rate_b2': (n - ne1) / n * 100,
        'n_samples':    n,
        'overall_tput': n / total_t,
        **lat_stats(rts),
        **throughput_stats(tputs),
    }


# ── Run combo ────────────────────────────────────────────────────────────────

def run_combo(model, samples, device, thr, bs, tms, eb1, eb2):
    """
    seg1 sample-by-sample → confidence 체크 → 미통과 샘플 큐 누적 → seg2 배치 실행.
    batch_size 또는 timeout_ms 조건 충족 시 flush.
    """
    rts, tputs, correct, exit_at = [], [], [], []
    q_feats, q_ts, q_te, q_lbls = [], [], [], []
    t = 0.0

    def flush():
        nonlocal t
        if not q_feats:
            return
        bsq = len(q_feats)
        _, _, preds_s2, seg2_ms = run_seg_batch(model, q_feats, eb1, eb2, 1)
        t += seg2_ms
        for k in range(bsq):
            rt = t - q_ts[k]
            rts.append(rt); tputs.append(bsq / rt)
            correct.append(int(preds_s2[k] == q_lbls[k]))
            exit_at.append(eb2)
        q_feats.clear(); q_ts.clear(); q_te.clear(); q_lbls.clear()

    for img, lbl in samples:
        feat, conf, pred, seg1_ms = run_seg1(model, img, device, eb1)
        ts = t; t += seg1_ms
        if conf >= thr:
            rt = seg1_ms
            rts.append(rt); tputs.append(1.0 / rt)
            correct.append(int(pred == lbl)); exit_at.append(eb1)
        else:
            q_feats.append(feat); q_ts.append(ts); q_te.append(t); q_lbls.append(lbl)
            if len(q_feats) >= bs or (t - q_te[0]) >= tms:
                flush()
    flush()
    return _result(bs, tms, thr, rts, tputs, correct, exit_at, t, eb1, eb2)


# ── Grid Search ───────────────────────────────────────────────────────────────

def run_grid(model, samples, device, thr, batch_sizes, timeout_ms_list, eb1, eb2):
    total = len(batch_sizes) * len(timeout_ms_list)
    results = []
    print(f"\n  {total} combos × {len(samples)} samples ...")
    idx = 0
    for bsz in batch_sizes:
        for tms in timeout_ms_list:
            st = run_combo(model, samples, device, thr, bsz, tms, eb1, eb2)
            results.append(st)
            idx += 1
            print(f"    [{idx:>3}/{total}] bs={bsz:>3} tms={tms:>5.1f}ms  "
                  f"acc={st['accuracy_pct']:.2f}%  avg={st['avg_ms']:.2f}ms  "
                  f"p99={st['p99_ms']:.2f}ms  "
                  f"exit=[{st['exit_rate_b1']:.0f}%/{st['exit_rate_b2']:.0f}%]")
    return results


# ── Save / Plot ───────────────────────────────────────────────────────────────

def save_csv(rows, path):
    fields = ['batch_size', 'timeout_ms', 'threshold', 'accuracy_pct',
              'exit_rate_b1', 'exit_rate_b2',
              'avg_ms', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'std_ms',
              'overall_tput',
              'avg_tput', 'p50_tput', 'p90_tput', 'p95_tput', 'p99_tput', 'std_tput',
              'n_samples']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    print(f"  CSV: {path}")


def _grid_mat(results, batch_sizes, timeout_ms_list, metric):
    mat = np.full((len(batch_sizes), len(timeout_ms_list)), np.nan)
    for st in results:
        bi = batch_sizes.index(st['batch_size'])
        ti = timeout_ms_list.index(st['timeout_ms'])
        mat[bi, ti] = st[metric]
    return mat


def plot_heatmap(results, plain_val, batch_sizes, timeout_ms_list,
                 metric, metric_label, out_path, device_label, threshold,
                 higher_better=False, raw_fmt='.2f', raw_unit='ms'):
    import matplotlib.colors as mcolors
    mat_raw = _grid_mat(results, batch_sizes, timeout_ms_list, metric)
    mat = ((mat_raw - plain_val) / abs(plain_val) * 100 if higher_better
           else (plain_val - mat_raw) / plain_val * 100)

    vmin_v = min(-15, np.nanmin(mat))
    vmax_v = max(15, np.nanmax(mat))
    norm = mcolors.TwoSlopeNorm(vmin=vmin_v, vcenter=0, vmax=vmax_v)
    cmap = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=(max(6, len(timeout_ms_list) * 1.4 + 1),
                                    max(4, len(batch_sizes) * 0.9 + 1)))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
    cb = plt.colorbar(im, ax=ax, label='Improvement over PlainViT (%)')
    cb.ax.tick_params(labelsize=9)
    ax.set_xticks(range(len(timeout_ms_list)))
    ax.set_xticklabels([f'{t}ms' for t in timeout_ms_list], fontsize=10)
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([f'bs={b}' for b in batch_sizes], fontsize=10)
    ax.set_xlabel('Timeout', fontsize=11); ax.set_ylabel('Batch Size', fontsize=11)
    ax.set_title(f'{metric_label} Improvement — 2-exit (real)\n'
                 f'(thr={threshold:.2f}, {device_label})', fontsize=11)
    for bi in range(len(batch_sizes)):
        for ti in range(len(timeout_ms_list)):
            v, raw = mat[bi, ti], mat_raw[bi, ti]
            if not np.isnan(v):
                rgba = cmap(norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = 'black' if lum > 0.45 else 'white'
                ax.text(ti, bi, f'{v:+.1f}%\n({raw:{raw_fmt}}{raw_unit})',
                        ha='center', va='center', fontsize=7.5,
                        color=tc, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  heatmap: {out_path}")


def plot_line_charts(results, plain_st, batch_sizes, timeout_ms_list,
                     out_dir, device_label, threshold):
    """
    2×3 figure (6 subplots).
    Row 0: avg_ms / p99_ms / avg_tput
    Row 1: p90_tput / p95_tput / p99_tput
    x=batch_size, lines=timeout.
    """
    tab10 = plt.get_cmap('tab10')
    colors = [tab10(i) for i in range(len(timeout_ms_list))]
    markers = ['o', 's', '^', 'D', 'v']

    metrics_cfg = [
        ('avg_ms',   'Avg Latency (ms)'),
        ('p99_ms',   'P99 Latency (ms)'),
        ('avg_tput', 'Avg Throughput (samples/ms)'),
        ('p90_tput', 'P90 Throughput (samples/ms)'),
        ('p95_tput', 'P95 Throughput (samples/ms)'),
        ('p99_tput', 'P99 Throughput (samples/ms)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f'2-exit Hybrid — Batch Size vs Metrics'
                 f'  (thr={threshold:.2f}, {device_label})', fontsize=13)

    for ax, (metric, ylabel) in zip(axes.flat, metrics_cfg):
        for ci, tms in enumerate(timeout_ms_list):
            subset = sorted([r for r in results if r['timeout_ms'] == tms],
                            key=lambda r: r['batch_size'])
            bszs = [r['batch_size'] for r in subset]
            vals = [r[metric] for r in subset]
            ax.plot(bszs, vals,
                    marker=markers[ci % len(markers)],
                    color=colors[ci],
                    label=f'{tms}ms',
                    linewidth=2, markersize=7, markeredgewidth=0.5,
                    markeredgecolor='white')

        if plain_st and metric in plain_st:
            ax.axhline(plain_st[metric], color='#222222', linestyle='--',
                       linewidth=1.8, label='PlainViT', alpha=0.85)

        ax.set_xlabel('Batch Size', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(batch_sizes)
        ax.legend(title='Timeout', fontsize=8, title_fontsize=9,
                  framealpha=0.9, edgecolor='#cccccc')
        ax.grid(alpha=0.25, linestyle='--', color='gray')
        ax.set_facecolor('#f9f9f9')
        ax.set_title(ylabel.split('(')[0].strip(), fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'hybrid_2exit_realrun_lineplot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  lineplot: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT 2-exit 실제 GPU 실행 기반 하이브리드 벤치마크'
    )
    parser.add_argument('--threshold',    type=float, required=True)
    parser.add_argument('--data-root',    type=str,   default='/home2/imagenet')
    parser.add_argument('--exit-blocks',  type=int,   nargs='+', default=[8, 12])
    parser.add_argument('--n-samples',    type=int,   default=1000)
    parser.add_argument('--warmup',       type=int,   default=50)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--batch-sizes',  type=int,   nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--timeout-ms',   type=float, nargs='+', default=[1., 2., 5., 10., 20.])
    parser.add_argument('--out-dir',      type=str,   default=None)
    parser.add_argument('--device-label', type=str,   default='RTX 5090')
    parser.add_argument('--skip-plain',   action='store_true')
    args = parser.parse_args()

    eb1, eb2 = args.exit_blocks[0], args.exit_blocks[-1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'hybrid_2exit_realrun_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device    : {device}  ({args.device_label})")
    print(f"Threshold : {args.threshold}")
    print(f"N-Samples : {args.n_samples}  (warmup={args.warmup}, seed={args.seed})")
    print(f"ExitBlocks: {args.exit_blocks}")
    print(f"BatchSizes: {args.batch_sizes}")
    print(f"TimeoutMs : {args.timeout_ms}")
    print(f"Output    : {out_dir}\n")

    # ── 모델 로드 ──
    ckpt = paths.latest_checkpoint('ee_vit_2exit')
    assert ckpt, "ee_vit_2exit checkpoint 없음"
    print(f"Checkpoint: {ckpt}")
    model = build_selective(args.exit_blocks).to(device)
    load_checkpoint(model, ckpt)
    model.eval()

    # ── 데이터 샘플링 ──
    print(f"\n데이터 로딩 ({args.n_samples + args.warmup}개) ...")
    dataset = build_val_dataset(args.data_root)
    all_samples = sample_data(dataset, args.n_samples + args.warmup, seed=args.seed)
    warmup_samples = all_samples[:args.warmup]
    eval_samples   = all_samples[args.warmup:]
    print(f"  warmup={len(warmup_samples)}, eval={len(eval_samples)}")

    # ── GPU 웜업 ──
    print(f"\nGPU 웜업 ({args.warmup}개) ...")
    with torch.no_grad():
        for img, _ in warmup_samples:
            x = img.unsqueeze(0).to(device)
            feat = model._embed(x)
            for bi in range(eb2):
                feat = model.blocks[bi](feat)
            _ = model.exit_heads[1](feat)
    torch.cuda.synchronize()

    # ── PlainViT 기준선 ──
    plain_st = None
    if not args.skip_plain:
        print(f"\nPlainViT 기준선 ({len(eval_samples)}개) ...")
        plain_model = build_plain().to(device)
        plain_model.eval()
        plain_lats, plain_correct = [], []
        with torch.no_grad():
            for img, lbl in eval_samples:
                x = img.unsqueeze(0).to(device)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); logits = plain_model(x); e.record()
                torch.cuda.synchronize()
                plain_lats.append(s.elapsed_time(e))
                plain_correct.append(int(logits.argmax(1).item() == lbl))
        del plain_model; torch.cuda.empty_cache()
        plain_tputs = [1.0 / l for l in plain_lats]
        plain_st = {
            'accuracy_pct': sum(plain_correct) / len(plain_correct) * 100,
            'overall_tput': len(plain_lats) / sum(plain_lats),
            **lat_stats(plain_lats),
            **throughput_stats(plain_tputs),
        }
        print(f"  acc={plain_st['accuracy_pct']:.2f}%  "
              f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms  "
              f"avg_tput={plain_st['avg_tput']:.4f}/ms")
        with open(os.path.join(out_dir, 'hybrid_2exit_realrun_plain.json'), 'w') as f:
            json.dump(plain_st, f, indent=2)

    # ── Grid Search ──
    grid = run_grid(model, eval_samples, device, args.threshold,
                    args.batch_sizes, args.timeout_ms, eb1, eb2)

    save_csv(grid, os.path.join(out_dir, 'hybrid_2exit_realrun.csv'))
    with open(os.path.join(out_dir, 'hybrid_2exit_realrun.json'), 'w') as f:
        json.dump(grid, f, indent=2)

    # ── Plots ──
    dl = args.device_label
    if plain_st:
        for metric, label, hb, rfmt, runit in [
            ('avg_ms',   'Avg Latency',    False, '.2f',  'ms'),
            ('p90_ms',   'P90 Latency',    False, '.2f',  'ms'),
            ('p95_ms',   'P95 Latency',    False, '.2f',  'ms'),
            ('p99_ms',   'P99 Latency',    False, '.2f',  'ms'),
            ('avg_tput', 'Avg Throughput', True,  '.4f',  '/ms'),
            ('p90_tput', 'P90 Throughput', True,  '.4f',  '/ms'),
            ('p95_tput', 'P95 Throughput', True,  '.4f',  '/ms'),
            ('p99_tput', 'P99 Throughput', True,  '.4f',  '/ms'),
        ]:
            plot_heatmap(
                grid, plain_st[metric], args.batch_sizes, args.timeout_ms,
                metric, label,
                os.path.join(out_dir, f'hybrid_2exit_realrun_{metric}_heatmap.png'),
                dl, args.threshold, higher_better=hb, raw_fmt=rfmt, raw_unit=runit)

        plot_line_charts(grid, plain_st, args.batch_sizes, args.timeout_ms,
                         out_dir, dl, args.threshold)

    best = min(grid, key=lambda x: x['avg_ms'])
    print(f"\nBest: bs={best['batch_size']} tms={best['timeout_ms']}ms  "
          f"avg={best['avg_ms']:.2f}ms  p99={best['p99_ms']:.2f}ms  "
          f"acc={best['accuracy_pct']:.2f}%")
    print(f"\nDone! → {out_dir}")


if __name__ == '__main__':
    main()
