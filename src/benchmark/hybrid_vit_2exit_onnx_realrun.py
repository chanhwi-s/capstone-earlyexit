"""
hybrid_vit_2exit_onnx_realrun.py — ONNX Runtime 기반 2-exit 하이브리드 벤치마크

PyTorch .pth 대신 ONNX Runtime (CUDA EP) 으로 추론.
  seg1.onnx  : static batch=BS1 (기본 8) — 이미지 BS1개 → feat [BS1,197,768] + logits
  seg2.onnx  : dynamic batch             — non-exit 샘플 누적 → seg2 flush
  plain_vit  : static batch=BS1          — 기준선 배치 추론

핵심 설계:
  - BS1개 이미지를 seg1 에 한 번에 통과 → 각 샘플 confidence 체크
  - 미탈출 샘플 feats를 GPU에 유지 (IOBinding + DLPack) → seg2 배치 flush
  - Baseline / Hybrid 모두 "BS1 단위 배치 시작" 구조로 동일한 조건

타이밍: torch.cuda.synchronize() + time.perf_counter()
         (ONNX Runtime CUDA EP는 자체 CUDA 스트림 사용 → device sync 로 측정)

사용법:
  cd src
  python benchmark/hybrid_vit_2exit_onnx_realrun.py --threshold 0.80
  python benchmark/hybrid_vit_2exit_onnx_realrun.py --threshold 0.80 \\
      --seg1-onnx /path/to/seg1.onnx --seg2-onnx /path/to/seg2.onnx \\
      --plain-onnx /path/to/plain_vit.onnx
"""

import os, sys, json, csv, argparse, random, time
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from benchmark.hybrid_vit_utils import lat_stats, throughput_stats

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime-gpu 설치 필요: pip install onnxruntime-gpu")


# ── ONNX Runtime 세션 ─────────────────────────────────────────────────────────

def build_session(onnx_path: str, device_id: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ('CUDAExecutionProvider', {'device_id': device_id}),
        'CPUExecutionProvider',
    ]
    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    ep = sess.get_providers()[0]
    print(f"  {os.path.basename(onnx_path):30s} → {ep}")
    return sess


def _ort_to_torch(ort_val: ort.OrtValue) -> torch.Tensor:
    """OrtValue(GPU) → torch GPU tensor. DLPack 우선, 실패 시 CPU copy fallback."""
    try:
        return torch.utils.dlpack.from_dlpack(ort_val.to_dlpack())
    except Exception:
        arr = ort_val.numpy()
        return torch.from_numpy(arr).cuda()


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def build_val_dataset(data_root: str):
    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(os.path.join(data_root, 'val'), transform=transform)


def sample_data(dataset, n: int, seed: int = 42):
    rng = random.Random(seed)
    idxs = rng.sample(range(len(dataset)), n)
    return [dataset[i] for i in idxs]


# ── GPU 실행 헬퍼 ─────────────────────────────────────────────────────────────

def _sync_time(fn):
    """fn() 실행 + device sync → elapsed_ms 반환."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def run_seg1_batch(sess: ort.InferenceSession,
                   imgs_t: torch.Tensor) -> tuple:
    """
    imgs_t: [BS1, 3, 224, 224] GPU tensor (contiguous float32)
    Returns: (feats [BS1, 197, 768] GPU tensor,
              logits_np [BS1, 1000] numpy,
              elapsed_ms float)
    """
    bs = imgs_t.shape[0]
    iob = sess.io_binding()
    iob.bind_input('image',
                   device_type='cuda', device_id=0,
                   element_type=np.float32, shape=(bs, 3, 224, 224),
                   buffer_ptr=imgs_t.data_ptr())
    iob.bind_output('feat_out',  device_type='cuda')
    iob.bind_output('ee_logits', device_type='cuda')

    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))

    outs       = iob.get_outputs()
    feats      = _ort_to_torch(outs[0])        # [BS1, 197, 768] GPU
    logits_np  = outs[1].numpy()               # [BS1, 1000] CPU (confidence 체크용)
    return feats, logits_np, elapsed_ms


def run_seg2_batch(sess: ort.InferenceSession,
                   feat_list: list) -> tuple:
    """
    feat_list: list of [1, 197, 768] GPU tensor
    Returns: (preds list[int], elapsed_ms float)
    """
    batch = torch.cat(feat_list, dim=0).contiguous()   # [B, 197, 768] GPU
    shape = tuple(batch.shape)

    iob = sess.io_binding()
    iob.bind_input('feat_in',
                   device_type='cuda', device_id=0,
                   element_type=np.float32, shape=shape,
                   buffer_ptr=batch.data_ptr())
    iob.bind_output('ee_logits', device_type='cuda')

    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))

    logits_np  = iob.get_outputs()[0].numpy()          # [B, 1000] CPU
    preds      = logits_np.argmax(axis=1).tolist()
    return preds, elapsed_ms


def run_plain_batch(sess: ort.InferenceSession,
                    imgs_t: torch.Tensor) -> tuple:
    """
    imgs_t: [BS1, 3, 224, 224] GPU tensor (contiguous float32)
    Returns: (preds list[int], elapsed_ms float)
    """
    bs = imgs_t.shape[0]
    iob = sess.io_binding()
    iob.bind_input('image',
                   device_type='cuda', device_id=0,
                   element_type=np.float32, shape=(bs, 3, 224, 224),
                   buffer_ptr=imgs_t.data_ptr())
    iob.bind_output('logits', device_type='cuda')

    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))

    logits_np  = iob.get_outputs()[0].numpy()
    preds      = logits_np.argmax(axis=1).tolist()
    return preds, elapsed_ms


def _conf(logits_row: np.ndarray) -> float:
    """단일 샘플 logits [1000] → max softmax confidence."""
    e = np.exp(logits_row - logits_row.max())
    return float((e / e.sum()).max())


# ── Combo / Grid ──────────────────────────────────────────────────────────────

def run_combo(seg1_sess, seg2_sess, samples, device,
              thr, bs1, bs2, tms):
    """
    bs1: seg1 고정 배치 (= baseline 배치, 예: 8)
    bs2: seg2 flush 배치 크기 (grid search 대상)
    tms: seg2 flush timeout (ms)

    bs1개씩 seg1 일괄 실행 → exit/non-exit 분류
    non-exit 누적 → bs2 or timeout 시 seg2 flush
    """
    rts, tputs, correct, exit_flags = [], [], [], []
    q_feats, q_ts, q_lbls = [], [], []
    t = 0.0

    def flush_seg2():
        nonlocal t
        if not q_feats:
            return
        bsq = len(q_feats)
        preds_s2, seg2_ms = run_seg2_batch(seg2_sess, q_feats)
        t += seg2_ms
        for k in range(bsq):
            rt = t - q_ts[k]
            rts.append(rt)
            tputs.append(bsq / rt)
            correct.append(int(preds_s2[k] == q_lbls[k]))
            exit_flags.append(2)
        q_feats.clear(); q_ts.clear(); q_lbls.clear()

    for i in range(0, len(samples), bs1):
        seg1_batch = samples[i : i + bs1]
        actual_bs1 = len(seg1_batch)

        imgs = torch.stack([s[0] for s in seg1_batch]).to(
            device, non_blocking=True).contiguous()
        feats, logits_np, seg1_ms = run_seg1_batch(seg1_sess, imgs)

        ts_batch = t          # 이 seg1 배치가 시작된 시점
        t += seg1_ms

        for k in range(actual_bs1):
            lbl  = seg1_batch[k][1]
            conf = _conf(logits_np[k])
            pred = int(logits_np[k].argmax())

            if conf >= thr:
                rt = seg1_ms
                rts.append(rt)
                tputs.append(actual_bs1 / seg1_ms)
                correct.append(int(pred == lbl))
                exit_flags.append(1)
            else:
                q_feats.append(feats[k : k + 1])    # [1, 197, 768] GPU
                q_ts.append(ts_batch)               # seg1 배치 시작 시점
                q_lbls.append(lbl)

                if len(q_feats) >= bs2 or (t - q_ts[0]) >= tms:
                    flush_seg2()

    flush_seg2()

    n   = len(rts)
    ne1 = sum(1 for f in exit_flags if f == 1)
    return {
        'batch_size':   bs2,
        'timeout_ms':   tms,
        'threshold':    thr,
        'accuracy_pct': sum(correct) / n * 100,
        'exit_rate_b1': ne1 / n * 100,
        'exit_rate_b2': (n - ne1) / n * 100,
        'n_samples':    n,
        'overall_tput': n / t,
        **lat_stats(rts),
        **throughput_stats(tputs),
    }


def run_grid(seg1_sess, seg2_sess, samples, device,
             thr, bs1, batch_sizes, timeout_ms_list):
    total   = len(batch_sizes) * len(timeout_ms_list)
    results = []
    skipped = []
    print(f"\n  {total} combos × {len(samples)} samples ...")
    idx = 0
    for bs2 in batch_sizes:
        for tms in timeout_ms_list:
            idx += 1
            try:
                st = run_combo(seg1_sess, seg2_sess, samples, device,
                               thr, bs1, bs2, tms)
                results.append(st)
                print(f"    [{idx:>3}/{total}] bs2={bs2:>4} tms={tms:>5.1f}ms  "
                      f"acc={st['accuracy_pct']:.2f}%  avg={st['avg_ms']:.2f}ms  "
                      f"p99={st['p99_ms']:.2f}ms  "
                      f"exit=[{st['exit_rate_b1']:.0f}%/{st['exit_rate_b2']:.0f}%]")
            except RuntimeError as e:
                if 'Failed to allocate memory' in str(e) or 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    skipped.append((bs2, tms))
                    print(f"    [{idx:>3}/{total}] bs2={bs2:>4} tms={tms:>5.1f}ms  "
                          f"[SKIP] GPU OOM — batch too large for VRAM")
                else:
                    raise
    if skipped:
        print(f"\n  OOM으로 스킵된 조합 {len(skipped)}개: {skipped}")
    return results


# ── 저장 / 플롯 ───────────────────────────────────────────────────────────────

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
                 bs1, higher_better=False, raw_fmt='.2f', raw_unit='ms'):
    mat_raw = _grid_mat(results, batch_sizes, timeout_ms_list, metric)
    mat     = ((mat_raw - plain_val) / abs(plain_val) * 100 if higher_better
                else (plain_val - mat_raw) / plain_val * 100)

    vmin_v = min(-15, np.nanmin(mat))
    vmax_v = max(15,  np.nanmax(mat))
    norm   = mcolors.TwoSlopeNorm(vmin=vmin_v, vcenter=0, vmax=vmax_v)
    cmap   = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=(max(6, len(timeout_ms_list) * 1.4 + 1),
                                    max(4, len(batch_sizes) * 0.9 + 1)))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')
    cb = plt.colorbar(im, ax=ax,
                      label=f'Improvement over PlainViT (bs={bs1}, ONNX) (%)')
    cb.ax.tick_params(labelsize=9)
    ax.set_xticks(range(len(timeout_ms_list)))
    ax.set_xticklabels([f'{t}ms' for t in timeout_ms_list], fontsize=10)
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([f'bs={b}' for b in batch_sizes], fontsize=10)
    ax.set_xlabel('Timeout', fontsize=11); ax.set_ylabel('Seg2 Batch Size', fontsize=11)
    ax.set_title(f'{metric_label} Improvement — 2-exit ONNX (real)\n'
                 f'(thr={threshold:.2f}, {device_label}, seg1 bs={bs1})', fontsize=11)
    for bi in range(len(batch_sizes)):
        for ti in range(len(timeout_ms_list)):
            v, raw = mat[bi, ti], mat_raw[bi, ti]
            if not np.isnan(v):
                rgba = cmap(norm(v))
                lum  = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc   = 'black' if lum > 0.45 else 'white'
                ax.text(ti, bi, f'{v:+.1f}%\n({raw:{raw_fmt}}{raw_unit})',
                        ha='center', va='center', fontsize=7.5,
                        color=tc, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  heatmap: {out_path}")


def plot_line_charts(results, plain_st, batch_sizes, timeout_ms_list,
                     out_dir, device_label, threshold, bs1):
    tab10   = plt.get_cmap('tab10')
    colors  = [tab10(i) for i in range(len(timeout_ms_list))]
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
    fig.suptitle(f'2-exit Hybrid ONNX — Seg2 Batch Size vs Metrics'
                 f'  (thr={threshold:.2f}, {device_label}, seg1 bs={bs1})',
                 fontsize=13)

    for ax, (metric, ylabel) in zip(axes.flat, metrics_cfg):
        for ci, tms in enumerate(timeout_ms_list):
            subset = sorted([r for r in results if r['timeout_ms'] == tms],
                            key=lambda r: r['batch_size'])
            ax.plot([r['batch_size'] for r in subset],
                    [r[metric]       for r in subset],
                    marker=markers[ci % len(markers)],
                    color=colors[ci], label=f'{tms}ms',
                    linewidth=2, markersize=7,
                    markeredgewidth=0.5, markeredgecolor='white')
        if plain_st and metric in plain_st:
            ax.axhline(plain_st[metric], color='#222222', linestyle='--',
                       linewidth=1.8,
                       label=f'PlainViT ONNX (bs={bs1})', alpha=0.85)
        ax.set_xlabel('Seg2 Batch Size', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(batch_sizes)
        ax.legend(title='Timeout', fontsize=8, title_fontsize=9,
                  framealpha=0.9, edgecolor='#cccccc')
        ax.grid(alpha=0.25, linestyle='--', color='gray')
        ax.set_facecolor('#f9f9f9')
        ax.set_title(ylabel.split('(')[0].strip(), fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'hybrid_2exit_onnx_realrun_lineplot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  lineplot: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def _find_onnx(model_name: str, filename: str) -> str | None:
    d = paths.onnx_dir(model_name)
    p = os.path.join(d, filename)
    return p if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser(
        description='SelectiveExitViT 2-exit ONNX Runtime 기반 하이브리드 벤치마크'
    )
    parser.add_argument('--threshold',     type=float, required=True)
    parser.add_argument('--data-root',     type=str,   default='/home2/imagenet')
    parser.add_argument('--n-samples',     type=int,   default=5000)
    parser.add_argument('--warmup',        type=int,   default=50,
                        help='seg1 배치 warmup 수 (기본: 50개 배치)')
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--baseline-batch-size', type=int, default=8,
                        help='seg1/plain 고정 배치 크기 (기본: 8)')
    parser.add_argument('--batch-sizes',   type=int,   nargs='+',
                        default=[8, 16, 32, 64, 128, 256, 512],
                        help='seg2 grid 탐색 배치 크기 목록')
    parser.add_argument('--timeout-ms',    type=float, nargs='+',
                        default=[1., 2., 5., 10., 20., 50., 100., 200., 500.])
    parser.add_argument('--seg1-onnx',     type=str,   default=None,
                        help='seg1.onnx 경로 (미지정 시 최신 exp 자동 탐색)')
    parser.add_argument('--seg2-onnx',     type=str,   default=None)
    parser.add_argument('--plain-onnx',    type=str,   default=None)
    parser.add_argument('--out-dir',       type=str,   default=None)
    parser.add_argument('--device-label',  type=str,   default='RTX 5090')
    parser.add_argument('--skip-plain',    action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs1     = args.baseline_batch_size
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'hybrid_2exit_onnx_realrun_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device      : {device}  ({args.device_label})")
    print(f"Threshold   : {args.threshold}")
    print(f"N-Samples   : {args.n_samples}  (warmup={args.warmup} batches, seed={args.seed})")
    print(f"Seg1/Plain BS: {bs1}  (static)")
    print(f"Seg2 Grid   : {args.batch_sizes}")
    print(f"TimeoutMs   : {args.timeout_ms}")
    print(f"Output      : {out_dir}\n")

    # ── ONNX 경로 결정 ──
    seg1_path  = (args.seg1_onnx  or _find_onnx('ee_vit_2exit', 'seg1.onnx'))
    seg2_path  = (args.seg2_onnx  or _find_onnx('ee_vit_2exit', 'seg2.onnx'))
    plain_path = (args.plain_onnx or _find_onnx('plain_vit',    'plain_vit.onnx'))

    missing = [(n, p) for n, p in [('seg1', seg1_path), ('seg2', seg2_path),
                                    ('plain', plain_path)] if not p]
    if missing and not args.skip_plain:
        for name, _ in missing:
            print(f"[ERROR] {name}.onnx 를 찾을 수 없습니다. "
                  f"--{name}-onnx 로 직접 지정하거나 export_vit_5090.sh 를 먼저 실행하세요.")
        return
    if not seg1_path or not seg2_path:
        print("[ERROR] seg1.onnx / seg2.onnx 필요. export_vit_5090.sh 를 먼저 실행하세요.")
        return

    # ── 세션 로드 ──
    print("ONNX 세션 로드 ...")
    seg1_sess  = build_session(seg1_path)
    seg2_sess  = build_session(seg2_path)
    plain_sess = build_session(plain_path) if (plain_path and not args.skip_plain) else None

    # ── n_samples → bs1 배수로 맞춤 ──
    n_total = (args.n_samples // bs1) * bs1
    if n_total != args.n_samples:
        print(f"  n_samples {args.n_samples} → {n_total}  (bs1={bs1} 배수로 조정)")

    # ── 데이터 샘플링 ──
    warmup_n = args.warmup * bs1
    print(f"\n데이터 로딩 ({n_total + warmup_n}개) ...")
    dataset     = build_val_dataset(args.data_root)
    all_samples = sample_data(dataset, n_total + warmup_n, seed=args.seed)
    warmup_samp = all_samples[:warmup_n]
    eval_samp   = all_samples[warmup_n:]
    print(f"  warmup={len(warmup_samp)}, eval={len(eval_samp)}")

    # ── GPU 웜업 ──
    print(f"\nGPU 웜업 ({args.warmup} seg1 배치) ...")
    for i in range(0, len(warmup_samp), bs1):
        batch = warmup_samp[i : i + bs1]
        if len(batch) < bs1:
            break
        imgs = torch.stack([s[0] for s in batch]).to(device, non_blocking=True).contiguous()
        run_seg1_batch(seg1_sess, imgs)
    torch.cuda.synchronize()

    # ── PlainViT 기준선 ──
    plain_st = None
    if plain_sess and not args.skip_plain:
        print(f"\nPlainViT 기준선 ONNX (bs={bs1}, {len(eval_samp)}개) ...")
        plain_lats, plain_correct = [], []
        wall_t = 0.0
        with torch.no_grad():
            for i in range(0, len(eval_samp), bs1):
                batch = eval_samp[i : i + bs1]
                if len(batch) < bs1:
                    break
                imgs = torch.stack([s[0] for s in batch]).to(
                    device, non_blocking=True).contiguous()
                lbls = [s[1] for s in batch]
                preds, batch_ms = run_plain_batch(plain_sess, imgs)
                wall_t += batch_ms
                for k in range(bs1):
                    plain_lats.append(batch_ms)
                    plain_correct.append(int(preds[k] == lbls[k]))

        plain_tputs = [bs1 / l for l in plain_lats]
        plain_st = {
            'baseline_batch_size': bs1,
            'accuracy_pct': sum(plain_correct) / len(plain_correct) * 100,
            'overall_tput': len(plain_lats) / wall_t,
            **lat_stats(plain_lats),
            **throughput_stats(plain_tputs),
        }
        print(f"  acc={plain_st['accuracy_pct']:.2f}%  "
              f"avg={plain_st['avg_ms']:.2f}ms  p99={plain_st['p99_ms']:.2f}ms  "
              f"avg_tput={plain_st['avg_tput']:.4f}/ms")
        with open(os.path.join(out_dir, 'hybrid_2exit_onnx_realrun_plain.json'), 'w') as f:
            json.dump(plain_st, f, indent=2)

    # ── Grid Search ──
    grid = run_grid(seg1_sess, seg2_sess, eval_samp, device,
                    args.threshold, bs1, args.batch_sizes, args.timeout_ms)

    save_csv(grid, os.path.join(out_dir, 'hybrid_2exit_onnx_realrun.csv'))
    with open(os.path.join(out_dir, 'hybrid_2exit_onnx_realrun.json'), 'w') as f:
        json.dump(grid, f, indent=2)

    # ── 플롯 ──
    dl = args.device_label
    if plain_st:
        for metric, label, hb, rfmt, runit in [
            ('avg_ms',       'Avg Latency',        False, '.2f', 'ms'),
            ('p90_ms',       'P90 Latency',         False, '.2f', 'ms'),
            ('p95_ms',       'P95 Latency',         False, '.2f', 'ms'),
            ('p99_ms',       'P99 Latency',         False, '.2f', 'ms'),
            ('overall_tput', 'Overall Throughput',  True,  '.4f', '/ms'),
            ('avg_tput',     'Avg Throughput',      True,  '.4f', '/ms'),
            ('p90_tput',     'P90 Throughput',      True,  '.4f', '/ms'),
            ('p95_tput',     'P95 Throughput',      True,  '.4f', '/ms'),
            ('p99_tput',     'P99 Throughput',      True,  '.4f', '/ms'),
        ]:
            plot_heatmap(
                grid, plain_st[metric], args.batch_sizes, args.timeout_ms,
                metric, label,
                os.path.join(out_dir, f'hybrid_2exit_onnx_realrun_{metric}_heatmap.png'),
                dl, args.threshold, bs1,
                higher_better=hb, raw_fmt=rfmt, raw_unit=runit)

        plot_line_charts(grid, plain_st, args.batch_sizes, args.timeout_ms,
                         out_dir, dl, args.threshold, bs1)

    best = min(grid, key=lambda x: x['avg_ms'])
    print(f"\nBest: seg2_bs={best['batch_size']} tms={best['timeout_ms']}ms  "
          f"avg={best['avg_ms']:.2f}ms  p99={best['p99_ms']:.2f}ms  "
          f"acc={best['accuracy_pct']:.2f}%")
    print(f"\nDone! → {out_dir}")


if __name__ == '__main__':
    main()
