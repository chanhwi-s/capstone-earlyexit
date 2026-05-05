"""
dynamic_benchmark.py — timeout-based dynamic batching goodput benchmark

static benchmark(no timeout, fixed bs2)와의 차이:
  - seg2를 dynamic batch ONNX(seg2_dynamic.onnx)로 실행 → 가변 batch 가능
  - timeout_ms: queue가 차지 않아도 대기 시간 초과 시 즉시 flush
  - SLO를 하나로 고정하고, timeout_ms를 sweep하여 goodput 변화 관찰

시뮬레이션 flush 조건 (seg1 배치 완료 시점마다 체크):
  1. len(queue) >= max_bs2   → 강제 flush (배치 꽉 참)
  2. t - queue[0].ts_seg1_done >= timeout_ms  → timeout flush

goodput(SLO) = count(rt ≤ SLO_ms) / wall_time_ms

사용법:
  cd src
  python benchmark/dynamic_benchmark.py \\
      --threshold 0.70 --slo-ms 100

  python benchmark/dynamic_benchmark.py \\
      --threshold 0.70 --slo-ms 100 \\
      --max-batch-sizes 32 64 128 \\
      --timeout-values 0 5 10 20 50 100 200 500
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError('onnxruntime-gpu 필요: pip install onnxruntime-gpu')


# ── ONNX Runtime 헬퍼 ─────────────────────────────────────────────────────────

def build_session(onnx_path: str, device_id: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        onnx_path, sess_options=opts,
        providers=[('CUDAExecutionProvider', {'device_id': device_id}),
                   'CPUExecutionProvider'])
    print(f'  {os.path.basename(onnx_path):45s} → {sess.get_providers()[0]}')
    return sess


def _sync_time(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _ort_to_torch(ort_val: ort.OrtValue) -> torch.Tensor:
    try:
        return torch.utils.dlpack.from_dlpack(ort_val.to_dlpack())
    except Exception:
        return torch.from_numpy(ort_val.numpy()).cuda()


def run_seg1_batch(sess, imgs_t: torch.Tensor):
    bs  = imgs_t.shape[0]
    iob = sess.io_binding()
    iob.bind_input('image', device_type='cuda', device_id=0,
                   element_type=np.float32, shape=(bs, 3, 224, 224),
                   buffer_ptr=imgs_t.data_ptr())
    iob.bind_output('feat_out',  device_type='cuda')
    iob.bind_output('ee_logits', device_type='cuda')
    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))
    outs = iob.get_outputs()
    return _ort_to_torch(outs[0]), outs[1].numpy(), elapsed_ms


def run_seg2_dynamic(sess, q_feats: list):
    """가변 batch seg2 실행. len(q_feats) 는 1 이상 임의 값."""
    batch = torch.cat(q_feats, dim=0).contiguous()
    iob   = sess.io_binding()
    iob.bind_input('feat_in', device_type='cuda', device_id=0,
                   element_type=np.float32, shape=tuple(batch.shape),
                   buffer_ptr=batch.data_ptr())
    iob.bind_output('ee_logits', device_type='cuda')
    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))
    logits_np  = iob.get_outputs()[0].numpy()
    return logits_np.argmax(axis=1).tolist(), elapsed_ms


def _conf(logits_row: np.ndarray) -> float:
    e = np.exp(logits_row - logits_row.max())
    return float((e / e.sum()).max())


# ── 데이터 ─────────────────────────────────────────────────────────────────────

def build_dataset(data_root: str):
    tf = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(os.path.join(data_root, 'val'), transform=tf)


def sample_data(dataset, n: int, seed: int = 42):
    rng = random.Random(seed)
    return [dataset[i] for i in rng.sample(range(len(dataset)), n)]


# ── 시뮬레이션 ─────────────────────────────────────────────────────────────────

def run_dynamic_combo(seg1_sess, seg2_sess, samples, device,
                      thr, bs1, max_bs2, timeout_ms):
    """
    Dynamic batching 시뮬레이션.

    flush 조건 (각 seg1 배치 완료 후 체크):
      1. len(queue) >= max_bs2
      2. timeout_ms < inf AND queue AND (t - queue[0]['ts_seg1_done']) >= timeout_ms

    마지막에 남은 queue는 강제 flush (discarded 아님 — dynamic의 핵심 이점).

    Returns:
        records  : list of {rt_ms, exit, correct, seg1_ms, queue_wait_ms, seg2_ms, flush_bs}
        wall_time: 총 시뮬레이션 시간 (ms)
    """
    records = []
    queue   = []   # list of {feat, ts_arrive, ts_seg1_done, lbl}
    t       = 0.0

    def flush():
        nonlocal t
        if not queue:
            return
        preds, seg2_ms_val = run_seg2_dynamic(seg2_sess, [item['feat'] for item in queue])
        t_seg2_start = t
        t += seg2_ms_val
        flush_bs = len(queue)
        for k, item in enumerate(queue):
            records.append({
                'rt_ms':         t - item['ts_arrive'],
                'exit':          2,
                'correct':       int(preds[k] == item['lbl']),
                'seg1_ms':       item['ts_seg1_done'] - item['ts_arrive'],
                'queue_wait_ms': t_seg2_start - item['ts_seg1_done'],
                'seg2_ms':       seg2_ms_val,
                'flush_bs':      flush_bs,
            })
        queue.clear()

    for i in range(0, len(samples), bs1):
        batch = samples[i : i + bs1]
        if len(batch) < bs1:
            break

        imgs = torch.stack([s[0] for s in batch]).to(device, non_blocking=True).contiguous()
        feats, logits_np, seg1_ms = run_seg1_batch(seg1_sess, imgs)

        ts_batch = t
        t       += seg1_ms
        t_seg1_done = t

        for k in range(bs1):
            lbl  = batch[k][1]
            conf = _conf(logits_np[k])
            pred = int(logits_np[k].argmax())
            if conf >= thr:
                records.append({
                    'rt_ms': seg1_ms, 'exit': 1, 'correct': int(pred == lbl),
                    'seg1_ms': seg1_ms, 'queue_wait_ms': 0.0,
                    'seg2_ms': 0.0, 'flush_bs': 0,
                })
            else:
                queue.append({'feat': feats[k : k + 1],
                              'ts_arrive': ts_batch,
                              'ts_seg1_done': t_seg1_done,
                              'lbl': lbl})

        # flush 조건 체크
        if len(queue) >= max_bs2:
            flush()
        elif timeout_ms < float('inf') and queue:
            if (t - queue[0]['ts_seg1_done']) >= timeout_ms:
                flush()

    flush()   # 남은 항목 강제 flush (static과의 핵심 차이)
    return records, t


def goodput_at_slo(records: list, wall_time: float, slo_ms: float) -> float:
    n = sum(1 for r in records if r['rt_ms'] <= slo_ms)
    return n / wall_time if wall_time > 0 else 0.0


# ── 플롯 ──────────────────────────────────────────────────────────────────────

def plot_results(sweep_results: dict, plain_gp: float, plain_avg_rt: float,
                 timeout_values, slo_ms: float,
                 out_dir: str, threshold: float, bs1: int, device_label: str):
    """
    sweep_results: {max_bs2: {timeout_ms: {'goodput': float, 'avg_rt': float, ...}}}

    좌: goodput (samples/ms) vs timeout_ms — 선 per max_bs2, plain 점선
    우: avg latency (ms) vs timeout_ms     — 선 per max_bs2, plain 점선
    """
    tab10    = plt.get_cmap('tab10')
    max_bs2s = sorted(sweep_results.keys())

    # timeout 축: 유한값만, 로그 스케일 여부 결정
    finite_timeouts = [t for t in timeout_values if t < float('inf')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Dynamic Batching — Goodput & Latency vs Timeout  '
        f'(thr={threshold:.2f}, SLO={slo_ms:.0f}ms, {device_label})',
        fontsize=12)

    for ax_idx, metric in enumerate(['goodput', 'avg_rt_ms']):
        ax     = axes[ax_idx]
        ylabel = ('Goodput (samples/ms)' if metric == 'goodput'
                  else 'Avg Response Time (ms)')
        title  = (f'Goodput at SLO={slo_ms:.0f}ms vs Timeout'
                  if metric == 'goodput'
                  else 'Avg Latency vs Timeout')
        ref_val = plain_gp if metric == 'goodput' else plain_avg_rt

        for i, mbs in enumerate(max_bs2s):
            xs, ys = [], []
            for to in finite_timeouts:
                entry = sweep_results[mbs].get(to)
                if entry:
                    xs.append(to)
                    ys.append(entry[metric])
            if xs:
                ax.plot(xs, ys, 'o-', color=tab10(i), linewidth=2,
                        markersize=5, label=f'max_bs2={mbs}')

        if ref_val is not None:
            label = (f'Plain (bs={bs1})' if metric == 'goodput'
                     else f'Plain avg ({plain_avg_rt:.1f}ms)')
            ax.axhline(ref_val, color='black', linestyle='--',
                       linewidth=2, label=label)

        ax.set_xlabel('Timeout (ms)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        if finite_timeouts and max(finite_timeouts) / max(min(finite_timeouts), 0.1) > 20:
            ax.set_xscale('log')

    plt.tight_layout()
    path = os.path.join(out_dir, 'dynamic_goodput_vs_timeout.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  plot: {path}')


def save_csv(sweep_results: dict, plain_gp: float, plain_avg_rt: float,
             slo_ms: float, out_path: str):
    rows = []
    for mbs, timeout_dict in sorted(sweep_results.items()):
        for to, entry in sorted(timeout_dict.items()):
            ratio_gp = (entry['goodput'] / plain_gp
                        if plain_gp and plain_gp > 0 else None)
            rows.append({
                'max_bs2':          mbs,
                'timeout_ms':       to if to < float('inf') else 'inf',
                'slo_ms':           slo_ms,
                'goodput':          round(entry['goodput'], 6),
                'goodput_ratio':    round(ratio_gp, 4) if ratio_gp else '',
                'avg_rt_ms':        round(entry['avg_rt_ms'], 4),
                'p99_rt_ms':        round(entry['p99_rt_ms'], 4),
                'exit1_rate_pct':   round(entry['exit1_rate_pct'], 2),
                'avg_flush_bs':     round(entry['avg_flush_bs'], 2),
                'n_samples':        entry['n_samples'],
            })
    if not rows:
        return
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'  CSV : {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def _find_onnx(model_name: str, filename: str):
    p = os.path.join(paths.onnx_dir(model_name), filename)
    return p if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser(
        description='timeout-based dynamic batching goodput benchmark'
    )
    parser.add_argument('--threshold',          type=float, required=True)
    parser.add_argument('--slo-ms',             type=float, default=100.0,
                        help='고정 SLO 임계값 (ms). 이 SLO 기준으로 goodput 비교.')
    parser.add_argument('--data-root',          type=str,   default='/home2/imagenet')
    parser.add_argument('--n-samples',          type=int,   default=5000)
    parser.add_argument('--seed',               type=int,   default=42)
    parser.add_argument('--baseline-batch-size',type=int,   default=32,
                        help='seg1/plain 고정 batch 크기')
    parser.add_argument('--max-batch-sizes',    type=int,   nargs='+',
                        default=[32, 64, 128],
                        help='seg2 queue 최대 batch 크기 목록')
    parser.add_argument('--timeout-values',     type=float, nargs='+',
                        default=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                        help='sweep할 timeout 값 목록 (ms). 0=즉시flush, inf=timeout없음')
    parser.add_argument('--warmup',             type=int,   default=50)
    parser.add_argument('--seg1-onnx',          type=str,   default=None)
    parser.add_argument('--seg2-dynamic-onnx',  type=str,   default=None)
    parser.add_argument('--plain-onnx',         type=str,   default=None)
    parser.add_argument('--onnx-dir',           type=str,   default=None)
    parser.add_argument('--out-dir',            type=str,   default=None)
    parser.add_argument('--device-label',       type=str,   default='RTX 5090')
    parser.add_argument('--skip-plain',         action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs1     = args.baseline_batch_size
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'dynamic_benchmark_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    onnx_dir   = args.onnx_dir or paths.onnx_dir('ee_vit_large_2exit')
    seg1_path  = args.seg1_onnx         or os.path.join(onnx_dir, 'seg1.onnx')
    seg2_path  = args.seg2_dynamic_onnx or os.path.join(onnx_dir, 'seg2_dynamic.onnx')
    plain_path = args.plain_onnx        or _find_onnx('plain_vit_large', 'plain_vit_large.onnx')

    print(f'Device          : {device}  ({args.device_label})')
    print(f'Threshold       : {args.threshold}')
    print(f'Fixed SLO       : {args.slo_ms} ms')
    print(f'N-Samples       : {args.n_samples}  (seed={args.seed})')
    print(f'seg1/Plain BS   : {bs1}')
    print(f'Max batch sizes : {args.max_batch_sizes}')
    print(f'Timeout values  : {args.timeout_values} ms')
    print(f'Output          : {out_dir}\n')

    # ── 세션 로드 ──
    print('ONNX 세션 로드 ...')
    for p, name in [(seg1_path, 'seg1'), (seg2_path, 'seg2_dynamic')]:
        if not os.path.exists(p):
            print(f'[ERROR] {name} ONNX 없음: {p}')
            return
    seg1_sess   = build_session(seg1_path)
    seg2_sess   = build_session(seg2_path)
    plain_sess  = None
    if not args.skip_plain and plain_path and os.path.exists(plain_path):
        plain_sess = build_session(plain_path)

    # ── 데이터 로드 ──
    n_total  = (args.n_samples // bs1) * bs1
    warmup_n = args.warmup * bs1
    print(f'\n데이터 로딩 ({n_total + warmup_n}개) ...')
    dataset     = build_dataset(args.data_root)
    all_samples = sample_data(dataset, n_total + warmup_n, seed=args.seed)
    warmup_samp = all_samples[:warmup_n]
    eval_samp   = all_samples[warmup_n:]

    # ── GPU 웜업 ──
    print(f'\nGPU 웜업 ({args.warmup} seg1 배치) ...')
    with torch.no_grad():
        for i in range(0, len(warmup_samp), bs1):
            batch = warmup_samp[i : i + bs1]
            if len(batch) < bs1: break
            imgs = torch.stack([s[0] for s in batch]).to(device, non_blocking=True).contiguous()
            run_seg1_batch(seg1_sess, imgs)
    torch.cuda.synchronize()

    # ── PlainViT 기준선 ──
    plain_gp     = None
    plain_avg_rt = None
    if plain_sess:
        from benchmark.hybrid_vit_2exit_goodput import run_plain
        print(f'\nPlainViT 기준선 ...')
        with torch.no_grad():
            plain_records, plain_wall = run_plain(plain_sess, eval_samp, device, bs1)
        plain_avg_rt = float(np.mean([r['rt_ms'] for r in plain_records]))
        plain_gp     = goodput_at_slo(plain_records, plain_wall, args.slo_ms)
        print(f'  avg_rt={plain_avg_rt:.2f}ms  goodput@{args.slo_ms:.0f}ms={plain_gp:.4f}/ms')

    # ── Sweep ──
    sweep_results = {}   # {max_bs2: {timeout_ms: stats_dict}}

    for mbs in args.max_batch_sizes:
        sweep_results[mbs] = {}
        for to_ms in args.timeout_values:
            label = f'max_bs2={mbs}, timeout={to_ms}ms'
            print(f'\n[{label}] 벤치마크 ({len(eval_samp)}개) ...')

            with torch.no_grad():
                records, wall_time = run_dynamic_combo(
                    seg1_sess, seg2_sess, eval_samp, device,
                    args.threshold, bs1, mbs, to_ms)

            n       = len(records)
            n_exit1 = sum(1 for r in records if r['exit'] == 1)
            acc     = sum(r['correct'] for r in records) / n * 100
            avg_rt  = float(np.mean([r['rt_ms'] for r in records]))
            p99_rt  = float(np.percentile([r['rt_ms'] for r in records], 99))
            gp      = goodput_at_slo(records, wall_time, args.slo_ms)

            nonexit = [r for r in records if r['exit'] == 2]
            avg_fbs = float(np.mean([r['flush_bs'] for r in nonexit])) if nonexit else 0.0

            print(f'  acc={acc:.2f}%  avg_rt={avg_rt:.2f}ms  p99={p99_rt:.2f}ms  '
                  f'goodput={gp:.4f}/ms  exit1={n_exit1/n*100:.1f}%  '
                  f'avg_flush_bs={avg_fbs:.1f}')

            sweep_results[mbs][to_ms] = {
                'goodput':        gp,
                'avg_rt_ms':      avg_rt,
                'p99_rt_ms':      p99_rt,
                'exit1_rate_pct': n_exit1 / n * 100,
                'avg_flush_bs':   avg_fbs,
                'accuracy_pct':   acc,
                'n_samples':      n,
            }

            with open(os.path.join(out_dir, f'records_mbs{mbs}_to{to_ms}.json'), 'w') as f:
                json.dump({'max_bs2': mbs, 'timeout_ms': to_ms,
                           'wall_time': wall_time, 'records': records}, f, indent=2)

    # ── 저장 및 플롯 ──
    save_csv(sweep_results, plain_gp, plain_avg_rt, args.slo_ms,
             os.path.join(out_dir, 'dynamic_summary.csv'))
    with open(os.path.join(out_dir, 'dynamic_summary.json'), 'w') as f:
        json.dump({'threshold': args.threshold, 'slo_ms': args.slo_ms,
                   'plain_goodput': plain_gp, 'plain_avg_rt_ms': plain_avg_rt,
                   'sweep': {str(k): {str(t): v for t, v in d.items()}
                              for k, d in sweep_results.items()}}, f, indent=2)

    plot_results(sweep_results, plain_gp, plain_avg_rt,
                 args.timeout_values, args.slo_ms,
                 out_dir, args.threshold, bs1, args.device_label)

    print(f'\nDone! → {out_dir}')


if __name__ == '__main__':
    main()
