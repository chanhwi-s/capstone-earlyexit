"""
hybrid_vit_2exit_goodput.py — 2-exit hybrid goodput benchmark (static seg2, SLO 기반)

기존 onnx_realrun과의 차이점:
  - seg2: dynamic → static batch (bs2 고정, seg2_bs{N}.onnx 사용)
  - timeout 없음: queue에 bs2개 쌓이면 flush
  - 마지막 partial queue (<bs2): static shape 제약으로 폐기
  - 모든 per-sample rt 저장 → post-hoc SLO 분석

goodput(SLO) = count(rt ≤ SLO_ms) / total_wall_time
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

try:
    from analysis.plot_latency_dist import plot_overview, plot_combined, plot_per_bs2
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


# ── ONNX Runtime 헬퍼 ─────────────────────────────────────────────────────────

def build_session(onnx_path: str, device_id: int = 0) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        onnx_path, sess_options=opts,
        providers=[('CUDAExecutionProvider', {'device_id': device_id}),
                   'CPUExecutionProvider'])
    print(f'  {os.path.basename(onnx_path):40s} → {sess.get_providers()[0]}')
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
    """Returns: (feats [bs1,197,768] GPU, logits_np [bs1,1000] CPU, elapsed_ms)"""
    bs = imgs_t.shape[0]
    iob = sess.io_binding()
    iob.bind_input('image', device_type='cuda', device_id=0,
                   element_type=np.float32, shape=(bs, 3, 224, 224),
                   buffer_ptr=imgs_t.data_ptr())
    iob.bind_output('feat_out',  device_type='cuda')
    iob.bind_output('ee_logits', device_type='cuda')
    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))
    outs = iob.get_outputs()
    return _ort_to_torch(outs[0]), outs[1].numpy(), elapsed_ms


def run_seg2_static(sess, q_feats: list, bs2: int):
    """
    static batch=bs2 seg2 실행. len(q_feats) == bs2 보장 필요.
    Returns: (preds list[int], elapsed_ms float)
    hidden_dim은 batch.shape에서 자동 추론 (ViT-B: 768, ViT-L: 1024)
    """
    batch = torch.cat(q_feats, dim=0).contiguous()   # [bs2, 197, hidden_dim]
    iob = sess.io_binding()
    iob.bind_input('feat_in', device_type='cuda', device_id=0,
                   element_type=np.float32, shape=tuple(batch.shape),
                   buffer_ptr=batch.data_ptr())
    iob.bind_output('ee_logits', device_type='cuda')
    elapsed_ms = _sync_time(lambda: sess.run_with_iobinding(iob))
    logits_np  = iob.get_outputs()[0].numpy()
    return logits_np.argmax(axis=1).tolist(), elapsed_ms


def run_plain_batch(sess, imgs_t: torch.Tensor):
    """Returns: (preds list[int], elapsed_ms float)"""
    bs = imgs_t.shape[0]
    iob = sess.io_binding()
    iob.bind_input('image', device_type='cuda', device_id=0,
                   element_type=np.float32, shape=(bs, 3, 224, 224),
                   buffer_ptr=imgs_t.data_ptr())
    iob.bind_output('logits', device_type='cuda')
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

def run_combo(seg1_sess, seg2_sess, samples, device, thr, bs1, bs2):
    """
    No timeout. Queue flush only when exactly bs2 non-exit samples accumulate.
    End-of-dataset partial queue (<bs2) is DISCARDED (static shape constraint).

    per-sample rt 분해:
      exit1  : rt = seg1_ms  (queue_wait=0, seg2_ms=0)
      exit2  : rt = seg1_ms + queue_wait + seg2_ms
                   - seg1_ms   : 해당 샘플이 속한 seg1 배치 실행 시간
                   - queue_wait: seg1 완료 후 seg2 flush 시작 전까지 대기
                   - seg2_ms   : seg2 배치 실행 시간 (flush 내 모두 동일)

    Returns:
        records   : list of {rt_ms, exit, correct, seg1_ms, queue_wait_ms, seg2_ms}
        wall_time : 총 시뮬레이션 시간 (ms)
        n_discard : 마지막에 폐기된 non-exit 샘플 수
    """
    records = []
    q_feats, q_ts, q_seg1_done, q_lbls = [], [], [], []
    t = 0.0

    def flush():
        nonlocal t
        preds, seg2_ms_val = run_seg2_static(seg2_sess, q_feats, bs2)
        t_seg2_start = t          # seg2 시작 시점 (t += seg2_ms 이전)
        t += seg2_ms_val
        for k in range(bs2):
            seg1_ms_k    = q_seg1_done[k] - q_ts[k]
            queue_wait_k = t_seg2_start - q_seg1_done[k]
            records.append({
                'rt_ms':         t - q_ts[k],
                'exit':          2,
                'correct':       int(preds[k] == q_lbls[k]),
                'seg1_ms':       seg1_ms_k,
                'queue_wait_ms': queue_wait_k,
                'seg2_ms':       seg2_ms_val,
            })
        q_feats.clear(); q_ts.clear(); q_seg1_done.clear(); q_lbls.clear()

    for i in range(0, len(samples), bs1):
        batch = samples[i : i + bs1]
        if len(batch) < bs1:
            break   # 마지막 incomplete seg1 배치 폐기

        imgs = torch.stack([s[0] for s in batch]).to(device, non_blocking=True).contiguous()
        feats, logits_np, seg1_ms = run_seg1_batch(seg1_sess, imgs)

        ts_batch = t
        t += seg1_ms
        t_seg1_done = t   # 이 seg1 배치 완료 시점

        for k in range(bs1):
            lbl  = batch[k][1]
            conf = _conf(logits_np[k])
            pred = int(logits_np[k].argmax())

            if conf >= thr:
                records.append({
                    'rt_ms':         seg1_ms,
                    'exit':          1,
                    'correct':       int(pred == lbl),
                    'seg1_ms':       seg1_ms,
                    'queue_wait_ms': 0.0,
                    'seg2_ms':       0.0,
                })
            else:
                q_feats.append(feats[k : k + 1])
                q_ts.append(ts_batch)
                q_seg1_done.append(t_seg1_done)
                q_lbls.append(lbl)
                if len(q_feats) == bs2:
                    flush()

    return records, t, len(q_feats)   # len(q_feats) = n_discarded


def run_plain(plain_sess, samples, device, bs1):
    """
    PlainViT baseline. 모든 샘플의 rt 저장.
    각 배치 내 bs1개 샘플은 동일한 rt = batch_ms 를 공유.

    Returns:
        records   : list of {rt_ms, correct}
        wall_time : 총 시간 (ms)
    """
    records = []
    wall_t  = 0.0
    for i in range(0, len(samples), bs1):
        batch = samples[i : i + bs1]
        if len(batch) < bs1:
            break
        imgs = torch.stack([s[0] for s in batch]).to(device, non_blocking=True).contiguous()
        lbls = [s[1] for s in batch]
        preds, batch_ms = run_plain_batch(plain_sess, imgs)
        wall_t += batch_ms
        for k in range(bs1):
            records.append({'rt_ms': batch_ms, 'correct': int(preds[k] == lbls[k])})
    return records, wall_t


# ── Goodput 분석 ──────────────────────────────────────────────────────────────

def goodput(records: list, wall_time: float, slo_ms: float) -> float:
    """goodput = count(rt ≤ slo_ms) / wall_time"""
    n = sum(1 for r in records if r['rt_ms'] <= slo_ms)
    return n / wall_time if wall_time > 0 else 0.0


def goodput_curve(records: list, wall_time: float, slo_values: list) -> list:
    return [goodput(records, wall_time, s) for s in slo_values]


# ── 저장 ──────────────────────────────────────────────────────────────────────

def save_csv(rows: list, path: str):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'  CSV : {path}')


# ── 플롯 ──────────────────────────────────────────────────────────────────────

def plot_goodput(hybrid_curves: dict, plain_gp: list,
                 slo_values, batch_sizes, out_path,
                 threshold, bs1, device_label):
    """
    좌: 절대 goodput (samples/ms) vs SLO
    우: goodput ratio (hybrid / plain) vs SLO
    """
    tab10 = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Goodput vs SLO — 2-exit Hybrid ONNX  (thr={threshold:.2f}, {device_label})',
        fontsize=12)

    # 절대값
    ax = axes[0]
    for i, bs2 in enumerate(batch_sizes):
        if bs2 not in hybrid_curves:
            continue
        ax.plot(slo_values, hybrid_curves[bs2],
                marker='o', color=tab10(i), linewidth=2, markersize=5,
                label=f'Hybrid bs2={bs2}')
    ax.plot(slo_values, plain_gp,
            'k--', linewidth=2, marker='s', markersize=5,
            label=f'PlainViT (bs={bs1})')
    ax.set_xlabel('SLO Threshold (ms)'); ax.set_ylabel('Goodput (samples/ms)')
    ax.set_title('Absolute Goodput')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 비율
    ax = axes[1]
    for i, bs2 in enumerate(batch_sizes):
        if bs2 not in hybrid_curves:
            continue
        ratio = [(h / p if p > 0 else 0.0)
                 for h, p in zip(hybrid_curves[bs2], plain_gp)]
        ax.plot(slo_values, ratio,
                marker='o', color=tab10(i), linewidth=2, markersize=5,
                label=f'bs2={bs2}')
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='Plain = 1.0')
    ax.set_xlabel('SLO Threshold (ms)'); ax.set_ylabel('Goodput Ratio (hybrid / plain)')
    ax.set_title('Goodput Ratio vs PlainViT')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


def plot_latency_decomposition(hybrid_data: dict, plain_avg_rt: float,
                               batch_sizes, out_path,
                               threshold, bs1, device_label):
    """
    bs2별 non-exit 샘플 평균 latency 분해 스택 바 차트.
    seg1_ms | queue_wait_ms | seg2_ms 로 rt를 구성 요소별로 시각화.
    plain avg_rt를 수평선으로 표시.
    """
    present = [b for b in batch_sizes if b in hybrid_data]
    if not present:
        return

    seg1_avgs  = []
    queue_avgs = []
    seg2_avgs  = []

    for bs2 in present:
        nonexit = [r for r in hybrid_data[bs2] if r['exit'] == 2]
        if not nonexit:
            seg1_avgs.append(0); queue_avgs.append(0); seg2_avgs.append(0)
            continue
        seg1_avgs.append(float(np.mean([r['seg1_ms']       for r in nonexit])))
        queue_avgs.append(float(np.mean([r['queue_wait_ms'] for r in nonexit])))
        seg2_avgs.append(float(np.mean([r['seg2_ms']        for r in nonexit])))

    x = np.arange(len(present))
    width = 0.5

    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.2 + 2), 5))
    b1 = ax.bar(x, seg1_avgs,  width, label='seg1',       color='#4c72b0')
    b2 = ax.bar(x, queue_avgs, width, bottom=seg1_avgs,   label='queue_wait', color='#dd8452')
    b3 = ax.bar(x, seg2_avgs,  width,
                bottom=[s + q for s, q in zip(seg1_avgs, queue_avgs)],
                label='seg2', color='#55a868')

    if plain_avg_rt is not None:
        ax.axhline(plain_avg_rt, color='black', linestyle='--', linewidth=1.5,
                   label=f'PlainViT avg rt ({plain_avg_rt:.1f}ms)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'bs2={b}' for b in present], fontsize=10)
    ax.set_xlabel('Seg2 Batch Size', fontsize=11)
    ax.set_ylabel('Avg Response Time (ms) — non-exit samples', fontsize=10)
    ax.set_title(
        f'Latency Decomposition — non-exit samples\n'
        f'(thr={threshold:.2f}, {device_label}, seg1 bs={bs1})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


def plot_avg_latency_vs_batchsize(hybrid_data: dict, plain_avg_rt: float,
                                   batch_sizes, out_path,
                                   threshold, bs1, device_label):
    """
    bs2별 전체 샘플(exit1 + exit2) 평균 latency 바 차트.
    plain avg latency를 점선으로 오버레이.
    우측: hybrid / plain 비율.
    """
    present = [b for b in batch_sizes if b in hybrid_data]
    if not present:
        return

    avg_lats = []
    for bs2 in present:
        rts = [r['rt_ms'] for r in hybrid_data[bs2]]
        avg_lats.append(float(np.mean(rts)) if rts else 0.0)

    x     = np.arange(len(present))
    width = 0.5
    tab10 = plt.get_cmap('tab10')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Avg Latency vs Batch Size — 2-exit Hybrid ONNX  (thr={threshold:.2f}, {device_label})',
        fontsize=12)

    # ── 좌: 절대 avg latency ────────────────────────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, avg_lats, width, color=[tab10(i) for i in range(len(present))],
                  alpha=0.85, label='Hybrid avg latency')
    if plain_avg_rt is not None:
        ax.axhline(plain_avg_rt, color='black', linestyle='--', linewidth=2,
                   label=f'PlainViT avg ({plain_avg_rt:.1f} ms)')
    for bar, val in zip(bars, avg_lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'bs2={b}' for b in present], fontsize=10)
    ax.set_xlabel('Seg2 Batch Size', fontsize=11)
    ax.set_ylabel('Avg Response Time (ms) — all samples', fontsize=11)
    ax.set_title('Avg Latency (All Samples)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    # ── 우: 비율 (hybrid / plain) ────────────────────────────────────────────────
    ax = axes[1]
    if plain_avg_rt and plain_avg_rt > 0:
        ratios = [lat / plain_avg_rt for lat in avg_lats]
        ratio_bars = ax.bar(x, ratios, width,
                            color=[tab10(i) for i in range(len(present))], alpha=0.85)
        ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Plain = 1.0')
        for bar, val in zip(ratio_bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}x', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Latency Ratio (hybrid / plain)', fontsize=11)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'PlainViT data unavailable',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'bs2={b}' for b in present], fontsize=10)
    ax.set_xlabel('Seg2 Batch Size', fontsize=11)
    ax.set_title('Latency Ratio vs PlainViT', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


def plot_latency_cdf(hybrid_data: dict, plain_records,
                     batch_sizes, slo_values, out_path,
                     threshold, bs1, device_label):
    """
    응답시간 CDF (log-scale x).
    SLO 기준선을 수직선으로 표시해 "각 SLO에서 몇 %가 통과하는지" 직접 확인 가능.
    """
    tab10 = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(10, 6))

    if plain_records:
        rts = sorted(r['rt_ms'] for r in plain_records)
        cdf = np.arange(1, len(rts) + 1) / len(rts)
        ax.plot(rts, cdf, 'k--', linewidth=2, label=f'PlainViT (bs={bs1})')

    for i, bs2 in enumerate(batch_sizes):
        if bs2 not in hybrid_data:
            continue
        rts = sorted(r['rt_ms'] for r in hybrid_data[bs2])
        cdf = np.arange(1, len(rts) + 1) / len(rts)
        ax.plot(rts, cdf, color=tab10(i), linewidth=2, label=f'Hybrid bs2={bs2}')

    # 주요 SLO 기준선 (최대 5개만 표시해 plot 혼잡 방지)
    slo_show = slo_values[::max(1, len(slo_values) // 5)][:5]
    for slo in slo_show:
        ax.axvline(slo, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.text(slo * 1.02, 0.02, f'{slo:.0f}ms', fontsize=7, color='gray')

    ax.set_xlabel('Response Time (ms)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title(
        f'Response Time CDF — 2-exit Hybrid ONNX  (thr={threshold:.2f}, {device_label})',
        fontsize=11)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def _find_onnx(model_name: str, filename: str):
    p = os.path.join(paths.onnx_dir(model_name), filename)
    return p if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser(
        description='2-exit hybrid goodput benchmark (static seg2, SLO-based)'
    )
    parser.add_argument('--threshold',           type=float, required=True)
    parser.add_argument('--data-root',           type=str,   default='/home2/imagenet')
    parser.add_argument('--n-samples',           type=int,   default=5000)
    parser.add_argument('--seed',                type=int,   default=42)
    parser.add_argument('--baseline-batch-size', type=int,   default=8,
                        help='seg1/plain 고정 batch 크기 (기본: 8)')
    parser.add_argument('--batch-sizes',         type=int,   nargs='+',
                        default=[8, 16, 32, 64],
                        help='seg2 static batch 목록. seg2_bs{N}.onnx 필요.')
    parser.add_argument('--slo-values',          type=float, nargs='+',
                        default=[10., 20., 30., 50., 75., 100., 150., 200., 300., 500.],
                        help='SLO 임계값 목록 (ms)')
    parser.add_argument('--warmup',              type=int,   default=50,
                        help='웜업 seg1 배치 수 (기본: 50)')
    parser.add_argument('--seg1-onnx',           type=str,   default=None)
    parser.add_argument('--plain-onnx',          type=str,   default=None)
    parser.add_argument('--onnx-dir',            type=str,   default=None,
                        help='seg2_bs{N}.onnx 디렉토리 (기본: 최신 exp 자동 탐색)')
    parser.add_argument('--out-dir',             type=str,   default=None)
    parser.add_argument('--device-label',        type=str,   default='RTX 5090')
    parser.add_argument('--skip-plain',          action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs1     = args.baseline_batch_size
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir or os.path.join(
        paths.EXPERIMENTS_DIR, 'eval', f'hybrid_2exit_goodput_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    seg1_path  = args.seg1_onnx  or _find_onnx('ee_vit_2exit', 'seg1.onnx')
    plain_path = args.plain_onnx or _find_onnx('plain_vit',    'plain_vit.onnx')
    seg2_dir   = args.onnx_dir   or paths.onnx_dir('ee_vit_2exit')

    print(f'Device        : {device}  ({args.device_label})')
    print(f'Threshold     : {args.threshold}')
    print(f'N-Samples     : {args.n_samples}  (seed={args.seed})')
    print(f'seg1/Plain BS : {bs1}  (static)')
    print(f'seg2 BS list  : {args.batch_sizes}')
    print(f'SLO values    : {args.slo_values} ms')
    print(f'seg2 dir      : {seg2_dir}')
    print(f'Output        : {out_dir}\n')

    # ── 세션 로드 ──
    print('ONNX 세션 로드 ...')
    if not seg1_path:
        print('[ERROR] seg1.onnx 없음'); return
    seg1_sess  = build_session(seg1_path)
    plain_sess = build_session(plain_path) if (plain_path and not args.skip_plain) else None
    if not plain_sess and not args.skip_plain:
        print('[WARN] plain_vit.onnx 없음. --skip-plain 으로 건너뜀.')

    # ── 데이터 로딩 ──
    n_total  = (args.n_samples // bs1) * bs1
    warmup_n = args.warmup * bs1
    if n_total != args.n_samples:
        print(f'  n_samples {args.n_samples} → {n_total}  (bs1={bs1} 배수 조정)')
    print(f'\n데이터 로딩 ({n_total + warmup_n}개) ...')
    dataset     = build_dataset(args.data_root)
    all_samples = sample_data(dataset, n_total + warmup_n, seed=args.seed)
    warmup_samp = all_samples[:warmup_n]
    eval_samp   = all_samples[warmup_n:]
    print(f'  warmup={len(warmup_samp)}, eval={len(eval_samp)}')

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
    plain_records, plain_wall = None, None
    if plain_sess:
        print(f'\nPlainViT 기준선 ({len(eval_samp)}개) ...')
        with torch.no_grad():
            plain_records, plain_wall = run_plain(plain_sess, eval_samp, device, bs1)
        plain_acc    = sum(r['correct'] for r in plain_records) / len(plain_records) * 100
        plain_avg_rt = float(np.mean([r['rt_ms'] for r in plain_records]))
        print(f'  acc={plain_acc:.2f}%  avg_rt={plain_avg_rt:.2f}ms  '
              f'overall_tput={len(plain_records)/plain_wall:.4f}/ms')
        print(f'  ※ SLO < {plain_avg_rt:.1f}ms 이면 plain goodput ≈ 0 (배치 단위 처리)')
        with open(os.path.join(out_dir, 'plain_records.json'), 'w') as f:
            json.dump({'records': plain_records, 'wall_time': plain_wall,
                       'accuracy_pct': plain_acc, 'avg_rt_ms': plain_avg_rt,
                       'overall_tput': len(plain_records) / plain_wall}, f, indent=2)

    # ── Hybrid 벤치마크 ──
    summary_rows   = []
    hybrid_curves  = {}   # bs2 → goodput list
    hybrid_records = {}   # bs2 → records list (CDF용)

    for bs2 in args.batch_sizes:
        seg2_path = os.path.join(seg2_dir, f'seg2_bs{bs2}.onnx')
        if not os.path.exists(seg2_path):
            print(f'\n[SKIP] bs2={bs2}: {seg2_path} 없음')
            print(f'         → python src/export/export_onnx_seg2_static.py --batch-sizes {bs2}')
            continue

        print(f'\n[bs2={bs2}] 로드 + 벤치마크 ({len(eval_samp)}개) ...')
        seg2_sess = build_session(seg2_path)

        with torch.no_grad():
            records, wall_time, n_discard = run_combo(
                seg1_sess, seg2_sess, eval_samp, device, args.threshold, bs1, bs2)

        del seg2_sess
        torch.cuda.empty_cache()

        n        = len(records)
        n_exit1  = sum(1 for r in records if r['exit'] == 1)
        n_exit2  = n - n_exit1
        acc      = sum(r['correct'] for r in records) / n * 100
        avg_rt   = float(np.mean([r['rt_ms'] for r in records]))
        p99_rt   = float(np.percentile([r['rt_ms'] for r in records], 99))

        # 폐기 비율 경고
        n_nonexit_total = n_exit2 + n_discard
        discard_pct = (n_discard / n_nonexit_total * 100) if n_nonexit_total > 0 else 0.0
        if n_nonexit_total == 0:
            print(f'  [WARNING] non-exit 샘플이 없습니다. threshold가 너무 낮지 않은지 확인.')
        elif n_discard == n_nonexit_total:
            print(f'  [WARNING] seg2 flush 한 번도 발생 안 함! '
                  f'non-exit {n_nonexit_total}개 전부 폐기. bs2를 줄이거나 n-samples 늘리세요.')
        elif discard_pct > 20:
            print(f'  [WARNING] non-exit 폐기율 {discard_pct:.1f}% (={n_discard}개). '
                  f'bs2를 줄이는 것을 고려.')
        else:
            print(f'  discarded={n_discard}  (non-exit 폐기율 {discard_pct:.1f}%)')

        # queueing delay 통계 (non-exit 샘플만)
        nonexit_recs = [r for r in records if r['exit'] == 2]
        if nonexit_recs:
            avg_queue = float(np.mean([r['queue_wait_ms'] for r in nonexit_recs]))
            p99_queue = float(np.percentile([r['queue_wait_ms'] for r in nonexit_recs], 99))
            avg_seg2  = float(np.mean([r['seg2_ms'] for r in nonexit_recs]))
        else:
            avg_queue = p99_queue = avg_seg2 = 0.0

        print(f'  samples={n}  exit1={n_exit1/n*100:.1f}%  exit2={n_exit2/n*100:.1f}%')
        print(f'  acc={acc:.2f}%  avg_rt={avg_rt:.2f}ms  p99_rt={p99_rt:.2f}ms  '
              f'overall_tput={n/wall_time:.4f}/ms')
        if nonexit_recs:
            print(f'  [non-exit 분해] avg_seg1={float(np.mean([r["seg1_ms"] for r in nonexit_recs])):.2f}ms  '
                  f'avg_queue_wait={avg_queue:.2f}ms  avg_seg2={avg_seg2:.2f}ms  '
                  f'p99_queue_wait={p99_queue:.2f}ms')

        gp_values = goodput_curve(records, wall_time, args.slo_values)
        hybrid_curves[bs2]  = gp_values
        hybrid_records[bs2] = records

        row = {
            'bs2':                bs2,
            'threshold':          args.threshold,
            'n_samples':          n,
            'n_discarded':        n_discard,
            'discard_pct':        round(discard_pct, 2),
            'exit_rate_b1_pct':   round(n_exit1 / n * 100, 2),
            'exit_rate_b2_pct':   round(n_exit2 / n * 100, 2),
            'accuracy_pct':       round(acc, 4),
            'avg_rt_ms':          round(avg_rt, 4),
            'p99_rt_ms':          round(p99_rt, 4),
            'overall_tput':       round(n / wall_time, 6),
            'avg_queue_wait_ms':  round(avg_queue, 4),
            'p99_queue_wait_ms':  round(p99_queue, 4),
            'avg_seg2_ms':        round(avg_seg2, 4),
        }
        for slo, gp in zip(args.slo_values, gp_values):
            row[f'goodput_slo{slo:.0f}ms'] = round(gp, 6)
            if plain_records:
                plain_gp_at_slo = goodput(plain_records, plain_wall, slo)
                row[f'goodput_ratio_slo{slo:.0f}ms'] = (
                    round(gp / plain_gp_at_slo, 4) if plain_gp_at_slo > 0 else float('inf'))
        summary_rows.append(row)

        # raw rt 저장 (SLO 재분석용)
        with open(os.path.join(out_dir, f'records_bs{bs2}.json'), 'w') as f:
            json.dump({'bs2': bs2, 'wall_time': wall_time, 'n_discarded': n_discard,
                       'records': records}, f, indent=2)

    # ── 저장 ──
    save_csv(summary_rows, os.path.join(out_dir, 'goodput_summary.csv'))
    with open(os.path.join(out_dir, 'goodput_summary.json'), 'w') as f:
        json.dump(summary_rows, f, indent=2)

    # ── 플롯 ──
    if hybrid_curves:
        plain_gp = goodput_curve(plain_records, plain_wall, args.slo_values) \
                   if plain_records else [0.0] * len(args.slo_values)

        plot_goodput(
            hybrid_curves, plain_gp, args.slo_values, args.batch_sizes,
            os.path.join(out_dir, 'goodput_vs_slo.png'),
            args.threshold, bs1, args.device_label)

        plot_latency_cdf(
            hybrid_records, plain_records, args.batch_sizes, args.slo_values,
            os.path.join(out_dir, 'latency_cdf.png'),
            args.threshold, bs1, args.device_label)

        plain_avg_rt = float(np.mean([r['rt_ms'] for r in plain_records])) \
                       if plain_records else None
        plot_latency_decomposition(
            hybrid_records, plain_avg_rt, args.batch_sizes,
            os.path.join(out_dir, 'latency_decomposition.png'),
            args.threshold, bs1, args.device_label)

        plot_avg_latency_vs_batchsize(
            hybrid_records, plain_avg_rt, args.batch_sizes,
            os.path.join(out_dir, 'avg_latency_vs_batchsize.png'),
            args.threshold, bs1, args.device_label)

        if _DIST_AVAILABLE:
            plain_rts = [r['rt_ms'] for r in plain_records] if plain_records else None
            dist_data = {
                bs2: {
                    'exit1':   [r['rt_ms'] for r in recs if r['exit'] == 1],
                    'exit2':   [r['rt_ms'] for r in recs if r['exit'] == 2],
                    'all':     [r['rt_ms'] for r in recs],
                    'n':       len(recs),
                    'n_exit1': sum(1 for r in recs if r['exit'] == 1),
                    'n_exit2': sum(1 for r in recs if r['exit'] == 2),
                }
                for bs2, recs in hybrid_records.items()
            }
            bs2_list = [b for b in args.batch_sizes if b in dist_data]

            plot_overview(
                dist_data, plain_rts, bs2_list,
                os.path.join(out_dir, 'latency_dist_overview.png'),
                args.threshold, args.device_label)

            plot_combined(
                dist_data, plain_rts, bs2_list,
                os.path.join(out_dir, 'latency_dist_combined.png'),
                args.threshold, args.device_label)

            for bs2 in bs2_list:
                plot_per_bs2(
                    dist_data, plain_rts, bs2,
                    os.path.join(out_dir, f'latency_dist_bs{bs2}.png'),
                    args.threshold, args.device_label)

    print(f'\nDone! → {out_dir}')


if __name__ == '__main__':
    main()
