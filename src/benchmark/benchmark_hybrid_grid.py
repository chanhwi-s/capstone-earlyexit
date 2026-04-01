"""
Hybrid Runtime Grid Search — batch_size × timeout_ms 파라미터 탐색

Hybrid 런타임(VEE seg1 + batched plain fallback)에서 두 핵심 파라미터를
grid search하여 최적 조합을 찾고 시각화합니다.

탐색 파라미터:
  - batch_size  : 배치 fallback 크기  (기본 후보: 1, 2, 4, 8, 16)
  - timeout_ms  : 배치 대기 타임아웃  (기본 후보: 5, 10, 20, 40 ms)

측정 지표:
  - Accuracy, Avg Latency (ms), P99 Latency (ms)
  - Avg Throughput (inf/sec), P99 Goodput (inf/sec)
  - Tail Latency Ratio (P99/P50)
  - Exit1 Rate (%), Fallback Rate (%)

저장 결과:
  experiments/eval/hybrid_grid/
  ├── grid_search_thr{thr}.json   ← 전체 수치 결과
  └── grid_search_thr{thr}.png    ← 히트맵 + 라인 플롯

사용법:
  # 기본 grid search (threshold=0.80)
  python benchmark_hybrid_grid.py --threshold 0.80 --num-samples 500

  # 커스텀 grid 지정
  python benchmark_hybrid_grid.py \\
      --batch-sizes 1 2 4 8 16 \\
      --timeout-ms  5 10 20 40 \\
      --threshold 0.80 --num-samples 500
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from profiling_utils import compute_latency_stats


# ── TRT Engine ───────────────────────────────────────────────────────────────

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


# ── 데이터 로드 ──────────────────────────────────────────────────────────────

def load_test_data(num_samples):
    from datasets.dataloader import get_dataloader
    from utils import load_config
    cfg = load_config('configs/train.yaml')
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'], batch_size=1,
        data_root=cfg['dataset']['data_root'], num_workers=0,
        seed=cfg['train']['seed'],
    )
    images, labels = [], []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images.append(img)
        labels.append(lbl[0].item())
    return images, labels


# ── Hybrid 단일 파라미터 조합 벤치마크 ───────────────────────────────────────

def bench_hybrid_once(vee_seg1, plain_engine, images, labels,
                      threshold, batch_size, timeout_ms):
    """한 가지 (batch_size, timeout_ms) 조합에 대해 hybrid 벤치마크 실행."""
    from infer.infer_trt_hybrid import HybridOrchestrator

    orch = HybridOrchestrator(vee_seg1, plain_engine,
                              batch_size=batch_size, timeout_ms=timeout_ms)
    run  = orch.run_stream(images, labels, threshold)

    n = len(labels)
    correct = sum(
        1 for i in range(n)
        if run['results'][i] is not None and
        run['results'][i]['pred'] == labels[i]
    )
    lats       = run['latencies_ms']
    exit1_rate = run['exit1_count']    / n * 100
    fb_rate    = run['fallback_count'] / n * 100
    stats      = compute_latency_stats(lats)

    return {
        'accuracy':     correct / n,
        'exit1_rate':   exit1_rate,
        'fallback_rate': fb_rate,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()},
    }


# ── Grid Search 실행 ─────────────────────────────────────────────────────────

def run_grid_search(vee_seg1, plain_engine, images, labels,
                    batch_sizes, timeout_ms_list, threshold):
    """(batch_size × timeout_ms) 전체 grid 실행.

    Returns:
        grid_results: dict[(bs, to_ms)] = result_dict
    """
    total = len(batch_sizes) * len(timeout_ms_list)
    done  = 0
    grid_results = {}

    print(f"\n{'='*70}")
    print(f"  Hybrid Grid Search  |  threshold={threshold}  |  총 {total}개 조합")
    print(f"{'='*70}")
    print(f"  {'bs':>4}  {'to_ms':>7}  {'acc':>7}  {'avg_ms':>8}  {'p99_ms':>8}  "
          f"{'tp':>8}  {'exit1%':>7}  {'fb%':>6}")
    print(f"  {'-'*70}")

    for bs in batch_sizes:
        for to_ms in timeout_ms_list:
            done += 1
            key  = (bs, to_ms)
            try:
                r = bench_hybrid_once(vee_seg1, plain_engine, images, labels,
                                      threshold, bs, to_ms)
                grid_results[key] = r
                print(f"  {bs:>4}  {to_ms:>7.1f}  "
                      f"{r['accuracy']:>7.4f}  {r['avg_ms']:>8.2f}  "
                      f"{r['p99_ms']:>8.2f}  {r['avg_throughput']:>8.1f}  "
                      f"{r['exit1_rate']:>7.1f}  {r['fallback_rate']:>6.1f}  "
                      f"  [{done}/{total}]")
            except Exception as e:
                print(f"  {bs:>4}  {to_ms:>7.1f}  ERROR: {e}")
                grid_results[key] = None

    return grid_results


# ── 시각화 ───────────────────────────────────────────────────────────────────

def plot_grid_search(grid_results, batch_sizes, timeout_ms_list, threshold, save_path):
    """Grid search 결과 시각화: 히트맵 + 라인 플롯."""
    valid = {k: v for k, v in grid_results.items() if v is not None}
    if not valid:
        print("[WARN] 유효한 grid 결과 없음, 플롯 생략")
        return

    bs_list = sorted(set(k[0] for k in valid))
    to_list = sorted(set(k[1] for k in valid))
    n_bs    = len(bs_list)
    n_to    = len(to_list)

    metrics_heatmap = [
        ('avg_ms',         'Avg Latency (ms)',      'YlOrRd'),
        ('p99_ms',         'P99 Latency (ms)',       'YlOrRd'),
        ('avg_throughput', 'Avg Throughput (inf/s)', 'YlGn'),
        ('p99_goodput',    'P99 Goodput (inf/s)',    'YlGn'),
        ('accuracy',       'Accuracy',               'Blues'),
        ('tail_ratio_p99_p50', 'P99/P50 Tail Ratio','PuRd'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Hybrid Runtime Grid Search  (threshold={threshold})', fontsize=14)

    for ax_idx, (metric, title, cmap) in enumerate(metrics_heatmap):
        ax = axes[ax_idx // 3][ax_idx % 3]

        # 2D 행렬 구성 (행=batch_size, 열=timeout_ms)
        mat = np.full((n_bs, n_to), np.nan)
        for bi, bs in enumerate(bs_list):
            for ti, to_ms in enumerate(to_list):
                key = (bs, to_ms)
                if key in valid and valid[key] is not None:
                    mat[bi, ti] = valid[key].get(metric, np.nan)

        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_to))
        ax.set_xticklabels([f'{t:.0f}' for t in to_list], fontsize=8)
        ax.set_yticks(range(n_bs))
        ax.set_yticklabels([str(b) for b in bs_list], fontsize=8)
        ax.set_xlabel('Timeout (ms)')
        ax.set_ylabel('Batch Size')
        ax.set_title(title)

        # 셀 내 수치 표시
        for bi in range(n_bs):
            for ti in range(n_to):
                val = mat[bi, ti]
                if not np.isnan(val):
                    fmt = f'{val:.3f}' if metric == 'accuracy' else f'{val:.3f}'
                    ax.text(ti, bi, fmt, ha='center', va='center',
                            fontsize=7, color='black')

        # 최적 셀 표시 (throughput/accuracy는 최대, latency/tail은 최소)
        if not np.all(np.isnan(mat)):
            if metric in ('avg_ms', 'p99_ms', 'tail_ratio_p99_p50'):
                opt = np.unravel_index(np.nanargmin(mat), mat.shape)
            else:
                opt = np.unravel_index(np.nanargmax(mat), mat.shape)
            ax.add_patch(plt.Rectangle(
                (opt[1] - 0.5, opt[0] - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=2.5, zorder=5,
            ))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"히트맵 저장: {save_path}")

    # ── 라인 플롯: batch_size별 avg_ms vs timeout_ms ──────────────────────────
    line_path = save_path.replace('.png', '_line.png')
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f'Hybrid Grid — Line Plots  (threshold={threshold})', fontsize=12)

    line_metrics = [
        ('avg_ms',         'Avg Latency (ms)',        False),
        ('avg_throughput', 'Avg Throughput (inf/s)',   True),
        ('accuracy',       'Accuracy',                 True),
    ]
    lcolors = plt.cm.tab10(np.linspace(0, 1, n_bs))

    for ax_idx, (metric, ylabel, higher_better) in enumerate(line_metrics):
        ax = axes2[ax_idx]
        for bi, bs in enumerate(bs_list):
            ys = []
            for to_ms in to_list:
                key = (bs, to_ms)
                val = valid.get(key, {})
                ys.append(val.get(metric, np.nan) if val else np.nan)
            ax.plot(to_list, ys, marker='o', label=f'bs={bs}',
                    color=lcolors[bi], linewidth=1.8)
        ax.set_xlabel('Timeout (ms)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{"↑" if higher_better else "↓"} {ylabel}')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(line_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"라인 플롯 저장: {line_path}")


# ── 최적 조합 요약 출력 ───────────────────────────────────────────────────────

def print_best_summary(grid_results):
    valid = {k: v for k, v in grid_results.items() if v is not None}
    if not valid:
        return

    print(f"\n{'='*60}")
    print(f"  Grid Search 최적 조합 요약")
    print(f"{'='*60}")

    criteria = [
        ('avg_ms',         '최소 Avg Latency',   False),
        ('p99_ms',         '최소 P99 Latency',   False),
        ('avg_throughput', '최대 Avg Throughput', True),
        ('accuracy',       '최대 Accuracy',       True),
        ('tail_ratio_p99_p50', '최소 Tail Ratio', False),
    ]

    for metric, desc, higher_better in criteria:
        best_key = None
        best_val = None
        for key, r in valid.items():
            val = r.get(metric)
            if val is None:
                continue
            if best_val is None or (higher_better and val > best_val) or \
                                   (not higher_better and val < best_val):
                best_key = key
                best_val = val
        if best_key:
            bs, to_ms = best_key
            fmt = f'{best_val:.4f}' if metric == 'accuracy' else f'{best_val:.2f}'
            print(f"  {desc:20s}: bs={bs:>3}, timeout={to_ms:>5.1f}ms  → {fmt}")

    print()


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Runtime Grid Search: batch_size × timeout_ms')
    parser.add_argument('--vee-seg1',    type=str, default=None)
    parser.add_argument('--plain',       type=str, default=None)
    parser.add_argument('--threshold',   type=float, default=0.80)
    parser.add_argument('--num-samples', type=int,   default=500,
                        help='평가 샘플 수 (grid 전체에 공유, 빠른 탐색을 위해 500 권장)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='탐색할 batch_size 후보 (예: 1 2 4 8 16)')
    parser.add_argument('--timeout-ms',  type=float, nargs='+', default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
                        help='탐색할 timeout 후보 (ms) (예: 5 10 20 40)')
    args = parser.parse_args()

    # 경로 자동 선택
    if args.vee_seg1 is None:
        args.vee_seg1 = paths.engine_path("vee_resnet18", "vee_seg1.engine")
    if args.plain    is None:
        args.plain    = paths.engine_path("plain_resnet18", "plain_resnet18.engine")

    # 엔진 로드
    print("\n=== TRT 엔진 로드 ===")
    if not os.path.exists(args.vee_seg1):
        print(f"[ERROR] VEE seg1 엔진 없음: {args.vee_seg1}")
        return
    if not os.path.exists(args.plain):
        print(f"[ERROR] Plain 엔진 없음: {args.plain}")
        return

    vee_seg1     = TRTEngine(args.vee_seg1)
    plain_engine = TRTEngine(args.plain)
    print(f"  VEE seg1  : {args.vee_seg1}")
    print(f"  Plain     : {args.plain}")

    # 데이터 로드
    print(f"\n테스트 데이터 로드 (n={args.num_samples})...")
    images, labels = load_test_data(args.num_samples)
    print(f"  로드 완료: {len(images)}개\n")

    # Grid Search
    grid_results = run_grid_search(
        vee_seg1, plain_engine, images, labels,
        batch_sizes=args.batch_sizes,
        timeout_ms_list=args.timeout_ms,
        threshold=args.threshold,
    )

    grid_results = run_grid_search(
        vee_seg1, plain_engine, images, labels,
        batch_sizes=args.batch_sizes,
        timeout_ms_list=args.timeout_ms,
        threshold=args.threshold,
    )

    # 최적 요약
    print_best_summary(grid_results)

    # 저장
    out_dir = paths.eval_dir("hybrid_grid")

    # JSON 저장
    json_data = {}
    for (bs, to_ms), r in grid_results.items():
        json_data[f"bs={bs}_to={to_ms}"] = {
            'batch_size': bs,
            'timeout_ms': to_ms,
            **(r if r is not None else {'error': 'failed'}),
        }
    json_path = os.path.join(out_dir, f"grid_search_thr{args.threshold:.2f}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON 저장: {json_path}")

    # 그래프 저장
    png_path = os.path.join(out_dir, f"grid_search_thr{args.threshold:.2f}.png")
    plot_grid_search(
        grid_results, args.batch_sizes, args.timeout_ms,
        args.threshold, png_path,
    )


if __name__ == '__main__':
    main()
