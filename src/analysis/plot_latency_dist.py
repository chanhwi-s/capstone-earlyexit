"""
plot_latency_dist.py — hybrid 2-exit 벤치마크 결과에서 latency distribution 시각화

bimodal 분포 (exit1=빠른 경로 / exit2=느린 경로) vs plain unimodal 대비를 강조.

사용법:
    python analysis/plot_latency_dist.py --result-dir <eval_dir>
    python analysis/plot_latency_dist.py --result-dir <eval_dir> --bs2 8 32
    python analysis/plot_latency_dist.py --result-dir <eval_dir> --no-plain

입력 파일 (--result-dir 내부):
    records_bs{N}.json  — hybrid per-sample 기록 (exit1/exit2 분리됨)
    plain_records.json  — plain baseline per-sample 기록

출력 (--result-dir 내부):
    latency_dist_overview.png   — bs2별 exit1/exit2/plain KDE 오버레이 서브플롯 그리드
    latency_dist_combined.png   — 모든 bs2 + plain을 한 축에 (대표 그래프)
    latency_dist_bs{N}.png      — bs2별 개별 상세 그래프 (--per-bs2 옵션)
"""

import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ── KDE 유틸 ──────────────────────────────────────────────────────────────────

def kde_curve(data: list, x_grid: np.ndarray) -> np.ndarray:
    if len(data) < 2:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(data, bw_method='scott')
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_hybrid(result_dir: str):
    """records_bs{N}.json → dict{bs2: {'exit1': [rt_ms], 'exit2': [rt_ms], 'all': [rt_ms]}}"""
    data = {}
    for path in sorted(glob.glob(os.path.join(result_dir, 'records_bs*.json'))):
        with open(path) as f:
            obj = json.load(f)
        bs2     = obj['bs2']
        records = obj['records']
        exit1   = [r['rt_ms'] for r in records if r['exit'] == 1]
        exit2   = [r['rt_ms'] for r in records if r['exit'] == 2]
        all_rt  = [r['rt_ms'] for r in records]
        data[bs2] = {'exit1': exit1, 'exit2': exit2, 'all': all_rt,
                     'n': len(records), 'n_exit1': len(exit1), 'n_exit2': len(exit2)}
        print(f'  loaded bs2={bs2:4d}: {len(records)} samples '
              f'(exit1={len(exit1)}, exit2={len(exit2)})')
    return data


def load_plain(result_dir: str):
    path = os.path.join(result_dir, 'plain_records.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        obj = json.load(f)
    rts = [r['rt_ms'] for r in obj['records']]
    print(f'  loaded plain: {len(rts)} samples  avg={np.mean(rts):.2f}ms')
    return rts


# ── 플롯 1: 서브플롯 그리드 ────────────────────────────────────────────────────

def plot_overview(hybrid_data: dict, plain_rts, bs2_list, out_path,
                  threshold=None, device_label=''):
    """
    bs2별 서브플롯. 각 서브플롯에 exit1 KDE(파랑) + exit2 KDE(주황) + plain KDE(검정 점선).
    bimodal vs unimodal 대비를 bs2별로 한눈에 확인.
    """
    n = len(bs2_list)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    title = '2-exit Hybrid Latency Distribution — bimodal (exit1 vs exit2)'
    if threshold is not None:
        title += f'  thr={threshold:.2f}'
    if device_label:
        title += f'  {device_label}'
    fig.suptitle(title, fontsize=13, y=1.01)

    all_rts_flat = []
    for bs2 in bs2_list:
        all_rts_flat.extend(hybrid_data[bs2]['all'])
    if plain_rts:
        all_rts_flat.extend(plain_rts)
    x_max = np.percentile(all_rts_flat, 99) * 1.1
    x_grid = np.linspace(0, x_max, 500)

    plain_kde = kde_curve(plain_rts, x_grid) if plain_rts else None

    for idx, bs2 in enumerate(bs2_list):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        d  = hybrid_data[bs2]

        kde_e1  = kde_curve(d['exit1'], x_grid)
        kde_e2  = kde_curve(d['exit2'], x_grid)
        kde_all = kde_curve(d['all'],   x_grid)

        r1 = d['n_exit1'] / d['n'] * 100
        r2 = d['n_exit2'] / d['n'] * 100

        ax.fill_between(x_grid, kde_e1,  alpha=0.25, color='#4c72b0')
        ax.fill_between(x_grid, kde_e2,  alpha=0.25, color='#dd8452')
        ax.plot(x_grid, kde_e1,  color='#4c72b0', linewidth=1.8,
                label=f'exit1 ({r1:.0f}%, fast path)')
        ax.plot(x_grid, kde_e2,  color='#dd8452', linewidth=1.8,
                label=f'exit2 ({r2:.0f}%, seg2 path)')
        ax.plot(x_grid, kde_all, color='gray',    linewidth=1.2,
                linestyle='-.', label='all samples')
        if plain_kde is not None:
            ax.plot(x_grid, plain_kde, 'k--', linewidth=1.8,
                    label=f'Plain (bs={d["n"] // max(1, d["n"] // 32)})')

        avg_e1 = np.mean(d['exit1']) if d['exit1'] else 0
        avg_e2 = np.mean(d['exit2']) if d['exit2'] else 0
        ax.set_title(f'bs2={bs2}  |  avg exit1={avg_e1:.1f}ms  exit2={avg_e2:.1f}ms',
                     fontsize=10)
        ax.set_xlabel('Response Time (ms)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, x_max)

    # 빈 서브플롯 숨기기
    for idx in range(len(bs2_list), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


# ── 플롯 2: 단일 축 combined ───────────────────────────────────────────────────

def plot_combined(hybrid_data: dict, plain_rts, bs2_list, out_path,
                  threshold=None, device_label=''):
    """
    모든 bs2의 전체 latency KDE를 한 축에 오버레이.
    plain을 검정 점선으로 표시. bimodal 구조가 가장 선명하게 보이는 대표 그래프.
    """
    tab10  = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_rts_flat = []
    for bs2 in bs2_list:
        all_rts_flat.extend(hybrid_data[bs2]['all'])
    if plain_rts:
        all_rts_flat.extend(plain_rts)
    x_max  = np.percentile(all_rts_flat, 99) * 1.1
    x_grid = np.linspace(0, x_max, 600)

    title = '2-exit Hybrid vs Plain — Latency Distribution'
    if threshold is not None:
        title += f'  (thr={threshold:.2f})'
    if device_label:
        title += f'  {device_label}'
    fig.suptitle(title, fontsize=12)

    # ── 좌: bs2별 전체 KDE + plain ──────────────────────────────────────────────
    ax = axes[0]
    for i, bs2 in enumerate(bs2_list):
        d   = hybrid_data[bs2]
        kde = kde_curve(d['all'], x_grid)
        ax.plot(x_grid, kde, color=tab10(i), linewidth=2,
                label=f'Hybrid bs2={bs2}')
        ax.fill_between(x_grid, kde, alpha=0.08, color=tab10(i))
    if plain_rts:
        kde_p = kde_curve(plain_rts, x_grid)
        ax.plot(x_grid, kde_p, 'k--', linewidth=2.5, label='PlainViT')
        ax.fill_between(x_grid, kde_p, alpha=0.08, color='black')
    ax.set_xlabel('Response Time (ms)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('All Samples (exit1 + exit2)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, x_max)

    # ── 우: 대표 bs2 하나 골라서 exit1/exit2 분리 KDE (bimodal 강조) ────────────
    ax   = axes[1]
    rep  = bs2_list[len(bs2_list) // 2]   # 중간 bs2를 대표로 선택
    d    = hybrid_data[rep]
    r1   = d['n_exit1'] / d['n'] * 100
    r2   = d['n_exit2'] / d['n'] * 100

    kde_e1  = kde_curve(d['exit1'], x_grid)
    kde_e2  = kde_curve(d['exit2'], x_grid)
    kde_all = kde_curve(d['all'],   x_grid)

    ax.fill_between(x_grid, kde_e1,  alpha=0.3,  color='#4c72b0')
    ax.fill_between(x_grid, kde_e2,  alpha=0.3,  color='#dd8452')
    ax.plot(x_grid, kde_e1,  color='#4c72b0', linewidth=2,
            label=f'exit1 — fast path ({r1:.0f}%)')
    ax.plot(x_grid, kde_e2,  color='#dd8452', linewidth=2,
            label=f'exit2 — seg2 path ({r2:.0f}%)')
    ax.plot(x_grid, kde_all, color='gray',    linewidth=1.5,
            linestyle='-.', label='combined (bimodal)')
    if plain_rts:
        kde_p = kde_curve(plain_rts, x_grid)
        ax.plot(x_grid, kde_p, 'k--', linewidth=2.5, label='PlainViT (unimodal)')
        ax.fill_between(x_grid, kde_p, alpha=0.08, color='black')

    avg_e1 = np.mean(d['exit1']) if d['exit1'] else 0
    avg_e2 = np.mean(d['exit2']) if d['exit2'] else 0
    ax.set_title(f'Bimodal Breakdown — bs2={rep}  '
                 f'(exit1 avg={avg_e1:.1f}ms, exit2 avg={avg_e2:.1f}ms)', fontsize=11)
    ax.set_xlabel('Response Time (ms)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim(0, x_max)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


# ── 플롯 3: bs2별 개별 상세 (옵션) ────────────────────────────────────────────

def plot_per_bs2(hybrid_data: dict, plain_rts, bs2: int, out_path,
                 threshold=None, device_label=''):
    d = hybrid_data[bs2]
    r1 = d['n_exit1'] / d['n'] * 100
    r2 = d['n_exit2'] / d['n'] * 100

    all_rts = d['all'] + (plain_rts or [])
    x_max   = np.percentile(all_rts, 99) * 1.1
    x_grid  = np.linspace(0, x_max, 600)

    fig, ax = plt.subplots(figsize=(9, 5))
    title = f'Latency Distribution — bs2={bs2}'
    if threshold is not None:
        title += f'  thr={threshold:.2f}'
    if device_label:
        title += f'  {device_label}'
    ax.set_title(title, fontsize=12)

    kde_e1  = kde_curve(d['exit1'], x_grid)
    kde_e2  = kde_curve(d['exit2'], x_grid)
    kde_all = kde_curve(d['all'],   x_grid)

    ax.fill_between(x_grid, kde_e1,  alpha=0.3,  color='#4c72b0')
    ax.fill_between(x_grid, kde_e2,  alpha=0.3,  color='#dd8452')
    ax.plot(x_grid, kde_e1,  color='#4c72b0', linewidth=2,
            label=f'exit1 — fast path ({r1:.0f}%, avg={np.mean(d["exit1"]):.1f}ms)' if d['exit1'] else 'exit1 (none)')
    ax.plot(x_grid, kde_e2,  color='#dd8452', linewidth=2,
            label=f'exit2 — seg2 path ({r2:.0f}%, avg={np.mean(d["exit2"]):.1f}ms)' if d['exit2'] else 'exit2 (none)')
    ax.plot(x_grid, kde_all, color='gray',    linewidth=1.5,
            linestyle='-.', label='combined (bimodal)')
    if plain_rts:
        kde_p = kde_curve(plain_rts, x_grid)
        ax.plot(x_grid, kde_p, 'k--', linewidth=2.5,
                label=f'PlainViT (unimodal, avg={np.mean(plain_rts):.1f}ms)')
        ax.fill_between(x_grid, kde_p, alpha=0.08, color='black')

    ax.set_xlabel('Response Time (ms)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, x_max)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  plot: {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='hybrid 2-exit latency distribution (bimodal) 시각화'
    )
    parser.add_argument('--result-dir', type=str, required=True,
                        help='records_bs*.json / plain_records.json 가 있는 디렉토리')
    parser.add_argument('--bs2', type=int, nargs='+', default=None,
                        help='그릴 bs2 목록. 미지정 시 자동 탐색.')
    parser.add_argument('--no-plain', action='store_true',
                        help='plain_records.json 을 로드하지 않음')
    parser.add_argument('--per-bs2', action='store_true',
                        help='bs2별 개별 상세 그래프도 저장')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--device-label', type=str, default='')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='출력 디렉토리 (기본: --result-dir 와 동일)')
    args = parser.parse_args()

    result_dir = args.result_dir
    out_dir    = args.out_dir or result_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n결과 디렉토리: {result_dir}')
    print('hybrid 기록 로드 ...')
    hybrid_data = load_hybrid(result_dir)
    if not hybrid_data:
        print('[ERROR] records_bs*.json 파일을 찾을 수 없습니다.'); return

    plain_rts = None
    if not args.no_plain:
        print('plain 기록 로드 ...')
        plain_rts = load_plain(result_dir)
        if plain_rts is None:
            print('  plain_records.json 없음 — plain 생략')

    bs2_list = sorted(args.bs2 or hybrid_data.keys())
    bs2_list = [b for b in bs2_list if b in hybrid_data]
    if not bs2_list:
        print('[ERROR] 유효한 bs2 데이터가 없습니다.'); return

    print(f'\nbs2 목록: {bs2_list}')
    print(f'출력 디렉토리: {out_dir}\n')

    # 서브플롯 그리드
    plot_overview(hybrid_data, plain_rts, bs2_list,
                  os.path.join(out_dir, 'latency_dist_overview.png'),
                  args.threshold, args.device_label)

    # 단일 combined
    plot_combined(hybrid_data, plain_rts, bs2_list,
                  os.path.join(out_dir, 'latency_dist_combined.png'),
                  args.threshold, args.device_label)

    # 개별 bs2 (옵션)
    if args.per_bs2:
        for bs2 in bs2_list:
            plot_per_bs2(hybrid_data, plain_rts, bs2,
                         os.path.join(out_dir, f'latency_dist_bs{bs2}.png'),
                         args.threshold, args.device_label)

    print('\nDone!')


if __name__ == '__main__':
    main()
