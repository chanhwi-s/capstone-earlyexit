"""
hybrid_vit_utils.py — ViT 하이브리드 런타임 공통 유틸리티

precompute_seg1 / precompute_seg / measure_seg_lut / bench_plain 등
2-exit / 3-exit 스크립트가 공유하는 핵심 함수 모음.

핵심 시뮬레이션 아이디어:
  1. 단일 패스로 필요한 모든 segment를 실행하되 feature는 저장하지 않음.
     conf / pred / seg1_time (스칼라)만 기록 → 메모리 ~1 MB (vs 피처 저장 시 ~30 GB).
  2. 각 segment의 batch latency를 batch_size별로 별도 측정 → LUT 구성.
  3. Grid search는 캐시된 스칼라 + LUT로 GPU 재실행 없이 순수 시뮬레이션.
  4. timeout 조건: 큐에서 가장 오래된 샘플의 대기 시간(누적 GPU 시간 기준)이
     timeout_ms 이상이면 flush (batch_size 미충족 시에도 강제 flush).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ── Segment 실행 ──────────────────────────────────────────────────────────────

def run_seg(model, feat: torch.Tensor, start_block: int, end_block: int, head_idx: int):
    """
    blocks[start_block .. end_block-1] + exit_heads[head_idx] 실행.
    feat: [B, 197, 768] (device). Returns (feat_out, logits).
    """
    with torch.no_grad():
        x = feat
        for bi in range(start_block, end_block):
            x = model.blocks[bi](x)
        logits = model.exit_heads[head_idx](x)
    return x, logits


# ── Seg1 전체 샘플 사전계산 ────────────────────────────────────────────────────

def precompute_seg1(model, loader: DataLoader, device,
                    seg1_end: int, warmup: int = 200) -> dict:
    """
    embedding + blocks[0..seg1_end) + exit_heads[0] 를 전체 val 샘플에 실행.

    seg1_end: exit_blocks[0] (예: 2-exit=8, 3-exit=6)

    Returns dict:
      feats      [N, 197, 768]  CPU tensor — seg1 출력 feature
      confs      [N]            float32    — max softmax (exit_head[0])
      preds      [N]            int32      — argmax (exit_head[0])
      labels     [N]            int32      — 정답 레이블
      seg1_times [N]            float64    — per-sample GPU 시간(ms)
    """
    model.eval()
    feats, confs, preds, labels, times = [], [], [], [], []

    with torch.no_grad():
        for i, (x, lbl) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            feat = model._embed(x)
            for bi in range(seg1_end):
                feat = model.blocks[bi](feat)
            logits = model.exit_heads[0](feat)
            e.record()
            torch.cuda.synchronize()

            if i >= warmup:
                times.append(s.elapsed_time(e))
                feats.append(feat.squeeze(0).cpu())   # [197, 768]
                confs.append(F.softmax(logits, dim=1).max(1).values.item())
                preds.append(logits.argmax(1).item())
                labels.append(lbl.item())

            if i >= warmup and (i - warmup + 1) % 5000 == 0:
                print(f"    {i - warmup + 1:>6} / {len(loader) - warmup}")

    return {
        'feats':      torch.stack(feats),
        'confs':      np.array(confs,  dtype=np.float32),
        'preds':      np.array(preds,  dtype=np.int32),
        'labels':     np.array(labels, dtype=np.int32),
        'seg1_times': np.array(times,  dtype=np.float64),
    }


def precompute_seg(model, device, feats_in: torch.Tensor,
                   start_block: int, end_block: int, head_idx: int,
                   chunk_size: int = 256) -> dict:
    """
    특정 segment를 사전계산된 feature에 대해 일괄 실행 (accuracy/exit 판단용).

    feats_in: [N, 197, 768] CPU tensor
    Returns dict: feats_out, confs, preds (모두 numpy 또는 CPU tensor)
    """
    model.eval()
    N = feats_in.shape[0]
    feats_out, confs_list, preds_list = [], [], []

    with torch.no_grad():
        for start in range(0, N, chunk_size):
            batch = feats_in[start:start + chunk_size].to(device)
            feat_out, logits = run_seg(model, batch, start_block, end_block, head_idx)
            torch.cuda.synchronize()
            feats_out.append(feat_out.cpu())
            confs_list.append(F.softmax(logits, dim=1).max(1).values.cpu().numpy())
            preds_list.append(logits.argmax(1).cpu().numpy())

    return {
        'feats': torch.cat(feats_out, dim=0),
        'confs': np.concatenate(confs_list).astype(np.float32),
        'preds': np.concatenate(preds_list).astype(np.int32),
    }


# ── Batch Latency LUT 측정 ────────────────────────────────────────────────────

def measure_seg_lut(model, device, start_block: int, end_block: int, head_idx: int,
                    batch_sizes: list, n_reps: int = 50) -> dict:
    """
    특정 segment의 batch_size별 GPU 레이턴시(ms)를 측정해 LUT 반환.
    dummy [B, 197, 768] 입력으로 순수 연산 시간만 측정.
    Returns {batch_size: median_latency_ms}.
    """
    model.eval()
    dummy = torch.randn(1, 197, 768, device=device)
    lut = {}

    with torch.no_grad():
        for bs in batch_sizes:
            x = dummy.expand(bs, -1, -1).contiguous()
            for _ in range(10):  # warmup
                run_seg(model, x, start_block, end_block, head_idx)
            torch.cuda.synchronize()

            lats = []
            for _ in range(n_reps):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run_seg(model, x, start_block, end_block, head_idx)
                e.record()
                torch.cuda.synchronize()
                lats.append(s.elapsed_time(e))

            lut[bs] = float(np.median(lats))
            print(f"    bs={bs:>3}: {lut[bs]:.3f} ms")

    return lut


def lut_lookup(lut: dict, bs: int) -> float:
    """batch_size에 해당하는 레이턴시 조회 (선형 보간 or nearest neighbor)."""
    if bs in lut:
        return lut[bs]
    keys = sorted(lut.keys())
    # Linear interpolation between two nearest keys
    for idx, k in enumerate(keys):
        if k >= bs:
            if idx == 0:
                return lut[k] * bs / k
            prev_k = keys[idx - 1]
            alpha = (bs - prev_k) / (k - prev_k)
            return lut[prev_k] + alpha * (lut[k] - lut[prev_k])
    # Extrapolate from largest key (linear scaling)
    k_max = keys[-1]
    return lut[k_max] * bs / k_max


# ── PlainViT 기준선 ───────────────────────────────────────────────────────────

def bench_plain(model, loader: DataLoader, device, warmup: int = 200):
    """PlainViT을 sample-by-sample로 실행. Returns (latencies_ms, correct)."""
    model.eval()
    latencies, correct = [], []
    with torch.no_grad():
        for i, (x, lbl) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            logits = model(x)
            e.record()
            torch.cuda.synchronize()
            if i >= warmup:
                latencies.append(s.elapsed_time(e))
                correct.append(int(logits.argmax(1).item() == lbl.item()))
    return latencies, correct


# ── 메모리 효율적 단일 패스 Precompute ───────────────────────────────────────
#
# 피처([B,197,768])를 RAM에 저장하지 않고 conf/pred/timing 스칼라만 기록.
# 50k 샘플 기준: 피처 저장 시 ~30 GB → 스칼라만 저장 시 ~1 MB.

def precompute_all_2exit(model, loader: DataLoader, device,
                          eb1: int, eb2: int, threshold: float,
                          warmup: int = 200) -> dict:
    """
    2-exit 모델의 모든 샘플을 단일 패스로 프로파일링.

    Seg1을 실행 → conf < threshold인 샘플만 즉시 Seg2 실행.
    피처는 저장하지 않고 conf/pred/timing 스칼라만 기록.

    Returns:
      labels     [N]       int32
      seg1_times [N]       float64  ms (Seg1 GPU 시간)
      confs_s1   [N]       float32  max softmax at exit_head[0]
      preds_s1   [N]       int32    argmax at exit_head[0]
      confs_s2   [M]       float32  (M = ne1 count)
      preds_s2   [M]       int32
    """
    model.eval()
    labels_all, seg1_times_all, confs_s1_all, preds_s1_all = [], [], [], []
    confs_s2_all, preds_s2_all = [], []

    with torch.no_grad():
        for i, (x, lbl) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            s.record()
            feat = model._embed(x)
            for bi in range(eb1):
                feat = model.blocks[bi](feat)
            logits_s1 = model.exit_heads[0](feat)
            e.record()
            torch.cuda.synchronize()

            if i < warmup:
                continue

            seg1_times_all.append(s.elapsed_time(e))
            labels_all.append(lbl.item())
            conf_s1 = F.softmax(logits_s1, dim=1).max(1).values.item()
            confs_s1_all.append(conf_s1)
            preds_s1_all.append(logits_s1.argmax(1).item())

            if conf_s1 < threshold:
                for bi in range(eb1, eb2):
                    feat = model.blocks[bi](feat)
                logits_s2 = model.exit_heads[1](feat)
                torch.cuda.synchronize()
                confs_s2_all.append(F.softmax(logits_s2, dim=1).max(1).values.item())
                preds_s2_all.append(logits_s2.argmax(1).item())

            n_done = i - warmup + 1
            if n_done % 5000 == 0:
                print(f"    {n_done:>6} / {len(loader) - warmup}")

    return {
        'labels':     np.array(labels_all,     dtype=np.int32),
        'seg1_times': np.array(seg1_times_all, dtype=np.float64),
        'confs_s1':   np.array(confs_s1_all,   dtype=np.float32),
        'preds_s1':   np.array(preds_s1_all,   dtype=np.int32),
        'confs_s2':   np.array(confs_s2_all,   dtype=np.float32),
        'preds_s2':   np.array(preds_s2_all,   dtype=np.int32),
    }


def precompute_all_3exit(model, loader: DataLoader, device,
                          eb1: int, eb2: int, eb3: int, threshold: float,
                          warmup: int = 200) -> dict:
    """
    3-exit 모델의 모든 샘플을 단일 패스로 프로파일링.

    Seg1 → conf < threshold → Seg2 → conf < threshold → Seg3 순으로 실행.
    피처는 저장하지 않고 스칼라만 기록.

    Returns:
      labels     [N]        int32
      seg1_times [N]        float64  ms (Seg1 GPU 시간)
      confs_s1   [N]        float32
      preds_s1   [N]        int32
      confs_s2   [M]        float32  (M = ne1 count, Seg1 비탈출)
      preds_s2   [M]        int32
      preds_s3   [K]        int32    (K = ne2 count, Seg2도 비탈출)
    """
    model.eval()
    labels_all, seg1_times_all, confs_s1_all, preds_s1_all = [], [], [], []
    confs_s2_all, preds_s2_all = [], []
    preds_s3_all = []

    with torch.no_grad():
        for i, (x, lbl) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)

            s.record()
            feat = model._embed(x)
            for bi in range(eb1):
                feat = model.blocks[bi](feat)
            logits_s1 = model.exit_heads[0](feat)
            e.record()
            torch.cuda.synchronize()

            if i < warmup:
                continue

            seg1_times_all.append(s.elapsed_time(e))
            labels_all.append(lbl.item())
            conf_s1 = F.softmax(logits_s1, dim=1).max(1).values.item()
            confs_s1_all.append(conf_s1)
            preds_s1_all.append(logits_s1.argmax(1).item())

            if conf_s1 < threshold:
                for bi in range(eb1, eb2):
                    feat = model.blocks[bi](feat)
                logits_s2 = model.exit_heads[1](feat)
                torch.cuda.synchronize()
                conf_s2 = F.softmax(logits_s2, dim=1).max(1).values.item()
                confs_s2_all.append(conf_s2)
                preds_s2_all.append(logits_s2.argmax(1).item())

                if conf_s2 < threshold:
                    for bi in range(eb2, eb3):
                        feat = model.blocks[bi](feat)
                    logits_s3 = model.exit_heads[2](feat)
                    torch.cuda.synchronize()
                    preds_s3_all.append(logits_s3.argmax(1).item())

            n_done = i - warmup + 1
            if n_done % 5000 == 0:
                print(f"    {n_done:>6} / {len(loader) - warmup}")

    return {
        'labels':     np.array(labels_all,     dtype=np.int32),
        'seg1_times': np.array(seg1_times_all, dtype=np.float64),
        'confs_s1':   np.array(confs_s1_all,   dtype=np.float32),
        'preds_s1':   np.array(preds_s1_all,   dtype=np.int32),
        'confs_s2':   np.array(confs_s2_all,   dtype=np.float32),
        'preds_s2':   np.array(preds_s2_all,   dtype=np.int32),
        'preds_s3':   np.array(preds_s3_all,   dtype=np.int32),
    }


# ── 통계 ─────────────────────────────────────────────────────────────────────

def lat_stats(response_times) -> dict:
    lat = np.array(response_times)
    return {
        'avg_ms': float(np.mean(lat)),
        'p50_ms': float(np.percentile(lat, 50)),
        'p90_ms': float(np.percentile(lat, 90)),
        'p95_ms': float(np.percentile(lat, 95)),
        'p99_ms': float(np.percentile(lat, 99)),
        'std_ms': float(np.std(lat)),
    }


def throughput_stats(throughputs) -> dict:
    """
    per-sample 유효 처리량(samples/ms) 통계.

    per-sample throughput = B / response_time
      - early exit (bs=1):     1 / response_time  (B=1)
      - batched seg (bs=B):    B / response_time  (실제 flush된 배치 크기)
    PlainViT baseline는 항상 B=1 → 1/lat 와 동일 기준으로 비교 가능.
    배치가 클수록 latency 악화에도 throughput이 올라갈 수 있음 (batching의 핵심).
    overall_tput(N/총시뮬시간)은 시스템 레벨 지표로 별도 반환.
    """
    tp = np.array(throughputs, dtype=np.float64)
    return {
        'avg_tput': float(np.mean(tp)),
        'p50_tput': float(np.percentile(tp, 50)),
        'p90_tput': float(np.percentile(tp, 90)),
        'p95_tput': float(np.percentile(tp, 95)),
        'p99_tput': float(np.percentile(tp, 99)),
        'std_tput': float(np.std(tp)),
    }
