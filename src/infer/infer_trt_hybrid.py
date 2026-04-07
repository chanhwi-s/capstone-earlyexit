"""
Hybrid TRT Runtime: VEE first-exit + Batched Plain Fallback

설계 개요:
  1. VEE seg1 엔진으로 first exit 시도 (batch=1)
  2. confidence < threshold → fallback queue에 추가
  3. fallback queue가 batch_size에 도달하거나 timeout이 경과하면
     plain_resnet 전체 엔진으로 batched inference 실행
  4. batch가 부족하면 zero-padding 후 유효 결과만 반환

최적화 포인트:
  - first exit 통과 샘플: 매우 빠른 single-segment latency
  - fallback 샘플: plain_resnet batched → amortized latency
  - 3-segment orchestration의 memory copy / alloc overhead 제거

사용법 (Orin):
  # 단일 설정 실행
  python infer_trt_hybrid.py \
      --vee-seg1 ../experiments/trt_engines/vee_resnet18/vee_seg1.engine \
      --plain    ../experiments/trt_engines/plain_resnet18/plain_resnet18.engine \
      --threshold 0.80 --batch-size 8 --timeout-ms 10

  # Grid search (batch_size × timeout 조합 전체 탐색)
  python infer_trt_hybrid.py \
      --vee-seg1 ../experiments/trt_engines/vee_resnet18/vee_seg1.engine \
      --plain    ../experiments/trt_engines/plain_resnet18/plain_resnet18.engine \
      --grid-search --threshold 0.80 --num-samples 1000
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from profiling_utils import compute_latency_stats, print_latency_report


# ── TRT Engine (기존 infer_trt.py 것과 동일 구조) ───────────────────────────

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

        print(f"[TRT] 로드: {os.path.basename(engine_path)}")
        print(f"      입력: {self.input_names}  출력: {self.output_names}")

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


# ── Hybrid Orchestrator ──────────────────────────────────────────────────────

class HybridOrchestrator:
    """
    VEE seg1 (first exit) + Plain ResNet (batched fallback) 하이브리드 런타임.

    동작 방식 (시뮬레이션):
      - 각 sample에 대해 seg1 실행 → confidence 확인
      - threshold 이상: exit1에서 즉시 반환
      - threshold 미만: fallback_queue에 추가
      - queue가 batch_size에 도달 OR timeout_ms 경과 시:
        zero-padding으로 batch 맞춰서 plain_resnet에 전달
        유효 sample만 결과 추출
    """

    def __init__(self, vee_seg1: TRTEngine, plain_engine: TRTEngine,
                 batch_size: int = 8, timeout_ms: float = 10.0):
        self.vee_seg1     = vee_seg1
        self.plain_engine = plain_engine
        self.batch_size   = batch_size
        self.timeout_ms   = timeout_ms

    def run_stream(self, images_list: list, labels_list: list,
                   threshold: float = 0.80):
        """
        스트리밍 추론 시뮬레이션.

        Args:
            images_list: [(1, 3, H, W) tensor, ...] — 순차 도착하는 이미지들
            labels_list: [int, ...] — 정답 레이블 (정확도 계산용)
            threshold:   confidence threshold

        Returns:
            dict: {
                'results':           list of per-sample dicts,
                'exit1_count':       int,
                'fallback_count':    int,
                'fallback_batches':  int,
                'latencies_ms':      list of float (per-sample end-to-end)
            }
        """
        n = len(images_list)
        results       = [None] * n
        latencies     = [0.0]  * n
        exit1_count   = 0
        fallback_count = 0
        fallback_batches = 0

        # fallback queue: (sample_idx, image_tensor)
        fallback_queue = []
        queue_start_time = None

        def _flush_fallback():
            """fallback queue를 batched plain inference로 처리."""
            nonlocal fallback_batches
            if not fallback_queue:
                return

            fallback_batches += 1
            valid_count = len(fallback_queue)

            # batch 구성: 유효 image + zero padding
            batch_images = []
            batch_indices = []
            for idx, img in fallback_queue:
                batch_images.append(img)
                batch_indices.append(idx)

            # zero-padding으로 batch_size 맞추기
            if valid_count < self.batch_size:
                pad_img = torch.zeros_like(batch_images[0])
                for _ in range(self.batch_size - valid_count):
                    batch_images.append(pad_img)

            batch_tensor = torch.cat(batch_images, dim=0)  # (B, 3, H, W)

            # plain_resnet batched inference
            t_fb_start = time.perf_counter()
            out = self.plain_engine.infer(batch_tensor)
            t_fb_end = time.perf_counter()

            fb_latency = (t_fb_end - t_fb_start) * 1000  # ms

            # 유효 sample만 결과 추출
            logits_key = self.plain_engine.output_names[0]
            all_logits = out[logits_key]  # (B, num_classes)

            for i, sample_idx in enumerate(batch_indices):
                logits_i = all_logits[i:i+1]
                conf_i   = F.softmax(logits_i, dim=1).max().item()
                pred_i   = logits_i.argmax(dim=1).item()

                results[sample_idx] = {
                    'pred':       pred_i,
                    'conf':       conf_i,
                    'exit':       'fallback',
                    'fb_batch':   valid_count,
                }
                # per-sample latency = seg1 시간 + 대기 시간 + fallback batch 시간 / 유효개수
                # (실제로는 arrival → result 시간이 중요하므로 대기 포함)
                latencies[sample_idx] += fb_latency

            fallback_queue.clear()

        # ── 스트리밍 루프 ──
        for i in range(n):
            img   = images_list[i]
            label = labels_list[i]

            t_start = time.perf_counter()

            # seg1 실행
            out1 = self.vee_seg1.infer(img)
            ee1_key = [k for k in out1 if 'ee1' in k.lower() or 'logit' in k.lower()]
            if ee1_key:
                ee1_logits = out1[ee1_key[0]]
            else:
                ee1_logits = list(out1.values())[-1]

            conf = F.softmax(ee1_logits, dim=1).max().item()

            t_seg1 = time.perf_counter()
            seg1_ms = (t_seg1 - t_start) * 1000

            if conf >= threshold:
                # ── Exit 1: 즉시 반환 ──
                pred = ee1_logits.argmax(dim=1).item()
                results[i] = {
                    'pred':  pred,
                    'conf':  conf,
                    'exit':  'exit1',
                }
                latencies[i] = seg1_ms
                exit1_count += 1
            else:
                # ── Fallback queue에 추가 ──
                latencies[i] = seg1_ms  # seg1 시간은 이미 소비

                if not fallback_queue:
                    queue_start_time = time.perf_counter()

                fallback_queue.append((i, img))
                fallback_count += 1

                # flush 조건: batch 충족 또는 timeout 경과
                elapsed = (time.perf_counter() - queue_start_time) * 1000
                if (len(fallback_queue) >= self.batch_size or
                        elapsed >= self.timeout_ms):
                    _flush_fallback()
                    queue_start_time = None

        # 잔여 queue flush
        _flush_fallback()

        return {
            'results':          results,
            'latencies_ms':     latencies,
            'exit1_count':      exit1_count,
            'fallback_count':   fallback_count,
            'fallback_batches': fallback_batches,
        }


# ── Hybrid VEE Orchestrator ──────────────────────────────────────────────────

class HybridVEEOrchestrator:
    """
    VEE seg1 (first exit) + VEE seg2 (batched fallback) 하이브리드 런타임.

    HybridOrchestrator와의 차이:
      - fallback 시 원본 이미지를 plain에 재투입하지 않고,
        seg1에서 이미 계산된 feat_layer1을 큐에 저장해뒀다가
        vee_seg2에 배치로 투입 → 중복 연산 없음.

    동작 방식:
      - vee_seg1(img) → feat_layer1, ee1_logits
      - conf >= threshold  → exit1 즉시 반환
      - conf <  threshold  → fallback_queue에 (idx, feat_layer1) 추가
      - queue가 batch_size 도달 OR timeout 경과 →
        feat들을 concatenate → vee_seg2(batch_feats) → 유효 결과 추출
    """

    def __init__(self, vee_seg1: TRTEngine, vee_seg2: TRTEngine,
                 batch_size: int = 8, timeout_ms: float = 10.0):
        self.vee_seg1   = vee_seg1
        self.vee_seg2   = vee_seg2
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms

    def run_stream(self, images_list: list, labels_list: list,
                   threshold: float = 0.80):
        """
        Args / Returns: HybridOrchestrator.run_stream과 동일한 인터페이스.
        """
        n = len(images_list)
        results          = [None] * n
        latencies        = [0.0]  * n
        exit1_count      = 0
        fallback_count   = 0
        fallback_batches = 0

        # fallback queue: (sample_idx, feat_layer1 tensor)
        fallback_queue   = []
        queue_start_time = None

        def _flush_fallback():
            nonlocal fallback_batches
            if not fallback_queue:
                return

            fallback_batches += 1
            valid_count  = len(fallback_queue)
            batch_feats  = []
            batch_indices = []

            for idx, feat in fallback_queue:
                batch_feats.append(feat)
                batch_indices.append(idx)

            # zero-padding으로 batch_size 맞추기
            if valid_count < self.batch_size:
                pad = torch.zeros_like(batch_feats[0])
                for _ in range(self.batch_size - valid_count):
                    batch_feats.append(pad)

            batch_tensor = torch.cat(batch_feats, dim=0)  # (B, C, H', W')

            # vee_seg2 배치 추론
            t_fb_start = time.perf_counter()
            out = self.vee_seg2.infer({'feat_layer1': batch_tensor})
            t_fb_end   = time.perf_counter()
            fb_latency = (t_fb_end - t_fb_start) * 1000

            logits_key = self.vee_seg2.output_names[0]
            all_logits = out[logits_key]  # (B, num_classes)

            for i, sample_idx in enumerate(batch_indices):
                logits_i = all_logits[i:i+1]
                conf_i   = F.softmax(logits_i, dim=1).max().item()
                pred_i   = logits_i.argmax(dim=1).item()
                results[sample_idx] = {
                    'pred':     pred_i,
                    'conf':     conf_i,
                    'exit':     'fallback_vee',
                    'fb_batch': valid_count,
                }
                latencies[sample_idx] += fb_latency

            fallback_queue.clear()

        # ── 스트리밍 루프 ──
        for i in range(n):
            img = images_list[i]

            t_start = time.perf_counter()
            out1    = self.vee_seg1.infer(img)

            # feat_layer1, ee1_logits 추출
            feat_key = [k for k in out1 if 'feat' in k.lower()]
            ee1_key  = [k for k in out1 if 'ee1'  in k.lower() or 'logit' in k.lower()]
            feat      = out1[feat_key[0]]  if feat_key else list(out1.values())[0]
            ee1_logits = out1[ee1_key[0]] if ee1_key  else list(out1.values())[-1]

            conf    = F.softmax(ee1_logits, dim=1).max().item()
            t_seg1  = time.perf_counter()
            seg1_ms = (t_seg1 - t_start) * 1000

            if conf >= threshold:
                pred = ee1_logits.argmax(dim=1).item()
                results[i] = {'pred': pred, 'conf': conf, 'exit': 'exit1'}
                latencies[i] = seg1_ms
                exit1_count += 1
            else:
                latencies[i] = seg1_ms  # seg1 시간은 이미 소비

                if not fallback_queue:
                    queue_start_time = time.perf_counter()

                fallback_queue.append((i, feat))
                fallback_count += 1

                elapsed = (time.perf_counter() - queue_start_time) * 1000
                if (len(fallback_queue) >= self.batch_size or
                        elapsed >= self.timeout_ms):
                    _flush_fallback()
                    queue_start_time = None

        _flush_fallback()

        return {
            'results':          results,
            'latencies_ms':     latencies,
            'exit1_count':      exit1_count,
            'fallback_count':   fallback_count,
            'fallback_batches': fallback_batches,
        }


# ── CIFAR-10 평가 ─────────────────────────────────────────────────────────────

def eval_cifar10_hybrid(orchestrator, threshold, num_samples):
    """CIFAR-10 테스트셋으로 하이브리드 런타임 평가."""
    from datasets.dataloader import get_dataloader
    from utils import load_config

    cfg = load_config('configs/train.yaml')
    _, test_loader, _ = get_dataloader(
        dataset=cfg['dataset']['name'],
        batch_size=1,
        data_root=cfg['dataset']['data_root'],
        num_workers=0,
        seed=cfg['train']['seed'],
    )

    images_list = []
    labels_list = []
    for i, (img, lbl) in enumerate(test_loader):
        if i >= num_samples:
            break
        images_list.append(img)
        labels_list.append(lbl[0].item())

    actual_n = len(images_list)
    print(f"  샘플 수: {actual_n}")

    run = orchestrator.run_stream(images_list, labels_list, threshold)

    # 정확도 계산
    correct = sum(
        1 for i in range(actual_n)
        if run['results'][i] is not None and
        run['results'][i]['pred'] == labels_list[i]
    )
    accuracy = correct / actual_n

    stats = compute_latency_stats(run['latencies_ms'])
    exit1_rate    = run['exit1_count']    / actual_n * 100
    fallback_rate = run['fallback_count'] / actual_n * 100

    return {
        'accuracy':          accuracy,
        'exit1_rate':        exit1_rate,
        'fallback_rate':     fallback_rate,
        'exit1_count':       run['exit1_count'],
        'fallback_count':    run['fallback_count'],
        'fallback_batches':  run['fallback_batches'],
        'latency_stats':     stats,
        'n':                 actual_n,
    }


# ── Grid Search ───────────────────────────────────────────────────────────────

def grid_search(vee_seg1, plain_engine, threshold, num_samples,
                batch_sizes=None, timeouts_ms=None):
    """batch_size × timeout_ms 조합에 대해 grid search 수행."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    if timeouts_ms is None:
        timeouts_ms = [2, 5, 10, 20, 50]

    all_results = []

    print(f"\n{'='*100}")
    print(f"Grid Search: batch_size × timeout_ms  (threshold={threshold}, n={num_samples})")
    print(f"{'='*100}")

    header = (
        f"{'BS':>4} {'TOut':>6} {'Acc':>7} {'Exit1%':>7} {'FB%':>7} "
        f"{'avg_ms':>8} {'p50_ms':>8} {'p90_ms':>8} {'p95_ms':>8} {'p99_ms':>8} "
        f"{'GP90':>8} {'Tail':>6} {'FBbat':>6}"
    )
    print(header)
    print("-" * len(header))

    for bs in batch_sizes:
        for to_ms in timeouts_ms:
            orch = HybridOrchestrator(vee_seg1, plain_engine,
                                      batch_size=bs, timeout_ms=to_ms)
            r = eval_cifar10_hybrid(orch, threshold, num_samples)
            s = r['latency_stats']

            row = {
                'batch_size':        bs,
                'timeout_ms':        to_ms,
                'accuracy':          r['accuracy'],
                'exit1_rate':        r['exit1_rate'],
                'fallback_rate':     r['fallback_rate'],
                'fallback_batches':  r['fallback_batches'],
                **s,
            }
            all_results.append(row)

            print(
                f"{bs:4d} {to_ms:6.1f} {r['accuracy']:7.4f} "
                f"{r['exit1_rate']:6.1f}% {r['fallback_rate']:6.1f}% "
                f"{s['avg_ms']:8.3f} {s['p50_ms']:8.3f} "
                f"{s['p90_ms']:8.3f} {s['p95_ms']:8.3f} {s['p99_ms']:8.3f} "
                f"{s['p90_goodput']:8.1f} {s['tail_ratio_p99_p50']:5.2f}x "
                f"{r['fallback_batches']:6d}"
            )

    # 최적 조합 찾기 (P90 goodput 기준)
    best = max(all_results, key=lambda x: x['p90_goodput'])
    print(f"\n★ 최적 (P90 Goodput 기준): BS={best['batch_size']}, "
          f"Timeout={best['timeout_ms']}ms → "
          f"P90 Goodput={best['p90_goodput']:.1f} inf/s, "
          f"Acc={best['accuracy']:.4f}")

    # P99 기준도 출력
    best_p99 = min(all_results, key=lambda x: x['p99_ms'])
    print(f"★ 최적 (P99 Latency 기준): BS={best_p99['batch_size']}, "
          f"Timeout={best_p99['timeout_ms']}ms → "
          f"P99={best_p99['p99_ms']:.3f}ms, "
          f"Acc={best_p99['accuracy']:.4f}")

    return all_results


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vee-seg1', type=str, default=None,
                        help='VEE seg1 engine 경로')
    parser.add_argument('--plain',    type=str, default=None,
                        help='Plain ResNet-18 engine 경로')
    parser.add_argument('--threshold',   type=float, default=0.80)
    parser.add_argument('--batch-size',  type=int,   default=8)
    parser.add_argument('--timeout-ms',  type=float, default=10.0)
    parser.add_argument('--num-samples', type=int,   default=1000)
    parser.add_argument('--grid-search', action='store_true',
                        help='batch_size × timeout grid search 실행')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32],
                        help='grid search용 batch_size 목록')
    parser.add_argument('--timeouts',    type=float, nargs='+',
                        default=[2, 5, 10, 20, 50],
                        help='grid search용 timeout_ms 목록')
    args = parser.parse_args()

    # ── 엔진 경로 자동 선택 ──
    if args.vee_seg1 is None:
        args.vee_seg1 = paths.engine_path("vee_resnet18", "vee_seg1.engine")
    if args.plain is None:
        args.plain = paths.engine_path("plain_resnet18", "plain_resnet18.engine")

    print("\n=== TRT 엔진 로드 ===")
    vee_seg1     = TRTEngine(args.vee_seg1)
    plain_engine = TRTEngine(args.plain)
    print()

    out_dir = paths.eval_dir("hybrid_runtime")

    if args.grid_search:
        # ── Grid Search ──
        results = grid_search(
            vee_seg1, plain_engine,
            threshold=args.threshold,
            num_samples=args.num_samples,
            batch_sizes=args.batch_sizes,
            timeouts_ms=args.timeouts,
        )

        # JSON 저장
        save_path = os.path.join(out_dir,
            f"grid_search_thr{args.threshold:.2f}.json")
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n결과 저장: {save_path}")

    else:
        # ── 단일 설정 실행 ──
        print(f"=== Hybrid Runtime 평가 ===")
        print(f"  threshold={args.threshold}, batch_size={args.batch_size}, "
              f"timeout={args.timeout_ms}ms, n={args.num_samples}")

        orch = HybridOrchestrator(
            vee_seg1, plain_engine,
            batch_size=args.batch_size,
            timeout_ms=args.timeout_ms,
        )
        r = eval_cifar10_hybrid(orch, args.threshold, args.num_samples)

        print(f"\n  정확도      : {r['accuracy']:.4f}  ({r['accuracy']*100:.2f}%)")
        print(f"  Exit1 비율  : {r['exit1_rate']:.1f}%  ({r['exit1_count']}개)")
        print(f"  Fallback 비율: {r['fallback_rate']:.1f}%  ({r['fallback_count']}개)")
        print(f"  Fallback 배치 수: {r['fallback_batches']}")

        print_latency_report(r['latency_stats'], "Hybrid Runtime")

        # JSON 저장
        save_data = {
            'threshold':    args.threshold,
            'batch_size':   args.batch_size,
            'timeout_ms':   args.timeout_ms,
            'accuracy':     r['accuracy'],
            'exit1_rate':   r['exit1_rate'],
            'fallback_rate': r['fallback_rate'],
            **r['latency_stats'],
        }
        save_path = os.path.join(out_dir,
            f"hybrid_thr{args.threshold:.2f}_bs{args.batch_size}_to{args.timeout_ms:.0f}.json")
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"결과 저장: {save_path}")


if __name__ == '__main__':
    main()
