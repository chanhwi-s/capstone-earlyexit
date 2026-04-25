"""
TRT 추론 엔진 — SelectiveExitViT & PlainViT  (Jetson AGX Orin 전용)

TRTEngine 은 infer_trt.py 에서 재사용 (torch 기반, pycuda-free).

세그먼트 I/O 규칙:
  seg1    : in="image"  → out="feat" + "ee_logits"
  seg_mid : in="feat"   → out="feat" + "ee_logits"   (3-exit 중간 세그먼트)
  seg_last: in="feat"   → out="ee_logits"
  plain   : in="image"  → out="logits"

사용법:
  from infer.infer_trt_vit_selective import (
      SelectiveViTTRT, PlainViTTRT,
      load_selective_vit_trt, load_plain_vit_trt,
  )

  vit_2exit = load_selective_vit_trt(exit_blocks=[8, 12])
  logits, exit_block, lat_ms = vit_2exit.infer(image_tensor, threshold=0.80)

  plain = load_plain_vit_trt()
  logits, lat_ms = plain.infer(image_tensor)
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths
from infer.infer_trt import TRTEngine     # torch 기반 TRT 래퍼 재사용


# ── SelectiveExitViT TRT Inference ───────────────────────────────────────────

class SelectiveViTTRT:
    """
    N개 TRT 세그먼트 엔진을 순서대로 실행하며 early exit 수행.

    Parameters
    ----------
    seg_engines  : list[TRTEngine]  (len = n_exits)
    exit_blocks  : list[int]        실제 블록 번호 (1-indexed, e.g. [8,12])
    """

    def __init__(self, seg_engines: list, exit_blocks: list):
        assert len(seg_engines) == len(exit_blocks), \
            "seg_engines와 exit_blocks 길이가 다릅니다."
        self.segs        = seg_engines
        self.exit_blocks = exit_blocks
        self.n_segs      = len(seg_engines)

    def infer(self, image: torch.Tensor, threshold: float):
        """
        Parameters
        ----------
        image     : (1, 3, 224, 224) CUDA 또는 CPU Tensor
        threshold : confidence threshold

        Returns
        -------
        (logits, exit_block, latency_ms)
          logits      : (1, num_classes) CPU Tensor
          exit_block  : 실제 블록 번호 (1-indexed)
          latency_ms  : float
        """
        t0   = time.perf_counter()
        feat = None

        for i, (seg, exit_block) in enumerate(zip(self.segs, self.exit_blocks)):
            is_first = (i == 0)
            is_last  = (i == self.n_segs - 1)

            # 입력 준비
            inp = {"image": image} if is_first else {"feat": feat}

            # TRT 실행
            out = seg.infer(inp)

            # 출력 파싱
            logits = out["ee_logits"]
            if not is_last:
                feat = out["feat"].cuda()   # 다음 세그먼트에 GPU 텐서 전달

            # Confidence 체크
            conf = F.softmax(logits, dim=1).max(dim=1).values.item()
            if conf >= threshold or is_last:
                lat_ms = (time.perf_counter() - t0) * 1000
                return logits, exit_block, lat_ms

        # fallback (이론상 도달 불가)
        lat_ms = (time.perf_counter() - t0) * 1000
        return logits, self.exit_blocks[-1], lat_ms


# ── PlainViT TRT Inference ────────────────────────────────────────────────────

class PlainViTTRT:
    """
    단일 TRT 엔진으로 PlainViT 추론.
    """

    def __init__(self, engine: TRTEngine):
        self.engine = engine

    def infer(self, image: torch.Tensor):
        """
        Returns
        -------
        (logits, latency_ms)
          logits     : (1, num_classes) CPU Tensor
          latency_ms : float
        """
        t0  = time.perf_counter()
        out = self.engine.infer({"image": image})
        lat_ms = (time.perf_counter() - t0) * 1000
        return out["logits"], lat_ms


# ── 로드 헬퍼 ─────────────────────────────────────────────────────────────────

def load_selective_vit_trt(exit_blocks: list,
                            engine_dir_override: str = None) -> SelectiveViTTRT:
    """
    TRT 엔진 파일을 자동 탐지하여 SelectiveViTTRT 생성.

    Parameters
    ----------
    exit_blocks          : e.g. [8, 12] or [6, 9, 12]
    engine_dir_override  : 엔진 디렉토리 직접 지정 (없으면 paths 자동)
    """
    n_exits    = len(exit_blocks)
    model_name = f"ee_vit_{n_exits}exit"
    eng_dir    = engine_dir_override or paths.engine_dir(model_name)

    seg_engines = []
    for i in range(1, n_exits + 1):
        epath = os.path.join(eng_dir, f"seg{i}.engine")
        if not os.path.exists(epath):
            raise FileNotFoundError(
                f"[ERROR] TRT 엔진 없음: {epath}\n"
                f"        orin_vit_pipeline.sh 를 먼저 실행하세요."
            )
        seg_engines.append(TRTEngine(epath))

    print(f"[TRT] {model_name} 엔진 {n_exits}개 로드 완료")
    return SelectiveViTTRT(seg_engines, exit_blocks)


def load_plain_vit_trt(engine_path_override: str = None) -> PlainViTTRT:
    """
    PlainViT TRT 엔진 로드.
    """
    epath = engine_path_override or paths.engine_path("plain_vit", "plain_vit.engine")
    if not os.path.exists(epath):
        raise FileNotFoundError(
            f"[ERROR] TRT 엔진 없음: {epath}\n"
            f"        orin_vit_pipeline.sh 를 먼저 실행하세요."
        )
    engine = TRTEngine(epath)
    print(f"[TRT] PlainViT 엔진 로드 완료")
    return PlainViTTRT(engine)
