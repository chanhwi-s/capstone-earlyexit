"""
SelectiveExitViT-L/16
ViT-Large/16 backbone 위에 지정된 블록 뒤에 EarlyExitHead를 삽입.

ViT-B/16 대비 차이:
  backbone : vit_large_patch16_224
  blocks   : 24 (B: 12)
  hidden   : 1024 (B: 768)
  feature  : [B, 197, 1024]

지원 구성:
  SelectiveExitViTLarge(exit_blocks=[12, 24])   ← 2-exit (block 12에 exit head)
  SelectiveExitViTLarge(exit_blocks=[8, 16, 24]) ← 3-exit

forward() 인터페이스는 ee_vit_selective.py와 동일:
  threshold=None  → 학습: list[logits_per_exit]
  threshold=float → 추론: (logits, exit_block_1indexed)

사용법:
  from models.ee_vit_large_selective import build_model_large, print_trainable_params
  model = build_model_large(exit_blocks=[12, 24], num_classes=1000)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

os.environ.setdefault('HF_HOME', '/home/cap10/.cache/huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/home/cap10/.cache/huggingface/hub')

from models.ee_vit_selective import EarlyExitHead   # 재사용


class SelectiveExitViTLarge(nn.Module):
    """
    ViT-L/16 with exit heads at selected Transformer blocks.

    Parameters
    ----------
    exit_blocks : list[int]
        1-indexed block positions for exit heads.
        Must include 24 (final block) as the last element.
        예: [12, 24]  or  [8, 16, 24]
    num_classes : int
        출력 클래스 수 (기본: 1000, ImageNet)
    """

    TOTAL_BLOCKS = 24
    HIDDEN_DIM   = 1024

    def __init__(self, exit_blocks: list, num_classes: int = 1000):
        super().__init__()

        assert len(exit_blocks) >= 1
        assert all(1 <= b <= self.TOTAL_BLOCKS for b in exit_blocks), \
            f"exit_blocks must be in [1, {self.TOTAL_BLOCKS}]"
        assert exit_blocks == sorted(exit_blocks), "exit_blocks must be ascending"
        assert exit_blocks[-1] == self.TOTAL_BLOCKS, \
            f"Last exit block must be {self.TOTAL_BLOCKS} (final block of ViT-L)"

        self.exit_blocks = list(exit_blocks)
        self.NUM_BLOCKS  = len(exit_blocks)
        self._exit_set   = set(b - 1 for b in exit_blocks)   # 0-indexed

        # ── Pretrained ViT-L/16 backbone ──────────────────────────────────
        vit = timm.create_model("vit_large_patch16_224", pretrained=True)

        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = vit.pos_drop
        self.blocks      = vit.blocks   # nn.Sequential(24 blocks)

        # ── Backbone Freeze ───────────────────────────────────────────────
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for p in self.blocks.parameters():
            p.requires_grad = False

        # ── Exit Heads (Trainable) ────────────────────────────────────────
        self.exit_heads = nn.ModuleList([
            EarlyExitHead(self.HIDDEN_DIM, num_classes)
            for _ in range(self.NUM_BLOCKS)
        ])

    @property
    def exit_block_labels(self) -> list:
        return [f'B{b}' for b in self.exit_blocks]

    @property
    def model_name(self) -> str:
        return f"ee_vit_large_{self.NUM_BLOCKS}exit"

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        return x

    def forward(self, x: torch.Tensor, threshold=None):
        x = self._embed(x)

        if threshold is None:
            # ── 학습 모드: 모든 exit logit 리스트 반환 ────────────────────
            logits_list   = []
            exit_head_idx = 0
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self._exit_set:
                    logits_list.append(self.exit_heads[exit_head_idx](x))
                    exit_head_idx += 1
            return logits_list

        else:
            # ── 추론 모드: confidence >= threshold 첫 exit에서 종료 ───────
            logits        = None
            exit_head_idx = 0
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self._exit_set:
                    logits = self.exit_heads[exit_head_idx](x)
                    conf   = F.softmax(logits, dim=1).max(dim=1).values
                    if conf.min().item() >= threshold:
                        return logits, self.exit_blocks[exit_head_idx]
                    exit_head_idx += 1
            return logits, self.exit_blocks[-1]


def build_model_large(exit_blocks: list, num_classes: int = 1000) -> SelectiveExitViTLarge:
    return SelectiveExitViTLarge(exit_blocks=exit_blocks, num_classes=num_classes)


def print_trainable_params(model: SelectiveExitViTLarge) -> None:
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    n_train   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\n{'─' * 64}")
    print(f"{'Trainable Parameters (ViT-L/16)':^64}")
    print(f"{'─' * 64}")
    for name, shape in trainable:
        print(f"  {name:<50} {str(list(shape)):>10}")
    print(f"{'─' * 64}")
    print(f"  Backbone     : vit_large_patch16_224  ({n_frozen:,} params, frozen)")
    print(f"  Exit blocks  : {model.exit_blocks}")
    print(f"  Trainable    : {n_train:>12,}  (exit heads only)")
    print(f"  Frozen       : {n_frozen:>12,}")
    print(f"{'─' * 64}\n")
