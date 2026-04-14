"""
Selective Early Exit ViT-B/16

Pretrained ViT-B/16 backbone 위에 지정된 블록 뒤에만 EarlyExitHead를 삽입.
기존 EEViT(12-exit)과 달리 원하는 블록만 선택 가능.

지원 구성 예:
  SelectiveExitViT(exit_blocks=[8, 12])     ← 2-exit
  SelectiveExitViT(exit_blocks=[6, 9, 12])  ← 3-exit

구조 (exit_blocks=[8, 12] 예):
  patch_embed + cls_token + pos_embed
  → block[0] ~ block[6]
  → block[7]  → exit_head[0]   (exit B8,  early exit)
  → block[8] ~ block[10]
  → block[11] → exit_head[1]   (exit B12, main exit)

Backbone : Frozen (requires_grad=False)
Exit heads: Trainable (requires_grad=True)

forward(x, threshold=None):
  threshold=None  → 학습 모드. n_exits개 exit logit 리스트 반환.
                    return [logits_1, ..., logits_n]
  threshold=float → 추론 모드. confidence >= threshold인 첫 exit에서 조기 종료.
                    return (logits, exit_block)   exit_block: 실제 블록 번호 (1-indexed)

사용법:
  from models.ee_vit_selective import SelectiveExitViT, build_model, print_trainable_params
  model = build_model(exit_blocks=[8, 12], num_classes=1000)
  model = build_model(exit_blocks=[6, 9, 12], num_classes=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Exit Head ──────────────────────────────────────────────────────────────────

class EarlyExitHead(nn.Module):
    """
    CLS 토큰을 분류하는 경량 head (EEViT와 동일).

    입력 x : [B, seq_len, hidden_dim]
    출력   : [B, num_classes] logits
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x[:, 0]))   # CLS 토큰 (index 0)


# ── SelectiveExitViT ───────────────────────────────────────────────────────────

class SelectiveExitViT(nn.Module):
    """
    ViT-B/16 with exit heads at selected Transformer blocks only.

    Parameters
    ----------
    exit_blocks : list[int]
        1-indexed block positions where exit heads are attached.
        Must include 12 (final block) as the last element.
        예: [8, 12] or [6, 9, 12]
    num_classes : int
        Number of output classes (default: 1000 for ImageNet).
    """

    TOTAL_BLOCKS = 12
    HIDDEN_DIM   = 768

    def __init__(self, exit_blocks: list, num_classes: int = 1000):
        super().__init__()

        assert len(exit_blocks) >= 1, "exit_blocks must have at least one entry"
        assert all(1 <= b <= self.TOTAL_BLOCKS for b in exit_blocks), \
            f"exit_blocks must be in range [1, {self.TOTAL_BLOCKS}]"
        assert exit_blocks == sorted(exit_blocks), "exit_blocks must be sorted ascending"
        assert exit_blocks[-1] == self.TOTAL_BLOCKS, \
            f"Last exit block must be {self.TOTAL_BLOCKS} (final block)"

        self.exit_blocks = list(exit_blocks)         # e.g. [8, 12] or [6, 9, 12]
        self.NUM_BLOCKS  = len(exit_blocks)          # trainer 호환: n_exits
        self._exit_set   = set(b - 1 for b in exit_blocks)  # 0-indexed set

        # ── Pretrained backbone 로드 ──────────────────────────────────────
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)

        self.patch_embed = vit.patch_embed
        self.cls_token   = vit.cls_token
        self.pos_embed   = vit.pos_embed
        self.pos_drop    = vit.pos_drop
        self.blocks      = vit.blocks   # nn.Sequential(12 blocks)

        # ── Backbone Freeze ───────────────────────────────────────────────
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for p in self.blocks.parameters():
            p.requires_grad = False

        # ── Exit Heads (Trainable) ────────────────────────────────────────
        # exit_heads[i] corresponds to exit_blocks[i] (0-indexed in exit list)
        self.exit_heads = nn.ModuleList([
            EarlyExitHead(self.HIDDEN_DIM, num_classes)
            for _ in range(self.NUM_BLOCKS)
        ])

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def exit_block_labels(self) -> list:
        """Visualization-friendly labels: ['B8', 'B12'] or ['B6', 'B9', 'B12']."""
        return [f'B{b}' for b in self.exit_blocks]

    @property
    def model_name(self) -> str:
        """Unique name for checkpoint directory: 'ee_vit_2exit' or 'ee_vit_3exit'."""
        return f"ee_vit_{self.NUM_BLOCKS}exit"

    # ── Internal ──────────────────────────────────────────────────────────

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B   = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        return x

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, threshold=None):
        x = self._embed(x)

        if threshold is None:
            # ── Training mode: return list of n_exits logit tensors ───────
            logits_list = []
            exit_head_idx = 0
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self._exit_set:
                    logits_list.append(self.exit_heads[exit_head_idx](x))
                    exit_head_idx += 1
            return logits_list   # len == NUM_BLOCKS

        else:
            # ── Inference mode: exit at first block with conf >= threshold ─
            logits = None
            exit_head_idx = 0
            for i, block in enumerate(self.blocks):
                x = block(x)
                if i in self._exit_set:
                    logits = self.exit_heads[exit_head_idx](x)
                    conf   = F.softmax(logits, dim=1).max(dim=1).values
                    if conf.min().item() >= threshold:
                        return logits, self.exit_blocks[exit_head_idx]  # 1-indexed block
                    exit_head_idx += 1
            # All blocks processed, return last exit
            return logits, self.exit_blocks[-1]


# ── 모델 생성 함수 ─────────────────────────────────────────────────────────────

def build_model(exit_blocks: list, num_classes: int = 1000) -> SelectiveExitViT:
    """
    Create a SelectiveExitViT model.

    Args:
        exit_blocks: list of 1-indexed block positions (must end with 12)
        num_classes: number of output classes
    """
    return SelectiveExitViT(exit_blocks=exit_blocks, num_classes=num_classes)


# ── 학습 가능한 파라미터 확인 유틸 ────────────────────────────────────────────

def print_trainable_params(model: SelectiveExitViT) -> None:
    trainable = [
        (n, p.shape)
        for n, p in model.named_parameters()
        if p.requires_grad
    ]
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"\n{'─' * 64}")
    print(f"{'Trainable Parameters':^64}")
    print(f"{'─' * 64}")
    for name, shape in trainable:
        print(f"  {name:<50} {str(list(shape)):>10}")
    print(f"{'─' * 64}")
    print(f"  Exit blocks : {model.exit_blocks}")
    print(f"  Trainable   : {n_trainable:>12,}")
    print(f"  Frozen      : {n_frozen:>12,}")
    print(f"{'─' * 64}\n")
