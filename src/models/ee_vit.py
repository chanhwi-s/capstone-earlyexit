"""
Early Exit ViT-B/16 (12-exit)

Pretrained ViT-B/16 backbone 위에 각 Transformer block 직후
EarlyExitHead (LayerNorm → Linear)를 삽입.

구조:
  patch_embed + cls_token + pos_embed
  → block[0]  → exit_head[0]   (exit  1)
  → block[1]  → exit_head[1]   (exit  2)
  ...
  → block[11] → exit_head[11]  (exit 12 = 최종 출력)

Backbone : Frozen (requires_grad=False)
Exit heads: Trainable (requires_grad=True)

forward(x, threshold=None):
  threshold=None  → 학습 모드. 12개 exit logit 리스트 반환.
                    return [logits_1, logits_2, ..., logits_12]
  threshold=float → 추론 모드. confidence >= threshold인 첫 exit에서 조기 종료.
                    return (logits, exit_idx)   exit_idx: 1~12 (1-indexed)

전제 조건:
  - timm의 vit_base_patch16_224 기본 설정 (no_embed_class=False, cls_token 사용)
  - patch_drop_rate=0.0, pre_norm=False (기본값)
  - 위 조건이 바뀌면 _embed() 수정 필요

사용법:
  from models.ee_vit import build_model, print_trainable_params
  model = build_model(num_classes=1000)
  print_trainable_params(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── Exit Head ──────────────────────────────────────────────────────────────────

class EarlyExitHead(nn.Module):
    """
    각 Transformer block 출력의 CLS 토큰을 분류하는 경량 head.

    입력 x : [B, seq_len, hidden_dim]
             (ViT-B/16 기준: seq_len=197, hidden_dim=768)
    출력   : [B, num_classes] logits

    LayerNorm → Linear(hidden_dim, num_classes)
    timm의 최종 head (norm → head) 구조를 동일하게 따름.
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x[:, 0]))   # CLS 토큰 (index 0)만 사용


# ── EE-ViT-B/16 ───────────────────────────────────────────────────────────────

class EEViT(nn.Module):
    """ViT-B/16 with 12 early exit heads (one per Transformer block)."""

    NUM_BLOCKS = 12
    HIDDEN_DIM = 768

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # ── Pretrained backbone 로드 ──────────────────────────────────────
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)

        # backbone 구성 요소 분리
        self.patch_embed = vit.patch_embed   # PatchEmbed: [B,196,768]
        self.cls_token   = vit.cls_token     # Parameter:  [1,  1,768]
        self.pos_embed   = vit.pos_embed     # Parameter:  [1,197,768]
        self.pos_drop    = vit.pos_drop      # Dropout (기본 p=0.0)
        self.blocks      = vit.blocks        # nn.Sequential, 12 TransformerBlocks

        # ── Backbone 전체 Freeze ──────────────────────────────────────────
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for p in self.blocks.parameters():
            p.requires_grad = False

        # ── Exit Heads (Trainable) ────────────────────────────────────────
        # exit_heads[i]: block[i] 직후에 적용 (i=0~11)
        self.exit_heads = nn.ModuleList([
            EarlyExitHead(self.HIDDEN_DIM, num_classes)
            for _ in range(self.NUM_BLOCKS)
        ])

    # ── 내부 유틸 ──────────────────────────────────────────────────────────

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patch embed → CLS 토큰 prepend → positional embed 적용.
        vit_base_patch16_224 기본 설정 (no_embed_class=False) 전제.
        """
        B   = x.shape[0]
        x   = self.patch_embed(x)                       # [B, 196, 768]
        cls = self.cls_token.expand(B, -1, -1)          # [B,   1, 768]
        x   = torch.cat([cls, x], dim=1)                # [B, 197, 768]
        x   = self.pos_drop(x + self.pos_embed)         # [B, 197, 768]
        return x

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, threshold=None):
        x = self._embed(x)

        if threshold is None:
            # ── 학습 모드: 12개 exit logit 모두 반환 ─────────────────────
            logits_list = []
            for i, block in enumerate(self.blocks):
                x = block(x)
                logits_list.append(self.exit_heads[i](x))
            return logits_list      # list of 12 tensors, each [B, num_classes]

        else:
            # ── 추론 모드: confidence >= threshold인 첫 exit에서 조기 종료 ─
            logits = None
            for i, block in enumerate(self.blocks):
                x      = block(x)
                logits = self.exit_heads[i](x)           # [B, num_classes]
                conf   = F.softmax(logits, dim=1).max(dim=1).values  # [B]
                if conf.min().item() >= threshold:
                    return logits, i + 1                 # 1-indexed exit number
            # 12블록 모두 통과 후에도 threshold 미달 → 마지막 exit 결과 반환
            return logits, self.NUM_BLOCKS


# ── 모델 생성 함수 ─────────────────────────────────────────────────────────────

def build_model(num_classes: int = 1000) -> EEViT:
    return EEViT(num_classes=num_classes)


# ── 학습 가능한 파라미터 확인 유틸 ────────────────────────────────────────────

def print_trainable_params(model: EEViT) -> None:
    """
    학습 가능한 파라미터만 출력.
    exit head만 학습되는지(backbone이 frozen인지) 확인용.
    """
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
    print(f"  Trainable : {n_trainable:>12,}")
    print(f"  Frozen    : {n_frozen:>12,}")
    print(f"{'─' * 64}\n")
