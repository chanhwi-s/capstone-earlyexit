"""
Plain ViT-B/16 (Pretrained Baseline)

timm의 vit_base_patch16_224 pretrained 모델을 래핑한 baseline.
EE-ViT와의 공정한 비교를 위해 동일한 pretrained weight에서 출발.

입력 : 3 × 224 × 224  (ImageNet 표준)
출력 : [B, num_classes] logits

사용법:
  from models.plain_vit import build_model
  model = build_model(num_classes=1000)
"""

import torch.nn as nn
import timm


class PlainViT(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # num_classes=1000(ImageNet)이면 pretrained head 그대로 사용.
        # 다른 num_classes이면 timm이 head만 재초기화한다.
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


def build_model(num_classes: int = 1000) -> PlainViT:
    return PlainViT(num_classes=num_classes)
