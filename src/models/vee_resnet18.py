"""
Very Early Exit ResNet-18  (VEE-ResNet-18)

기존 EE-ResNet-18과 다른 점:
  - Exit head가 layer1 직후 (가장 이른 시점)에 1개만 존재
  - layer2/layer3 뒤에는 exit head 없이 layer4까지 쭉 진행
  - 출력: 2개 (ee1_logits, main_logits)

설계 근거:
  1. EE-ResNet-18 실험에서 layer3 뒤 exit(EE2)의 exit rate가 ~3.7%로 현저히 낮음
     → exit head 2개까지는 불필요
  2. layer1 직후에 exit을 배치해 최대한 빠른 조기 종료 유도
  3. 쉬운 샘플은 layer1 exit에서 탈출, 어려운 샘플은 full backbone으로 처리

Feature map 크기 (CIFAR-10 32×32 입력 기준):
  stem   : 64 × 16 × 16  (conv1 stride=2)
  maxpool: 64 ×  8 ×  8
  layer1 : 64 ×  8 ×  8  ← exit1 (ExitHead: AdaptiveAvgPool→64→num_classes)
  layer2 : 128 ×  4 ×  4
  layer3 : 256 ×  2 ×  2
  layer4 : 512 ×  1 ×  1 ← main_fc (ExitHead: AdaptiveAvgPool→512→num_classes)

forward(x, threshold=None):
  threshold=None  → 학습 모드. [out_ee1, out_main] 리스트 반환
  threshold=float → 추론 모드. (logits, exit_idx) 반환.  exit_idx: 1 or 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks (ee_resnet18, plain_resnet18과 동일) ────────────────────

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ExitHead(nn.Module):
    """AdaptiveAvgPool → Flatten → Linear"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ── Very Early Exit ResNet-18 ───────────────────────────────────────────────

class ResNet_VEE(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_planes = 64

        # ── Stem ──
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Residual Layers ──
        self.layer1 = self._make_layer(block, 64,  layers[0])          # 64  ch
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 128 ch
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 256 ch
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 512 ch

        # ── Heads ──
        self.exit1   = ExitHead(64,  num_classes)   # layer1 직후 (Very Early!)
        self.main_fc = ExitHead(512, num_classes)   # layer4 직후 (최종)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, threshold=None):
        # ── Stem ──
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ── Layer 1 → Very Early Exit ──
        x = self.layer1(x)
        out_ee1 = self.exit1(x)

        if threshold is not None:
            conf = F.softmax(out_ee1, dim=1).max(dim=1).values
            if conf.min().item() >= threshold:
                return out_ee1, 1

        # ── Layer 2 → 3 → 4 → Main (exit 없이 직진) ──
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out_main = self.main_fc(x)

        if threshold is not None:
            return out_main, 2

        # 학습 모드: 두 출력 모두 반환
        return [out_ee1, out_main]


# ── 모델 생성 함수 ─────────────────────────────────────────────────────────────

def build_model(num_classes=1000):
    """Random 초기화된 VEE-ResNet-18 반환."""
    return ResNet_VEE(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
