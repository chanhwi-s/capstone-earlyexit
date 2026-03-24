import torch
import torch.nn as nn
import torch.nn.functional as F


# ── plain_resnet18과 동일한 building block ───────────────────────────────────

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


"""
plain_resnet18과 완전히 동일한 BasicBlock
2개의 3x3 conv + residual connection
"""
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


# ── Early Exit Head ──────────────────────────────────────────────────────────

"""
Early Exit Head
각 중간 레이어의 feature map을 받아 클래스 확률로 변환한다.

input : C x H x W  (C: 해당 레이어의 출력 채널 수)
AdaptiveAvgPool2d  → C x 1 x 1
Flatten            → C
Linear             → num_classes
"""
class ExitHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        return self.head(x)


# ── ResNet-18 with Early Exit ────────────────────────────────────────────────

"""
plain_resnet18.ResNet과 동일한 backbone 구조에
layer2, layer3 출력 직후 ExitHead를 추가한 모델.

Exit point별 feature map 크기 (224x224 입력 기준):
  exit1  : layer2 출력  128 x 28 x 28
  exit2  : layer3 출력  256 x 14 x 14
  main   : layer4 출력  512 x  7 x  7

forward(x, threshold=None) 동작:
  threshold=None  → 학습 모드. 세 exit의 logit을 모두 계산해 리스트로 반환.
                    return [out_ee1, out_ee2, out_main]

  threshold=float → 추론 모드. 각 exit에서 softmax confidence를 확인하고
                    배치 내 모든 샘플의 최소 confidence >= threshold이면
                    해당 exit에서 조기 종료.
                    return (logits, exit_idx)  exit_idx: 1, 2, 3
                    ※ 샘플별 exit rate 측정 시 batch_size=1 권장
"""
class ResNet_EE(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_planes = 64

        # ── Stem (plain_resnet18과 동일) ──────────────────────────────────
        # input : 3  x 224 x 224
        # output: 64 x 112 x 112
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=False)

        # input : 64 x 112 x 112
        # output: 64 x  56 x  56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ── Residual Layers (plain_resnet18과 동일) ───────────────────────
        # layer1: 64  x 56 x 56
        # layer2: 128 x 28 x 28  ← exit1
        # layer3: 256 x 14 x 14  ← exit2
        # layer4: 512 x  7 x  7  ← main_fc
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # ── Early Exit Heads ──────────────────────────────────────────────
        # plain_resnet18의 avgpool + fc 를 exit point마다 ExitHead로 대체
        self.exit1   = ExitHead(128, num_classes)   # layer2 출력 후
        self.exit2   = ExitHead(256, num_classes)   # layer3 출력 후
        self.main_fc = ExitHead(512, num_classes)   # layer4 출력 후 (최종)

    def _make_layer(self, block, planes, blocks, stride=1):
        # plain_resnet18._make_layer와 완전히 동일
        downsample = None

        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
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

        # ── Layer 1 ──
        x = self.layer1(x)

        # ── Layer 2 → Exit 1 ──────────────────────────────────────────────
        x = self.layer2(x)
        out_ee1 = self.exit1(x)

        if threshold is not None:
            conf = F.softmax(out_ee1, dim=1).max(dim=1).values  # shape: (B,)
            if conf.min().item() >= threshold:
                return out_ee1, 1

        # ── Layer 3 → Exit 2 ──────────────────────────────────────────────
        x = self.layer3(x)
        out_ee2 = self.exit2(x)

        if threshold is not None:
            conf = F.softmax(out_ee2, dim=1).max(dim=1).values
            if conf.min().item() >= threshold:
                return out_ee2, 2

        # ── Layer 4 → Main Output ─────────────────────────────────────────
        x = self.layer4(x)
        out_main = self.main_fc(x)

        if threshold is not None:
            return out_main, 3

        # 학습 모드: 세 출력 모두 반환
        return [out_ee1, out_ee2, out_main]


# ── 모델 생성 함수 ────────────────────────────────────────────────────────────

def build_model(num_classes=1000):
    """
    random 초기화된 ResNet_EE 반환.
    plain_resnet18.build_model()과 동일한 backbone 구조.
    """
    return ResNet_EE(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes
    )
