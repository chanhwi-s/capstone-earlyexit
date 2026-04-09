import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Bottleneck Block (plain_resnet50과 동일) ─────────────────────────────────

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


# ── Early Exit Head ──────────────────────────────────────────────────────────

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


# ── ResNet-50 with Early Exit (4 exits) ─────────────────────────────────────

"""
backbone은 plain_resnet50과 동일하되, 각 residual layer 출력 직후 ExitHead 부착.

Exit point별 채널 수 (224x224 입력 기준):
  exit1  : layer1 출력   256 x 56 x 56   (64  * expansion 4)  ← 3 bottlenecks
  exit2  : layer2 출력   512 x 28 x 28   (128 * expansion 4)  ← 4 bottlenecks
  exit3  : layer3 출력  1024 x 14 x 14   (256 * expansion 4)  ← 6 bottlenecks
  main   : layer4 출력  2048 x  7 x  7   (512 * expansion 4)  ← 3 bottlenecks

forward(x, threshold=None) 동작:
  threshold=None  → 학습 모드. 4개 exit 모두 logit 반환.
                    return [out_ee1, out_ee2, out_ee3, out_main]

  threshold=float → 추론 모드. 각 exit에서 softmax max confidence 확인.
                    배치 내 모든 샘플의 min confidence >= threshold이면 조기 종료.
                    return (logits, exit_idx)  exit_idx: 1, 2, 3, 4
                    ※ 샘플별 exit rate 측정 시 batch_size=1 권장
"""
class ResNet_EE(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_planes = 64

        # Stem
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block,  64, layers[0])            # 256ch out
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 512ch out
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 1024ch out
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 2048ch out

        # Exit Heads (각 layer 출력 직후)
        self.exit1   = ExitHead( 64 * block.expansion, num_classes)   # 256
        self.exit2   = ExitHead(128 * block.expansion, num_classes)   # 512
        self.exit3   = ExitHead(256 * block.expansion, num_classes)   # 1024
        self.main_fc = ExitHead(512 * block.expansion, num_classes)   # 2048

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, threshold=None):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1 → Exit 1
        x = self.layer1(x)
        out_ee1 = self.exit1(x)

        if threshold is not None:
            conf = F.softmax(out_ee1, dim=1).max(dim=1).values
            if conf.min().item() >= threshold:
                return out_ee1, 1

        # Layer 2 → Exit 2
        x = self.layer2(x)
        out_ee2 = self.exit2(x)

        if threshold is not None:
            conf = F.softmax(out_ee2, dim=1).max(dim=1).values
            if conf.min().item() >= threshold:
                return out_ee2, 2

        # Layer 3 → Exit 3
        x = self.layer3(x)
        out_ee3 = self.exit3(x)

        if threshold is not None:
            conf = F.softmax(out_ee3, dim=1).max(dim=1).values
            if conf.min().item() >= threshold:
                return out_ee3, 3

        # Layer 4 → Main
        x = self.layer4(x)
        out_main = self.main_fc(x)

        if threshold is not None:
            return out_main, 4

        # 학습 모드: 4개 출력 모두 반환
        return [out_ee1, out_ee2, out_ee3, out_main]


def build_model(num_classes=1000):
    return ResNet_EE(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
