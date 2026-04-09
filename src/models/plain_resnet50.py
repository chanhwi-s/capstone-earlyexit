import torch
import torch.nn as nn
import torch.nn.functional as F


"""
ResNet-50의 Bottleneck Block
1x1 → 3x3 → 1x1 conv 구조, expansion=4

채널 변화 예시 (planes=64일 때):
  입력: in_planes
  1x1 conv: planes (64)       ← 채널 압축
  3x3 conv: planes (64)
  1x1 conv: planes * 4 (256)  ← 채널 확장
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        # 1x1 conv: 채널 압축
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # 3x3 conv: spatial feature 추출
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # 1x1 conv: 채널 확장 (planes → planes * expansion)
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


"""
Plain ResNet-50

stem → layer1(3 blocks) → layer2(4 blocks) → layer3(6 blocks) → layer4(3 blocks) → fc

feature map 크기 변화 (224x224 입력 기준):
  stem     : 3  x 224 x 224 → 64  x 56  x 56
  layer1   : 64  x 56 x 56  → 256 x 56  x 56   (stride=1, expansion 만 변화)
  layer2   : 256 x 56 x 56  → 512 x 28  x 28   (stride=2)
  layer3   : 512 x 28 x 28  → 1024x 14  x 14   (stride=2)
  layer4   : 1024x 14 x 14  → 2048x  7  x  7   (stride=2)
  avgpool  : 2048x  7 x  7  → 2048x  1  x  1
  fc       : 2048            → num_classes
"""
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_planes = 64

        # Stem: 7x7 conv, stride=2 → maxpool stride=2
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block,  64, layers[0])           # 256 ch out
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512 ch out
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024 ch out
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048 ch out

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # 채널 수 또는 spatial size가 달라지는 경우 projection shortcut 사용
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
