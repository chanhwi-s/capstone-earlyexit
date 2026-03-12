import torch
import torch.nn as nn
import torch.nn.functional as F

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
ResNet18을 위한 Basic Block class
ResNet18에서의 모든 basic block은 2개의 3x3 conv filter와 residual connection으로 구성됨
Layer 구조
- 3x3 conv filter
- batch norm
- ReLU
- 3x3 conv filter
- batch norm
- + identity (residual connection)
- ReLU
"""
class BasicBlock(nn.Module):
    expansion=1 # block 출력 채널 확장 비율, ResNet18에서는 채널 확장이 없으므로 1로 설정

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

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
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_planes=64

        """
        맨 앞 stem layer
        7x7 conv filter X 64, stride=2, padding=3
        input: 3 x 224 x 224
        output: 64 x 112 x 112
        """
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        """
        3x3 maxpooling, stride=2, padding=1
        input: 64 x 112 x 112
        output: 64 x 56 x 56 
        """
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        """
        4개의 residual block
        shape 변화
        input: 64 x 56 x 56
        output1: 64 x 56 x 56
        output2: 128 x 28 x 28
        output3: 256 x 14 x 14
        output4: 512 x 7 x 7
        final output: 
        """
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        """
        Spatial dimension 제거
        input: 512 x 7 x 7
        output: 512 x 1 x 1 (512 차원 vector가 됨)
        """
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        """
        input: 512 dim vector
        output: num_classes dim vector
        """
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None   # downsample block 기본값은 None

        """
        stride로 인해 feature map spatial size가 변경되거나 channel 수가 변경될 경우
        identity와 F(x)의 shape이 달라지므로 downsample projection을 사용한다

        ex) layer2 block
        conv1: 128 filters (3 x 3 x 64) filter, stride=2, padding=1
        conv2: 128 filters (3 x 3 x 128) filter, stride=1, padding=1
        input(x) shape: 64 x 56 x 56
        output(F(x)) shape: 128 x 28 x 28

        여기서 residual connection(output = F(x) + x)을 하려면 입/출력 차원을 맞춰주어야 한다.
        따라서 downsample block을 128 x (64 x 1 x 1) filter, stride=2로 설계하여 출력 차원을 맞춰준다
        identity shape: 128 x 28 x 28 이 되므로, conv 2개를 지난 feature map에 더하기 가능
        """
        if stride != 1 or self.in_planes != planes: 
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes,
                    kernel_size=1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(planes)
            )
        
        layers = []
        """
        각 residual layer의 첫번째 block에서는 downsample 여부에 따라 layer 추가
        ResNet18 기준
        layer1에서는 모두 3x3 conv, stride=1, padding=1, downsample=None인 block 4개 추가,
        layer2,3,4에서는 첫번째는 3x3 conv, stride=2, padding=1, downsample= 1x1 conv, stride=2 block 1개 추가 후
        3x3 conv, stride=1, padding=1, downsample=None인 block 3개 추가
        """
        layers.append(block(self.in_planes, planes, stride, downsample))
        
        # 첫번째 layer 이후 in_planes(입력 채널)과 planes(출력 채널)은 무조건 같음
        self.in_planes = planes

        # 남은 block 개수(blocks - 1) 만큼 basic block 추가
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
    
def resnet18(num_classes=1000):
    return ResNet(
        BasicBlock,
        [2,2,2,2],
        num_classes=num_classes
    )
