import torch
import torch.nn as nn
import torchvision.models as models

class ExitHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        """
        Early Exit Head
        input: in_channels x n x n
        output(AdaptivePool2d): in_channels x 1 x 1
        output(Flatten): in_channels
        output(Linear): num_classes
        """
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class ResNet18_PT_EE(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=False):
        super().__init__()

        backbone = models.resnet18(weights="IMAGENET1K_V1")

        backbone.fc = nn.Identity() # backbone fc layer 제거

        self.backbone = backbone
        
        """
        layer2, layer3 에서만 Early Exit Head 추가
        layer2 output: 128 x 56 x 56 -> ExitHead(128, num_classes)
        layer3 output: 256 x 28 x 28 -> ExitHead(256, num_classes)
        """
        self.exit1 = ExitHead(128, num_classes)
        self.exit2 = ExitHead(256, num_classes)
        self.main_fc = ExitHead(512, num_classes)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False # gradient 흐르지 않도록 하여 pretrained parameter 고정
        
    def forward(self, x):
        # stem layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # layer1
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        out_ee1 = self.exit1(x)

        x = self.backbone.layer3(x)
        out_ee2 = self.exit2(x)

        x = self.backbone.layer4(x)
        out_main = self.main_fc(x)

        return [out_ee1, out_ee2, out_main]


model = ResNet18_PT_EE(num_classes=10)
x = torch.randn(4, 3, 224, 224)

outs = model(x)
print([o.shape for o in outs])