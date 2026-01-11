# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/03/16} ${17:14}
# @Function: Realization of xception architecture

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile


class SeperableConv2d(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, **kwargs):
        super(SeperableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(inputchannel, inputchannel, kernel_size, groups=inputchannel, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(inputchannel, outputchannel, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MiddleFLowBlock(nn.Module):
    def __init__(self):
        super(MiddleFLowBlock, self).__init__()
        self.shortcut = nn.Sequential()
        self.conv1    = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        shortcut = self.shortcut(x)
        return shortcut + residual

class MiddleFlow(nn.Module):
    def __init__(self, block):
        super(MiddleFlow, self).__init__()
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())
        return nn.Sequential(*flows)


class ExitFLow(nn.Module):
    def __init__(self):
        super(ExitFLow, self).__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, kernel_size=1, stride=2),
            nn.BatchNorm2d(1024)
        )
        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, kernel_size=3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
#        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output   = shortcut + residual
        output   = self.conv(output)
#        output   = self.avgpool(output)
        return output

class Xception(nn.Module):
    def __init__(self, block, bands=3, num_class=100):
        super(Xception, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )   
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
        )
        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.BatchNorm2d(256),
        )

        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, kernel_size=3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)   #16倍下采样，stride设为1，32倍设为2
        )

        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, kernel_size=1, stride=1),       #16倍下采样，stride设为1，32倍设为2
            nn.BatchNorm2d(728)
        )
        self.middel_flow = MiddleFlow(block)
        self.exit_flow   = ExitFLow()
#        self.fc          = nn.Linear(2048, num_class)
        self._initialize_weights()

    def forward(self, x):

        x        = self.conv1(x)
        x        = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x        = residual + shortcut

        low_level_feature = x

        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x        = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x        = residual + shortcut
        x        = self.middel_flow(x)
        x        = self.exit_flow(x)

        high_level_feature = x

        return low_level_feature, high_level_feature
#        x        = x.view(x.size(0), -1)
#        x        = self.fc(x)
#        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def xception(bands=3):
    model = Xception(MiddleFLowBlock, bands)
    return model


# Test Model Structure and Outputsize
if __name__ == "__main__":
     model         = xception(bands=3)
     device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model.to(device)
     x             = torch.randn(1, 3, 224, 224).to(device)
     print("Input shape:", x.shape)
     output        = model(x)
     flops, params = profile(model, inputs=(x, ))
     print('flops: ', flops, 'params: ', params)
     print("Output shape:", output[0].shape, output[1].shape)
     summary(model, (3, 224, 224)) 