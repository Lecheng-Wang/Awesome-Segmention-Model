# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/03/16 17:14 
# @Function: Realization of xception architecture

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile


class SeperableConv2d(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, dilation=1, **kwargs):
        super(SeperableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(inputchannel, inputchannel, kernel_size, groups=inputchannel, dilation=dilation, bias=False, **kwargs)
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output   = shortcut + residual
        output   = self.conv(output)
        output   = self.avgpool(output)
        return output

class Xception(nn.Module):
    def __init__(self, block, bands=3, num_class=100, downsample_ratio=32):
        super(Xception, self).__init__()

        cfgs = {
            8:  {'conv1_stride':1, 'conv1_dilation':2, 'conv3_stride':1, 'conv3_dilation': 4},
            16: {'conv1_stride':1, 'conv1_dilation':2, 'conv3_stride':2, 'conv3_dilation': 1},
            32: {'conv1_stride':2, 'conv1_dilation':1, 'conv3_stride':2, 'conv3_dilation': 1}
        }

        downsample_cfg = cfgs[downsample_ratio]

        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=3, padding=downsample_cfg['conv1_dilation'], 
                      stride=downsample_cfg['conv1_stride'], dilation=downsample_cfg['conv1_dilation'],  bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )   
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, kernel_size=3, 
                            padding=downsample_cfg['conv3_dilation'], dilation=downsample_cfg['conv3_dilation']),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, kernel_size=3,
                            padding=downsample_cfg['conv3_dilation'], dilation=downsample_cfg['conv3_dilation']),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=downsample_cfg['conv3_stride'], padding=1),
        )
        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=downsample_cfg['conv3_stride']),
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, kernel_size=1, stride=2),
            nn.BatchNorm2d(728)
        )
        self.middel_flow = MiddleFlow(block)
        self.exit_flow   = ExitFLow()
        self.fc          = nn.Linear(2048, num_class)

    def forward(self, x):

        x        = self.conv1(x)
        x        = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x        = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x        = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x        = residual + shortcut
        x        = self.middel_flow(x)
        x        = self.exit_flow(x)
        x        = x.view(x.size(0), -1)
        x        = self.fc(x)
        return x


def xception(bands=3, num_classes=1000):
    model = Xception(MiddleFLowBlock, bands, num_classes, downsample_ratio=32)
    return model


# Test Model Structure and Outputsize
if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = xception(bands=3, num_classes=1000).to(device)
    x               = torch.randn(1, 3, 299, 299).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(1, 3, 299, 299), device=device.type)