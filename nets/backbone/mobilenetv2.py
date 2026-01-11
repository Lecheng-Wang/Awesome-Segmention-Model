# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2023/9/20  18:23
# @Function: Realization of mobileNetv2 architecture

import math
import os
import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

# Common 3×3 2D_Conv Layer
def conv_3x3_bn(inputchannel, outputchannel, stride):
    return nn.Sequential(
        nn.Conv2d(inputchannel, outputchannel, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(outputchannel),
        nn.ReLU6(inplace=True)
    )
# Common 1×1 2D_Conv Layer
def conv_1x1_bn(inputchannel, outputchannel):
    return nn.Sequential(
        nn.Conv2d(inputchannel, outputchannel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(outputchannel),
        nn.ReLU6(inplace=True)
    )

# The Single Block in MobileNetV2(expand_ratio aims to expand channels of input_features)
class InvertedResidual(nn.Module):
    def __init__(self, inputchannel, outputchannel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        expand_channels = round(inputchannel * expand_ratio)
        self.use_res_connect = (stride == 1) and (inputchannel == outputchannel)
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(expand_channels, outputchannel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outputchannel),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inputchannel, expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),

                nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels, bias=False),
                nn.BatchNorm2d(expand_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(expand_channels, outputchannel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outputchannel),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Construct the MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, bands=3, num_classes=1000, downsample_factor=32):
        super(MobileNetV2, self).__init__()
        Block         = InvertedResidual
        bands         = bands
        input_channel = 32
        last_channel  = 1280

        Block_Setting = [
            [1, 16,  1, 1],
            [6, 24,  2, 2],
            [6, 32,  3, 2],
            [6, 64,  4, 2],
            [6, 96,  3, 1],
            [6, 160, 3, 2],  # 设置为2后下采样倍率为32, 设置为1后下采样率为16
            [6, 320, 1, 1],
        ]
        self.features = [conv_3x3_bn(bands, input_channel, stride=2)]

        # t：Times of the Channels will be expanded in this Block group
        # c：OutputChannel of Features in this Block
        # n：Repeat times of Block in this group
        # s：The stride of GroupConv in Block(Only first Block need to set, aiming to Supsample the size of Features)
        for t, c, n, s in Block_Setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(Block(input_channel, output_channel, stride=s, expand_ratio=t))
                else:
                    self.features.append(Block(input_channel, output_channel, stride=1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, last_channel))
        self.features   = nn.Sequential(*self.features)

        # 新增用于下采样倍率设置
        self.total_idx = len(self.features)
        self.down_idx  = [2, 4, 7, 14]
        from functools import partial
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(last_channel, num_classes)
                                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)



    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x


def mobilenetv2(bands=3, num_classes=1000, **kwargs):
    model = MobileNetV2(bands=bands, num_classes=num_classes, downsample_factor=16, **kwargs)
    return model


# Test Model Structure and Outputsize
if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = mobilenetv2(bands=3, num_classes=1000).to(device)
    x               = torch.randn(2, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 224, 224), batch_dim=0)