# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/03/16 17:14
# @Function: Realization of resnext architecture

import torch
import torch.nn as nn


class _BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride, downsample=None, 
                 groups=1, base_channels=64, dilation=1):
        super(_BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        
        # 添加dilation参数
        padding = dilation  # 保持空间尺寸不变的padding
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        
        # 第二个卷积层也添加dilation
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride, downsample=None, groups=1, base_channels=64, dilation=1):
        super(_Bottleneck, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        
        # 计算中间通道数
        channels = int(out_channels * (base_channels / 64.0)) * groups
        
        # 第一个1x1卷积
        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 3x3卷积添加dilation
        padding = dilation  # 保持空间尺寸不变的padding
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), padding, groups=groups, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # 最后一个1x1卷积
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels * self.expansion))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(self, arch_cfg, block, groups=1, channels_per_group=64, bands=3, num_classes=1000, downsample_ratio=32):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.groups = groups
        self.base_channels = channels_per_group
        self.downsample_ratio = downsample_ratio
        
        # 验证下采样倍率是否有效
        if downsample_ratio not in [8, 16, 32]:
            raise ValueError("Invalid downsample ratio. Supported values are 8, 16, 32.")
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(bands, self.in_channels, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))
        
        # 根据下采样倍率设置各层的步长和空洞率
        if downsample_ratio == 32:
            # 标准32倍下采样
            self.layer1 = self._make_layer(arch_cfg[0], block, 64, 1, dilation=1)
            self.layer2 = self._make_layer(arch_cfg[1], block, 128, 2, dilation=1)
            self.layer3 = self._make_layer(arch_cfg[2], block, 256, 2, dilation=1)
            self.layer4 = self._make_layer(arch_cfg[3], block, 512, 2, dilation=1)
        elif downsample_ratio == 16:
            # 16倍下采样 - 在layer4使用空洞卷积
            self.layer1 = self._make_layer(arch_cfg[0], block, 64, 1, dilation=1)
            self.layer2 = self._make_layer(arch_cfg[1], block, 128, 2, dilation=1)
            self.layer3 = self._make_layer(arch_cfg[2], block, 256, 2, dilation=1)  # 步长改为1
            self.layer4 = self._make_layer(arch_cfg[3], block, 512, 1, dilation=2)  # 使用空洞卷积
        elif downsample_ratio == 8:
            # 8倍下采样 - 在layer3和layer4使用空洞卷积
            self.layer1 = self._make_layer(arch_cfg[0], block, 64, 1, dilation=1)
            self.layer2 = self._make_layer(arch_cfg[1], block, 128, 2, dilation=1)
            self.layer3 = self._make_layer(arch_cfg[2], block, 256, 1, dilation=2)  # 使用空洞卷积
            self.layer4 = self._make_layer(arch_cfg[3], block, 512, 1, dilation=4)  # 使用更大的空洞卷积

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, repeat_times, block, channels, stride=1, dilation=1):
        downsample = None
        # 当步长不为1或输入输出通道数不匹配时需要下采样
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, 
                         (1, 1), (stride, stride), (0, 0), bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        # 第一个block处理步长和通道变化
        layers.append(block(
            self.in_channels, channels, stride, downsample, 
            self.groups, self.base_channels, dilation
        ))
        self.in_channels = channels * block.expansion
        
        # 后续blocks步长固定为1，使用相同的dilation
        for _ in range(1, repeat_times):
            layers.append(block(
                self.in_channels, channels, 1, None, 
                self.groups, self.base_channels, dilation
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始下采样
        out = self.conv1(x)        # 1/2
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)     # 1/4
        
        # 各层处理
        out = self.layer1(out)      # 1/4
        out = self.layer2(out)      # 1/8 (如果stride=2)
        out = self.layer3(out)      # 1/16 (如果stride=2) 或 1/8 (如果使用空洞)
        out = self.layer4(out)      # 1/32 (如果stride=2) 或 1/8-1/16 (如果使用空洞)
        
        # 全局平均池化和分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out


def resnext50_32x4d(bands=3, num_classes=1000, **kwargs):
    model = ResNeXt([3, 4, 6, 3], _Bottleneck, 32, 4, bands, num_classes, downsample_ratio=32, **kwargs)
    return model

def resnext101_32x8d(bands=3, num_classes=1000, **kwargs):
    model = ResNeXt([3, 4, 23, 3], _Bottleneck, 32, 8, bands, num_classes, downsample_ratio=32, **kwargs)
    return model

def resnext101_64x4d(bands=3, num_classes=1000, **kwargs):
    model = ResNeXt([3, 4, 23, 3], _Bottleneck, 64, 4, bands, num_classes, downsample_ratio=32, **kwargs)
    return model



if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = resnext50_32x4d().to(device)
    x               = torch.randn(2, 3, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 256, 256), batch_dim=0)