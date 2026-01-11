# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/07/02  19:33
# @Function: Realization of efficientnet architecture

import torch
import torch.nn as nn
import math

# Swish激活函数（EfficientNet默认使用）
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 内存高效的Swish实现（减少内存消耗）
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

# Squeeze-and-Excitation模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# MBConv模块
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate):
        super(MBConvBlock, self).__init__()
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.use_residual = in_channels == out_channels and stride == 1
        reduced_channels  = max(1, int(in_channels * se_ratio)) if self.has_se else None
        
        # 扩展阶段
        expand_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, bias=False),
                nn.BatchNorm2d(expand_channels),
                Swish()
            )
        else:
            self.expand_conv = None
        
        # 深度卷积阶段
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expand_channels, expand_channels, kernel_size, stride, kernel_size//2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            Swish()
        )
        
        # Squeeze-and-Excitation
        if self.has_se:
            self.se_block = SEBlock(expand_channels, reduced_channels)
        else:
            self.se_block = None
        
        # 输出阶段
        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # DropConnect (随机丢弃路径)
        self.drop_connect_rate = drop_connect_rate
        if self.use_residual and drop_connect_rate > 0:
            self.dropout = nn.Dropout2d(drop_connect_rate, inplace=True)
        else:
            self.dropout = None

    def forward(self, inputs):
        x = inputs
        
        # 扩展通道
        if self.expand_conv is not None:
            x = self.expand_conv(x)
        
        # 深度卷积
        x = self.depthwise_conv(x)
        
        # SE注意力机制
        if self.se_block is not None:
            x = self.se_block(x)
        
        # 投影到输出通道
        x = self.project_conv(x)
        
        # 残差连接
        if self.use_residual:
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + inputs
        
        return x

# EfficientNet主网络
class EfficientNet(nn.Module):
    # 配置参数 (width, depth, resolution, dropout)
    model_config = {
        # (width_coeff, depth_coeff, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    
    # 基础网络配置 (stage重复次数, 输出通道, 卷积核大小, 步长, 扩展比例, SE比例)
    base_config = [
        # t, c, n, k, s, se
        [1, 16,  1, 3, 1, 0.25],   # stage1
        [6, 24,  2, 3, 2, 0.25],   # stage2
        [6, 40,  2, 5, 2, 0.25],   # stage3
        [6, 80,  3, 3, 2, 0.25],   # stage4
        [6, 112, 3, 5, 1, 0.25],   # stage5
        [6, 192, 4, 5, 2, 0.25],   # stage6
        [6, 320, 1, 3, 1, 0.25]    # stage7
    ]

    def __init__(self, bands=3, num_classes=1000, model_name='efficientnet-b0', drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()
        
        # 获取模型缩放系数
        w, d, res, p    = self.model_config[model_name]
        self.resolution = res
        
        # 计算通道数和重复次数（应用宽度/深度系数）
        def scale_channels(c):
            return math.ceil(c * w)
        def scale_repeats(r):
            return math.ceil(r * d)
        
        # 构建初始卷积层
        out_channels   = scale_channels(32)
        self.stem_conv = nn.Sequential(
            nn.Conv2d(bands, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        
        # 构建中间阶段
        self.blocks = nn.ModuleList([])
        in_channels = out_channels
        
        # 根据配置创建各阶段
        for config in self.base_config:
            t, c, n, k, s, se = config
            out_channels      = scale_channels(c)
            repeats           = scale_repeats(n)
            
            for i in range(repeats):
                stride = s if i == 0 else 1  # 每个block只有第一个使用指定步长
                block  = MBConvBlock(in_channels, out_channels, k, stride, t, se, drop_connect_rate)
                self.blocks.append(block)
                in_channels = out_channels
        
        final_channels  = scale_channels(1280)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish()
        )
        
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(p)
        self.classifier = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.stem_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# 创建EfficientNet实例
def efficientnet(bands=3, num_classes=1000):
    model = EfficientNet(bands, num_classes, model_name='efficientnet-b0')
    return model

if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = efficientnet(bands=3, num_classes=1000).to(device)
    x               = torch.randn(1, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(1, 3, 224, 224), device=device.type)