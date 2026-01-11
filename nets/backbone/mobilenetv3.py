# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2025/7/2  18:38
# @Function   : mobilenetv3
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace=True)/6
        return out
    
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x+3, inplace=True)/6
        return out

class relu(nn.Module):
    def forward(self, x):
        out = F.relu(x)
        return out

class SeModule(nn.Module):
    def __init__(self, in_planes, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//reduction, in_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_planes),
            hsigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    def __init__(self, kernel_size, in_planes, expand_planes, out_planes, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride    = stride
        self.se        = semodule
        self.conv1     = nn.Conv2d(in_planes, expand_planes, kernel_size=1, stride=1, padding=0)
        self.bn1       = nn.BatchNorm2d(expand_planes)
        self.nolinear1 = nolinear
        self.conv2     = nn.Conv2d(expand_planes, expand_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_planes, bias=False)
        self.bn2       = nn.BatchNorm2d(expand_planes)
        self.nolinear2 = nolinear
        self.conv3     = nn.Conv2d(expand_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3       = nn.BatchNorm2d(out_planes)
        self.short_cut = nn.Sequential()
        #若stride>1说明输出尺寸会变小（下采样），若输入通道数和输出通道数不一致，则需要使用1X1卷积改变维度
        if stride == 1 and in_planes != out_planes:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nolinear1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nolinear2(out)
        if self.se != None:
            out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.short_cut(x) if self.stride == 1 else out
        return out 

class MobileNetV3_Large(nn.Module):
    def __init__(self, bands=3, num_classes=19):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(bands, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.hs1   = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, relu(), None, 1),
            Block(3, 16, 64, 24, relu(), None, 2),
            Block(3, 24, 72, 24, relu(), None, 1),
            Block(5, 24, 72, 40, relu(), SeModule(72), 2),
            Block(5, 40, 120, 40, relu(), SeModule(120), 1),
            Block(5, 40, 120, 40, relu(), SeModule(120), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(480), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(672), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(672), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),
            Block(5, 160, 960, 160, hswish(), SeModule(960), 1),
        )
        self.conv2    = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2      = nn.BatchNorm2d(960)
        self.hs2      = hswish()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3    = nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.hs3      = hswish()
        self.conv4    = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
    
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs1(out)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hs2(out)
        out = self.avg_pool(out)
        #out = out.view(out.size(0), -1)
        out = self.conv3(out)
        out = self.hs3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        return out

class MobileNetV3_Small(nn.Module):
    def __init__(self, bands=3, num_classes=19):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(bands, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.hs1   = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, relu(), SeModule(16), 2),
            Block(3, 16, 72, 24, relu(), None, 2),
            Block(3, 24, 88, 24, relu(), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(96), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(240), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(120), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(144), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(288), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(576), 1),
        )
        self.conv2    = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2      = nn.BatchNorm2d(576)
        self.hs2      = hswish()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3    = nn.Conv2d(576, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.hs3      = hswish()
        self.conv4    = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs1(out)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.hs2(out)
        out = self.avg_pool(out)
        #out = out.view(out.size(0), -1)
        out = self.conv3(out)
        out = self.hs3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        return out


def mobilenetv3_s(bands=3, num_classes=1000):
    model = MobileNetV3_Small(bands=bands, num_classes=num_classes)
    return model

def mobilenetv3_l(bands=3, num_classes=1000):
    model = MobileNetV3_Large(bands=bands, num_classes=num_classes)
    return model



if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = mobilenetv3_l(bands=3, num_classes=1000).to(device)
    x               = torch.randn(2, 3, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(2, 3, 256, 256), device=device.type)

