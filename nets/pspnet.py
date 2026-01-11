# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/16} ${03:06}
# @Function   : PSPNet
# @Description: Realization of PSPNet architecture


import torch
import torch.nn.functional as F

from torch                    import nn
from .netforpspnet.mobilenetv2 import mobilenetv2
from .netforpspnet.resnet      import resnet50


class Resnet(nn.Module):
    def __init__(self, in_channels=6, dilate_scale=8):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(in_channels=in_channels)
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1   = model.conv1[0]
        self.bn1     = model.conv1[1]
        self.relu1   = model.conv1[2]
        self.conv2   = model.conv1[3]
        self.bn2     = model.conv1[4]
        self.relu2   = model.conv1[5]
        self.conv3   = model.conv1[6]
        self.bn3     = model.bn1
        self.relu3   = model.relu
        self.maxpool = model.maxpool
        self.layer1  = model.layer1
        self.layer2  = model.layer2
        self.layer3  = model.layer3
        self.layer4  = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding  = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding  = (dilate, dilate)

    def forward(self, x):
        x     = self.relu1(self.bn1(self.conv1(x)))
        x     = self.relu2(self.bn2(self.conv2(x)))
        x     = self.relu3(self.bn3(self.conv3(x)))
        x     = self.maxpool(x)
        x     = self.layer1(x)
        x     = self.layer2(x)
        x_aux = self.layer3(x)
        x     = self.layer4(x_aux)
        return x_aux, x

class MobileNetV2(nn.Module):
    def __init__(self,downsample_factor=8, in_channels=6):
        super(MobileNetV2, self).__init__()
        from functools import partial
        model          = mobilenetv2(in_channels=in_channels)
        self.features  = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx  = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding  = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding  = (dilate, dilate)

    def forward(self, x):
        x_aux = self.features[:14](x)
        x     = self.features[14:](x_aux)
        return x_aux, x
 
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels    = in_channels // len(pool_sizes)
        self.stages     = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn    = norm_layer(out_channels)
        relu  = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        b,c,h,w  = features.size()
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output   = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, bands=6,  backbone="resnet50", aux_branch=False):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = Resnet(in_channels=bands)
            aux_channel = 1024
            out_channel = 2048
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor,in_channels=bands)
            aux_channel = 96
            out_channel = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x   = self.backbone(x)
        output     = self.master_branch(x)
        output     = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output


if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = PSPNet(num_classes=3, bands=6, downsample_factor=8, backbone="resnet50").to(device)
    x               = torch.randn(2, 6, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)