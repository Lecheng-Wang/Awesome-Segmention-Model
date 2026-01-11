# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/5/8} ${03:06}
# @Function   : Unet
# @Description: Realization of Unet architecture


import torch
import torch.nn as nn

from .netforunet.resnet           import resnet18,resnet34,resnet50,resnet101,resnet152
from .netforunet.vgg              import vgg11,vgg13,vgg16,vgg19
from .attentions.attention_module import SENet_Block,ECANet_Block,CBAM_Block,ViT_Block,Self_Attention


class unetUp(nn.Module):
    def __init__(self, inputchannels, outputchannels, batch_norm=True):
        super(unetUp, self).__init__()
        if batch_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inputchannels, outputchannels, kernel_size=3, padding=1),
                nn.BatchNorm2d(outputchannels),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(outputchannels, outputchannels, kernel_size=3, padding=1),
                nn.BatchNorm2d(outputchannels),
                nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inputchannels, outputchannels, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(outputchannels, outputchannels, kernel_size=3, padding=1),
                nn.ReLU()
            )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x1, x2):
        x = torch.cat([x1, self.up(x2)], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Unet(nn.Module):
    def __init__(self, bands=6, num_classes=3, BN=True, backbone='vgg13', atten_type=None):
        super(Unet, self).__init__()
        self.atten_type = atten_type
        if backbone == 'vgg11':
            self.vgg    = vgg11(bands=bands, batch_norm=BN)
            in_filters  = [192, 384, 768, 1024]
        
        elif backbone == 'vgg13':
            self.vgg    = vgg13(bands=bands, batch_norm=BN)
            in_filters  = [192, 384, 768, 1024]

        elif backbone == 'vgg16':
            self.vgg    = vgg16(bands=bands, batch_norm=BN)
            in_filters  = [192, 384, 768, 1024]

        elif backbone == 'vgg19':
            self.vgg    = vgg19(bands=bands, batch_norm=BN)
            in_filters  = [192, 384, 768, 1024]

        elif backbone == "resnet18":
            self.resnet = resnet18(downsample_factor=32, bands=bands)
            in_filters  = [192, 320, 640, 768]

        elif backbone == "resnet34":
            self.resnet = resnet34(downsample_factor=32, bands=bands)
            in_filters  = [192, 320, 640, 768]

        elif backbone == "resnet50":
            self.resnet = resnet50(downsample_factor=32, bands=bands)
            in_filters  = [192, 512, 1024, 3072]
        
        elif backbone == "resnet101":
            self.resnet = resnet101(downsample_factor=32, bands=bands)
            in_filters  = [192, 512, 1024, 3072]

        elif backbone == "resnet152":
            self.resnet = resnet152(downsample_factor=32, bands=bands)
            in_filters  = [192, 512, 1024, 3072]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg11/13/16/19, resnet18/34/50/101/152.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]

        if atten_type==None:
            pass

        elif atten_type=="senet":
            self.attention_block1 = SENet_Block(in_channels=1024)
            self.attention_block2 = SENet_Block(in_channels=512)
            self.attention_block3 = SENet_Block(in_channels=256)
            self.attention_block4 = SENet_Block(in_channels=64)

        elif atten_type=="ecanet":
            self.attention_block1 = ECANet_Block(in_channels=1024)
            self.attention_block2 = ECANet_Block(in_channels=512)
            self.attention_block3 = ECANet_Block(in_channels=256)
            self.attention_block4 = ECANet_Block(in_channels=64)

        elif atten_type=="cbam":
            self.attention_block1 = CBAM_Block(in_channels=1024)
            self.attention_block2 = CBAM_Block(in_channels=512)
            self.attention_block3 = CBAM_Block(in_channels=256)
            self.attention_block4 = CBAM_Block(in_channels=64)

        elif atten_type=="vit":
            self.attention_block1 = ViT_Block(in_channels=1024, patch_size=1, d_model=512, num_heads=8, dropout=0.1)
            self.attention_block2 = ViT_Block(in_channels=512,  patch_size=2, d_model=256, num_heads=8, dropout=0.1)
            self.attention_block3 = ViT_Block(in_channels=256,  patch_size=4, d_model=128, num_heads=8, dropout=0.1)
            self.attention_block4 = ViT_Block(in_channels=128,  patch_size=8, d_model=64,  num_heads=8, dropout=0.1)

        elif atten_type=="self_atten":
            self.attention_block1 = Self_Attention(in_channels=1024, patch_size=8, d_model=512, num_heads=8, dropout=0.1)
            self.attention_block2 = Self_Attention(in_channels=512,  patch_size=4, d_model=256, num_heads=8, dropout=0.1)
            self.attention_block3 = Self_Attention(in_channels=256,  patch_size=2, d_model=128, num_heads=8, dropout=0.1)
            self.attention_block4 = Self_Attention(in_channels=64,   patch_size=1, d_model=64,  num_heads=8, dropout=0.1)
        
        else:
            raise ValueError('Unsupported attention mechanism :`{}`,You can only use senet, ecanet, cbam, vit, self_atten'.format(atten_type))

        self.up_concat4 = unetUp(in_filters[3], out_filters[3], batch_norm=BN)
        self.up_concat3 = unetUp(in_filters[2], out_filters[2], batch_norm=BN)
        self.up_concat2 = unetUp(in_filters[1], out_filters[1], batch_norm=BN)
        self.up_concat1 = unetUp(in_filters[0], out_filters[0], batch_norm=BN)

        if backbone in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(out_filters[0]),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(out_filters[0]),
                nn.ReLU()
            )
        else:
            self.up_conv = None

        self.final    = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone
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

    def forward(self, inputs):
        if self.backbone in ['vgg11','vgg13','vgg16','vgg19']:
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        
        if self.atten_type == None:
            pass
        else:
            feat4=self.attention_block1(feat4)
            feat3=self.attention_block2(feat3)
            feat2=self.attention_block3(feat2)
            feat1=self.attention_block4(feat1)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)        
        return final


# Test Model Structure and Outputsize
if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = Unet(bands=10, num_classes=3, BN=True, backbone="vgg11", atten_type=None).to(device)
    x               = torch.randn(2, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (10, 256, 256), batch_dim=0)