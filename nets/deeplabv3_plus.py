# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/03/15} ${19:33}
# @Function   : DeeplabV3+
# @Description: Realization of deeplabv3plus architecture


import torch
import torch.nn            as nn
import torch.nn.functional as F

from .netfordeeplabv3plus.xception    import xception
from .netfordeeplabv3plus.mobilenetv2 import mobilenetv2
from .netfordeeplabv3plus.resnet      import resnet18,resnet34,resnet50,resnet101,resnet152
from .netfordeeplabv3plus.vgg         import vgg11,vgg13,vgg16,vgg19
from .netfordeeplabv3plus.Inceptionv1 import GoogLeNet
from .netfordeeplabv3plus.Inceptionv2 import InceptionV2
from .netfordeeplabv3plus.Inceptionv3 import InceptionV3
from .netfordeeplabv3plus.Inceptionv4 import InceptionV4
from .attentions.attention_module     import SENet_Block,ECANet_Block,CBAM_Block,ViT_Block,Self_Attention


# Construct ASPP Block
class ASPP(nn.Module):
    def __init__(self, inputchannel, outputchannel, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(inputchannel, outputchannel, kernel_size=1, stride=1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(outputchannel),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inputchannel, outputchannel, kernel_size=3, stride=1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(outputchannel),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(inputchannel, outputchannel, kernel_size=3, stride=1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(outputchannel),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(inputchannel, outputchannel, kernel_size=3, stride=1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(outputchannel),
            nn.ReLU(inplace=True)
        )
        self.branch5_conv = nn.Conv2d(inputchannel, outputchannel, kernel_size=1, stride=1, padding=0, bias=True)
        self.branch5_bn   = nn.BatchNorm2d(outputchannel)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(outputchannel*5, outputchannel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(outputchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1          = self.branch1(x)
        conv3x3_1        = self.branch2(x)
        conv3x3_2        = self.branch3(x)
        conv3x3_3        = self.branch4(x)

        global_feature   = torch.mean(x, 2, True)
        global_feature   = torch.mean(global_feature, 3, True)
        global_feature   = self.branch5_conv(global_feature)
        global_feature   = self.branch5_bn(global_feature)
        global_feature   = self.branch5_relu(global_feature)
        global_feature   = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat      = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result           = self.conv_cat(feature_cat)
        return result


# Construct DeeplabV3+ architecture
class DeepLab(nn.Module):
    def __init__(self, bands=3, num_classes=100, backbone="mobilenet", atten_type=None):
        super(DeepLab, self).__init__()
        self.atten_type = atten_type
        if backbone == "xception":
            self.backbone       = xception(bands=bands)
            high_level_channels = 2048
            low_level_channels  = 128

        elif backbone == "mobilenet":
            self.backbone       = mobilenetv2(bands=bands)
            high_level_channels = 320
            low_level_channels  = 24

        elif backbone == "resnet":
            self.backbone       = resnet50(bands=bands)
            high_level_channels = 2048
            low_level_channels  = 256

        elif backbone == "vggnet":
            self.backbone       = vgg16(bands=bands)
            high_level_channels = 512
            low_level_channels  = 256

        elif backbone == "inception":
            self.backbone       = InceptionV3(bands=bands)
            high_level_channels = 2048
            low_level_channels  = 192
        else:
            raise ValueError('Unsupported backbone:`{}`,You can only use mobilenet, xception, vggnet, resnet, inception'.format(backbone))

        self.aspp = ASPP(inputchannel=high_level_channels, outputchannel=256, rate=16)

        if self.atten_type == None:
            pass

        elif self.atten_type == "senet":
            self.attention_block1 = SENet_Block(in_channels=256)
            self.attention_block2 = SENet_Block(in_channels=48)

        elif self.atten_type == "ecanet":
            self.attention_block1 = ECANet_Block(in_channels=256)
            self.attention_block2 = ECANet_Block(in_channels=48)

        elif self.atten_type == "cbam":
            self.attention_block1 = CBAM_Block(in_channels=256)
            self.attention_block2 = CBAM_Block(in_channels=48)

        elif self.atten_type == "vit":
            self.attention_block1 = ViT_Block(in_channels=256, patch_size=1, d_model=512, num_heads=8, dropout=0.1)
            self.attention_block2 = ViT_Block(in_channels=48,  patch_size=2, d_model=128, num_heads=8, dropout=0.1)

        elif self.atten_type == "self_atten":
            self.attention_block1 = Self_Attention(in_channels=256, patch_size=1, d_model=512, num_heads=8, dropout=0.1)
            self.attention_block2 = Self_Attention(in_channels=48,  patch_size=2, d_model=128, num_heads=8, dropout=0.1)

        else:
            raise ValueError('Unsupported attention mechanism :`{}`,You can only use senet, ecanet, cbam, vit, self_atten'.format(atten_type))

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.cls_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

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
        H, W                = x.size(2), x.size(3)
        low_feat, high_feat = self.backbone(x)
        high_feat           = self.aspp(high_feat)
        low_feat            = self.shortcut_conv(low_feat)
        
        if self.atten_type == None:
            pass
        else:
            high_feat     = self.attention_block1(high_feat)
            low_feat      = self.attention_block2(low_feat)

        high_feat         = F.interpolate(high_feat, size=(low_feat.size(2), low_feat.size(3)), mode='bilinear', align_corners=True)
        conv_out          = self.cat_conv(torch.cat((high_feat, low_feat), dim=1))
        output            = self.cls_conv(conv_out)
        output            = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=True)
        return output


# Test Model Structure and Outputsize
if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = DeepLab(bands=10, num_classes=3, backbone="mobilenet", atten_type=None).to(device)
    x               = torch.randn(2, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (10, 256, 256), batch_dim=0)