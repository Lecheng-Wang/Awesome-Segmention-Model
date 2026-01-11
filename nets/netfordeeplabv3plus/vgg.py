# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/03/16  17:14
# @Function: Realization of vggnet architecture

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features   = features
#        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
#        self.classifier = nn.Sequential(
#            nn.Linear(512 * 7 * 7, 4096),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(4096, 4096),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(4096, num_classes)
#		)
        self._initialize_weights()

    def forward(self, x):

## used in images classify
#        x     = self.features(x)
#        x     = self.avgpool(x)
#        x     = torch.flatten(x, 1)
#        x     = self.classifier(x)
#        return x


## used in deeplabv3+ based on vgg11 
#        low_level_feature  = self.features[  :14](x)
#        high_level_feature = self.features[14:-1](low_level_feature)
#        return low_level_feature, high_level_feature

## used in deeplabv3+ based on vgg13 
#        low_level_feature  = self.features[  :20](x)
#        high_level_feature = self.features[20:-1](low_level_feature)
#        return low_level_feature, high_level_feature

# used in deeplabv3+ based on vgg16 
        low_level_feature  = self.features[  :23](x)
        high_level_feature = self.features[23:-1](low_level_feature)
        return low_level_feature, high_level_feature

## used in deeplabv3+ based on vgg19
#        low_level_feature  = self.features[  :26](x)
#        high_level_feature = self.features[26:-1](low_level_feature)
#        return low_level_feature, high_level_feature

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


def make_layers(layer_config, batch_norm=False, bands=3):
    in_channel = bands
    layers     = []
    for value in layer_config:
        if value == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channel, value, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channel = value
    return nn.Sequential(*layers)

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def vgg11(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg11_config, batch_norm=batch_norm, bands=bands), **kwargs)   
    return model

def vgg13(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg13_config, batch_norm=batch_norm, bands=bands), **kwargs)   
    return model

def vgg16(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg16_config, batch_norm=batch_norm, bands=bands), **kwargs)   
    return model

def vgg19(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg19_config, batch_norm=batch_norm, bands=bands), **kwargs)   
    return model

# Test Model Structure and Outputsize
if __name__ == "__main__":
    model         = vgg16(bands=3, batch_norm=True)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x             = torch.randn(1, 3, 224, 224).to(device)  # Assume inputsize 3×224×224 RGB image
    print("Input shape:", x.shape)
    output        = model(x)
    flops, params = profile(model, inputs=(x, ))
    print('flops: ', flops/1e9, 'params: ', params/1e6)
    print("Low level output shape:", output[0].shape)
    print("High level output shape:", output[1].shape)
    summary(model, (3, 224, 224))