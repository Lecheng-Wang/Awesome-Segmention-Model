# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/03/16} ${17:14}
# @Function: Realization of vggnet architecture

import torch
import torch.nn   as nn

from torchinfo    import summary
from thop         import profile

class VGG(nn.Module):
    def __init__(self, features, net_type='vgg13',BN=True, num_classes=1000):
        super(VGG, self).__init__()
        self.features   = features
        self.BN         = BN
        self.net_type   = net_type
#        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
#        self.classifier = nn.Sequential(
#            nn.Linear(512 * 7 * 7, 4096),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(4096, 4096),
#            nn.ReLU(True),
#            nn.Dropout(0.5),
#            nn.Linear(4096, num_classes)
#       )

    def forward(self, x):
        if self.net_type=='vgg11' and self.BN==False:
        # vgg11 without layer
            feat1 = self.features[  :2 ](x)
            feat2 = self.features[ 2:5 ](feat1)
            feat3 = self.features[ 5:10](feat2)
            feat4 = self.features[10:15](feat3)
            feat5 = self.features[15:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]
        
        elif self.net_type=='vgg11' and self.BN==True:
        # vgg11 with BN layer
            feat1 = self.features[  :3 ](x)
            feat2 = self.features[ 3:7 ](feat1)
            feat3 = self.features[ 7:14](feat2)
            feat4 = self.features[14:21](feat3)
            feat5 = self.features[21:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]

        elif self.net_type=='vgg13' and self.BN==False:
        # vgg13 without BN layer
            feat1 = self.features[  :4 ](x)
            feat2 = self.features[4 :9 ](feat1)
            feat3 = self.features[9 :14](feat2)
            feat4 = self.features[14:19](feat3)
            feat5 = self.features[19:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]

        elif self.net_type=='vgg13' and self.BN==True:
        # vgg13 with BN layer
            feat1 = self.features[  :6 ](x)
            feat2 = self.features[6 :13](feat1)
            feat3 = self.features[13:20](feat2)
            feat4 = self.features[20:27](feat3)
            feat5 = self.features[27:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]

        elif self.net_type=='vgg16' and self.BN==False:
        #vgg16 without BN layer 
            feat1 = self.features[  :4 ](x)
            feat2 = self.features[4 :9 ](feat1)
            feat3 = self.features[9 :16](feat2)
            feat4 = self.features[16:23](feat3)
            feat5 = self.features[23:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]

        elif self.net_type=='vgg16' and self.BN==True:
        # vgg16 with BN layer 
            feat1 = self.features[  :6 ](x)
            feat2 = self.features[6 :13](feat1)
            feat3 = self.features[13:23](feat2)
            feat4 = self.features[23:33](feat3)
            feat5 = self.features[33:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]
            
        elif self.net_type=='vgg19' and self.BN==False:
        # vgg19 without BN layer 
            feat1 = self.features[  :4 ](x)
            feat2 = self.features[4 :9 ](feat1)
            feat3 = self.features[9 :18](feat2)
            feat4 = self.features[18:27](feat3)
            feat5 = self.features[27:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]

        elif self.net_type=='vgg19' and self.BN==True:
        # vgg19 with BN layer 
            feat1 = self.features[  :6 ](x)
            feat2 = self.features[6 :13](feat1)
            feat3 = self.features[13:26](feat2)
            feat4 = self.features[26:39](feat3)
            feat5 = self.features[39:-1](feat4)
            return [feat1, feat2, feat3, feat4, feat5]  
            
# used in images classify
#        x     = self.features(x)
#        x     = self.avgpool(x)
#        x     = torch.flatten(x, 1)
#        x     = self.classifier(x)
#        return x


def make_layers(layer_config, batch_norm=False, bands=3):
    in_channel = bands
    layers     = []
    for value in layer_config:
        if value == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channel, value, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(value), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channel = value
    return nn.Sequential(*layers)

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def vgg11(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg11_config, batch_norm=batch_norm, bands=bands),net_type='vgg11',BN=batch_norm,**kwargs)   
    return model

def vgg13(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg13_config, batch_norm=batch_norm, bands=bands),net_type='vgg13',BN=batch_norm,**kwargs)   
    return model

def vgg16(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg16_config, batch_norm=batch_norm, bands=bands),net_type='vgg16',BN=batch_norm,**kwargs)
    return model

def vgg19(bands=3, batch_norm=True, **kwargs):
    model = VGG(make_layers(vgg19_config, batch_norm=batch_norm, bands=bands),net_type='vgg19',BN=batch_norm,**kwargs)
    return model

# Test Model Structure and Outputsize
if __name__ == "__main__":
    model         = vgg19(bands=6, batch_norm=True)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x             = torch.randn(1, 6, 224, 224).to(device)  # Assume inputsize 3×224×224 RGB image
    print("Input shape:", x.shape)
    output        = model(x)
    flops, params = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', flops/1e9, 'Params(M): ', params/1e6)
    print("Output shape:", output[0].shape ,output[1].shape, output[2].shape, output[3].shape, output[4].shape)
    summary(model, (6, 224, 224), batch_dim=0)
