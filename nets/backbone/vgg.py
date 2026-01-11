# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/03/16 17:14
# @Function: Realization of vggnet architecture

import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features   = features
        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
		)

    def forward(self, x):
        x     = self.features(x)
        x     = self.avgpool(x)
        x     = torch.flatten(x, 1)
        x     = self.classifier(x)
        return x




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


configs = {
	'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def vggnet(bands=3, num_classes=1000, backbone='vgg11', batch_norm=True, **kwargs):
    model = VGG(make_layers(configs[backbone], batch_norm=batch_norm, bands=bands), num_classes=num_classes, **kwargs)   
    return model



# Test Model Structure and Outputsize
if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = vggnet(bands=3, num_classes=1000, backbone='vgg11').to(device)
    x               = torch.randn(1, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(1, 3, 224, 224), device=device.type)