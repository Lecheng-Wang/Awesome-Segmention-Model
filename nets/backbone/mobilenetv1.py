# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2025/7/2  18:39
# @Function   : mobilenetv1
# @Description: 

import torch
import torch.nn            as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class MobileNet(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    
    def __init__(self, bands=3, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1  = nn.Conv2d(bands, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
    
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride     = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes  = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layers(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def mobilenetv1(bands=3, num_classes=1000):
    model = MobileNet(bands=bands, num_classes=num_classes)
    return model



if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = mobilenetv1(bands=3, num_classes=1000).to(device)
    x               = torch.randn(2, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(2, 3, 224, 224), device=device.type)