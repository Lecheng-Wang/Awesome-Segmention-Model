# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/03/15  21:33
# @Function: Realization of inceptionv2 architecture

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class BasicConv2d(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inputchannel, outputchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(outputchannel)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inceptionv2Module(nn.Module):
    def __init__(self, inputchannel, out_1x1, out_3x3reduce, out_3x3, double_out_3x3reduce, double_out_3x3, out_pool):
        super(Inceptionv2Module, self).__init__()
        # 1x1 branch
        self.branch1 = BasicConv2d(inputchannel, out_1x1, kernel_size=1)
        
        # 1x1 to 3x3 branch
        self.branch2 = nn.Sequential(
            BasicConv2d(inputchannel,  out_3x3reduce, kernel_size=1),
            BasicConv2d(out_3x3reduce, out_3x3,       kernel_size=3, padding=1)
        )
        
        # 1x1 to 3x3 to 3x3 branch (replaces 5x5)
        self.branch3 = nn.Sequential(
            BasicConv2d(inputchannel,         double_out_3x3reduce, kernel_size=1),
            BasicConv2d(double_out_3x3reduce, double_out_3x3reduce, kernel_size=3, padding=1),
            BasicConv2d(double_out_3x3reduce, double_out_3x3,       kernel_size=3, padding=1)
        )
        
        # 3x3 pool to 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inputchannel, out_pool, kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionV2(nn.Module):
    def __init__(self, bands=3, num_classes=1000):
        super(InceptionV2, self).__init__()
        self.stem = nn.Sequential(
            BasicConv2d(bands, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(64,  64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.inception3a = Inceptionv2Module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inceptionv2Module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inceptionv2Module(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inceptionv2Module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inceptionv2Module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inceptionv2Module(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inceptionv2Module(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = Inceptionv2Module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inceptionv2Module(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout     = nn.Dropout(0.2)
        self.fc          = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



# Test Model Structure and Outputsize
if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = InceptionV2(bands=3, num_classes=1000).to(device)
    x               = torch.randn(2, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 224, 224), batch_dim=0)
