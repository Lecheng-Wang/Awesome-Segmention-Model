# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/03/15} ${22:00}
# @Function: Realization of inceptionv4 architecture

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class BasicConv2d(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inputchannel, outputchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(outputchannel, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_3a(nn.Module):
    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #原论文paddings是0, 此处应该是1
        self.conv    = BasicConv2d(64, 96, kernel_size=3, stride=2, padding=1) #原论文padding是0, 此处应该是1

    def forward(self, x):
        x0  = self.maxpool(x)
        x1  = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_4a(nn.Module):
    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64,  96, kernel_size=3, stride=1, padding=1) #原论文paddings是0, 此处应该是1
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64,  64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64,  64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64,  96, kernel_size=(3,3), stride=1, padding=1) #原论文paddings是0, 此处应该是1
        )

    def forward(self, x):
        x0  = self.branch0(x)
        x1  = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_5a(nn.Module):
    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv    = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1

    def forward(self, x):
        x0  = self.conv(x)
        x1  = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out

class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64,  96, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64,  96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96,  96, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1
        )
        
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192,  224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224,  256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192,  192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192,  224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224,  224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224,  256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192,  192, kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256,  256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256,  320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320,  320, kernel_size=3,     stride=2, padding=1)#原论文paddings是0, 此处应该是1
        )

        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#原论文paddings是0, 此处应该是1

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_C(nn.Module):
    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0    = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        
        self.branch1_0  = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384,  256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384,  256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch2_0  = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1  = BasicConv2d(384,  448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2  = BasicConv2d(448,  512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512,  256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512,  256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch3    = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0    = self.branch0(x)
        
        x1_0  = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1    = torch.cat((x1_1a, x1_1b), 1)

        x2_0  = self.branch2_0(x)
        x2_1  = self.branch2_1(x2_0)
        x2_2  = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2    = torch.cat((x2_3a, x2_3b), 1)

        x3    = self.branch3(x)

        out   = torch.cat((x0, x1, x2, x3), 1)
        return out

class InceptionV4(nn.Module):
    def __init__(self, bands=3, num_classes=1000):
        super(InceptionV4, self).__init__()
        self.conv1  = BasicConv2d(bands, 32, kernel_size=3, stride=2, padding=1) # 原论文padding是0, 此处应该改为1            
        self.conv2  = BasicConv2d(32,    32, kernel_size=3, stride=1, padding=1) # 原论文padding是0, 此处应该是1
        self.conv3  = BasicConv2d(32,    64, kernel_size=3, stride=1, padding=1)
        self.mixed1 = Mixed_3a()
        self.mixed2 = Mixed_4a()
        self.mixed3 = Mixed_5a()

        self.features = nn.Sequential(
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.8)
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.mixed1(x)
        x = self.mixed2(x)
        x = self.mixed3(x)
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classif(x) 
        return x

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

# Test Model Structure and Outputsize
if __name__ == "__main__":
    model         = InceptionV4(bands=3)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x             = torch.randn(1, 3, 224, 224).to(device)
    print("Input shape:", x.shape)
    output        = model(x)
    flops, params = profile(model, inputs=(x, ))
    print('flops: ', flops, 'params: ', params)
    print("Output shape:", output.shape)
    summary(model, (3, 224, 224)) 