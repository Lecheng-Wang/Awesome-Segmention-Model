# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : 2025/03/15  21:33
# @Function: Realization of inceptionv3 architecture

import torch
import torch.nn as nn

class InceptionV3(nn.Module):
    def __init__(self, bands=3, num_classes=1000, aux_logits=True, dropout=0.5):
        super(InceptionV3, self).__init__()
        self.aux_logits    = aux_logits
        self.Conv2d_1a_3x3 = BasicConv2d(bands, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1      = nn.MaxPool2d(3, 2, 1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192,kernel_size=3, stride=1, padding=1)
        self.maxpool2      = nn.MaxPool2d(3, 2, 1)

        self.Mixed_5b = InceptionA(192, 32)
        self.Mixed_5c = InceptionA(256, 64)
        self.Mixed_5d = InceptionA(288, 64)

        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, 128)
        self.Mixed_6c = InceptionC(768, 160)
        self.Mixed_6d = InceptionC(768, 160)
        self.Mixed_6e = InceptionC(768, 192)

        if aux_logits:
            self.aux = InceptionAux(768, num_classes)
        else:
            self.aux = None

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(dropout, True)
        self.fc       = nn.Linear(2048, num_classes)


    def forward(self, x):

        out = self.Conv2d_1a_3x3(x)
        out = self.Conv2d_2a_3x3(out)
        out = self.Conv2d_2b_3x3(out)
        out = self.maxpool1(out)
        out = self.Conv2d_3b_1x1(out)
        out = self.Conv2d_4a_3x3(out)
        out = self.maxpool2(out)

        out = self.Mixed_5b(out)
        out = self.Mixed_5c(out)
        out = self.Mixed_5d(out)

        out = self.Mixed_6a(out)

        out = self.Mixed_6b(out)
        out = self.Mixed_6c(out)
        out = self.Mixed_6d(out)
        out = self.Mixed_6e(out)

        if self.aux is not None:
            aux1 = self.aux(out)

        out  = self.Mixed_7a(out)
        out  = self.Mixed_7b(out)
        out  = self.Mixed_7c(out)

        out  = self.avgpool(out)
        out  = torch.flatten(out, 1)
        out  = self.dropout(out)
        aux2 = self.fc(out)

        return aux2, aux1



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1   = BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.avgpool     = nn.AvgPool2d((3, 3), (1, 1), (1, 1))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        out = torch.cat(out, 1)

        return out


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3      = BasicConv2d(in_channels, 384, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.maxpool        = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

    def forward(self, x):
        branch3x3    = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool  = self.maxpool(x)

        out = [branch3x3, branch3x3dbl, branch_pool]
        out = torch.cat(out, dim=1)
        return out


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1   = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch7x7_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))

        self.avgpool     = nn.AvgPool2d((3, 3), (1, 1), (1, 1))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = self.avgpool(x)
        branch_pool = self.branch_pool(branch_pool)

        out = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        out = torch.cat(out, 1)

        return out


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1   = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch3x3_2   = BasicConv2d(192, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.maxpool       = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

    def forward(self, x):
        branch3x3   = self.branch3x3_1(x)
        branch3x3   = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = self.maxpool(x)
        out         = [branch3x3, branch7x7x3, branch_pool]
        out         = torch.cat(out, dim=1)
        return out


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1       = BasicConv2d(in_channels, 320, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch3x3_1     = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch3x3_2a    = BasicConv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch3x3_2b    = BasicConv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch3x3dbl_1  = BasicConv2d(in_channels, 448, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch3x3dbl_2  = BasicConv2d(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.avgpool         = nn.AvgPool2d((3, 3), (1, 1), (1, 1))
        self.branch_pool     = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1    = self.branch1x1(x)
        branch3x3    = self.branch3x3_1(x)
        branch3x3    = torch.cat([self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = torch.cat([self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1)
        branch_pool  = self.avgpool(x)
        branch_pool  = self.branch_pool(branch_pool)
        out          = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        out          = torch.cat(out, dim=1)
        return out


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool1 = nn.AvgPool2d((5, 5), (3, 3))
        self.conv0    = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv1    = BasicConv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(768, num_classes)
        self.conv1.stddev = 0.01
        self.fc.stddev    = 0.001

    def forward(self, x):
        out = self.avgpool1(x)
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.avgpool2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def inception_v3(bands=3, num_classes=1000, **kwargs):
    model = InceptionV3(bands, num_classes, **kwargs)
    return model


if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = inception_v3(bands=3, num_classes=1000).to(device)
    x               = torch.randn(2, 3, 256, 256).to(device)
    output,_          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 256, 256), batch_dim=0)