# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/16} ${12:29}
# @Function   : FCN
# @Description: Realization of fcn8s/16s/32s architecture


import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_FCN(nn.Module):
    def __init__(self, in_channels=3):
        super(VGG16_FCN, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)  #1/2
        
        # Block2: 2个卷积层
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)  #1/4
        
        # Block3: 3个卷积层
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)  #1/8
        
        # Block4: 3个卷积层
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)  #1/16
        
        # Block5: 3个卷积层
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)  #1/32
        
        # 全连接层转换为卷积层
        self.fc6   = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7   = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
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
        x     = self.relu1_1(self.conv1_1(x))
        x     = self.relu1_2(self.conv1_2(x))
        x     = self.pool1(x)
        x     = self.relu2_1(self.conv2_1(x))
        x     = self.relu2_2(self.conv2_2(x))
        x     = self.pool2(x)
        x     = self.relu3_1(self.conv3_1(x))
        x     = self.relu3_2(self.conv3_2(x))
        x     = self.relu3_3(self.conv3_3(x))
        pool3 = x
        x     = self.pool3(x)
        x     = self.relu4_1(self.conv4_1(x))
        x     = self.relu4_2(self.conv4_2(x))
        x     = self.relu4_3(self.conv4_3(x))
        pool4 = x
        x     = self.pool4(x)
        x     = self.relu5_1(self.conv5_1(x))
        x     = self.relu5_2(self.conv5_2(x))
        x     = self.relu5_3(self.conv5_3(x))
        pool5 = x
        x     = self.pool5(x)
        x     = self.relu6(self.fc6(x))
        x     = self.drop6(x)
        x     = self.relu7(self.fc7(x))
        fc7   = self.drop7(x)
        
        return pool3, pool4, pool5, fc7


class FCN32s(nn.Module):
    def __init__(self, bands=3, num_classes=21):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes
        self.base        = VGG16_FCN(bands)
        self.score_fr    = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.upscore32   = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)
        self._initialize_upsample()

    def _initialize_upsample(self):
        kernel = self._bilinear_kernel(self.num_classes, self.num_classes, 64)
        self.upscore32.weight.data.copy_(kernel)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og     = (torch.arange(kernel_size).float() - center) / factor
        filt   = (1 - torch.abs(og)).unsqueeze(0)
        kernel = filt.t() @ filt
        kernel = kernel.expand(in_channels, out_channels, kernel_size, kernel_size)
        return kernel.detach()

    def forward(self, x):
        _, _, _, fc7 = self.base(x)
        h            = self.score_fr(fc7)
        h            = self.upscore32(h)
        _, _, hh, ww = x.size()
        return h[:, :, :hh, :ww]


class FCN16s(nn.Module):
    def __init__(self, bands=3, num_classes=21):
        super(FCN16s, self).__init__()
        self.num_classes = num_classes
        self.base        = VGG16_FCN(bands)
        self.score_fr    = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upscore2    = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore16   = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8, bias=False)
        self._initialize_upsample()

    def _initialize_upsample(self):
        kernel2  = self._bilinear_kernel(self.num_classes, self.num_classes, 4)
        kernel16 = self._bilinear_kernel(self.num_classes, self.num_classes, 32)
        self.upscore2.weight.data.copy_(kernel2)
        self.upscore16.weight.data.copy_(kernel16)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og     = (torch.arange(kernel_size).float() - center) / factor
        filt   = (1 - torch.abs(og)).unsqueeze(0)
        kernel = filt.t() @ filt
        kernel = kernel.expand(in_channels, out_channels, kernel_size, kernel_size)
        return kernel.detach()

    def forward(self, x):
        _, pool4, _, fc7 = self.base(x)
        h                = self.score_fr(fc7)
        h                = self.upscore2(h)
        score_pool4      = self.score_pool4(pool4)
        if h.size() != score_pool4.size():
            h = F.interpolate(
                h, 
                size=score_pool4.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        h = score_pool4 + h
        h = self.upscore16(h)
        _, _, hh, ww = x.size()
        return h[:, :, :hh, :ww]


class FCN8s(nn.Module):
    def __init__(self, bands=3, num_classes=21):
        super(FCN8s, self).__init__()
        self.num_classes   = num_classes
        self.base          = VGG16_FCN(bands)
        self.score_fr      = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4   = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3   = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upscore2      = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,  stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,  stride=2, padding=1, bias=False)
        self.upscore8      = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self._initialize_upsample()

    def _initialize_upsample(self):
        kernel2 = self._bilinear_kernel(self.num_classes, self.num_classes, 4)
        kernel8 = self._bilinear_kernel(self.num_classes, self.num_classes, 16)
        self.upscore2.weight.data.copy_(kernel2)
        self.upscore_pool4.weight.data.copy_(kernel2)
        self.upscore8.weight.data.copy_(kernel8)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og     = (torch.arange(kernel_size).float() - center) / factor
        filt   = (1 - torch.abs(og)).unsqueeze(0)
        kernel = filt.t() @ filt
        kernel = kernel.expand(in_channels, out_channels, kernel_size, kernel_size)
        return kernel.detach()

    def forward(self, x):
        pool3, pool4, _, fc7 = self.base(x)
        h                    = self.score_fr(fc7)
        h                    = self.upscore2(h)        
        score_pool4          = self.score_pool4(pool4)
        if h.size() != score_pool4.size():
            h = F.interpolate(
                h, 
                size=score_pool4.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        h           = score_pool4 + h
        h           = self.upscore_pool4(h)
        score_pool3 = self.score_pool3(pool3)
        if h.size() != score_pool3.size():
            h = F.interpolate(
                h, 
                size=score_pool3.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        h            = score_pool3 + h
        h            = self.upscore8(h)
        _, _, hh, ww = x.size()
        return h[:, :, :hh, :ww]

if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = FCN8s(bands=6, num_classes=3).to(device)
    x               = torch.randn(2, 6, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)
