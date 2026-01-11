# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/14} ${21:52}
# @Function   : SegNet 
# @Description: Realization of SegNet architecture

import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, bands=3, num_classes=21):
        super(SegNet, self).__init__()
        
        # 编码器 (VGG16结构)
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.enc_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # 解码器
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec_block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        )
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
        # 编码器路径
        x = self.enc_block1(x)
        size1 = x.size()
        x, indices1 = self.pool1(x)
        
        x = self.enc_block2(x)
        size2 = x.size()
        x, indices2 = self.pool2(x)
        
        x = self.enc_block3(x)
        size3 = x.size()
        x, indices3 = self.pool3(x)
        
        x = self.enc_block4(x)
        size4 = x.size()
        x, indices4 = self.pool4(x)
        
        x = self.enc_block5(x)
        size5 = x.size()
        x, indices5 = self.pool5(x)
        
        # 解码器路径
        x = self.unpool5(x, indices5, output_size=size5)
        x = self.dec_block5(x)
        
        x = self.unpool4(x, indices4, output_size=size4)
        x = self.dec_block4(x)
        
        x = self.unpool3(x, indices3, output_size=size3)
        x = self.dec_block3(x)
        
        x = self.unpool2(x, indices2, output_size=size2)
        x = self.dec_block2(x)
        
        x = self.unpool1(x, indices1, output_size=size1)
        x = self.dec_block1(x)
        
        return x

if __name__ == "__main__":
    from torchinfo     import summary
    from thop          import profile
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = SegNet(bands=6, num_classes=3).to(device)
    x             = torch.randn(2, 6, 256, 256).to(device)
    output        = model(x)
    flops, params = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)