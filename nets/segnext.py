# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/29} ${21:44}
# @Function   : SegNeXt
# @Description: Realization of SegNeXt network

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch0 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        )
        self.conv_fusion = nn.Conv2d(channels, channels, kernel_size=1)
        self.att         = nn.Sequential(nn.Conv2d(channels, channels // 4, 1),
		                                 nn.BatchNorm2d(channels // 4),
										 nn.GELU(),
										 nn.Conv2d(channels // 4, channels, 1),
										 nn.Sigmoid()
										 )

    def forward(self, x):
        out = self.branch0(x) + self.branch1(x) + self.branch2(x) + self.branch3(x)
        out = self.conv_fusion(out)
        att = self.att(out)
        return x * att

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(dim * 4, dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MSCABlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.norm1     = nn.BatchNorm2d(dim)
        self.conv1     = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.att       = MSCA(dim)
        self.norm2     = nn.BatchNorm2d(dim)
        self.mlp       = MLP(dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = x + self.drop_path(self.att(self.conv1(self.norm1(x))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2), 
			nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
			nn.GELU()
        )

    def forward(self, x):
        return self.stem(x)

class SegNeXt(nn.Module):
    def __init__(self, bands=3, num_classes=21, backbone='B'):
        super().__init__()
        configs = {
            'T': {'base': 32, 'layers': [3, 3, 5,  2], 'channels': [32, 64,  160, 256], 'decoder_dim': 256},
            'S': {'base': 64, 'layers': [2, 2, 4,  2], 'channels': [64, 128, 320, 512], 'decoder_dim': 256},
            'B': {'base': 64, 'layers': [3, 3, 12, 3], 'channels': [64, 128, 320, 512], 'decoder_dim': 512},
            'L': {'base': 64, 'layers': [3, 5, 27, 3], 'channels': [64, 128, 320, 512], 'decoder_dim': 1024}
        }
        cfg           = configs[backbone]
        self.channels = cfg['channels']

        self.stem   = Stem(in_channels=bands, out_channels=self.channels[0])
        self.stage1 = self._make_stage(self.channels[0], self.channels[0], cfg['layers'][0], stride=1)
        self.stage2 = self._make_stage(self.channels[0], self.channels[1], cfg['layers'][1], stride=2)
        self.stage3 = self._make_stage(self.channels[1], self.channels[2], cfg['layers'][2], stride=2)
        self.stage4 = self._make_stage(self.channels[2], self.channels[3], cfg['layers'][3], stride=2)

        self.decoder = nn.Sequential(
            nn.Conv2d(sum(self.channels), cfg['decoder_dim'], kernel_size=1),
            nn.BatchNorm2d(cfg['decoder_dim']),
            nn.GELU(),
            nn.Conv2d(cfg['decoder_dim'], num_classes, kernel_size=1)
        )

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        if stride == 2:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), 
				nn.GELU()
            ))
        else:
            out_channels = in_channels
        for _ in range(blocks):
            layers.append(MSCABlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        size1 = x.shape[2:]
        x     = self.stem(x)
        s1    = self.stage1(x)
        s2    = self.stage2(s1)
        s3    = self.stage3(s2)
        s4    = self.stage4(s3)
        up    = lambda feat: F.interpolate(feat, size=size1, mode='bilinear', align_corners=True)
        fused = torch.cat([up(s1), up(s2), up(s3), up(s4)], dim=1)
        return self.decoder(fused)




if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = SegNeXt(bands=10, num_classes=3, backbone='T').to(device)
    x               = torch.randn(2, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (10, 256, 256), batch_dim=0)
