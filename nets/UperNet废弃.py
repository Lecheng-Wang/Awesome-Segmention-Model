# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/16} ${12:29}
# @Function   : UperNet
# @Description: Realization of UperNet based on swin_transformer architecture

import torch
import torch.nn            as nn
import torch.nn.functional as F
from timm.layers                  import DropPath, trunc_normal_
from timm.models.swin_transformer import SwinTransformerBlock

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1        = nn.Linear(in_features, hidden_features)
        self.act        = nn.GELU()
        self.fc2        = nn.Linear(hidden_features, out_features)
        self.drop       = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim       = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm_layer(4 * dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"输入的H={H}和W={W}必须能被2整除"
        x0 = x[:, :, 0::2, 0::2]  # (B, C, H/2, W/2)
        x1 = x[:, :, 1::2, 0::2]  # (B, C, H/2, W/2)
        x2 = x[:, :, 0::2, 1::2]  # (B, C, H/2, W/2)
        x3 = x[:, :, 1::2, 1::2]  # (B, C, H/2, W/2)
        x  = torch.cat([x0, x1, x2, x3], 1)  # (B, 4*C, H/2, W/2)
        x  = x.permute(0, 2, 3, 1).contiguous()  # (B, H/2, W/2, 4*C)
        x  = self.norm(x)
        x  = self.reduction(x)  # (B, H/2, W/2, 2*C)
        x  = x.permute(0, 3, 1, 2).contiguous()  # (B, 2*C, H/2, W/2)
        return x

class SwinTransformerBackbone(nn.Module):
    def __init__(self, img_size=512, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.embed_dim         = embed_dim
        self.img_size          = img_size
        self.input_resolutions = [
            (img_size // 4, img_size // 4),       # 阶段0
            (img_size // 8, img_size // 8),       # 阶段1
            (img_size // 16, img_size // 16),     # 阶段2
            (img_size // 32, img_size // 32)      # 阶段3
        ]
        self.stages = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                block = SwinTransformerBlock(
                    dim              = embed_dim * (2 ** i),
                    input_resolution = self.input_resolutions[i],
                    num_heads        = num_heads[i],
                    window_size      = 7,
                    shift_size       = 0 if (j % 2 == 0) else 7 // 2,
                    mlp_ratio        = 4.,
                    drop_path        = 0.1 if j > 0 else 0.0
                )
                stage_blocks.append(block)
            self.stages.append(nn.Sequential(*stage_blocks))
        
        self.downsamples = nn.ModuleList()
        for i in range(3):
            downsample = PatchMerging(dim = embed_dim * (2 ** i),norm_layer=nn.LayerNorm)
            self.downsamples.append(downsample)

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm        = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x          = self.patch_embed(x)
        B, C, H, W = x.shape
        x          = x.flatten(2).transpose(1, 2)
        x          = self.norm(x)
        H_ = W_    = int(x.shape[1] ** 0.5)
        x          = x.reshape(B, H_, W_, C)
        features   = []
        current_C  = self.embed_dim
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                features.append(x.permute(0, 3, 1, 2).contiguous())
                x         = x.permute(0, 3, 1, 2).contiguous()
                x         = self.downsamples[i](x)
                current_C = current_C * 2
                H_        = H_ // 2
                W_        = W_ // 2
                x         = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
            else:
                features.append(x.permute(0, 3, 1, 2).contiguous())
        
        return features

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.features = nn.ModuleList()
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
                )
            )
        self.out_channels = in_dim + len(bins) * reduction_dim
    
    def forward(self, x):
        x_size = x.size()
        out    = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs     = nn.ModuleList()
        
        for i in range(len(in_channels)):
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
            )
            
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features):
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        for i in range(len(laterals)-2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i+1], size=laterals[i].shape[2:], mode='bilinear', align_corners=True)

        outputs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        return outputs


class UperNet(nn.Module):
    def __init__(self, img_size=224, bands=3, num_classes=21, backbone='Swin_T'):
        super().__init__()
        configs = {
            'Swin_T': {'embed_dim': 96, 'depths': [2,2,6, 2], 'num_heads': [3,6, 12,24], 'in_channels': [96, 192,384,768]},
            'Swin_S': {'embed_dim': 96, 'depths': [2,2,18,2], 'num_heads': [3,6, 12,24], 'in_channels': [96, 192,384,768]},
            'Swin_B': {'embed_dim': 128,'depths': [2,2,18,2], 'num_heads': [4,8, 16,32], 'in_channels': [128,256,512,1024]},
            'Swin_L': {'embed_dim': 192,'depths': [2,2,18,2], 'num_heads': [6,12,24,48], 'in_channels': [192,384,768,1536]}
        }
        config = configs[backbone]
        self.backbone = SwinTransformerBackbone(
            img_size  = img_size,
            in_chans  = bands,
            embed_dim = config["embed_dim"],
            depths    = config["depths"],
            num_heads = config["num_heads"]
        )
        in_channels = config["in_channels"]

        self.ppm        = PPM(in_channels[-1], 512 // 4)
        ppm_out_dim     = self.ppm.out_channels
        fpn_in_channels = in_channels[:-1] + [ppm_out_dim]
        self.fpn        = FPN(fpn_in_channels, out_channels=256)
        
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(256 * len(fpn_in_channels), 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x):
        features     = self.backbone(x)
        ppm_out      = self.ppm(features[-1])
        fpn_inputs   = features[:-1] + [ppm_out]
        fpn_outputs  = self.fpn(fpn_inputs)
        target_size  = fpn_outputs[0].shape[2:]
        fpn_features = []

        for feature in fpn_outputs:
            fpn_features.append(F.interpolate(feature, size=target_size,mode='bilinear', align_corners=True))
        
        fused = torch.cat(fpn_features, dim=1)
        fused = self.fpn_fuse(fused)
        out   = self.cls_seg(fused)
        out   = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return out


if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = UperNet(num_classes=3, bands=6, img_size=256, backbone='Swin_L').to(device)
    x               = torch.randn(2, 6, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)
