# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/10} ${16:07}
# @Function   : SETR
# @Description: Realization of SETR architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.img_size   = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.n_patches  = (img_size // patch_size) ** 2
        self.proj       = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, d_model, H/patch, W/patch)
        x = x.flatten(2)  # (B, d_model, N)
        x = x.transpose(1, 2)  # (B, N, d_model)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.attn   = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with pre-norm
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.dropout(attn_output)
        
        # MLP with pre-norm
        norm_x = self.norm2(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, depth, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class SETR_PUP_Decoder(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        # Progressive upsampling with 4 stages
        self.stages = nn.ModuleList()
        channels = [in_dim, 256, 128, 64, 32]
        
        for i in range(4):
            self.stages.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], 1),
                nn.GroupNorm(8, channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            ))
        
        self.final_conv = nn.Conv2d(channels[-1], n_classes, kernel_size=1)

    def forward(self, x, img_size):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).view(B, C, H, W)  # Convert to 2D feature map
        
        # Progressive upsampling
        for stage in self.stages:
            x = stage(x)
        
        # Final convolution and resize to original image size
        x = self.final_conv(x)
        x = F.interpolate(x, size=img_size, mode='bilinear', align_corners=True)
        return x

class SETR(nn.Module):
    def __init__(self, img_size=224, patch_size=16, bands=3, num_classes=21, backbone='Base', mlp_ratio=4.0, dropout=0.1, decoder_type='PUP'):
        super().__init__()
        configs = {
            'Base':  {'depth': 12, 'd_model': 768,  'num_heads': 12},
            'Large': {'depth': 24, 'd_model': 1024, 'num_heads': 16}
        }
        config           = configs[backbone]
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, bands, config["d_model"])
        self.n_patches   = self.patch_embed.n_patches
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.n_patches, config["d_model"]))
        self.encoder     = TransformerEncoder(config["d_model"], config["depth"], config["num_heads"], mlp_ratio, dropout)
        
        if decoder_type == 'PUP':
            self.decoder = SETR_PUP_Decoder(config["d_model"], num_classes)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        img_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        x = self.encoder(x)
        x = self.decoder(x, img_size)
        return x

if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = SETR(img_size=256, bands=10, num_classes=3, backbone='Base').to(device)
    x               = torch.randn(1, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(1, 10, 256, 256), device=device.type)