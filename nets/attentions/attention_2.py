# encoding = utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchInternalAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, position_encoding='learnable'):
        super().__init__()
        self.patch_size        = patch_size
        self.embed_dim         = embed_dim
        self.num_heads         = num_heads
        self.position_encoding = position_encoding
        self.in_channels       = in_channels
        
        if embed_dim % num_heads != 0:
            new_embed_dim  = math.ceil(embed_dim / num_heads) * num_heads
            embed_dim      = new_embed_dim
            self.embed_dim = embed_dim
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        
        if position_encoding == 'learnable':
            self.position_emb = nn.Parameter(torch.randn(1, patch_size*patch_size, embed_dim))
        elif position_encoding == 'sinusoidal':
            self.register_buffer('position_emb', self.sinusoidal_position_encoding(patch_size, embed_dim))
        else:
            raise ValueError("position_encoding must be 'learnable' or 'sinusoidal'")
        
        self.attn      = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) 
        self.to_weight = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        
        if H % P != 0 or W % P != 0:
            new_H = (H // P + 1) * P
            new_W = (W // P + 1) * P
            x     = F.pad(x, (0, new_W - W, 0, new_H - H))
            H, W  = new_H, new_W


        num_patches  = (H // P) * (W // P)        
        x            = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x            = x.unfold(1, P, P).unfold(2, P, P)   # [B, num_h, num_w, C, P, P]
        num_h, num_w = H // P, W // P
        x            = x.contiguous().view(B, num_patches, P, P, C)  # [B, num_patches, P, P, C]
        x            = x.permute(0, 1, 4, 2, 3).contiguous()  # [B, num_patches, C, P, P]
        
        patches_emb = self.patch_embed(x.view(B * num_patches, C, P, P))
        patches_emb = patches_emb.view(B, num_patches, self.embed_dim, P, P)
        
        seq = patches_emb.permute(0, 1, 3, 4, 2).contiguous()
        seq = seq.view(B, num_patches, P*P, self.embed_dim)
        
        position_emb = self.position_emb.expand(B, num_patches, -1, -1)
        seq          = seq + position_emb
        seq_flat     = seq.view(B * num_patches, P*P, self.embed_dim)
        
        attn_output, _ = self.attn(seq_flat, seq_flat, seq_flat)
        
        pixel_weights_flat = self.to_weight(attn_output)  # [B * num_patches, P*P, 1]
        pixel_weights = pixel_weights_flat.view(B, num_patches, P, P, 1)
        
        pixel_weights_full = pixel_weights.permute(0, 1, 2, 3, 4).contiguous().view(B, num_h, num_w, P, P, 1)
        pixel_weights_full = pixel_weights_full.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, 1)
        pixel_weights_full = pixel_weights_full.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        if pixel_weights_full.size(2) != H or pixel_weights_full.size(3) != W:
            pixel_weights_full = pixel_weights_full[:, :, :H, :W]
        
        return pixel_weights_full

    def sinusoidal_position_encoding(self, patch_size, embed_dim):
        num_positions = patch_size * patch_size
        position = torch.arange(0, num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pos_enc = torch.zeros(1, num_positions, embed_dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_enc


# 测试示例
if __name__ == "__main__":
    x = torch.randn(2, 64, 224, 224)
    
    patch_attn = PatchInternalAttention(
        in_channels=64,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        position_encoding='learnable'
    )
    
    pixel_weights = patch_attn(x)
    
    print("输入形状:", x.shape)
    print("像素权重形状:", pixel_weights.shape)
    
    # 可视化
    import matplotlib.pyplot as plt
    
    weight_map = pixel_weights[0, 0].detach().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(weight_map, cmap='hot')
    plt.title("Pixel Attention Weights")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('patch_internal_attention.png', dpi=300)
    plt.show()