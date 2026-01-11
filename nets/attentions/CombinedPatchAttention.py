# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2025/7/14 18:14
# @Function   : InnerPatchAttention and InterPatchAttention
# @Description: 
'''             Combine Inner-Patch and Inter-Patch Multi-Head Self-Attention to model both local and global
                contextual relationships in the sequence of image patches, 
                aiming to achieve finer segmentation results.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InterPatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, position_encoding='learnable'):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.position_encoding = position_encoding
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if position_encoding == 'learnable':
            self.position_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        elif position_encoding == 'sinusoidal':
            self.register_buffer('position_emb', None)
        else:
            raise ValueError("position_encoding must be 'learnable' or 'sinusoidal'")
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.to_weight = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W  = x.shape
        P           = self.patch_size
        patches     = self.patch_embed(x)
        num_patches = patches.size(2) * patches.size(3)
        patch_emb   = patches.flatten(2).permute(0, 2, 1)  # [B, N, embed_dim]
        
        if self.position_encoding == 'sinusoidal':
            position_emb = self.sinusoidal_position_encoding(num_patches, self.embed_dim)
            position_emb = position_emb.to(x.device).unsqueeze(0).expand(B, -1, -1)
        else:
            position_emb = self.position_emb.expand(B, num_patches, -1)
        
        patch_emb = patch_emb + position_emb
        
        attn_output, _ = self.attn(patch_emb, patch_emb, patch_emb)  # [B, N, embed_dim]
        patch_weights  = self.to_weight(attn_output)  # [B, N, 1]
        H_out          = H // P
        W_out          = W // P
        patch_weights  = patch_weights.view(B, H_out, W_out, 1).permute(0, 3, 1, 2)
        
        return patch_weights


    def sinusoidal_position_encoding(self, num_patches, embed_dim):
        position         = torch.arange(0, num_patches).unsqueeze(1)
        div_term         = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_enc          = torch.zeros(num_patches, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc

    def get_attention_map(self, x):
        with torch.no_grad():
            return self.forward(x)


class InnerPatchAttention(nn.Module):
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
        
        position_emb   = self.position_emb.expand(B, num_patches, -1, -1)
        seq            = seq + position_emb
        seq_flat       = seq.view(B * num_patches, P*P, self.embed_dim)
        attn_output, _ = self.attn(seq_flat, seq_flat, seq_flat)
        
        pixel_weights_flat = self.to_weight(attn_output)  # [B * num_patches, P*P, 1]
        pixel_weights      = pixel_weights_flat.view(B, num_patches, P, P, 1)
        pixel_weights_full = pixel_weights.permute(0, 1, 2, 3, 4).contiguous().view(B, num_h, num_w, P, P, 1)
        pixel_weights_full = pixel_weights_full.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, 1)
        pixel_weights_full = pixel_weights_full.permute(0, 3, 1, 2)  # [B, 1, H, W]
        
        if pixel_weights_full.size(2) != H or pixel_weights_full.size(3) != W:
            pixel_weights_full = pixel_weights_full[:, :, :H, :W]
        
        return pixel_weights_full

    def sinusoidal_position_encoding(self, patch_size, embed_dim):
        num_positions       = patch_size * patch_size
        position            = torch.arange(0, num_positions).unsqueeze(1).float()
        div_term            = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_enc             = torch.zeros(1, num_positions, embed_dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_enc

    def get_attention_map(self, x):
        with torch.no_grad():
            return self.forward(x)



class CombinedPatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16,
                       inner_embed_dim=256, inner_num_heads=8,
                       inter_embed_dim=256, inter_num_heads=8, position_encoding='learnable'):
        super().__init__()
        self.patch_size      = patch_size
        self.inter_attention = InterPatchAttention(
                                    in_channels       = in_channels,
                                    patch_size        = patch_size,
                                    embed_dim         = inter_embed_dim,
                                    num_heads         = inter_num_heads,
                                    position_encoding = position_encoding)
        
        self.inner_attention = InnerPatchAttention(
                                    in_channels       = in_channels,
                                    patch_size        = patch_size,
                                    embed_dim         = inner_embed_dim,
                                    num_heads         = inner_num_heads,
                                    position_encoding = position_encoding)

    def forward(self, x):
        patch_weight_map       = self.inter_attention(x)
        pixel_weight_map       = self.inner_attention(x)
        patch_weight_upsampled = F.interpolate(patch_weight_map, size=(x.shape[2], x.shape[3]), mode='nearest')
        combined_attention     = patch_weight_upsampled * pixel_weight_map  # [B, 1, H, W]

        return combined_attention



if __name__ == "__main__":
    import gdal
    import matplotlib.pyplot as plt
    gdal.UseExceptions()


    #img_data = gdal.Open('167.tif').ReadAsArray()
    #x = torch.tensor(img_data).unsqueeze(0)
    x = torch.randn(1, 10, 256, 256)    
    model = CombinedPatchAttention(
        in_channels       = 10,
        patch_size        = 16,
        inner_embed_dim   = 512, inner_num_heads = 16,
        inter_embed_dim   = 512, inter_num_heads = 16,
        position_encoding = 'sinusoidal'
    )

    with torch.no_grad():
        patch_attn_map      = model.inter_attention(x)
        pixel_attn_map      = model.inner_attention(x)
        patch_attn_upsample = F.interpolate(patch_attn_map, size=(x.shape[2], x.shape[3]), mode='nearest') # nearest
        combined_attn_map   = patch_attn_upsample * pixel_attn_map  # [B, 1, H, W]
        enhanced_feat_map   = x * combined_attn_map

    input_feat        = x[0, 0].detach().cpu().numpy()                   # 原始特征图第0通道
    patch_attn        = patch_attn_map[0, 0].detach().cpu().numpy()      # patch-level attention
    patch_attn_up     = patch_attn_upsample[0, 0].detach().cpu().numpy() # 上采样后的patch attention
    pixel_attn        = pixel_attn_map[0, 0].detach().cpu().numpy()      # pixel attention
    final_attn        = combined_attn_map[0, 0].detach().cpu().numpy()   # 最终融合attention
    enhanced_feat     = enhanced_feat_map[0, 0].detach().cpu().numpy()   # 增强后的特征图第0通道


    plt.figure(figsize=(18, 10))

    plt.subplot(231)
    plt.imshow(input_feat, cmap='gray')
    plt.title("Input Feature Map(Channel0)")
    plt.clim(0, 1.6)  # 设置颜色映射范围
    plt.colorbar()

    plt.subplot(232)
    plt.imshow(patch_attn, cmap='hot')
    plt.title("InterPatch Attention(H/p x W/p)")
    plt.clim(0, 1)  # 设置颜色映射范围
    plt.colorbar()

    plt.subplot(233)
    plt.imshow(patch_attn_up, cmap='hot')
    plt.title("Upsampled InterPatch Attention")
    plt.clim(0, 1)  # 设置颜色映射范围
    plt.colorbar()

    plt.subplot(234)
    plt.imshow(pixel_attn, cmap='plasma')
    plt.title("InnerPatch Attention")
    plt.clim(0, 1)  # 设置颜色映射范围
    plt.colorbar()

    plt.subplot(235)
    plt.imshow(final_attn, cmap='plasma')
    plt.title("Combined Attention")
    plt.clim(0, 1)  # 设置颜色映射范围
    plt.colorbar()

    plt.subplot(236)
    plt.imshow(enhanced_feat, cmap='gray')
    plt.title("Feature After Attention")
    plt.clim(0, 1.6)  # 设置颜色映射范围
    plt.colorbar()

    plt.tight_layout()

    plt.subplots_adjust(
    top=0.946,
    bottom=0.057,
    left=0.036,
    right=0.981,
    hspace=0.222,
    wspace=0.146)
    plt.savefig("hierarchical_attention_full.png", dpi=300)
    plt.show()

