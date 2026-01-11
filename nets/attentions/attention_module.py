# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2025/6/8 16:38
# @Function   : attention block
# @Description: Reslization of SENet、ECANet、CBAM、Vit、Self_Attention

import torch
import torch.nn as nn
import math

# 1、SENet attention block realization
class SENet_Block(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(SENet_Block, self).__init__()
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.fc        = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio, False),
            nn.ReLU(),
            nn.Linear(in_channels//ratio, in_channels, False),
            nn.Sigmoid()
        )

    def forward (self, x):
        B, C, H, W = x.size()
        avg        = self.avg_pool(x).view([B, C])
        out        = self.fc(avg).view([B, C, 1, 1])
        return x*out



# 2、ECANet attention block realization
class ECANet_Block(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet_Block, self).__init__()
        kernel_size   = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size   = kernel_size if kernel_size % 2 else kernel_size + 1
        padding       = kernel_size//2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward (self, x):
        B, C, H, W = x.size()
        avg        = self.avg_pool(x).view([B, 1, C])
        out        = self.conv(avg)
        out        = self.sigmoid(out).view([B, C, 1, 1])
        return x*out



# 3、CBAM attention block realization
class Channel_Attention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(Channel_Attention, self).__init__()
        self.Max_pool = nn.AdaptiveMaxPool2d(1)
        self.Avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio, False),
            nn.ReLU(),
            nn.Linear(in_channels//ratio, in_channels, False)
        )
        self.sigmoid  = nn.Sigmoid()

    def forward(self,x):
        B, C, H, W   = x.size()
        Max_pool_out = self.Max_pool(x).view([B, C])
        Avg_pool_out = self.Avg_pool(x).view([B, C])
        Max_fc_out   = self.fc(Max_pool_out)
        Avg_fc_out   = self.fc(Avg_pool_out)
        out          = Max_fc_out + Avg_fc_out
        out          = self.sigmoid(out).view([B, C, 1, 1])
        return x*out

class Spatial_Attention(nn.Module):
    def __init__ (self, kernel_size=3):
        super(Spatial_Attention, self).__init__()
        padding      = kernel_size//2
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward (self, x):
        B, C, H, W     = x.size()
        Max_pool_out,_ = torch.max(x,  dim=1, keepdim=True)
        Mean_pool_out  = torch.mean(x, dim=1, keepdim=True)
        pool_out       = torch.cat([Max_pool_out, Mean_pool_out], dim=1)
        out            = self.conv(pool_out)
        out            = self.sigmoid(out)
        return x*out

class CBAM_Block(nn.Module):
    def __init__ (self, in_channels, ratio=4, kernel_size=3):
        super(CBAM_Block, self).__init__()
        self.Channel_Attention = Channel_Attention(in_channels, ratio)
        self.Spatial_Attention = Spatial_Attention(kernel_size)

    def forward (self, x):
        x = self.Channel_Attention(x)
        x = self.Spatial_Attention(x)
        return x



# 4、Vision Transformer attention block realization
class ViT_Block(nn.Module):
    def __init__(self, in_channels, patch_size, d_model=None, image_size=256, num_heads=4, dropout=0.1, downsample=False):
        super(ViT_Block, self).__init__()
        if isinstance(patch_size, int):
            self.patch_h = self.patch_w = patch_size
        else:
            self.patch_h, self.patch_w  = patch_size
        self.d_model             = d_model or in_channels * patch_size * patch_size
        self.image_size          = image_size
        self.in_channels         = in_channels
        self.patch_size          = patch_size
        self.num_heads           = num_heads
        self.downsample          = downsample
        self.patch_dim           = self.in_channels*self.patch_h*self.patch_w
        self.num_patches         = (self.image_size//self.patch_h)*(self.image_size//self.patch_w)

        if self.patch_dim != self.d_model:
            self.projection      = nn.Linear(self.patch_dim, self.d_model, bias=False)
            self.projection_back = nn.Linear(self.d_model, self.patch_dim, bias=False)
        else:
            self.projection      = nn.Identity()
            self.projection_back = nn.Identity()
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # Transformer layer
        self.transformer    = nn.TransformerEncoderLayer(
            d_model         = self.d_model,
            nhead           = self.num_heads,
            dim_feedforward = 4 * self.d_model,
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)


    def forward(self, x):
        B, C, H, W  = x.size()

        if H % self.patch_h != 0 or W % self.patch_w != 0:
            raise ValueError(f"Image size ({H},{W}) must be divisible by patch_size {self.patch_size}")

        # images divided into patches
        num_patches_h = H // self.patch_h
        num_patches_w = W // self.patch_w
        num_patches   = num_patches_h * num_patches_w
        patch_dim     = C * self.patch_h * self.patch_w
        
        x = x.view(B, C, num_patches_h, self.patch_h, num_patches_w, self.patch_w).permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, num_patches, patch_dim)        # [B, num_patches, patch_dim]
        x = self.projection(x)                       # [B, num_patches, patch_dim]->[B, num_patches, d_model]
        x = x + self.positional_encoding
        x = self.transformer(x)                      # [B, num_patches, d_model]
        x = self.projection_back(x)                  # [B, num_patches, d_model]  ->[B, num_patches, patch_dim]

        if self.downsample:
            x = x.view(B, num_patches_h, num_patches_w, C*self.patch_h*self.patch_w).permute(0, 3, 1, 2).contiguous()
            # [B, C*patch_h*patch_w, num_patches_h, num_patches_w]
        else:
            x = x.view(B, num_patches_h, num_patches_w, C, self.patch_h, self.patch_w).permute(0, 3, 1, 4, 2, 5).contiguous()
            x = x.view(B, C, H, W)
            # [B, C, num_patches_h, patch_h, num_patches_w, patch_w]
        return x




# Self_Attention block realization
class Self_Attention(nn.Module):
    def __init__(self, in_channels, patch_size, d_model=None, num_heads=4, dropout=0.1):
        super(Self_Attention, self).__init__()
        if isinstance(patch_size, int):
            self.patch_h = self.patch_w = patch_size
        else:
            self.patch_h, self.patch_w  = patch_size
        
        self.in_channels         = in_channels
        self.d_model             = d_model or in_channels
        self.patch_size          = patch_size
        self.num_heads           = num_heads
        
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj   = nn.Linear(in_channels, self.d_model, bias=False)
        self.k_proj   = nn.Linear(in_channels, self.d_model, bias=False)
        self.v_proj   = nn.Linear(in_channels, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, in_channels, bias=False)
        
        self.scale    = (self.d_model // num_heads) ** -0.5
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % self.patch_h != 0 or W % self.patch_w != 0:
            raise ValueError(f"Image size ({H},{W}) must be divisible by patch_size {self.patch_size}")
        # images divided into patches 
        num_patches_h = H // self.patch_h
        num_patches_w = W // self.patch_w
        num_patches   = num_patches_h * num_patches_w
        
        x       = x.view(B, C, num_patches_h, self.patch_h, num_patches_w, self.patch_w)
        x       = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, num_patches_h, num_patches_w, patch_h, patch_w, C]
        patches = x.view(B, num_patches, -1, C)             # [B, num_patches, patch_h*patch_w, C]
        
        Q = self.q_proj(patches)                            # [B, num_patches, patch_pixels, d_model]
        K = self.k_proj(patches)
        V = self.v_proj(patches)
        
        head_dim = self.d_model // self.num_heads
        Q = Q.view(B, num_patches, -1, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)  # [B, heads, num_patches, patch_pixels, head_dim]
        K = K.view(B, num_patches, -1, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)
        V = V.view(B, num_patches, -1, self.num_heads, head_dim).permute(0, 3, 1, 2, 4)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, heads, num_patches, patch_pixels, patch_pixels]
        attn_probs  = self.dropout(torch.softmax(attn_scores, dim=-1))
        attn_output = torch.matmul(attn_probs, V)                        # [B, heads, num_patches, patch_pixels, head_dim]
        
        attn_output = attn_output.permute(0, 2, 3, 1, 4).contiguous()    # [B, num_patches, patch_pixels, heads, head_dim]
        attn_output = attn_output.view(B, num_patches, -1, self.d_model) # [B, num_patches, patch_pixels, d_model]
        
        output = self.out_proj(attn_output)                              # [B, num_patches, patch_pixels, in_channels]
        
        output = output.view(B, num_patches_h, num_patches_w, self.patch_h, self.patch_w, C)
        output = output.permute(0, 5, 1, 3, 2, 4).contiguous()           # [B, C, num_patches_h, patch_h, num_patches_w, patch_w]
        return output.view(B, C, H, W)


if __name__=='__main__':

    # test
    x = torch.randn(4, 64, 64, 64)
#    print(x[:,34,32,12])

    model1 = Self_Attention(64, 4, 32, 4, 0.1)
    out1 = model1(x)
#    print(out1[:,34,32,12])

    model2 = ViT_Block(in_channels=64, patch_size=2, d_model=32, image_size=64, num_heads=4, dropout=0.1, downsample=True)
    out2 = model2(x)
#    print(out2[:,34,32,12])

    print(out1.shape)
    print(out2.shape)

