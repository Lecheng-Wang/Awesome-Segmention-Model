# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${DATE} ${TIME}
# @Function   : function to 
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class PatchInternalAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, position_encoding='learnable'):
        super().__init__()
        self.patch_size        = patch_size
        self.embed_dim         = embed_dim
        self.num_heads         = num_heads
        self.position_encoding = position_encoding

        if embed_dim % num_heads != 0:
            new_embed_dim  = math.ceil(embed_dim / num_heads) * num_heads
            embed_dim      = new_embed_dim
            self.embed_dim = embed_dim
        
        # 1. Patch嵌入层
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        
        # 2. 位置编码
        if position_encoding == 'learnable':
            self.position_emb = nn.Parameter(torch.randn(1, patch_size*patch_size, embed_dim))
        elif position_encoding == 'sinusoidal':
            self.register_buffer('position_emb', self.sinusoidal_position_encoding(patch_size, embed_dim))
        else:
            raise ValueError("position_encoding must be 'learnable' or 'sinusoidal'")
        
        # 3. 多头自注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 4. 权重生成层
        self.to_weight = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # 5. 标准化层
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P          = self.patch_size
        
        if H % P != 0 or W % P != 0:
            new_H = (H // P) * P
            new_W = (W // P) * P
            x     = x[:, :, :new_H, :new_W]
            H, W  = new_H, new_W
        
        num_patches  = (H // P) * (W // P)
        num_h, num_w = H // P, W // P
        
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, num_patches, C, P, P)
        
        patches_emb = self.patch_embed(x.view(B * num_patches, C, P, P))
        patches_emb = patches_emb.view(B, num_patches, self.embed_dim, P, P)
        
        seq = patches_emb.permute(0, 1, 3, 4, 2).contiguous()
        seq = seq.view(B, num_patches, P*P, self.embed_dim)
        
        position_emb = self.position_emb.expand(B, num_patches, -1, -1)
        seq = seq + position_emb
        
        seq_flat       = seq.view(B * num_patches, P*P, self.embed_dim)
        seq_flat_norm  = self.norm(seq_flat)
        attn_output, _ = self.attn(seq_flat_norm, seq_flat_norm, seq_flat_norm)
        
        pixel_weights_flat = self.to_weight(attn_output)
        
        pixel_weights = pixel_weights_flat.view(B, num_patches, P, P)
        
        pixel_weights_full = torch.zeros(B, H, W, device=x.device)
        for i in range(num_h):
            for j in range(num_w):
                idx = i * num_w + j
                pixel_weights_full[:, i*P:(i+1)*P, j*P:(j+1)*P] = pixel_weights[:, idx]
        
        return pixel_weights_full.unsqueeze(1)  # [B, 1, H, W]

    def sinusoidal_position_encoding(self, patch_size, embed_dim):
        num_positions = patch_size * patch_size
        position = torch.arange(0, num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pos_enc = torch.zeros(1, num_positions, embed_dim)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_enc


class PatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, 
                 position_encoding='learnable'):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.position_encoding = position_encoding
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, 
                                     stride=patch_size)
        
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
        """输入: [B, C, H, W], 输出: [B, 1, H//P, W//P] patch级权重图"""
        B, C, H, W = x.shape
        P = self.patch_size
        
        # 确保尺寸兼容
        if H % P != 0 or W % P != 0:
            new_H = (H // P) * P
            new_W = (W // P) * P
            x = x[:, :, :new_H, :new_W]
            H, W = new_H, new_W
        
        # 1. 划分Patch并进行嵌入
        patches = self.patch_embed(x)  # [B, embed_dim, H//P, W//P]
        
        # 2. 展平为序列 [B, num_patches, embed_dim]
        patch_emb = patches.flatten(2).permute(0, 2, 1)  # [B, N, embed_dim]
        
        # 3. 添加位置编码
        if self.position_encoding == 'sinusoidal':
            num_patches = patch_emb.size(1)
            position_emb = self.sinusoidal_position_encoding(num_patches, self.embed_dim)
            position_emb = position_emb.to(x.device).unsqueeze(0)  # [1, N, embed_dim]
        else:
            num_patches = patch_emb.size(1)
            position_emb = self.position_emb.repeat(1, num_patches, 1)
        
        patch_emb = patch_emb + position_emb
        
        # 4. 多头自注意力计算
        attn_output, _ = self.attn(patch_emb, patch_emb, patch_emb)  # [B, N, embed_dim]
        
        # 5. 生成patch权重
        patch_weights = self.to_weight(attn_output)  # [B, N, 1]
        
        # 6. 重塑为空间权重图
        H_out = H // P
        W_out = W // P
        patch_weights = patch_weights.view(B, H_out, W_out, 1).permute(0, 3, 1, 2)  # [B, 1, H_out, W_out]
        
        return patch_weights

    def sinusoidal_position_encoding(self, num_patches, embed_dim):
        position = torch.arange(0, num_patches).unsqueeze(1)  # [num_patches, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pos_enc = torch.zeros(num_patches, embed_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        return pos_enc


class CombinedAttention(nn.Module):
    """结合patch间和patch内注意力的统一模型"""
    def __init__(self, in_channels, patch_size=16, 
                 intra_embed_dim=256, intra_num_heads=8,
                 inter_embed_dim=256, inter_num_heads=8,
                 position_encoding='learnable'):
        super().__init__()
        self.patch_size = patch_size
        
        # Patch内部注意力模块 (像素级)
        self.intra_attention = PatchInternalAttention(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=intra_embed_dim,
            num_heads=intra_num_heads,
            position_encoding=position_encoding
        )
        
        # Patch间注意力模块 (整体权重)
        self.inter_attention = PatchAttention(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=inter_embed_dim,
            num_heads=inter_num_heads,
            position_encoding=position_encoding
        )
        
        # 可学习的融合因子
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))
        
        # 最终特征变换
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        """输入: [B, C, H, W], 输出: [B, C, H, W] 增强的特征图"""
        # 1. 计算patch内的像素级权重
        intra_weights = self.intra_attention(x)  # [B, 1, H, W]
        
        # 2. 计算patch间的整体权重
        inter_weights = self.inter_attention(x)  # [B, 1, H//P, W//P]
        
        # 3. 上采样patch间权重到原始空间尺寸
        inter_weights_upsampled = F.interpolate(inter_weights, scale_factor=self.patch_size, 
                                               mode='nearest')  # [B, 1, H, W]
        
        # 4. 结合两种注意力权重
        combined_weights = self.alpha * intra_weights + self.beta * inter_weights_upsampled
        
        # 5. 使用组合权重增强特征图
        enhanced_features = x * combined_weights
        
        # 6. 应用最终的变换
        output = self.transform(enhanced_features)
        
        return output, intra_weights, inter_weights, combined_weights


# 测试示例
if __name__ == "__main__":
    # 创建测试输入
    x = torch.randn(2, 64, 224, 224)
    
    # 实例化组合注意力模块
    combined_attn = CombinedAttention(
        in_channels=64,
        patch_size=16,
        intra_embed_dim=256,
        intra_num_heads=8,
        inter_embed_dim=256,
        inter_num_heads=8,
        position_encoding='learnable'
    )
    
    # 前向传播
    output, intra_weights, inter_weights, combined_weights = combined_attn(x)
    
    print("输入形状:", x.shape)
    print("输出特征形状:", output.shape)
    print("Patch内权重形状:", intra_weights.shape)
    print("Patch间权重形状:", inter_weights.shape)
    print("组合权重形状:", combined_weights.shape)
    
    # 可视化结果
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 显示原始特征
    axes[0, 0].imshow(x[0, 0].detach().cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title("Original Feature Map")
    axes[0, 0].axis('off')
    
    # 显示patch内权重
    intra_img = intra_weights[0, 0].detach().cpu().numpy()
    im_intra = axes[0, 1].imshow(intra_img, cmap='hot')
    axes[0, 1].set_title("Intra-Patch Attention Weights")
    fig.colorbar(im_intra, ax=axes[0, 1])
    axes[0, 1].axis('off')
    
    # 显示patch间权重（上采样后）
    inter_img = F.interpolate(inter_weights, scale_factor=16, mode='nearest')[0, 0].detach().cpu().numpy()
    im_inter = axes[1, 0].imshow(inter_img, cmap='hot')
    axes[1, 0].set_title("Inter-Patch Attention Weights")
    fig.colorbar(im_inter, ax=axes[1, 0])
    axes[1, 0].axis('off')
    
    # 显示组合权重
    combined_img = combined_weights[0, 0].detach().cpu().numpy()
    im_combined = axes[1, 1].imshow(combined_img, cmap='hot')
    axes[1, 1].set_title("Combined Attention Weights")
    fig.colorbar(im_combined, ax=axes[1, 1])
    axes[1, 1].axis('off')
    
    # 显示增强后的特征
    axes[2, 0].imshow(output[0, 0].detach().cpu().numpy(), cmap='viridis')
    axes[2, 0].set_title("Enhanced Feature Map")
    axes[2, 0].axis('off')
    
    # 显示差异
    diff = torch.abs(x[0, 0] - output[0, 0]).detach().cpu().numpy()
    axes[2, 1].imshow(diff, cmap='plasma')
    axes[2, 1].set_title("Attention Difference")
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('combined_attention_results.png', dpi=300)
    plt.show()