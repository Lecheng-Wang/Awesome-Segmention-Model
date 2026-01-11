import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchAttention(nn.Module):
    def __init__(self, in_channels, patch_size=16, embed_dim=256, num_heads=8, position_encoding='learnable'):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.position_encoding = position_encoding
        
        # 1. Patch Embedding (使用卷积实现)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. 位置编码
        if position_encoding == 'learnable':
            self.position_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        elif position_encoding == 'sinusoidal':
            self.register_buffer('position_emb', None)
        else:
            raise ValueError("position_encoding must be 'learnable' or 'sinusoidal'")
        
        # 3. 多头自注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 4. 输出层 (生成patch权重)
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
            position_emb = position_emb.to(x.device).unsqueeze(0).expand(B, -1, -1)  # 扩展到整个批次
        else:
            position_emb = self.position_emb.expand(B, num_patches, -1)  # 扩展到整个批次
        
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


# 测试示例
if __name__ == "__main__":
    # 输入特征图 [batch, channels, height, width]
    x = torch.randn(2, 64, 224, 224)
    
    # 创建模块 (可学习位置编码)
    patch_attn_learnable = PatchAttention(
        in_channels=64,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        position_encoding='learnable'
    )
    
    # 创建模块 (正弦位置编码)
    patch_attn_sinusoidal = PatchAttention(
        in_channels=64,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        position_encoding='sinusoidal'
    )
    
    # 前向传播
    weights_learnable  = patch_attn_learnable(x)
    weights_sinusoidal = patch_attn_sinusoidal(x)
    
    print("输入形状:", x.shape)
    print("学习位置编码输出形状:", weights_learnable.shape)   # [2, 1, 14, 14]
    print("正弦位置编码输出形状:", weights_sinusoidal.shape)  # [2, 1, 14, 14]
    
    # 可视化注意力图
    import matplotlib.pyplot as plt
    
    # 获取第一个样本的注意力图
    attn_map1 = weights_learnable[0, 0].detach().cpu().numpy()
    attn_map2 = weights_sinusoidal[0, 0].detach().cpu().numpy()
    input_map = x[0, 0].detach().cpu().numpy()  # 取第一个通道
    
    # 修复可视化布局
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(input_map, cmap='gray')
    plt.title("Input feature map")
    plt.subplot(132)
    plt.imshow(attn_map1, cmap='hot')
    plt.title("Learnable Position Attention")
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(attn_map2, cmap='hot')
    plt.title("Sinusoidal Position Attention")
    plt.colorbar()
    plt.tight_layout()
    plt.show()