# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/11} ${13:35}
# @Function   : Segformer
# @Description: Realization of Segformer architecture

import torch
import torch.nn  as nn
import math
from functools      import partial
from timm.layers    import DropPath, trunc_normal_


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # 展平为序列 [B, N, E]
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1        = nn.Linear(in_features, hidden_features)
        self.act        = act_layer()
        self.fc2        = nn.Linear(hidden_features, out_features)
        self.drop       = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = head_dim ** -0.5
        
        self.q         = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv        = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio  = sr_ratio

        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes  = num_classes
        self.depths       = depths
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans,      embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        self.blocks1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], 
                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                 drop_path=dpr[i], sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        
        self.blocks2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], 
                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                 drop_path=dpr[i+depths[0]], sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        
        self.blocks3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], 
                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                 drop_path=dpr[i+depths[0]+depths[1]], sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        
        self.blocks4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], 
                 qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, 
                 drop_path=dpr[i+depths[0]+depths[1]+depths[2]], sr_ratio=sr_ratios[3])
            for i in range(depths[3])])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        outs = []
        
        # Stage 1
        x, H, W = self.patch_embed1(x), x.shape[2]//4, x.shape[3]//4
        for blk in self.blocks1:
            x = blk(x, H, W)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        
        # Stage 2
        x, H, W = self.patch_embed2(x.permute(0,2,1).reshape(B, -1, H, W)), H//2, W//2
        for blk in self.blocks2:
            x = blk(x, H, W)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        
        # Stage 3
        x, H, W = self.patch_embed3(x.permute(0,2,1).reshape(B, -1, H, W)), H//2, W//2
        for blk in self.blocks3:
            x = blk(x, H, W)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        
        # Stage 4
        x, H, W = self.patch_embed4(x.permute(0,2,1).reshape(B, -1, H, W)), H//2, W//2
        for blk in self.blocks4:
            x = blk(x, H, W)
        outs.append(x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous())
        
        return outs  # 返回多尺度特征图

class SegformerHead(nn.Module):
    """ Segformer解码器头 """
    def __init__(self, in_channels, embedding_dim, num_classes):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels
        
        # 1x1卷积减少通道数
        self.linear_c4 = nn.Conv2d(c4_in, embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(c3_in, embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(c2_in, embedding_dim, 1)
        self.linear_c1 = nn.Conv2d(c1_in, embedding_dim, 1)
        
        # 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim*4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 分类头
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # 上采样并减少通道
        n, _, h, w = c1.shape
        _c4 = self.linear_c4(c4)
        _c4 = nn.functional.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3)
        _c3 = nn.functional.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2)
        _c2 = nn.functional.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1)
        
        # 特征融合
        _c = self.fusion(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        # 分类
        x = self.classifier(_c)
        return x

class Segformer(nn.Module):
    def __init__(self, bands=3, num_classes=19, backbone='b0'):
        super().__init__()
        # 模型变体配置 (B0-B5)
        configs = {
            'b0': {'embed_dims': [32,64, 160,256], 'depths': [2,2,2, 2], 'num_heads': [1,2,5,8], 'mlp_ratios': [4,4,4,4]},
            'b1': {'embed_dims': [64,128,320,512], 'depths': [2,2,2, 2], 'num_heads': [1,2,5,8], 'mlp_ratios': [4,4,4,4]},
            'b2': {'embed_dims': [64,128,320,512], 'depths': [3,3,6, 3], 'num_heads': [1,2,5,8], 'mlp_ratios': [4,4,4,4]},
            'b3': {'embed_dims': [64,128,320,512], 'depths': [3,3,18,3], 'num_heads': [1,2,5,8], 'mlp_ratios': [4,4,4,4]},
            'b4': {'embed_dims': [64,128,320,512], 'depths': [3,8,27,3], 'num_heads': [1,2,5,8], 'mlp_ratios': [4,4,4,4]},
            'b5': {'embed_dims': [64,128,320,512], 'depths': [3,6,40,3], 'num_heads': [1,2,5,8], 'mlp_ratios': [4,4,4,4]}
        }
        config = configs[backbone]
        
        self.backbone = MixVisionTransformer(
            in_chans   = bands,
            embed_dims = config['embed_dims'],
            num_heads  = config['num_heads'],
            mlp_ratios = config['mlp_ratios'],
            depths     = config['depths'],
            sr_ratios  = [8, 4, 2, 1]
        )
        
        self.decode_head = SegformerHead(
            in_channels   = config['embed_dims'],
            embedding_dim = 256,
            num_classes   = num_classes
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
        features = self.backbone(x)
        logits   = self.decode_head(features)
        logits   = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits



if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = Segformer(bands=10, num_classes=3, backbone='b0').to(device)
    x               = torch.randn(2, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (10, 256, 256), batch_dim=0)