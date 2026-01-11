# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/28} ${15:52}
# @Function   : SCTNet implementation
# @Description: Dual-branch network with Transformer and CNN branches


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import OrderedDict



class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, g=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )


class CFBlock(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.dwconv  = ConvBNReLU(dim, dim, k=3, p=1, g=dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, 1, bias=False)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, 1, bias=False)
        self.bn      = nn.BatchNorm2d(dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.bn(x)
        return x + shortcut


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        x        = self.proj(x)
        H_p, W_p = x.shape[2:]
        tokens   = x.flatten(2).transpose(1, 2)
        return tokens, H_p, W_p


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden_dim, dim), nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, in_ch=3, patch_size=16, embed_dim=512, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size)
        self.pos_emb     = nn.Parameter(torch.zeros(1, 4096, embed_dim))
        self.blocks      = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm        = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_emb, std=.02)

    def forward(self, x):
        tokens, H_p, W_p = self.patch_embed(x)
        tokens           = tokens + self.pos_emb[:, :tokens.size(1), :]
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens   = self.norm(tokens)
        feat_map = tokens.transpose(1, 2).reshape(x.size(0), -1, H_p, W_p)
        return feat_map


class SIAM(nn.Module):
    def __init__(self, c_cnn, c_trans):
        super().__init__()
        self.conv_trans = nn.Conv2d(c_trans, c_cnn, kernel_size=1, bias=False)

    def forward(self, f_cnn, f_trans):
        return f_cnn, self.conv_trans(f_trans)


class DAPPM(nn.Module):
    def __init__(self, in_c, inter_c, out_c):
        super().__init__()
        self.scale0      = ConvBNReLU(in_c, inter_c, k=1, p=0)
        self.scale1      = nn.Sequential(nn.AvgPool2d(5, stride=2, padding=2), ConvBNReLU(in_c, inter_c, k=1, p=0))
        self.scale2      = nn.Sequential(nn.AvgPool2d(9, stride=4, padding=4), ConvBNReLU(in_c, inter_c, k=1, p=0))
        self.scale3      = nn.Sequential(nn.AvgPool2d(17, stride=8, padding=8), ConvBNReLU(in_c, inter_c, k=1, p=0))
        self.scale4      = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNReLU(in_c, inter_c, k=1, p=0))
        self.process     = nn.ModuleList([ConvBNReLU(inter_c, inter_c) for _ in range(4)])
        self.compression = ConvBNReLU(inter_c * 5, out_c, k=1, p=0)
        self.shortcut    = ConvBNReLU(in_c, out_c, k=1, p=0)

    def forward(self, x):
        size   = x.shape[-2:]
        x_list = [self.scale0(x)]
        for i, scale in enumerate([self.scale1, self.scale2, self.scale3, self.scale4]):
            x_up = F.interpolate(scale(x), size, mode='bilinear', align_corners=False)
            x_list.append(self.process[i](x_up + x_list[-1]))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class SCTNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=3, backbone_dims=(64, 128, 256, 512), 
	                   patch_size=16, trans_embed_dim=512, trans_depth=6, trans_heads=8):
        super().__init__()
        # --- CNN backbone ---
        self.stem       = ConvBNReLU(in_ch, backbone_dims[0], k=7, s=2, p=3)
        self.cnn_stages = nn.ModuleList()
        in_dim          = backbone_dims[0]
        for dim in backbone_dims[1:]:
            self.cnn_stages.append(nn.Sequential(ConvBNReLU(in_dim, dim, k=3, s=2, p=1), CFBlock(dim), CFBlock(dim)))
            in_dim = dim
        self.cnn_out_dim = backbone_dims[-1]

        # --- Transformer branch ---
        self.transformer = SimpleTransformer(in_ch, patch_size, trans_embed_dim, trans_depth, trans_heads)
        self.siam        = SIAM(self.cnn_out_dim, trans_embed_dim)

        # --- DAPPM & SegHead ---
        self.spp      = DAPPM(self.cnn_out_dim, backbone_dims[0]*2, backbone_dims[1]*2)  # 512→256
        head_ch       = backbone_dims[1] + backbone_dims[1]*2  # 128 + 256 = 384
        self.seg_head = nn.Sequential(OrderedDict([
            ('conv1', ConvBNReLU(head_ch, 256, k=3, p=1)),
            ('drop', nn.Dropout2d(0.1)),
            ('conv_out', nn.Conv2d(256, num_classes, 1))
        ]))

    def forward(self, x, return_align_feat=False):
        ori_size   = x.shape[-2:]
        x1         = self.stem(x)                   # 1/2
        x2         = self.cnn_stages[0](x1)         # 1/4  (C=128)
        x3         = self.cnn_stages[1](x2)         # 1/8  (C=256)
        x4         = self.cnn_stages[2](x3)         # 1/16 (C=512)
        x5         = F.avg_pool2d(x4, 2, stride=2)  # 1/32 (C=512)
        x6         = self.spp(x5)                   # 1/32 (C=256)
        x6_up      = F.interpolate(x6, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x_cat      = torch.cat([x2, x6_up], dim=1)  # 1/8,  C=128+256=384
        logits_1_8 = self.seg_head(x_cat)
        logits     = F.interpolate(logits_1_8, size=ori_size, mode='bilinear', align_corners=False)

        if self.training and return_align_feat:
            f_trans                    = self.transformer(x)
            f_trans_up                 = F.interpolate(f_trans, size=x4.shape[-2:], mode='bilinear', align_corners=False)
            f_cnn_align, f_trans_align = self.siam(x4, f_trans_up)
            print(f_cnn_align.shape,f_trans_align.shape)
            return logits, f_cnn_align, f_trans_align
        return logits

# 1、L2损失函数对齐
#loss_align = F.mse_loss(f_cnn_aligned, f_trans_aligned)

# 2、KL散度对齐
#def channel_wise_distillation(f_cnn, f_trans, T=1.0):
#    s = F.softmax(f_cnn / T, dim=1)
#    t = F.softmax(f_trans / T, dim=1)
#    return F.kl_div(s.log(), t, reduction='batchmean') * (T ** 2)
#
#loss_align = channel_wise_distillation(f_cnn_aligned, f_trans_aligned)

# 3、Cosine相似度对齐
#cos = nn.CosineSimilarity(dim=1)
#loss_align = 1-cos(f_cnn_aligned, f_trans_aligned).mean()


if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = SCTNet(in_ch=6, num_classes=3).to(device)
    x               = torch.randn(2, 6, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)