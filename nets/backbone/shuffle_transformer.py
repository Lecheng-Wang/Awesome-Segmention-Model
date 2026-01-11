# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/6/29} ${20:35}
# @Function   : shuffle transformer
# @Description: Realization of shuffle transformer network

import torch
from torch       import nn
from einops      import rearrange, repeat
from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0., stride=False):
        super().__init__()
        self.stride     = stride
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1        = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act        = act_layer()
        self.fc2        = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop       = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads              = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim                    = dim // self.num_heads
        self.ws                     = window_size
        self.shuffle                = shuffle
        self.scale                  = qk_scale or head_dim ** -0.5
        self.to_qkv                 = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.attn_drop              = nn.Dropout(attn_drop)
        self.proj                   = nn.Conv2d(dim, dim, 1)
        self.proj_drop              = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h        = torch.arange(self.ws)
            coords_w        = torch.arange(self.ws)
            coords          = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index   = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv        = self.to_qkv(x)

        if self.shuffle:
            q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        else:
            q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        if self.shuffle:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
        else:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class Block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=1, shuffle=False, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, stride=False, relative_pos_embedding=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(
            dim, num_heads = num_heads, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop      = attn_drop, proj_drop=drop, relative_pos_embedding=relative_pos_embedding)

        self.local_conv = nn.Conv2d(dim, dim, window_size, 1, 0, groups=dim, bias=qkv_bias)
        pad_total       = window_size - 1
        pad_left        = pad_total // 2
        pad_right       = pad_total - pad_left
        self.pad        = (pad_left, pad_right, pad_left, pad_right)
        
        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2      = norm_layer(dim)
        mlp_hidden_dim  = int(dim * mlp_ratio)
        self.mlp        = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop, stride=stride)
        self.norm3      = norm_layer(dim)

    def forward(self, x):
        x        = x + self.drop_path(self.attn(self.norm1(x)))
        x_norm   = self.norm2(x)
        x_padded = nn.functional.pad(x_norm, self.pad)
        x_local  = self.local_conv(x_padded)
        x        = x + x_local
        x        = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim       = dim
        self.out_dim   = out_dim
        self.norm      = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class StageModule(nn.Module):
    def __init__(self, layers, dim, out_dim, num_heads, window_size=1, shuffle=True, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, relative_pos_embedding=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        if dim != out_dim:
            self.patch_partition = PatchMerging(dim, out_dim)
        else:
            self.patch_partition = None

        num = layers // 2
        self.layers = nn.ModuleList([])
        for idx in range(num):
            the_last = (idx==num-1)
            self.layers.append(nn.ModuleList([
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=False, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding),
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
                      relative_pos_embedding=relative_pos_embedding)
            ]))

    def forward(self, x):
        if self.patch_partition:
            x = self.patch_partition(x)
            
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=3, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, inter_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x


class ShuffleTransformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, token_dim=32, embed_dim=96, mlp_ratio=4., layers=[2,2,6,2], num_heads=[3,6,12,24], 
                relative_pos_embedding=True, shuffle=True, window_size=7, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                has_pos_embed=False, **kwargs):
        super().__init__()
        self.num_classes   = num_classes
        self.num_features  = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.has_pos_embed = has_pos_embed
        dims = [i*32 for i in num_heads]

        self.to_token = PatchEmbedding(in_chans=in_chans, inter_channel=token_dim, out_channels=embed_dim)



        if self.has_pos_embed:
            num_patches = (img_size // 4) ** 2
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        self.stage1 = StageModule(layers[0], embed_dim, dims[0], num_heads[0], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2],
                                  relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3], window_size=window_size, shuffle=shuffle, 
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3],
                                  relative_pos_embedding=relative_pos_embedding)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dims[3], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x          = self.to_token(x)
        b, c, h, w = x.shape

        if self.has_pos_embed:
            x = x + self.pos_embed.view(1, h, w, c).permute(0, 3, 1, 2)
            x = self.pos_drop(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def Shuffle_Transformer(img_size=224, wind_size=7, in_chans=3, num_classes=1000, backbone='Tiny'):
    configs = {
        'Tiny':  {'embed_dim': 96,  'layers': [2, 2, 6,  2], 'num_heads': [3, 6, 12, 24]},
        'Small': {'embed_dim': 96,  'layers': [2, 2, 18, 2], 'num_heads': [3, 6, 12, 24]},
        'Base':  {'embed_dim': 128, 'layers': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32]}
    }
    config = configs[backbone]
    model  = ShuffleTransformer(img_size=img_size, in_chans=in_chans, num_classes=num_classes, 
                               embed_dim=config["embed_dim"], layers=config["layers"], num_heads=config["num_heads"],
                               window_size=wind_size,
                               token_dim=32, mlp_ratio=4., relative_pos_embedding=True, shuffle=True, qkv_bias=False, qk_scale=None, 
                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0., has_pos_embed=False)

    return model


if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = Shuffle_Transformer(img_size=224, wind_size=7, in_chans=3, num_classes=1000, backbone='Base').to(device)
    x               = torch.randn(2, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 224, 224), batch_dim=0)