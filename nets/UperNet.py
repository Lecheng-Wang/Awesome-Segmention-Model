# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${DATE} ${TIME}
# @Function   : function to 
# @Description: 



import torch
import torch.nn as nn
import numpy    as np
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_


# Common 3×3_Conv Block
def conv3x3(inputchannel, outputchannel, stride=1, groups=1, dilation=1):
    padding = dilation
    return nn.Conv2d(
        inputchannel, outputchannel, kernel_size=3, stride=stride,
        padding=padding, groups=groups, dilation=dilation, bias=False
    )

# Common 1×1_Conv Block
def conv1x1(inputchannel, outputchannel, stride=1):
    return nn.Conv2d(
        inputchannel, outputchannel, kernel_size=1, 
        stride=stride, bias=False
    )

# Block Type1(Which is commonly used in Less layers ResNet, Such as ResNet18 and ResNet34)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inputchannel, outputchannel, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # 3×3 Conv
        self.conv1 = conv3x3(inputchannel, outputchannel, stride, groups, dilation)
        self.bn1 = norm_layer(outputchannel)
        self.relu = nn.ReLU(inplace=True)
        # 3×3 Conv
        self.conv2 = conv3x3(outputchannel, outputchannel, 1, groups, dilation)
        self.bn2 = norm_layer(outputchannel)
        # Downsample Section in the end of Block Group.
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # Residual Connect
        out += identity
        out = self.relu(out)

        return out

# Block Type2(Which is commonly used in More layers ResNet, Such as ResNet50、ResNet101 and ResNet152)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inputchannel, outputchannel, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(outputchannel * (base_width / 64.)) * groups
        # 1×1 Conv 
        self.conv1 = conv1x1(inputchannel, width)
        self.bn1 = norm_layer(width)
        # 3×3 Conv
        self.conv2 = conv3x3(width, width, stride, groups, dilation=dilation)
        self.bn2 = norm_layer(width)
        # 1x1 Conv
        self.conv3 = conv1x1(width, outputchannel * self.expansion)
        self.bn3 = norm_layer(outputchannel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        # Residual Connect
        out += identity
        out = self.relu(out)

        return out

# Construct the ResNet
class ResNet(nn.Module):
    def __init__(self, block_type, block_config, num_classes=1000, bands=3, downsample_factor=32):
        super(ResNet, self).__init__()
        self.inputchannel      = 64
        self.downsample_factor = downsample_factor
        
        # Initial layers
        self.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 修正padding
        
        # Configure layers based on downsample factor
        if downsample_factor == 32:
            # 标准32倍下采样
            self.layer1 = self._make_layer(block_type, 64, block_config[0], stride=1, dilation=1)
            self.layer2 = self._make_layer(block_type, 128, block_config[1], stride=2, dilation=1)
            self.layer3 = self._make_layer(block_type, 256, block_config[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(block_type, 512, block_config[3], stride=2, dilation=1)
        elif downsample_factor == 16:
            # 16倍下采样
            self.layer1 = self._make_layer(block_type, 64, block_config[0], stride=1, dilation=1)
            self.layer2 = self._make_layer(block_type, 128, block_config[1], stride=2, dilation=1)
            self.layer3 = self._make_layer(block_type, 256, block_config[2], stride=2, dilation=1)
            self.layer4 = self._make_layer(block_type, 512, block_config[3], stride=1, dilation=2)
        elif downsample_factor == 8:
            # 8倍下采样
            self.layer1 = self._make_layer(block_type, 64, block_config[0], stride=1, dilation=1)
            self.layer2 = self._make_layer(block_type, 128, block_config[1], stride=2, dilation=1)
            self.layer3 = self._make_layer(block_type, 256, block_config[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(block_type, 512, block_config[3], stride=1, dilation=4)
        else:
            raise ValueError("Unsupported downsample_factor. Use 8, 16, or 32.")
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, Block_type, outputchannel, blocks_num, stride=1, dilation=1):
        downsample = None
        groups     = 1
        base_width = 64
        
        # 当下采样或通道数变化时创建downsample模块
        if stride != 1 or self.inputchannel != outputchannel * Block_type.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inputchannel, outputchannel * Block_type.expansion, stride),
                nn.BatchNorm2d(outputchannel * Block_type.expansion)
            )
        
        layers = []
        # 第一个block处理步长和通道变化
        layers.append(Block_type(
            self.inputchannel, outputchannel, stride, downsample,
            groups, base_width, dilation
        ))
        self.inputchannel = outputchannel * Block_type.expansion
        
        # 后续blocks
        for _ in range(1, blocks_num):
            layers.append(Block_type(
                self.inputchannel, outputchannel, 
                stride=1, downsample=None,
                groups=groups, base_width=base_width,
                dilation=dilation
            ))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet50(bands=3, num_classes=1000, downsample_factor=32, **kwargs):
    model = ResNet(block_type=Bottleneck, block_config=[3, 4, 6, 3], num_classes=num_classes, bands=bands, downsample_factor=downsample_factor)    
    return model

# 下面是Swin_Transformer的部分实现
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x          = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows    = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv      = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v  = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W     = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask    = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W    = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim              = dim
        self.reduction        = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm             = norm_layer(4 * dim)

    def forward(self, x):
        H, W    = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x  = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x  = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x  = self.norm(x)
        x  = self.reduction(x)

        return x



class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, fused_window_process=False):

        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.depth            = depth
        self.blocks           = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                 num_heads=num_heads, window_size=window_size,
                                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                                 mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 drop=drop, attn_drop=attn_drop,
                                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                 norm_layer=norm_layer,
                                                 fused_window_process=fused_window_process)for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, features):
        for blk in self.blocks:
            x = blk(x)
        features.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size                = to_2tuple(img_size)
        patch_size              = to_2tuple(patch_size)
        patches_resolution      = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size           = img_size
        self.patch_size         = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches        = patches_resolution[0] * patches_resolution[1]

        self.in_chans           = in_chans
        self.embed_dim          = embed_dim
        self.proj               = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x



class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, 
                 fused_window_process=False, only_features=True, **kwargs):
        super().__init__()

        self.num_classes  = num_classes
        self.num_layers   = len(depths)
        self.embed_dim    = embed_dim
        self.ape          = ape
        self.patch_norm   = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio    = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size   = img_size,
                                      patch_size = patch_size,
                                      in_chans   = in_chans,
                                      embed_dim  = embed_dim,
                                      norm_layer = norm_layer if self.patch_norm else None)
        num_patches             = self.patch_embed.num_patches
        patches_resolution      = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.only_features      = only_features

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm    = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        features = []

        for layer in self.layers:
            x = layer(x, features)
        if not self.only_features:
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            return x
        else:
            #print(len(features))
            return features

    def forward(self, x):
        x = self.forward_features(x)
        if not self.only_features:
            x = self.head(x)
        return x



class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if droupout == False:
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        else: 
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))

        self.layer4    = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer5    = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        out1 = self.layer1(x)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            out1, inds = self.layer4(out1)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            out1 = self.layer3(out1)
            out1, inds = self.layer4(out1)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            out1 = self.layer5(out1)
            return out1


class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        '''
        torch.nn.init.normal_(self.layer.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layerprob.weight, mean=0, std=1)
        '''

    def forward(self, x):
        #print('ClassifyBlock: ')
        #print('x shape: ', x.shape)
        out = self.layer(x)   
        #print('out shape: ', out.shape)
        #print('breakpoint 1:' )
        #breakpoint()
        #out = torch.permute(out, (0,3,1,2))
        #print('out shape: ', out.shape)
   #     #print('out shape: ', out.shape)
        #print('out[0,:,0,10]: ', out[0,:,0:2,10])
        #print('torch.sum(out[0,:,0,10]): ', torch.sum(out[0,:,0,10]))
        return out

class PSPhead(nn.Module):
    def __init__(self, input_dim=1024, output_dims=256, final_output_dims=1024, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool),
                nn.Conv2d(input_dim, output_dims, kernel_size=1),
                nn.BatchNorm2d(output_dims),
                nn.PReLU(num_parameters=1, init=0.25)
            ) for pool in pool_scales
        ])

        self.bottleneck = nn.Sequential(nn.Conv2d(input_dim + output_dims*len(pool_scales), final_output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_output_dims),
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))


    def forward(self, x):
        x = x.permute((0,3,1,2))
        ppm_outs = []
        ppm_outs.append(x)
        for ppm in self.ppm_modules:
            #ppm_out = Resize((x.shape[2], x.shape[3]), interpolation=InterpolationMode.BILINEAR)
            #print('ppm(x).shape: ', ppm(x).shape)
            ppm_out = F.interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        #print('ppm_outs shape: ', ppm_outs.shape)
        ppm_head_out = self.bottleneck(ppm_outs)
        #ppm_head_out = ppm_head_out.permute((0,2,3,1))
        #print('ppm_head_out shape: ', ppm_head_out.shape)
        return ppm_head_out

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[96, 192, 384, 768], fpn_out=256):
        super(FPN_fuse, self).__init__()
        
        # 1. 所有层都做 1x1 Conv 通道统一：从 feature_channels[i] -> fpn_out
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, fpn_out, kernel_size=1)
            for in_channels in feature_channels
        ])

        # 2. 每一层平滑卷积（除了最后一层不需要）
        self.smooth_conv = nn.ModuleList([
            nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
            for _ in range(len(feature_channels) - 1)
        ])

        # 3. 融合卷积
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: list of 4 tensors, each shape (B, C, H, W)，C=96/192/384/768
        
        # Step 1：通道统一到 fpn_out（例如 256）
        feats = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Step 2：FPN 自顶向下路径（从 P4 向 P1）
        for i in range(len(feats) - 1, 0, -1):
            upsample = F.interpolate(feats[i], size=feats[i - 1].shape[2:], mode='bilinear', align_corners=False)
            feats[i - 1] = feats[i - 1] + upsample  # top-down fuse
            feats[i - 1] = self.smooth_conv[i - 1](feats[i - 1])  # smooth

        # Step 3：上采样所有特征图到 P1 的大小后 concat 融合
        out_size = feats[0].shape[2:]
        feats = [F.interpolate(f, size=out_size, mode='bilinear', align_corners=False) for f in feats]
        out = self.conv_fusion(torch.cat(feats, dim=1))  # concat + fuse
        return out

class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes, bands=3, backbone='swinnet', use_aux=True, fpn_out=256, freeze_bn=False, head_out=128, **_):
        super(UperNet, self).__init__()

        if backbone == 'resnet50':
            feature_channels = [64, 128, 256, 512]
            self.backbone = resnet50(bands)
        elif backbone == 'swinnet':
            feature_channels = [96, 192, 384, 768]
            self.backbone = SwinTransformer(img_size=256,
                                patch_size=4,
                                in_chans=bands,
                                num_classes=16,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=8,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0,
                                drop_path_rate=0.1,
                                ape=False,
                                #norm_layer=layernorm,
                                patch_norm=True,
                                fused_window_process=False)

        else:
            feature_channels = [256, 512, 1024, 2048]

        #self.PPN = PSPModule(feature_channels[-1])
        self.PPMhead       = PSPhead(input_dim=768, output_dims=96, final_output_dims=768)
        self.FPN           = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head          = ConvLayer(fpn_out, head_out, kernel_size=3, stride=1, padding=1,  output=64, layertype=3, droupout=True)
        self.ClassifyBlock = ClassifyBlock(64, num_classes)
        #self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        #if freeze_bn: self.freeze_bn()
        #if freeze_backbone: 
        #    set_trainable([self.backbone], False)
        self.num_classes = num_classes


    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        for i in range(len(features)):
            h = int(np.sqrt(features[i].shape[1]))
            features[i] = features[i].view(features[i].shape[0], h, h, features[i].shape[2])
            if i != len(features) - 1:
               features[i] = features[i].permute(0,3,1,2)
            #print('after backbone, features i shape: ', features[i].shape)
        #print('features[-1] shape before PPMhead: ', features[-1].shape)
        features[-1] = self.PPMhead(features[-1])
        #print('features[-1] shape after PPMhead: ', features[-1].shape)
        x = self.FPN(features)
        #print('after FPN x.shape: ', x.shape)
        x = self.head(x)
        #print('after head ConvLayer x.shape: ', x.shape)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        #print('after interpolate x.shape: ', x.shape)
        x = self.ClassifyBlock(x)
        #print('after ClassifyBlock x.shape: ', x.shape)
        return x


if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = UperNet(bands=10, num_classes=3).to(device)
    x               = torch.randn(2, 10, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (10, 256, 256), batch_dim=0)