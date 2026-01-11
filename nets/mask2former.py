import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, x):
        B, C, H, W = x.shape
        mask       = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)

        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * torch.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * torch.pi

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



class SimpleBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=1024):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)




class SimplePixelDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.position_embedding = PositionEmbeddingSine(out_channels // 2)

    def forward(self, x):
        features = self.conv(x)
        pos      = self.position_embedding(features)
        return features, pos



class MaskTransformerDecoder(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=6, num_queries=100, num_classes=21):
        super().__init__()
        self.num_queries        = num_queries
        self.query_embed        = nn.Embedding(num_queries, hidden_dim)
        self.transformer_layers = nn.ModuleList([
             nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads) for _ in range(num_layers)
             ])
        self.class_head         = nn.Linear(hidden_dim, num_classes)
        self.mask_embed_head    = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, pos_embed):
        B, C, H, W = x.shape
        src        = x.flatten(2).permute(2, 0, 1)
        pos        = pos_embed.flatten(2).permute(2, 0, 1)
        tgt        = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        for layer in self.transformer_layers:
            tgt = layer(tgt, src + pos)

        pred_logits = self.class_head(tgt)           # [N_queries, B, num_classes]
        pred_masks  = self.mask_embed_head(tgt)       # [N_queries, B, C]
        return pred_logits.transpose(0, 1), pred_masks.transpose(0, 1)



class Mask2Former(nn.Module):
    def __init__(self,bands=3, num_classes=21):
        super().__init__()
        self.backbone            = SimpleBackbone(in_channels=bands, out_channels=2048)
        self.pixel_decoder       = SimplePixelDecoder(in_channels=2048, out_channels=256)
        self.transformer_decoder = MaskTransformerDecoder(hidden_dim=256,
                                                          num_heads=8,
                                                          num_layers=6,
                                                          num_queries=100,
                                                          num_classes=num_classes)

    def forward(self, x):
        B, C, H, W               = x.shape
        feats                    = self.backbone(x)
        features, pos            = self.pixel_decoder(feats)
        class_logits, mask_embed = self.transformer_decoder(features, pos)
        mask_logits              = torch.einsum("bqc, bchw -> bqhw", mask_embed, features)  # [B, Q, H, W]

        class_probs              = F.softmax(class_logits, dim=-1)  # [B, Q, C]
        mask_probs               = mask_logits.sigmoid()             # [B, Q, H, W]
        segmentation_logits      = torch.einsum("bqc, bqhw -> bchw", class_probs, mask_probs)  # [B, C, H, W]
        segmentation_logits      = F.interpolate(segmentation_logits, size=(H, W), mode='bilinear', align_corners=True)
        return segmentation_logits

# ------------------------
# Test Forward
# ------------------------
if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = Mask2Former(bands=6,  num_classes=3).to(device)
    x               = torch.randn(2, 6, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs(G): ', (flops/1e9/x.shape[0]), 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)
