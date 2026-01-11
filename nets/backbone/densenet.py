# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2025/6/29  20:35
# @Function   : DenseNet
# @Description: Realization of DenseNet network

import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size, drop_rate)
            self.layers.append(layer)

    def forward(self, init_features):
        features = init_features
        for layer in self.layers:
            features = layer(features)
        return features

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, in_chans=3, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        num_features = num_init_features
        self.blocks  = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.blocks.append(block)
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2  # 压缩50%
                )
                self.blocks.append(trans)
                num_features = num_features // 2
        
        self.final_bn   = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        

    def forward(self, x):
        x = self.features(x)
        for block in self.blocks:
            x = block(x)
        
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def densenet121(inchans=3, num_classes=1000, **kwargs):
    return DenseNet(in_chans          = inchans, 
	                growth_rate       = 32,
					block_config      = (6,12,24,16),
					num_init_features = 64,
					num_classes       = num_classes,
					**kwargs)

def densenet161(inchans=3, num_classes=1000, **kwargs):
    return DenseNet(in_chans          = inchans, 
	                growth_rate       = 48,
					block_config      = (6,12,36,24),
					num_init_features = 64,
					num_classes       = num_classes,
					**kwargs)

def densenet169(inchans=3, num_classes=1000, **kwargs):
    return DenseNet(in_chans          = inchans, 
	                growth_rate       = 32,
					block_config      = (6,12,32,32),
					num_init_features = 64,
					num_classes       = num_classes,
					**kwargs)

def densenet201(inchans=3, num_classes=1000, **kwargs):
    return DenseNet(in_chans          = inchans, 
	                growth_rate       = 32,
					block_config      = (6,12,48,32),
					num_init_features = 64,
					num_classes       = num_classes,
					**kwargs)

def densenet264(inchans=3, num_classes=1000, **kwargs):
    return DenseNet(in_chans          = inchans, 
	                growth_rate       = 32,
					block_config      = (6,12,64,48),
					num_init_features = 64,
					num_classes       = num_classes,
					**kwargs)
# 示例使用
if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = densenet264(inchans=3, num_classes=21).to(device)
    x               = torch.randn(2, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9), 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 224, 224), batch_dim=0)