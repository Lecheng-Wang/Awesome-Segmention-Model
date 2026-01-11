# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/5/15} ${20:36}
# @Function: function to initial parameters in model

import torch
import torch.nn as nn

def weights_init(net, init_type='kaiming', init_gain=0.02, a=0, slope=0):
    def init_func(m):
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=a, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialize method: [{init_type}] is not supported!')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, slope)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, slope)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    print(f'initialize network with {init_type.upper()} type.')
    net.apply(init_func)
    return net