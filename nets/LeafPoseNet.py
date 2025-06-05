import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.nn import init
import matplotlib.pyplot as plt


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class convbnrelu(nn.Sequential):
    def __init__(self, inp, oup, ker=3, stride=1, groups=1):

        super(convbnrelu, self).__init__(
            nn.Conv2d(inp, oup, ker, stride, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

class SuperInvBottleneck(nn.Module):
    # 6 is the upperbound for expansion
    # 7 is the upperbound for kernel_size
    def __init__(self, inplanes, planes,  ker_size, expansion, stride=1,):
        super(SuperInvBottleneck, self).__init__()
        feature_dim = round(inplanes * expansion)
        layers = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True),

            nn.Conv2d(feature_dim, feature_dim, ker_size, stride, ker_size // 2, groups=feature_dim, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True),

            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.conv = nn.Sequential(*layers)
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.use_residual_connection:
            out = out + residual
        return out
    
class SuperSepConv2dv4(nn.Module):
    def __init__(self, inp, oup, ker=3, stride=1, c1 = True, c2 = True):
        super(SuperSepConv2dv4, self).__init__()
        if c1:
            gro1 = inp
        else:
            gro1 = 3
        
        if c2:
            gro2 = 1
        else:
            gro2 = 3

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, 1, 1, 0, groups=gro2, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace = True),

            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=gro1,  bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace = True),

            nn.Conv2d(inp, oup, 1, 1, 0, groups=gro2, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv(x)

class LeafPoseNet(nn.Module):
    def __init__(self, width_mult=1.0, round_nearest=2):
        super(LeafPoseNet, self).__init__()
        input_channel = 24
        inverted_residual_setting = [
            # t, c, n, s
            [6, 36, 4, 2],
            [6, 48, 6, 2],
            [6, 72, 6, 2]
        ]
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        
        self.first = nn.Sequential(                
            convbnrelu(3, 24, ker=3, stride=2),    # convbnrelu=Conv2d,BatchNorm2d,eLU6
            convbnrelu(24, 24, ker=3, stride=1, groups=24),  #convbnrelu=Conv2d,BatchNorm2d,eLU6
            nn.Conv2d(24, input_channel, 1, 1, 0, bias=False),   #Conv2d
            nn.BatchNorm2d(input_channel)                        #BatchNorm2d
        )

        # building inverted residual blocks
        self.stage = []
        for cnt in range(len(inverted_residual_setting)):  #cnt=0,1,2
            t, c, n, s = inverted_residual_setting[cnt]
            layer = []
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                layer.append(SuperInvBottleneck(input_channel, output_channel, ker_size=7, expansion=6, stride = stride ))
                input_channel = output_channel
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
        self.stage = nn.ModuleList(self.stage)

        self.dropout_1 = nn.ModuleList(
            [nn.Dropout2d(p=0.1),  #p=0.1 
             nn.Dropout2d(p=0.2),
             nn.Dropout2d(p=0.3),
             nn.Dropout2d(p=0.3)]
            )

        self.deconv_1 = nn.ModuleList(
            [SuperSepConv2dv4(120, 48),
             SuperSepConv2dv4(84, 36),
             SuperSepConv2dv4(60, 24)]
        )

        self.deconv_2 = nn.ModuleList(
            [SuperSepConv2dv4(84, 36),
             SuperSepConv2dv4(60, 24)]
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(60, 60, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace = True),
            nn.Conv2d(60, 60, 5, 1,2, groups=60,  bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace = True),
            nn.Conv2d(60, 48, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(48, 48, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True),
            nn.Conv2d(48, 48, 5, 1,2, groups=48,  bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True),
            nn.Conv2d(48, 24, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(24, 24, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace = True),
            nn.Conv2d(24, 24, 5, 1,2, groups=24,  bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace = True),
            nn.Conv2d(24, 3, 1, 1, 0, groups=1, bias=False),
            nn.Sigmoid(),)

        self.final_layer = nn.Sequential(
            nn.Conv2d(60, 60, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace = True),
            nn.Conv2d(60, 60, 5, 1,2, groups=60,  bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace = True),
            nn.Conv2d(60, 48, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(48, 48, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True),
            nn.Conv2d(48, 48, 5, 1,2, groups=48,  bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace = True),
            nn.Conv2d(48, 24, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(24, 24, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace = True),
            nn.Conv2d(24, 24, 5, 1,2, groups=24,  bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace = True),
            nn.Conv2d(24, 3, 1, 1, 0, groups=1, bias=False),)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        shape = [x.shape[2], x.shape[3]]
        x1 = self.first(x)
        x1 = self.dropout_1[0](x1)
        #print(f"x1 shape: {x1.shape}")
        x2 = self.stage[0](x1)       #4 blocks
        x2 = self.dropout_1[1](x2)  
        #print(f"x2 shape: {x2.shape}")
        x3 = self.stage[1](x2)        #6 blocks
        x3 = self.dropout_1[2](x3)   
        #print(f"x3 shape: {x3.shape}")
        x4 = self.stage[2](x3)           #6 blocks
        x4 = self.dropout_1[3](x4)   
        #print(f"x4 shape: {x4.shape}")
        x4 = F.interpolate(x4, size = [shape[0]//8, shape[1]//8], mode='nearest')
        x4 = self.deconv_1[0](torch.cat([x4, x3], dim=1))

        x3 = F.interpolate(x3, size = [shape[0]//4, shape[1]//4], mode='nearest')
        x3 = self.deconv_1[1](torch.cat([x2, x3], dim=1))

        x2 = F.interpolate(x2, size = [shape[0]//2, shape[1]//2], mode='nearest')
        x2 = self.deconv_1[2](torch.cat([x2, x1], dim=1))

        x4 = F.interpolate(x4, size = [shape[0]//4, shape[1]//4], mode='nearest')
        x4 = self.deconv_2[0](torch.cat([x3, x4], dim=1))

        x3 = F.interpolate(x3, size = [shape[0]//2, shape[1]//2], mode='nearest')
        x3 = self.deconv_2[1](torch.cat([x2, x3], dim=1))

        x4 = F.interpolate(x4, size = [shape[0]//2, shape[1]//2], mode='nearest')

        attention = self.spatial_attention(torch.cat([x3, x4], dim=1))
        final_outputs = self.final_layer(torch.cat([x3, x4], dim=1))

        final_outputs = final_outputs*attention  
        #print(f"Final output tensor shape: {final_outputs.shape}")        
        return final_outputs