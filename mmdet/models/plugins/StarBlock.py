import torch
import torch.nn as nn

class ConvBn(nn.Sequential):
    def __init__(self, inplanes, outplanes, kernel_size, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv',nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn',nn.BatchNorm2d(outplanes))
            nn.init.constant_(self.bn.weight,1)
            nn.init.constant_(self.bn.bias,0)
    
class StarBlock(nn.Module):
    
    def __init__(self, inplanes, outplanes, mlp_ratio=2, stride=1) -> None:
        super().__init__()
        self.f1 = ConvBn(inplanes, inplanes*mlp_ratio, 1, with_bn=False)
        self.f2 = ConvBn(inplanes, inplanes*mlp_ratio, 1, with_bn=False)
        self.g = ConvBn(mlp_ratio * inplanes, inplanes, 1, with_bn=True)
        self.act = nn.ReLU6()
    
    def forward(self, x):
        input = x
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.g(x)
        x += input        
        return x